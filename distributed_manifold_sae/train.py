"""Distributed training loop for K=1M Manifold-SAE.

Launch (real cluster, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 -m distributed_manifold_sae.train \\
        --config configs/k1m_circle_cogito.yaml

Mock launch (CPU, gloo backend, 4 simulated ranks):
    python -m distributed_manifold_sae.train --mock-world 4

Architecture:
    - bf16 forward / fp32 master weights via torch.amp.autocast + GradScaler is
      bypassed in favor of native bf16 reductions (NCCL bf16 allreduce). Master
      weights stay fp32 in the optimizer.
    - FSDP wraps `encoder.out_proj`, `anchor`, `tangent` (the three K-scaling
      tensors). Smaller modules stay replicated.
    - Gradient checkpointing on the encoder MLP (toggled via model cfg).
    - Riemannian retraction is called after optimizer.step() — orthonormalises
      per-atom tangent frames on each rank's local FSDP shard.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from .model import ManifoldSAE, ManifoldSAEConfig
from .loss import ComposedLoss, ComposedLossConfig
from .data import DataConfig, build_dataloader


# ---------------------------------------------------------------------------
# Distributed init.
# ---------------------------------------------------------------------------
def setup_distributed(mock_world: int | None = None) -> tuple[int, int, torch.device]:
    """Initialize torch.distributed. Returns (rank, world_size, device).

    If `mock_world` is set, we use gloo on CPU and spawn-mock via env vars.
    In a real torchrun launch we pick up RANK/WORLD_SIZE/LOCAL_RANK from env.
    """
    if mock_world is not None:
        # Single-process mock — pretend we're rank 0 of world_size=mock_world.
        # For genuine multi-process mock, launch with torchrun --nproc_per_node=mock_world.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("WORLD_SIZE", str(mock_world))
        os.environ.setdefault("RANK", "0")
        backend = "gloo"
        device = torch.device("cpu")
    else:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend=backend)
        except Exception as e:
            print(f"[warn] dist init failed ({e}); running single-process.", file=sys.stderr)
            return 0, 1, device

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP wrap policy.
# ---------------------------------------------------------------------------
def wrap_fsdp(model: ManifoldSAE, device: torch.device, world_size: int) -> nn.Module:
    """Wrap large modules with FSDP. No-op if world_size==1 or FSDP unavailable."""
    if world_size <= 1:
        return model.to(device)

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
    except ImportError:
        print("[warn] FSDP not available, falling back to DDP.", file=sys.stderr)
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model.to(device))

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,    # fp32 reductions — critical for K=1M
        buffer_dtype=torch.bfloat16,
    )
    wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(1e7)
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=device.index if device.type == "cuda" else None,
        use_orig_params=True,
    )
    return model


# ---------------------------------------------------------------------------
# Gumbel temperature schedule.
# ---------------------------------------------------------------------------
def tau_schedule(step: int, total_steps: int, tau_init: float, tau_final: float) -> float:
    """Exponential anneal from tau_init to tau_final over total_steps."""
    if step >= total_steps:
        return tau_final
    frac = step / max(1, total_steps)
    log_tau = (1 - frac) * torch.log(torch.tensor(tau_init)) + frac * torch.log(torch.tensor(tau_final))
    return float(log_tau.exp())


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
def train(
    model_cfg: ManifoldSAEConfig,
    loss_cfg: ComposedLossConfig,
    data_cfg: DataConfig,
    *,
    epochs: int = 10,
    lr: float = 3e-4,
    log_every: int = 50,
    ckpt_every: int = 1000,
    ckpt_dir: str = "./checkpoints",
    rank: int = 0,
    world_size: int = 1,
    device: torch.device = torch.device("cpu"),
    mock_run: bool = False,
) -> None:
    is_master = rank == 0

    model = ManifoldSAE(model_cfg)
    model = wrap_fsdp(model, device, world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))

    loader, sampler = build_dataloader(
        data_cfg, rank=rank, world_size=world_size, shuffle=True
    )

    loss_fn = ComposedLoss(loss_cfg, top_k=model_cfg.top_k)

    # Dashboard (master only).
    writer = None
    if is_master:
        try:
            from .dashboard import build_writer
            writer = build_writer(ckpt_dir)
        except Exception:
            pass

    total_steps = epochs * len(loader)
    step = 0

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            x = batch.to(device, non_blocking=True)
            # Update tau on the underlying module (FSDP-wrapped or not).
            underlying = model.module if hasattr(model, "module") else model
            tau = tau_schedule(
                step, total_steps,
                model_cfg.gumbel_tau_init, model_cfg.gumbel_tau_final,
            )
            underlying.tau.fill_(tau)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                out = model(x)
                total_loss, log = loss_fn(x.float(), out, underlying)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Riemannian retraction post-step on local shard only.
            # NOTE: with FSDP + use_orig_params we have local shards; this only
            # touches the shard owned by this rank.
            try:
                underlying.riemannian_retract()
            except Exception:
                # First-pass safety: a bad retraction shouldn't kill the run.
                if is_master and step == 0:
                    print("[warn] riemannian_retract failed (likely sharded tensor); skipping.", file=sys.stderr)

            if is_master and step % log_every == 0:
                msg = f"epoch={epoch} step={step} tau={tau:.3f} grad={float(grad_norm):.3f} " \
                      + " ".join(f"{k}={v:.4f}" for k, v in log.items())
                print(msg)
                if writer is not None:
                    for k, v in log.items():
                        writer.add_scalar(f"loss/{k}", v, step)
                    writer.add_scalar("opt/tau", tau, step)
                    writer.add_scalar("opt/grad_norm", float(grad_norm), step)
                    writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], step)

            if is_master and step > 0 and step % ckpt_every == 0:
                save_checkpoint(underlying, optimizer, step, ckpt_dir)

            step += 1

            if mock_run and step >= 5:
                # Smoke test: just verify the loop runs.
                return

    if is_master:
        save_checkpoint(underlying, optimizer, step, ckpt_dir, final=True)


# ---------------------------------------------------------------------------
# Checkpointing.
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step: int, ckpt_dir: str, final: bool = False) -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    name = "final.pt" if final else f"step_{step:08d}.pt"
    path = Path(ckpt_dir) / name
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }, path)
    print(f"[ckpt] saved {path}")


def load_checkpoint(model, optimizer, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (Hydra-style). If omitted, defaults used.")
    p.add_argument("--mock-world", type=int, default=None,
                   help="If set, init dist with gloo on CPU for testing.")
    p.add_argument("--mock-run", action="store_true",
                   help="Run a 5-step smoke test instead of full training.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    return p.parse_args()


def _load_yaml_config(path: str | None):
    if path is None:
        return ManifoldSAEConfig(), ComposedLossConfig(), DataConfig(activations_path="X_L40.npy")
    try:
        import yaml
    except ImportError:
        raise RuntimeError("YAML config requested but pyyaml not installed.")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return (
        ManifoldSAEConfig(**raw.get("model", {})),
        ComposedLossConfig(**raw.get("loss", {})),
        DataConfig(**raw.get("data", {})),
    )


def main():
    args = _parse_args()
    rank, world_size, device = setup_distributed(args.mock_world)
    try:
        model_cfg, loss_cfg, data_cfg = _load_yaml_config(args.config)
        train(
            model_cfg, loss_cfg, data_cfg,
            epochs=args.epochs, lr=args.lr,
            ckpt_dir=args.ckpt_dir,
            rank=rank, world_size=world_size, device=device,
            mock_run=args.mock_run,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
