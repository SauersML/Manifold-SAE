"""Curve SAE vs vanilla TopK SAE on real LM residual-stream activations.

Single-process, both phases on GPU. The gamfit dual-cuBLAS conflict on
Colab is resolved upstream via LD_PRELOAD in colab_run.txt — once only
one cuBLAS is mapped in the process, gamfit's safety check passes and
its CUDA dispatch can run REML on the GPU.

Run from a Colab T4 cell as::

    LD_PRELOAD=<pip cublas> python -m experiments.llm_real

or import and call ``main(Config(...))`` to override defaults.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn


# ---------------------------------------------------------------------------
# Transitional bridge: bypass gamfit's CUDA dual-stack safety check on
# environments (Colab, several cloud images) that have both /usr/local/cuda*
# and the torch-bundled nvidia/cublas-cu12 reachable. The proper fix is
# upstream in gam (commit downgrades the assert to a once-per-process
# warning). Until a new gamfit wheel containing that fix lands on PyPI,
# this shim no-ops the check so `pip install gamfit` from PyPI keeps working.
def _bypass_gamfit_cuda_check() -> None:
    import gamfit._cuda as _gc

    _gc.assert_no_cuda_library_conflicts = lambda context: None
    try:
        import gamfit._binding as _gb

        _gb.assert_no_cuda_library_conflicts = lambda context: None
        if hasattr(_gb.rust_module, "cache_clear"):
            _gb.rust_module.cache_clear()
    except ImportError:
        pass


_bypass_gamfit_cuda_check()
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # Default: Qwen2.5-0.5B (pure text causal LM, Apache 2.0, 24 layers,
    # hidden_size=896, Qwen2 tokenizer). Verified to work with AutoModel +
    # standard forward hooks. Qwen3.5 is multimodal and behaves differently.
    model_name: str = "Qwen/Qwen2.5-0.5B"
    layer: int = 12  # mid for 24-layer Qwen2.5-0.5B
    n_tokens: int = 200_000
    seq_len: int = 256
    text_dataset: str = "wikitext"
    text_subset: str = "wikitext-2-raw-v1"
    text_split: str = "train"
    min_text_len: int = 200
    max_texts: int = 4000
    n_features: int = 4096
    top_k: int = 32
    n_steps: int = 4000
    batch_size: int = 1024
    lr: float = 1e-3
    sae_n_basis: int = 10
    sae_intrinsic_rank: int = 2
    sae_sparsity_weight: float = 3e-4
    sae_ortho_weight: float = 1e-3
    eval_n: int = 8192
    # Use an absolute path outside the repo so checkpoints survive
    # `rm -rf /content/Manifold-SAE` in the Colab setup cell.
    output_dir: str = "/content/runs/LLM_REAL"
    seed: int = 0
    # Warm-start: if a checkpoint exists at <output_dir>/checkpoint.pt with
    # a matching config, resume (model + optimizer + step count). Set
    # resume=False to force a fresh run.
    resume: bool = True


# ---------------------------------------------------------------------------
# Activation harvest
# ---------------------------------------------------------------------------


def harvest_activations(cfg: Config, device: torch.device) -> torch.Tensor:
    """Collect residual-stream activations from one layer using a forward
    hook on that block's output. Avoids ``output_hidden_states=True`` which
    materializes all 25 layers' activations in memory per batch.
    """
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    print(f"[harvest] loading {cfg.model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()

    # Find the transformer-block module list and pick our target layer.
    # Supports gpt2 (.h[i]), Qwen2/Qwen3/Llama/Mistral (.layers[i]).
    blocks = None
    for attr in ("h", "layers", "encoder_layer"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            break
    if blocks is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    if blocks is None:
        raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")
    if cfg.layer < 0 or cfg.layer >= len(blocks):
        raise ValueError(f"layer {cfg.layer} out of range for model with {len(blocks)} blocks")

    captured = {}

    def hook(_module, _inputs, output):
        # Block outputs are typically a tuple (hidden_state, ...) or a single tensor.
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = blocks[cfg.layer].register_forward_hook(hook)

    print(f"[harvest] loading text: {cfg.text_dataset}/{cfg.text_subset}", flush=True)
    ds = load_dataset(cfg.text_dataset, cfg.text_subset, split=cfg.text_split)
    texts = [t for t in ds["text"] if len(t) > cfg.min_text_len][: cfg.max_texts]

    chunks = []
    n_collected = 0
    torch.set_grad_enabled(False)
    print(f"[harvest] collecting {cfg.n_tokens:,} tokens at layer {cfg.layer} (block {cfg.layer} of {len(blocks)})", flush=True)
    try:
        for i in range(0, len(texts), 8):
            if n_collected >= cfg.n_tokens:
                break
            batch = tok(
                texts[i : i + 8],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.seq_len,
            ).to(device)
            model(**batch)
            h = captured["h"]
            mask = batch["attention_mask"].bool()
            chunks.append(h[mask].cpu().float())
            n_collected += int(mask.sum())
    finally:
        handle.remove()
    torch.set_grad_enabled(True)
    X = torch.cat(chunks, dim=0)[: cfg.n_tokens]
    print(f"[harvest] X shape: {tuple(X.shape)}  mean={X.mean():.3f}  std={X.std():.3f}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return X


# ---------------------------------------------------------------------------
# Vanilla TopK SAE baseline
# ---------------------------------------------------------------------------


class VanillaSAE(nn.Module):
    def __init__(self, D: int, F: int, top_k: int) -> None:
        super().__init__()
        self.F = F
        self.top_k = top_k
        H = 4 * D
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H)
        self.act = nn.GELU()
        self.head = nn.Linear(H, F)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        vals, idx = torch.topk(z, self.top_k, dim=1)
        gate = torch.zeros_like(z).scatter_(1, idx, vals)
        return gate @ self.W_dec, gate


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _ckpt_signature(cfg: Config, label: str) -> dict:
    return {
        "label": label,
        "model_name": cfg.model_name,
        "layer": cfg.layer,
        "n_features": cfg.n_features,
        "top_k": cfg.top_k,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "sae_n_basis": cfg.sae_n_basis,
        "sae_intrinsic_rank": cfg.sae_intrinsic_rank,
    }


def train_sae(sae, X, cfg: Config, label: str, ckpt_path: Path, is_curve: bool = False, sae_cfg=None) -> float:
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    start_step = 0
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sig = _ckpt_signature(cfg, label)
        if ckpt.get("sig") == sig:
            sae.load_state_dict(ckpt["sae"])
            opt.load_state_dict(ckpt["opt"])
            start_step = int(ckpt["step"])
            print(f"  [{label}] resumed from {ckpt_path} at step {start_step}", flush=True)
        else:
            print(f"  [{label}] checkpoint exists but signature changed — starting fresh", flush=True)

    n = X.shape[0]
    t0 = time.time()
    log_every = max(cfg.n_steps // 10, 1)
    if start_step >= cfg.n_steps:
        print(f"  [{label}] already trained for {start_step} steps (target {cfg.n_steps}); skipping", flush=True)
        return 0.0
    for step in range(start_step, cfg.n_steps):
        idx = torch.randint(0, n, (cfg.batch_size,), device=X.device)
        batch = X[idx]
        opt.zero_grad(set_to_none=True)
        if is_curve:
            from manifold_sae.losses import total_loss
            out = sae(batch)
            losses = total_loss(out, batch, sae_cfg)
            loss = losses["total"]
            mse = losses["mse"]
        else:
            recon, z = sae(batch)
            mse = ((recon - batch) ** 2).mean()
            loss = mse + 3e-4 * z.abs().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"  [{label} step {step:5d}] mse={mse.item():.4e}", flush=True)
    torch.save(
        {"sae": sae.state_dict(), "opt": opt.state_dict(), "step": cfg.n_steps, "sig": _ckpt_signature(cfg, label)},
        ckpt_path,
    )
    print(f"  [{label}] saved checkpoint to {ckpt_path}", flush=True)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config | None = None) -> int:
    if cfg is None:
        cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device}  output_dir={out_dir}", flush=True)

    # Cache harvested activations so re-runs are fast.
    act_path = out_dir / "activations.pt"
    act_sig = {"model_name": cfg.model_name, "layer": cfg.layer, "n_tokens": cfg.n_tokens,
               "text_dataset": cfg.text_dataset, "text_subset": cfg.text_subset, "seq_len": cfg.seq_len}
    if cfg.resume and act_path.exists():
        cached = torch.load(act_path, map_location="cpu", weights_only=False)
        if cached.get("sig") == act_sig:
            X = cached["X"]
            print(f"[harvest] loaded cached activations from {act_path}: shape={tuple(X.shape)}", flush=True)
        else:
            X = harvest_activations(cfg, device)
            torch.save({"X": X, "sig": act_sig}, act_path)
    else:
        X = harvest_activations(cfg, device)
        torch.save({"X": X, "sig": act_sig}, act_path)
    mu = X.mean(0, keepdim=True)
    sigma = float(X.std().item())
    X_n = ((X - mu) / max(sigma, 1e-6)).to(device)
    var = float(X_n.var().item())
    D = X_n.shape[1]

    print("\n[vanilla] training", flush=True)
    vanilla = VanillaSAE(D, cfg.n_features, cfg.top_k).to(device)
    t_v = train_sae(vanilla, X_n, cfg, "vanilla", out_dir / "vanilla_ckpt.pt", is_curve=False)

    print("\n[curve] training (gamfit REML each batch, GPU)", flush=True)
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    sae_cfg = ManifoldSAEConfig(
        input_dim=D, n_features=cfg.n_features, n_basis=cfg.sae_n_basis,
        top_k=cfg.top_k, intrinsic_rank=cfg.sae_intrinsic_rank,
        sparsity_weight=cfg.sae_sparsity_weight, ortho_weight=cfg.sae_ortho_weight,
        encoder_type="linear", continuous_amp=True,
    )
    curve = ManifoldSAE(sae_cfg).to(device)
    t_c = train_sae(curve, X_n, cfg, "curve", out_dir / "curve_ckpt.pt", is_curve=True, sae_cfg=sae_cfg)

    print("\n[eval]", flush=True)
    eval_batch = X_n[: min(cfg.eval_n, X_n.shape[0])]
    with torch.no_grad():
        recon_v, z_v = vanilla(eval_batch)
        mse_v = float(((recon_v - eval_batch) ** 2).mean())
        alive_v = int((z_v > 0).any(0).sum())
        out_c = curve(eval_batch)
        mse_c = float(((out_c.reconstruction - eval_batch) ** 2).mean())
        alive_c = int((out_c.amplitudes > 1e-3).any(0).sum())
    print(f"vanilla: MSE={mse_v:.4f}  expl={1-mse_v/var:.3f}  alive={alive_v}/{cfg.n_features}  train_s={t_v:.0f}", flush=True)
    print(f"curve  : MSE={mse_c:.4f}  expl={1-mse_c/var:.3f}  alive={alive_c}/{cfg.n_features}  train_s={t_c:.0f}", flush=True)

    print("\n[snapshot] lock-and-cache the curve SAE", flush=True)
    curve.update_snapshot(eval_batch)
    curve.inference_mode = True
    with torch.no_grad():
        out_inf = curve(eval_batch)
        mse_inf = float(((out_inf.reconstruction - eval_batch) ** 2).mean())
    print(f"locked snapshot recon MSE={mse_inf:.4f}", flush=True)

    report = {
        "config": asdict(cfg),
        "var": var,
        "vanilla": {"mse": mse_v, "explained": 1 - mse_v / var, "alive": alive_v, "train_seconds": t_v},
        "curve": {"mse": mse_c, "explained": 1 - mse_c / var, "alive": alive_c, "train_seconds": t_c, "locked_mse": mse_inf},
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out_dir / 'results.json'}", flush=True)
    print(json.dumps(report, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
