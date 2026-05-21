"""Curve SAE vs vanilla TopK SAE on real LM residual-stream activations.

Three-phase orchestration to dodge Colab's dual-CUDA-stack issue
(both system and torch-bundled cuBLAS mapped → gamfit refuses to load):

  1. parent process (GPU OK): harvest residual-stream activations from
     gpt2, train vanilla TopK SAE, dump activations + vanilla results.
  2. parent process spawns a subprocess with CUDA_VISIBLE_DEVICES=''
     so the child's torch doesn't map CUDA libs at all. The child
     loads gamfit cleanly, trains the curve SAE on CPU, dumps results.
  3. parent merges the two result files and prints the head-to-head.

Run from Colab with::

    !cd /content/Manifold-SAE && python -m experiments.llm_real

or directly::

    !cd /content/Manifold-SAE && python -m experiments.llm_real --phase=curve_only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn


@dataclass
class Config:
    model_name: str = "gpt2"
    layer: int = 6
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
    n_steps_curve: int = 2000  # curve SAE on CPU is slow; fewer steps
    batch_size: int = 1024
    batch_size_curve: int = 256
    lr: float = 1e-3
    sae_n_basis: int = 10
    sae_intrinsic_rank: int = 2
    sae_sparsity_weight: float = 3e-4
    sae_ortho_weight: float = 1e-3
    eval_n: int = 8192
    output_dir: str = "runs/LLM_REAL"
    seed: int = 0


# ---------------------------------------------------------------------------
# Activation harvest (parent, GPU)
# ---------------------------------------------------------------------------


def harvest_activations(cfg: Config, device: torch.device) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    print(f"[harvest] loading {cfg.model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()

    print(f"[harvest] loading text: {cfg.text_dataset}/{cfg.text_subset}", flush=True)
    ds = load_dataset(cfg.text_dataset, cfg.text_subset, split=cfg.text_split)
    texts = [t for t in ds["text"] if len(t) > cfg.min_text_len][: cfg.max_texts]

    chunks = []
    n_collected = 0
    torch.set_grad_enabled(False)
    print(f"[harvest] collecting {cfg.n_tokens:,} tokens at layer {cfg.layer}", flush=True)
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
        out = model(**batch, output_hidden_states=True)
        h = out.hidden_states[cfg.layer]
        mask = batch["attention_mask"].bool()
        chunks.append(h[mask].cpu().float())
        n_collected += int(mask.sum())
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
# Training loops
# ---------------------------------------------------------------------------


def train_vanilla(sae, X, cfg: Config, label: str = "vanilla") -> float:
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    n = X.shape[0]
    t0 = time.time()
    log_every = max(cfg.n_steps // 10, 1)
    for step in range(cfg.n_steps):
        idx = torch.randint(0, n, (cfg.batch_size,), device=X.device)
        batch = X[idx]
        opt.zero_grad(set_to_none=True)
        recon, z = sae(batch)
        mse = ((recon - batch) ** 2).mean()
        loss = mse + 3e-4 * z.abs().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"  [{label} step {step:5d}] mse={mse.item():.4e}", flush=True)
    return time.time() - t0


def train_curve(sae, X, cfg: Config, sae_cfg, label: str = "curve") -> float:
    from manifold_sae.losses import total_loss

    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    n = X.shape[0]
    t0 = time.time()
    log_every = max(cfg.n_steps_curve // 10, 1)
    for step in range(cfg.n_steps_curve):
        idx = torch.randint(0, n, (cfg.batch_size_curve,), device=X.device)
        batch = X[idx]
        opt.zero_grad(set_to_none=True)
        out = sae(batch)
        losses = total_loss(out, batch, sae_cfg)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"  [{label} step {step:5d}] mse={losses['mse'].item():.4e}", flush=True)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def run_gpu_phase(cfg: Config, out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu-phase] device={device}", flush=True)

    X = harvest_activations(cfg, device)
    mu = X.mean(0, keepdim=True)
    sigma = float(X.std().item())
    X_n = (X - mu) / max(sigma, 1e-6)
    var = float(X_n.var().item())
    torch.save({"X_n": X_n, "var": var, "mu": mu, "sigma": sigma}, out_dir / "activations.pt")
    print(f"[gpu-phase] saved activations to {out_dir / 'activations.pt'}", flush=True)

    print("\n[vanilla] training", flush=True)
    vanilla = VanillaSAE(X_n.shape[1], cfg.n_features, cfg.top_k).to(device)
    X_n_dev = X_n.to(device)
    t_v = train_vanilla(vanilla, X_n_dev, cfg)

    print("\n[vanilla] evaluating", flush=True)
    eval_batch = X_n_dev[: min(cfg.eval_n, X_n_dev.shape[0])]
    with torch.no_grad():
        recon_v, z_v = vanilla(eval_batch)
        mse_v = float(((recon_v - eval_batch) ** 2).mean())
        alive_v = int((z_v > 0).any(0).sum())

    print(f"[vanilla] MSE={mse_v:.4f}  expl={1-mse_v/var:.3f}  alive={alive_v}/{cfg.n_features}", flush=True)
    return {
        "var": var,
        "vanilla": {"mse": mse_v, "explained": 1 - mse_v / var, "alive": alive_v, "train_seconds": t_v, "n_features": cfg.n_features},
    }


def run_curve_phase(cfg: Config, out_dir: Path) -> dict:
    """Runs in a subprocess with CUDA_VISIBLE_DEVICES='' so torch doesn't
    map CUDA libs and gamfit's Rust loader doesn't trip its dual-stack
    safety check. Loads activations from disk, trains curve SAE on CPU.
    """
    device = torch.device("cpu")
    print(f"[curve-phase] device={device} (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}')", flush=True)

    data = torch.load(out_dir / "activations.pt", weights_only=False)
    X_n = data["X_n"]
    var = float(data["var"])
    D = X_n.shape[1]

    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    sae_cfg = ManifoldSAEConfig(
        input_dim=D,
        n_features=cfg.n_features,
        n_basis=cfg.sae_n_basis,
        top_k=cfg.top_k,
        intrinsic_rank=cfg.sae_intrinsic_rank,
        sparsity_weight=cfg.sae_sparsity_weight,
        ortho_weight=cfg.sae_ortho_weight,
        encoder_type="linear",
        continuous_amp=True,
    )
    curve = ManifoldSAE(sae_cfg).to(device)
    print(f"[curve] training ({cfg.n_steps_curve} steps, batch={cfg.batch_size_curve})", flush=True)
    t_c = train_curve(curve, X_n, cfg, sae_cfg)

    eval_batch = X_n[: min(cfg.eval_n, X_n.shape[0])]
    print("[curve] evaluating", flush=True)
    with torch.no_grad():
        out_c = curve(eval_batch)
        mse_c = float(((out_c.reconstruction - eval_batch) ** 2).mean())
        alive_c = int((out_c.amplitudes > 1e-3).any(0).sum())

    print("[curve] lock-and-cache", flush=True)
    curve.update_snapshot(eval_batch)
    curve.inference_mode = True
    with torch.no_grad():
        out_inf = curve(eval_batch)
        mse_inf = float(((out_inf.reconstruction - eval_batch) ** 2).mean())

    print(f"[curve] MSE={mse_c:.4f}  expl={1-mse_c/var:.3f}  alive={alive_c}/{cfg.n_features}  locked_mse={mse_inf:.4f}", flush=True)
    return {
        "curve": {"mse": mse_c, "explained": 1 - mse_c / var, "alive": alive_c, "train_seconds": t_c, "locked_mse": mse_inf, "n_features": cfg.n_features},
    }


# ---------------------------------------------------------------------------
# Main: orchestrate phases
# ---------------------------------------------------------------------------


def main(cfg: Config | None = None, phase: str = "auto") -> int:
    if cfg is None:
        cfg = Config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] phase={phase}  output_dir={out_dir}", flush=True)

    if phase == "curve_only":
        # Subprocess invocation. CUDA_VISIBLE_DEVICES is set externally.
        report = run_curve_phase(cfg, out_dir)
        (out_dir / "curve_results.json").write_text(json.dumps(report, indent=2, default=float))
        return 0

    # phase == "auto": run GPU phase, then spawn CPU subprocess for curve.
    gpu_report = run_gpu_phase(cfg, out_dir)
    (out_dir / "gpu_results.json").write_text(json.dumps(gpu_report, indent=2, default=float))

    print("\n[main] spawning curve-SAE subprocess (CPU, no CUDA libs)", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-m", "experiments.llm_real", "--phase=curve_only"],
        env=env, check=False,
    )
    if result.returncode != 0:
        print(f"[main] curve subprocess failed (exit {result.returncode})", flush=True)
        return result.returncode

    curve_report = json.loads((out_dir / "curve_results.json").read_text())
    full = {"config": asdict(cfg), **gpu_report, **curve_report}
    (out_dir / "results.json").write_text(json.dumps(full, indent=2, default=float))

    print("\n========== FINAL ==========", flush=True)
    print(json.dumps(full, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="auto", choices=["auto", "curve_only"])
    args = parser.parse_args()
    sys.exit(main(phase=args.phase))
