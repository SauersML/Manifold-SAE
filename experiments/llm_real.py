"""Curve SAE vs vanilla TopK SAE on real LM residual-stream activations.

Harvests activations from gpt2-small (or any HF causal LM) on a text
corpus, trains both architectures at matched dictionary size, reports
explained variance + alive features + lock-and-cache feedforward check.

Run from a Colab T4 cell as::

    !cd /content/Manifold-SAE && python -m experiments.llm_real

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
    batch_size: int = 1024
    lr: float = 1e-3
    sae_n_basis: int = 10
    sae_intrinsic_rank: int = 2
    sae_sparsity_weight: float = 3e-4
    sae_ortho_weight: float = 1e-3
    eval_n: int = 8192
    output_dir: str = "runs/LLM_REAL"
    seed: int = 0


# ---------------------------------------------------------------------------
# Activation harvest
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
    torch.cuda.empty_cache() if device.type == "cuda" else None
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


def train_sae(sae, X, cfg: Config, label: str, is_curve: bool = False, sae_cfg=None) -> float:
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    n = X.shape[0]
    t0 = time.time()
    log_every = max(cfg.n_steps // 10, 1)
    for step in range(cfg.n_steps):
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
    return time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config = Config()) -> int:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = harvest_activations(cfg, device)
    mu = X.mean(0, keepdim=True)
    sigma = float(X.std().item())
    X_n = ((X - mu) / max(sigma, 1e-6)).to(device)
    D = X_n.shape[1]
    var = float(X_n.var().item())

    print("\n[vanilla] training", flush=True)
    vanilla = VanillaSAE(D, cfg.n_features, cfg.top_k).to(device)
    t_v = train_sae(vanilla, X_n, cfg, "vanilla", is_curve=False)

    print("\n[curve] training (gamfit REML each batch)", flush=True)
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
    t_c = train_sae(curve, X_n, cfg, "curve", is_curve=True, sae_cfg=sae_cfg)

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
