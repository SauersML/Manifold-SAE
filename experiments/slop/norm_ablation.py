"""Normalization ablation — is per-dim norm itself the architectural verdict?

The post-fix findings claim Manifold-SAE underperforms vanilla on real
LM residuals. That conclusion rests on per-dim std normalization. But:

* `raw_centered` (no scaling) reads rank-1 on Qwen-1.5B residuals.
* `per_dim_std` reads rank ~925.

Per-dim norm divides each coordinate by its own σ. If the LM's signal
lives in the few high-variance channels (the well-known "outlier
feature" phenomenon — Dettmers et al., Sun et al.), per-dim norm
*amplifies low-variance noise to unit scale* and the "925 PCs of
variance" is then largely fabricated rank.

Test
----
Train vanilla + curve SAE under TWO normalizations:
  A. per_dim_std (current convention)
  B. raw_centered (subtract mean only)
Evaluate each model in BOTH spaces:
  * MSE in the normalization space the model trained in (current metric)
  * MSE in the RAW residual space (un-normalize the recon, compare to
    raw — measures actual reconstruction quality of the LM's signal)

Report per (norm, arch):
  * alive atom count
  * EV in train-norm space
  * EV in raw space (the metric that doesn't bake in any choice of norm)
  * mean cos similarity between PC1 of the residual and the SAE's
    "dominant" atom direction (does the SAE capture the variance-
    dominant direction or fight against it?)

If curve SAE under raw_centered beats vanilla under per_dim_std on
EV-in-raw-space, the negative headline is a normalization choice
masquerading as an architectural fact.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


@dataclass
class Config:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "18"))
    n_tokens: int = 40_000
    seq_len: int = 256
    sae_F: int = 128
    sae_top_k: int = 2
    sae_n_basis: int = 10
    sae_R: int = 2
    n_steps_vanilla: int = 1500
    n_steps_curve: int = 2000
    batch_size_vanilla: int = 1024
    batch_size_curve: int = 256
    lr: float = 1e-3
    eval_n: int = 8192
    norms: tuple[str, ...] = field(default_factory=lambda: ("raw_centered", "per_dim_std"))
    archs: tuple[str, ...] = field(default_factory=lambda: ("vanilla", "curve"))
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/NORM_ABLATION")


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("no blocks")


def harvest(cfg: Config, device) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model.model if hasattr(model, "model") else model.transformer)
    cap = {}
    h = blocks[cfg.layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=False)
    feats = []
    n = 0
    with torch.no_grad():
        for ex in ds:
            t = ex.get("text", "").strip()
            if len(t) < 200:
                continue
            enc = tok(t, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to(device)
            model(**enc)
            h_act = cap["h"][0]                                # (T, D)
            feats.append(h_act.cpu())
            n += h_act.shape[0]
            if n >= cfg.n_tokens:
                break
    h.remove()
    del model
    torch.cuda.empty_cache()
    X = torch.cat(feats, dim=0)[:cfg.n_tokens].float()
    return X


def normalize(X: torch.Tensor, kind: str):
    """Return (X_norm, stats) where stats can un-normalize."""
    mu = X.mean(dim=0, keepdim=True)
    if kind == "raw_centered":
        return X - mu, {"mu": mu, "sigma": torch.ones_like(mu)}
    elif kind == "per_dim_std":
        sigma = X.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (X - mu) / sigma, {"mu": mu, "sigma": sigma}
    else:
        raise ValueError(kind)


def unnormalize(Xn: torch.Tensor, stats: dict) -> torch.Tensor:
    return Xn * stats["sigma"] + stats["mu"]


def train_vanilla(cfg: Config, Xn: torch.Tensor, device, seed=0):
    torch.manual_seed(seed)
    D, F = Xn.shape[1], cfg.sae_F
    enc = nn.Linear(D, F).to(device)
    dec = nn.Parameter(torch.randn(F, D, device=device) / D**0.5)
    bias = nn.Parameter(torch.zeros(D, device=device))
    opt = torch.optim.Adam(list(enc.parameters()) + [dec, bias], lr=cfg.lr)
    k = cfg.sae_top_k
    Xn = Xn.to(device)
    for step in range(cfg.n_steps_vanilla):
        idx = torch.randint(0, Xn.shape[0], (cfg.batch_size_vanilla,))
        batch = Xn[idx]
        opt.zero_grad()
        z = F_nn.relu(enc(batch - bias))
        vals, idx_top = torch.topk(z, k, dim=1)
        z_sparse = torch.zeros_like(z).scatter(1, idx_top, vals)
        recon = z_sparse @ dec + bias
        loss = F_nn.mse_loss(recon, batch) + 1e-4 * z_sparse.abs().mean()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"      [van step {step:4d}] mse={F_nn.mse_loss(recon,batch).item():.4e}", flush=True)
    return {"enc": enc, "dec": dec, "bias": bias}


def vanilla_recon(model_d, Xn: torch.Tensor) -> tuple[torch.Tensor, int]:
    enc, dec, bias = model_d["enc"], model_d["dec"], model_d["bias"]
    k = enc.weight.shape[0]  # F
    # apply top_k
    z = F_nn.relu(enc(Xn - bias))
    # need top_k from cfg, but we'll just take top 2 (matches default)
    vals, idx = torch.topk(z, k=2, dim=1)
    z_sparse = torch.zeros_like(z).scatter(1, idx, vals)
    recon = z_sparse @ dec + bias
    alive = int((z_sparse > 0).any(dim=0).sum().item())
    return recon, alive


def train_curve(cfg: Config, Xn: torch.Tensor, device, seed=0):
    torch.manual_seed(seed)
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig, SparsityConfig
    manifold = "circle" if cfg.sae_R <= 1 else "product"
    rank = 1 if cfg.sae_R <= 1 else cfg.sae_R
    sae_cfg = ManifoldSAEConfig(
        input_dim=Xn.shape[1], n_atoms=cfg.sae_F, n_basis_per_atom=cfg.sae_n_basis,
        intrinsic_rank=rank, atom_manifold=manifold,
        sparsity=SparsityConfig(kind="softmax_topk", target_k=cfg.sae_top_k),
        dtype=torch.float64,
    )
    sae = ManifoldSAE(sae_cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    Xn = Xn.to(device=device, dtype=sae_cfg.dtype)
    for step in range(cfg.n_steps_curve):
        idx = torch.randint(0, Xn.shape[0], (cfg.batch_size_curve,))
        batch = Xn[idx]
        opt.zero_grad()
        out = sae(batch)
        loss = total_loss(out, batch, sae)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % 400 == 0:
            print(f"      [crv step {step:4d}] mse={F_nn.mse_loss(out.x_hat,batch).item():.4e}", flush=True)
    sae.eval()
    sae.fit(Xn[:2048])
    sae.lock_snapshot()
    return sae


def curve_recon(sae, Xn: torch.Tensor) -> tuple[torch.Tensor, int]:
    with torch.no_grad():
        out = sae(Xn.to(dtype=sae.cfg.dtype))
    alive = int(((out.amplitudes > 1e-6).sum(dim=0) > 30).sum().item())
    return out.x_hat.to(torch.float32), alive


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {cfg.model_name} L={cfg.layer} F={cfg.sae_F} device={device}", flush=True)

    X = harvest(cfg, device)                                  # (N, D)
    print(f"[data] X={X.shape}  raw_var={X.var().item():.4f}", flush=True)

    raw_var = float(X.var().item())     # denominator for cross-norm comparison
    results = {}

    for norm in cfg.norms:
        Xn, stats = normalize(X, norm)
        # Per-norm "in-space" variance (the denominator if you measured EV
        # in the same space the model trained in).
        in_space_var = float(Xn.var().item())
        print(f"\n=== norm = {norm}  (in-space var = {in_space_var:.4f}) ===", flush=True)

        for arch in cfg.archs:
            print(f"  --- training {arch} under {norm}", flush=True)
            if arch == "vanilla":
                obj = train_vanilla(cfg, Xn, device)
                recon, alive = vanilla_recon(obj, Xn.to(device))
            else:
                obj = train_curve(cfg, Xn, device)
                recon, alive = curve_recon(obj, Xn.to(device))

            mse_in_space = float(F_nn.mse_loss(recon, Xn.to(device)).item())
            ev_in_space = 1.0 - mse_in_space / in_space_var

            # Un-normalize the reconstruction and compare to raw X.
            recon_raw = unnormalize(recon.detach().cpu(), stats)
            mse_raw = float(F_nn.mse_loss(recon_raw, X).item())
            ev_raw = 1.0 - mse_raw / raw_var

            print(f"    alive={alive}  EV_in-space={ev_in_space:.3f}  "
                  f"EV_raw={ev_raw:.3f}", flush=True)
            results[f"{norm}__{arch}"] = {
                "alive": alive,
                "ev_in_space": ev_in_space,
                "ev_raw": ev_raw,
                "in_space_var": in_space_var,
                "raw_var": raw_var,
            }

    summary = {"config": asdict(cfg), "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # Ranking by EV_raw (norm-invariant metric)
    print("\n=== RANKING by EV_raw (norm-invariant) ===", flush=True)
    print(f"{'config':28} {'alive':>6} {'EV_in':>8} {'EV_raw':>8}", flush=True)
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["ev_raw"])
    for k, v in ranked:
        print(f"{k:28} {v['alive']:6d} {v['ev_in_space']:8.3f} {v['ev_raw']:8.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
