"""Synthetic 2D recovery v3 — fixes the NaN-isotropy + negative-EV bugs in v2.

Bugs in v2 results:
  1. v1_isotropy crashed: `s_var.sqrt() / t_var.sqrt()` from sae_2d.py
     has infinite gradient at t_var=0, blowing up encoder → NaN positions.
     Fix here: smooth isotropy via (t_var − s_var)² / (t_var + s_var + ε),
     which is C¹ everywhere and zero at t_var = s_var.
  2. v0_baseline reported EV = −0.224 even with train MSE 4e-4. Root cause:
     evaluate() runs through `inference_mode=True` (locked snapshot), but
     the snapshot batch (2048 tokens) didn't cover the soft-rescale range
     of the full 30k-sample population. Tokens outside the locked range
     get clamped, recon collapses, EV goes negative.
     Fix here: ALSO report training-mode EV so we separate "did the model
     learn anything" from "did the snapshot generalize".

Same 8-variant grid as v2 but with corrected isotropy + dual-mode EV.
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
    d_ambient: int = 256
    grid_size: int = 8
    n_grids: int = 4
    sparsity_per_token: int = 2
    n_samples: int = 30_000
    n_steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    sae_F: int = 16
    sae_top_k: int = 4
    sae_n_basis: int = 8
    sae_R: int = 2
    noise: float = 0.02
    seed: int = 0
    variants: tuple[str, ...] = field(default_factory=lambda: (
        "v0_baseline",
        "v1_isotropy_safe",
        "v2_ortho",
        "v3_coverage",
        "v4_deeper_enc",
        "v5_lower_K",
        "v6_combined",
    ))
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/SYNTHETIC_2D_V3")


def plant_2d_manifolds(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    D, G = cfg.d_ambient, cfg.n_grids
    bases, curvatures = [], []
    for _ in range(G):
        Q, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        bases.append(Q); curvatures.append(rng.uniform(0.3, 0.8))
    active = np.zeros((cfg.n_samples, G), dtype=bool)
    for n in range(cfg.n_samples):
        idx = rng.choice(G, size=cfg.sparsity_per_token, replace=False)
        active[n, idx] = True
    t_coords = rng.uniform(0.0, 1.0, (cfg.n_samples, G))
    s_coords = rng.uniform(0.0, 1.0, (cfg.n_samples, G))
    amps = rng.uniform(0.7, 1.3, (cfg.n_samples, G))
    X = np.zeros((cfg.n_samples, D), dtype=np.float32)
    for g in range(G):
        m = active[:, g]
        if not m.any():
            continue
        u = t_coords[m, g] + curvatures[g] * np.sin(2 * np.pi * s_coords[m, g])
        v = s_coords[m, g] + curvatures[g] * np.cos(2 * np.pi * t_coords[m, g])
        contrib = (u[:, None] * bases[g][:, 0:1].T + v[:, None] * bases[g][:, 1:2].T)
        X[m] += (amps[m, g, None] * contrib).astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * cfg.noise
    return {"X": torch.from_numpy(X), "active": active,
            "t_coords": t_coords, "s_coords": s_coords,
            "amps": amps, "bases": bases, "curvatures": curvatures}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = float(np.sqrt((rx*rx).sum() * (ry*ry).sum()))
    return float((rx*ry).sum() / denom) if denom > 0 else 0.0


def _coverage_kl(positions: torch.Tensor, mask: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    centers = torch.linspace(0.0, 1.0, n_bins, device=positions.device, dtype=positions.dtype)
    width = 1.0 / max(n_bins - 1, 1)
    diff = positions.unsqueeze(-1) - centers.view(1, 1, -1)
    bin_w = torch.exp(-0.5 * (diff / (width + 1e-8)) ** 2)
    mw = mask.unsqueeze(-1) * bin_w
    p = mw.sum(dim=0)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    uniform = 1.0 / n_bins
    kl = (p * (torch.log(p.clamp(min=1e-12)) - np.log(uniform))).sum(dim=-1)
    return kl.mean()


def _safe_isotropy_loss(positions_t: torch.Tensor, positions_s: torch.Tensor,
                        amp: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Smooth isotropy: (var_t - var_s)^2 / (var_t + var_s + eps).

    C^1 everywhere (no sqrt), zero iff var_t = var_s, scale-invariant in
    a quadratic sense. Restrict to atoms firing in this batch.
    """
    mask = (amp > 1e-6).float()
    n = mask.sum(dim=0) + eps
    mu_t = (positions_t * mask).sum(dim=0) / n
    mu_s = (positions_s * mask).sum(dim=0) / n
    var_t = (((positions_t - mu_t) ** 2) * mask).sum(dim=0) / n
    var_s = (((positions_s - mu_s) ** 2) * mask).sum(dim=0) / n
    fire = (mask.mean(dim=0) > 0.05).float()
    iso = ((var_t - var_s) ** 2) / (var_t + var_s + eps)
    return (iso * fire).sum() / (fire.sum() + eps)


def _ortho_loss(dirs: torch.Tensor) -> torch.Tensor:
    F = dirs.shape[0]
    flat = dirs.reshape(F, -1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    G = flat @ flat.t()
    G = G - torch.diag(torch.diagonal(G))
    return (G ** 2).mean()


def _replace_encoder_with_deeper(sae) -> None:
    enc = sae.encoder
    D, H = enc.input_dim, enc.hidden_dim
    sae.encoder.fc1 = nn.Sequential(
        nn.Linear(D, H), nn.GELU(),
        nn.Linear(H, H),
    ).to(next(sae.parameters()).device)


def train_2d_variant(cfg: Config, X: torch.Tensor, device: torch.device, variant: str) -> dict:
    from manifold_sae.sae_2d import ManifoldSAE2D, ManifoldSAE2DConfig
    K = 4 if variant in ("v5_lower_K", "v6_combined") else cfg.sae_n_basis
    sae_cfg = ManifoldSAE2DConfig(
        input_dim=cfg.d_ambient, n_features=cfg.sae_F, n_basis=K,
        top_k=cfg.sae_top_k, intrinsic_rank=cfg.sae_R, continuous_amp=True,
    )
    sae = ManifoldSAE2D(sae_cfg).to(device)
    if variant in ("v4_deeper_enc", "v6_combined"):
        _replace_encoder_with_deeper(sae)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    use_iso = variant in ("v1_isotropy_safe", "v6_combined")
    use_ortho = variant in ("v2_ortho", "v6_combined")
    use_cov = variant in ("v3_coverage", "v6_combined")

    X = X.to(device)
    for step in range(cfg.n_steps):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size,))
        batch = X[idx]
        opt.zero_grad()
        out = sae(batch)
        mse = F_nn.mse_loss(out.reconstruction, batch)
        sparsity = out.amplitudes.abs().mean()
        loss = mse + 3e-4 * sparsity
        if use_iso:
            loss = loss + 5e-3 * _safe_isotropy_loss(out.positions_t, out.positions_s, out.amplitudes)
        if use_ortho:
            loss = loss + 1e-3 * _ortho_loss(out.directions)
        if use_cov:
            loss = loss + 1e-2 * (_coverage_kl(out.positions_t, out.amplitudes)
                                   + _coverage_kl(out.positions_s, out.amplitudes))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    [{variant} step {step:5d}] mse={mse.item():.4e}", flush=True)
    sae.eval()
    # Snapshot on a LARGER reference batch (full 30k) so soft_min/max
    # locks across the actual distribution, not a small subsample.
    snap_n = min(8192, X.shape[0])
    sae.update_snapshot(X[:snap_n])
    return {"sae": sae}


def evaluate_2d(cfg: Config, sae, data: dict, device: torch.device) -> dict:
    X = data["X"].to(device)
    # TRAINING-mode evaluation (per-batch REML)
    sae.inference_mode = False
    with torch.no_grad():
        out_train = sae(X)
    mse_train = F_nn.mse_loss(out_train.reconstruction, X).item()
    # LOCKED-mode evaluation
    sae.inference_mode = True
    with torch.no_grad():
        out_locked = sae(X)
    mse_locked = F_nn.mse_loss(out_locked.reconstruction, X).item()
    var = float(X.var().item())
    ev_train = 1 - mse_train / var
    ev_locked = 1 - mse_locked / var

    out = out_train  # use training-mode atom usage for recovery
    fire = (out.amplitudes > 1e-6).sum(dim=0).cpu().numpy()
    pos_t = out.positions_t.cpu().numpy()
    pos_s = out.positions_s.cpu().numpy()
    ratios = out.intrinsic_dim_ratio.cpu().numpy()
    per_grid = []
    for g in range(cfg.n_grids):
        m = data["active"][:, g]
        if m.sum() < 50:
            continue
        gt_t, gt_s = data["t_coords"][m, g], data["s_coords"][m, g]
        best, best_atom = 0.0, -1
        for k in range(cfg.sae_F):
            if fire[k] < 30:
                continue
            sc = (_spearman(pos_t[m, k], gt_t) ** 2 + _spearman(pos_s[m, k], gt_s) ** 2) / 2
            if sc > best:
                best, best_atom = sc, k
        per_grid.append({"grid": g, "best_atom": best_atom, "score": best})
    return {
        "ev_train": ev_train,
        "ev_locked": ev_locked,
        "alive": int((fire > 30).sum()),
        "per_grid": per_grid,
        "mean_per_grid": float(np.mean([g["score"] for g in per_grid])) if per_grid else 0.0,
        "ratio_median": float(np.median(ratios)),
        "ratio_frac_2d": float((ratios > 0.5).mean()),
    }


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} out={out_dir}", flush=True)
    data = plant_2d_manifolds(cfg)
    print(f"  X={data['X'].shape}", flush=True)

    all_results = {}
    for variant in cfg.variants:
        print(f"\n=== variant: {variant} ===", flush=True)
        try:
            obj = train_2d_variant(cfg, data["X"], device, variant)
            report = evaluate_2d(cfg, obj["sae"], data, device)
            all_results[variant] = report
            print(f"  EV_train={report['ev_train']:.3f}  EV_locked={report['ev_locked']:.3f}  "
                  f"alive={report['alive']}/{cfg.sae_F}  "
                  f"mean_per_grid={report['mean_per_grid']:.3f}  "
                  f"frac_2d_atoms={report['ratio_frac_2d']:.2f}", flush=True)
            for g in report["per_grid"]:
                print(f"    grid {g['grid']}: best_atom={g['best_atom']} score={g['score']:.3f}", flush=True)
        except Exception as e:
            print(f"  [{variant}] FAILED: {type(e).__name__}: {e}", flush=True)
            all_results[variant] = {"error": str(e)}

    summary = {"config": asdict(cfg), "results": all_results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== RANKING by mean_per_grid ===", flush=True)
    ranked = sorted(((v, r) for v, r in all_results.items() if "mean_per_grid" in r),
                    key=lambda kv: -kv[1]["mean_per_grid"])
    print(f"{'variant':22} {'EVtrain':>8} {'EVlock':>8} {'alive':>6} {'per_grid':>9} {'frac2d':>7}", flush=True)
    for v, r in ranked:
        print(f"{v:22} {r['ev_train']:8.3f} {r['ev_locked']:8.3f} {r['alive']:6d} "
              f"{r['mean_per_grid']:9.3f} {r['ratio_frac_2d']:7.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
