"""Falsifiable continuous-feature-recovery benchmark.

The fair critique of `experiments/realistic_scaling.py` is that curve SAE
has more decoder parameters per atom (rank R > 1) so of course it
explains more variance at matched dictionary size F. This benchmark
addresses that head-on by:

1. Matching **total decoder parameter count**, not F. Vanilla SAE gets
   `F_vanilla = R · F_curve` atoms so each architecture has the same
   `F · D · R` decoder-budget.

2. Reporting a metric that doesn't reward expressiveness in general,
   only the *right kind* of expressiveness: the ability to encode a
   continuous scalar latent as a single atom's signal.

Specifically: ground-truth data is generated from a *known* scalar
latent `z ∈ [0, 1]` per sample. Each sample is `C(z) + noise` where
`C: [0, 1] → ℝ^D` is a smooth (possibly non-monotone) 1D curve in
residual space. After training, for each atom we compute the Spearman
correlation between (atom-signal, z) where atom-signal is:

  * curve SAE atom k: position `t_k` (the architecture's claim)
  * vanilla SAE atom k: TopK activation magnitude

If the underlying curve C is *non-monotone* (e.g. a U-curve), no single
vanilla atom's activation can have high |Spearman(activation, z)|: the
activation is a linear projection of `C(z) + noise`, which is
non-monotone in z by construction. The best vanilla atom's |ρ| is
upper-bounded well below 1. Curve SAE atoms have a free t-parameter and
*can* in principle achieve |ρ| ≈ 1.

This is the architectural claim in falsifiable form. If curve SAE atoms
DON'T beat vanilla atoms on |Spearman(signal, z)| when curves are
non-monotone, the architecture is wrong for this kind of data.

Three regimes are tested:

* monotone — `C(z) = z · v` for a random direction `v`. Both should win.
* non-monotone — `C(z) = sin(2πz) · v_1 + cos(2πz) · v_2`. A 1D loop
  in 2D. Curve SAE should win cleanly; vanilla has no chance.
* mixed — Half the curves are monotone, half non-monotone.

For each regime, both architectures are trained at matched decoder
parameter count and we report best-atom |Spearman(signal, z)|.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check
from manifold_sae._cluster_bridge import require_cuda_if_env

bypass_gamfit_cuda_check()


# ---------------------------------------------------------------------------
# Data: planted single-latent 1D manifolds in residual space
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    D: int                                          # ambient dim
    n_curves: int                                   # number of GT curves
    curve_kind: str                                 # "monotone" | "non_monotone" | "mixed"
    sparsity_per_token: int
    n_samples: int
    # Matched-decoder-param budget. Curve has F_curve atoms of rank R.
    F_curve: int
    R: int
    top_k: int
    # Vanilla gets F_vanilla = R * F_curve atoms (matched total params).
    # If you want matched-F instead, set F_vanilla_override.
    F_vanilla_override: int | None = None
    n_steps_vanilla: int = 4000
    n_steps_curve: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    n_basis: int = 12
    noise: float = 0.03
    seed: int = 0


SCENARIOS = (
    Scenario("monotone_D256",     D=256, n_curves=16, curve_kind="monotone",
             sparsity_per_token=3, n_samples=40_000, F_curve=16, R=2, top_k=4),
    Scenario("non_monotone_D256", D=256, n_curves=16, curve_kind="non_monotone",
             sparsity_per_token=3, n_samples=40_000, F_curve=16, R=2, top_k=4),
    Scenario("mixed_D256",        D=256, n_curves=16, curve_kind="mixed",
             sparsity_per_token=3, n_samples=40_000, F_curve=16, R=2, top_k=4),
)


def _random_dir(D: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(D)
    return v / np.linalg.norm(v)


def make_curve(D: int, kind: str, rng: np.random.Generator):
    """Return a callable z -> ℝ^D for z ∈ [0,1], and a label string."""
    if kind == "monotone":
        v = _random_dir(D, rng)
        return (lambda z: z[..., None] * v), "monotone"
    if kind == "non_monotone":
        v1, v2 = _random_dir(D, rng), _random_dir(D, rng)
        # 1D loop in the (v1, v2) plane: cos/sin parametrization
        return (lambda z: (np.cos(2 * np.pi * z))[..., None] * v1
                          + (np.sin(2 * np.pi * z))[..., None] * v2), "non_monotone"
    raise ValueError(f"unknown curve_kind {kind}")


def generate_data(s: Scenario):
    rng = np.random.default_rng(s.seed)
    curves = []
    labels = []
    for k in range(s.n_curves):
        if s.curve_kind == "mixed":
            kind = "monotone" if k % 2 == 0 else "non_monotone"
        else:
            kind = s.curve_kind
        c, lbl = make_curve(s.D, kind, rng)
        curves.append(c)
        labels.append(lbl)

    active = np.zeros((s.n_samples, s.n_curves), dtype=bool)
    for n in range(s.n_samples):
        idx = rng.choice(s.n_curves, size=s.sparsity_per_token, replace=False)
        active[n, idx] = True
    z = rng.uniform(0.0, 1.0, (s.n_samples, s.n_curves))    # the planted scalar latent per (sample, curve)
    amps = rng.uniform(0.7, 1.3, (s.n_samples, s.n_curves))

    X = np.zeros((s.n_samples, s.D), dtype=np.float32)
    for k in range(s.n_curves):
        m = active[:, k]
        if not m.any(): continue
        contrib = amps[m, k, None] * curves[k](z[m, k])
        X[m] += contrib.astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * s.noise
    return {
        "X": torch.from_numpy(X),
        "z": z,
        "active": active,
        "amps": amps,
        "curve_labels": labels,
    }


# ---------------------------------------------------------------------------
# SAEs
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

    def forward(self, x: torch.Tensor):
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        vals, idx = torch.topk(z, self.top_k, dim=1)
        gate = torch.zeros_like(z).scatter_(1, idx, vals)
        return gate @ self.W_dec, gate


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = float(np.sqrt((rx*rx).sum() * (ry*ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def train_vanilla(s: Scenario, X: torch.Tensor, device: torch.device) -> nn.Module:
    F = s.F_vanilla_override if s.F_vanilla_override else s.R * s.F_curve
    sae = VanillaSAE(s.D, F, s.top_k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=s.lr)
    X = X.to(device)
    for step in range(s.n_steps_vanilla):
        idx = torch.randint(0, X.shape[0], (s.batch_size,))
        batch = X[idx]
        opt.zero_grad()
        recon, _ = sae(batch)
        loss = F_nn.mse_loss(recon, batch)
        loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  [van step {step}] mse={loss.item():.4e}", flush=True)
    sae.eval()
    return sae


def train_curve(s: Scenario, X: torch.Tensor, device: torch.device) -> nn.Module:
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    cfg = ManifoldSAEConfig(
        input_dim=s.D, n_features=s.F_curve, n_basis=s.n_basis,
        top_k=s.top_k, intrinsic_rank=s.R, encoder_type="linear",
        continuous_amp=True,
    )
    sae = ManifoldSAE(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=s.lr)
    X = X.to(device)
    for step in range(s.n_steps_curve):
        idx = torch.randint(0, X.shape[0], (s.batch_size,))
        batch = X[idx]
        opt.zero_grad()
        out = sae(batch)
        loss = total_loss(out, batch, cfg)["total"]
        loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  [crv step {step}] mse={F_nn.mse_loss(out.reconstruction, batch).item():.4e}", flush=True)
    sae.eval()
    sae.update_snapshot(X[:min(2048, X.shape[0])])
    sae.inference_mode = True
    return sae


def best_atom_spearman_vanilla(sae: VanillaSAE, X: torch.Tensor, z_per_curve: np.ndarray,
                                active: np.ndarray, device: torch.device) -> dict:
    """For each ground-truth curve k, compute the best vanilla atom's
    |Spearman(activation, z)| restricted to samples where curve k is active.
    """
    with torch.no_grad():
        _, gate = sae(X.to(device))
    g = gate.cpu().numpy()                                # (N, F_v)
    F_v = g.shape[1]
    K = z_per_curve.shape[1]
    best = np.zeros(K)
    for k in range(K):
        mask = active[:, k]
        if mask.sum() < 50:
            continue
        z_k = z_per_curve[mask, k]
        rhos = [abs(spearman(g[mask, f], z_k)) for f in range(F_v)]
        best[k] = max(rhos) if rhos else 0.0
    return {
        "per_curve_best_rho": best.tolist(),
        "mean_best_rho": float(best.mean()),
        "median_best_rho": float(np.median(best)),
    }


def best_atom_spearman_curve(sae: nn.Module, X: torch.Tensor, z_per_curve: np.ndarray,
                              active: np.ndarray, device: torch.device) -> dict:
    """For each GT curve, find the curve-SAE atom whose position t_k best
    tracks z restricted to samples where the GT curve is active.
    """
    with torch.no_grad():
        out = sae(X.to(device))
    pos = out.positions.cpu().numpy()
    F_c = pos.shape[1]
    K = z_per_curve.shape[1]
    best = np.zeros(K)
    for k in range(K):
        mask = active[:, k]
        if mask.sum() < 50:
            continue
        z_k = z_per_curve[mask, k]
        rhos = [abs(spearman(pos[mask, f], z_k)) for f in range(F_c)]
        best[k] = max(rhos) if rhos else 0.0
    return {
        "per_curve_best_rho": best.tolist(),
        "mean_best_rho": float(best.mean()),
        "median_best_rho": float(np.median(best)),
    }


def run_scenario(s: Scenario, output_dir: Path) -> dict:
    print(f"\n=========== Scenario: {s.name} ({s.curve_kind}) ===========", flush=True)
    print(f"  D={s.D} n_curves={s.n_curves} sparsity={s.sparsity_per_token}", flush=True)
    F_vanilla = s.F_vanilla_override if s.F_vanilla_override else s.R * s.F_curve
    print(f"  curve SAE F={s.F_curve} R={s.R}  |  vanilla SAE F={F_vanilla}", flush=True)
    print(f"  decoder params: curve {s.F_curve * s.D * s.R}, vanilla {F_vanilla * s.D}", flush=True)

    data = generate_data(s)
    X = data["X"]
    mu = X.mean(0, keepdim=True); sigma = float(X.std().item())
    X_n = (X - mu) / max(sigma, 1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[vanilla] training", flush=True)
    sae_v = train_vanilla(s, X_n, device)
    print("[curve] training", flush=True)
    sae_c = train_curve(s, X_n, device)

    res_v = best_atom_spearman_vanilla(sae_v, X_n, data["z"], data["active"], device)
    res_c = best_atom_spearman_curve(sae_c, X_n, data["z"], data["active"], device)
    print(f"[eval] vanilla best atom |ρ(activation, z)|: mean={res_v['mean_best_rho']:.3f} "
          f"median={res_v['median_best_rho']:.3f}", flush=True)
    print(f"[eval] curve   best atom |ρ(t_k,     z)|:    mean={res_c['mean_best_rho']:.3f} "
          f"median={res_c['median_best_rho']:.3f}", flush=True)
    print(f"[eval] Δ mean |ρ|:    {res_c['mean_best_rho']-res_v['mean_best_rho']:+.3f}", flush=True)

    with torch.no_grad():
        rec_v, _ = sae_v(X_n.to(device))
        rec_c = sae_c(X_n.to(device)).reconstruction
    mse_v = F_nn.mse_loss(rec_v, X_n.to(device)).item()
    mse_c = F_nn.mse_loss(rec_c, X_n.to(device)).item()
    var = float(X_n.var().item())
    expl_v = 1 - mse_v / var
    expl_c = 1 - mse_c / var
    print(f"[eval] reconstruction expl: vanilla={expl_v:.3f}  curve={expl_c:.3f}", flush=True)

    out_dir = output_dir / s.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": asdict(s),
        "vanilla": {**res_v, "explained": expl_v, "F": F_vanilla},
        "curve": {**res_c, "explained": expl_c, "F": s.F_curve, "R": s.R},
        "delta_mean_rho": res_c["mean_best_rho"] - res_v["mean_best_rho"],
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main() -> int:
    require_cuda_if_env()
    output_dir = Path(os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "runs/CONTINUOUS_RECOVERY"))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    for s in SCENARIOS:
        all_reports[s.name] = run_scenario(s, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(all_reports, indent=2, default=float))
    print(f"\n[done] {output_dir / 'summary.json'}")
    print("\n=== Summary ===")
    for name, r in all_reports.items():
        v = r["vanilla"]["mean_best_rho"]
        c = r["curve"]["mean_best_rho"]
        print(f"  {name}: vanilla |ρ|={v:.3f}  curve |ρ|={c:.3f}  Δ={c-v:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
