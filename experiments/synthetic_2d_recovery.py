"""Synthetic 2D-manifold recovery — does Manifold-SAE 2D natively
recover planted 2D structures?

Plant a known 2D manifold in `ℝ^D` (a 5×5 grid surface), train
Manifold-SAE 2D, and report whether atoms recover the underlying
2-coordinate structure.

The architectural claim: each `g_k(t, s)` atom can natively span a 2D
manifold. A single atom should cover the planted grid with non-trivial
variance in both t and s axes (intrinsic_dim_ratio ≈ 1).

For comparison, also train Manifold-SAE 1D — the 1D architecture should
need multiple atoms to cover the same 2D structure, demonstrating the
2D extension's value.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: E402

import numpy as np
import torch
import torch.nn.functional as F_nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


@dataclass
class Config:
    d_ambient: int = 256
    grid_size: int = 8                # 8×8 = 64 grid points per manifold
    n_grids: int = 4                  # 4 independent 2D manifolds
    sparsity_per_token: int = 2       # how many manifolds active per token
    n_samples: int = 30_000
    n_steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3

    # 2D arch
    sae_F: int = 16
    sae_top_k: int = 4
    sae_n_basis: int = 8
    sae_R: int = 2

    noise: float = 0.02
    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/SYNTHETIC_2D",
    )
    seed: int = 0


def plant_2d_manifolds(cfg: Config) -> dict:
    """Plant `n_grids` independent 2D grid surfaces in `ℝ^D`. Each grid
    lives in its own random 2D linear subspace (D × 2), warped by a
    fixed nonlinear function so the surface isn't flat.
    """
    rng = np.random.default_rng(cfg.seed)
    D = cfg.d_ambient
    G = cfg.n_grids
    K_grid = cfg.grid_size

    # Each grid has a 2-direction ambient embedding (D, 2) + a curvature
    # multiplier so the surface isn't a flat plane.
    bases = []
    curvatures = []
    for g in range(G):
        Q, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        bases.append(Q)
        # Curvature: simple cos×sin perturbation
        curvatures.append(rng.uniform(0.3, 0.8))

    # Sample tokens.
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
        if not m.any(): continue
        # Warped 2D coords: u = t + curvature * sin(2π·s),
        #                    v = s + curvature * cos(2π·t)
        u = t_coords[m, g] + curvatures[g] * np.sin(2 * np.pi * s_coords[m, g])
        v = s_coords[m, g] + curvatures[g] * np.cos(2 * np.pi * t_coords[m, g])
        # Embed via the grid's 2D subspace
        contrib = (u[:, None] * bases[g][:, 0:1].T + v[:, None] * bases[g][:, 1:2].T)
        X[m] += (amps[m, g, None] * contrib).astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * cfg.noise

    return {
        "X": torch.from_numpy(X),
        "active": active,
        "t_coords": t_coords,
        "s_coords": s_coords,
        "amps": amps,
        "bases": bases,
        "curvatures": curvatures,
    }


def train_2d(cfg: Config, X: torch.Tensor, device: torch.device):
    from manifold_sae.sae_2d import ManifoldSAE2D, ManifoldSAE2DConfig

    sae_cfg = ManifoldSAE2DConfig(
        input_dim=cfg.d_ambient, n_features=cfg.sae_F, n_basis=cfg.sae_n_basis,
        top_k=cfg.sae_top_k, intrinsic_rank=cfg.sae_R, continuous_amp=True,
    )
    sae = ManifoldSAE2D(sae_cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    X = X.to(device)
    for step in range(cfg.n_steps):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size,))
        batch = X[idx]
        opt.zero_grad()
        out = sae(batch)
        mse = F_nn.mse_loss(out.reconstruction, batch)
        # Light sparsity + ortho priors (skipping coverage/isotropy for v1)
        sparsity = out.amplitudes.abs().mean()
        loss = mse + 3e-4 * sparsity
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  [2d step {step:5d}] mse={mse.item():.4e}", flush=True)
    sae.eval()
    sae.update_snapshot(X[: min(2048, X.shape[0])])
    sae.inference_mode = True
    return sae


def train_1d_baseline(cfg: Config, X: torch.Tensor, device: torch.device):
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    # Same parameter budget as 2D: F atoms × R rank × D = same decoder params.
    sae_cfg = ManifoldSAEConfig(
        input_dim=cfg.d_ambient, n_features=cfg.sae_F, n_basis=cfg.sae_n_basis,
        top_k=cfg.sae_top_k, intrinsic_rank=cfg.sae_R,
        encoder_type="linear", continuous_amp=True,
    )
    sae = ManifoldSAE(sae_cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    X = X.to(device)
    for step in range(cfg.n_steps):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size,))
        batch = X[idx]
        opt.zero_grad()
        out = sae(batch)
        loss = total_loss(out, batch, sae_cfg)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            mse = F_nn.mse_loss(out.reconstruction, batch).item()
            print(f"  [1d step {step:5d}] mse={mse:.4e}", flush=True)
    sae.eval()
    sae.update_snapshot(X[: min(2048, X.shape[0])])
    sae.inference_mode = True
    return sae


def evaluate(cfg: Config, sae_2d, sae_1d, data: dict, device: torch.device) -> dict:
    X = data["X"].to(device)
    with torch.no_grad():
        out_2d = sae_2d(X)
        out_1d = sae_1d(X)

    mse_2d = F_nn.mse_loss(out_2d.reconstruction, X).item()
    mse_1d = F_nn.mse_loss(out_1d.reconstruction, X).item()
    var = float(X.var().item())
    expl_2d = 1 - mse_2d / var
    expl_1d = 1 - mse_1d / var

    # Per-atom intrinsic-dim ratio for 2D atoms
    ratios_2d = out_2d.intrinsic_dim_ratio.cpu().numpy()

    # Per-atom firing count
    fire_2d = (out_2d.amplitudes > 1e-6).sum(dim=0).cpu().numpy()
    fire_1d = (out_1d.amplitudes > 1e-6).sum(dim=0).cpu().numpy()

    # For each GT grid: which 2D atom recovers it best? Use Mardia-style
    # 2D rank correlation between (atom_t, atom_s) and (gt_t, gt_s)
    # restricted to firing samples for that grid.
    pos_t_2d = out_2d.positions_t.cpu().numpy()
    pos_s_2d = out_2d.positions_s.cpu().numpy()
    grid_best_2d = []
    grid_best_1d_pair = []
    for g in range(cfg.n_grids):
        m = data["active"][:, g]
        if m.sum() < 50: continue
        gt_t = data["t_coords"][m, g]
        gt_s = data["s_coords"][m, g]
        # For each 2D atom, compute squared rank-corr with gt_t, gt_s
        best_2d_score = 0.0
        best_2d_atom = -1
        for k in range(cfg.sae_F):
            if fire_2d[k] < 30: continue
            score = (
                _spearman(pos_t_2d[m, k], gt_t) ** 2 +
                _spearman(pos_s_2d[m, k], gt_s) ** 2
            ) / 2
            if score > best_2d_score:
                best_2d_score = score
                best_2d_atom = k
        grid_best_2d.append({"grid": g, "best_atom": best_2d_atom, "score": best_2d_score})

        # For 1D: best PAIR of atoms — atom_a tracks gt_t, atom_b tracks gt_s
        pos_1d = out_1d.positions.cpu().numpy()
        amp_1d = out_1d.amplitudes.cpu().numpy()
        rho_t_per_atom = []
        rho_s_per_atom = []
        for k in range(cfg.sae_F):
            if (amp_1d[m, k] > 1e-6).sum() < 30:
                rho_t_per_atom.append(0.0); rho_s_per_atom.append(0.0); continue
            rho_t_per_atom.append(abs(_spearman(pos_1d[m, k], gt_t)))
            rho_s_per_atom.append(abs(_spearman(pos_1d[m, k], gt_s)))
        rho_t_per_atom = np.array(rho_t_per_atom)
        rho_s_per_atom = np.array(rho_s_per_atom)
        best_t = float(rho_t_per_atom.max())
        best_s = float(rho_s_per_atom.max())
        grid_best_1d_pair.append({
            "grid": g, "best_atom_for_t": int(rho_t_per_atom.argmax()),
            "best_atom_for_s": int(rho_s_per_atom.argmax()),
            "rho_t": best_t, "rho_s": best_s,
            "pair_score": (best_t**2 + best_s**2) / 2,
        })

    return {
        "explained_2d": expl_2d,
        "explained_1d": expl_1d,
        "atoms_2d_alive": int((fire_2d > 30).sum()),
        "atoms_1d_alive": int((fire_1d > 30).sum()),
        "intrinsic_dim_ratio_2d": ratios_2d.tolist(),
        "n_atoms_2d_using_both_axes": int((ratios_2d > 0.5).sum()),
        "n_atoms_2d_1d_like": int((ratios_2d < 0.2).sum()),
        "per_grid_2d": grid_best_2d,
        "per_grid_1d_pair": grid_best_1d_pair,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = float(np.sqrt((rx*rx).sum() * (ry*ry).sum()))
    return float((rx*ry).sum() / denom) if denom > 0 else 0.0


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} output_dir={out_dir}", flush=True)

    print("[data] planting 2D manifolds", flush=True)
    data = plant_2d_manifolds(cfg)
    print(f"  X={data['X'].shape}, n_grids={cfg.n_grids}, grid_size={cfg.grid_size}", flush=True)

    print("\n[2D] training Manifold-SAE 2D", flush=True)
    sae_2d = train_2d(cfg, data["X"], device)
    print("\n[1D] training Manifold-SAE 1D baseline", flush=True)
    sae_1d = train_1d_baseline(cfg, data["X"], device)

    print("\n[eval]", flush=True)
    report = evaluate(cfg, sae_2d, sae_1d, data, device)
    print(f"  Manifold-SAE 2D: expl={report['explained_2d']:.3f}  "
          f"alive={report['atoms_2d_alive']}/{cfg.sae_F}  "
          f"atoms-using-both-axes={report['n_atoms_2d_using_both_axes']}", flush=True)
    print(f"  Manifold-SAE 1D: expl={report['explained_1d']:.3f}  "
          f"alive={report['atoms_1d_alive']}/{cfg.sae_F}", flush=True)
    for g_2d, g_1d in zip(report["per_grid_2d"], report["per_grid_1d_pair"]):
        print(f"    grid {g_2d['grid']}: 2D best-atom score={g_2d['score']:.3f}"
              f"  |  1D best-pair score={g_1d['pair_score']:.3f}"
              f"  ρ_t={g_1d['rho_t']:.2f} ρ_s={g_1d['rho_s']:.2f}", flush=True)

    (out_dir / "results.json").write_text(json.dumps({
        "config": asdict(cfg), "report": report,
    }, indent=2, default=float))
    print(f"\n[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
