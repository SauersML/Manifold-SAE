"""Orthogonal-manifolds recovery experiment for Manifold-SAE.

Question
--------
Plant several *distinct* manifold types (line, circle, helix, lissajous, ...)
in **mutually orthogonal** subspaces of a high-dimensional ambient space, so the
ground-truth geometry is unambiguous: each curve lives in its own non-interacting
block of R^D. Then train a Manifold-SAE on the sparse-sum activations and ask:

1. **Shape recovery** — does each planted curve reappear as a learned atom?
   (Hungarian/greedy-matched symmetric Chamfer, gauge-free.)
2. **Subspace recovery** — does the matched atom live in the *same* ambient
   subspace it was planted in? (Mean cosine of principal angles between the
   planted block and the learned atom's realized subspace.)
3. **Orthogonality respect** — does the learned atom keep its energy *inside*
   its planted block, or does it leak into other features' orthogonal blocks?
   (Fraction of curve energy outside the matched planted subspace.)

Unlike ``synthetic_recovery.py`` (independent random projections, which overlap),
the planted subspaces here are exactly orthogonal, so leakage is unambiguous.

No CLI. Edit ``Config`` at the bottom, or ``from experiments.orthogonal_manifolds
import Config, main; main(Config(...))``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from manifold_sae.data_synthetic import SyntheticDataset, CURVE_TYPES, chamfer_distance


# A diverse default platter: one of each distinct manifold type, easy -> hard.
ALL_CURVE_INDICES = list(range(len(CURVE_TYPES)))


@dataclass(frozen=True)
class Config:
    d_ambient: int = 128
    # Which distinct manifold types to plant (indices into CURVE_TYPES). Default
    # = all 12 distinct types, each in its own orthogonal block.
    curve_indices: tuple[int, ...] = tuple(ALL_CURVE_INDICES)
    n_samples: int = 16384
    sparsity: float = 0.25
    noise: float = 0.02
    # Architecture.
    n_basis: int = 12
    intrinsic_rank: int = 3
    top_k: int = 3
    slack_features: int = 4          # dictionary surplus over planted count
    ortho_weight: float = 1e-2
    # Training.
    n_steps: int = 4000
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 0
    # Curriculum: ramp GT active-count from 1 -> full over curriculum_steps.
    curriculum_start_active: int = 1
    curriculum_steps: int = 2000
    # Ground truth: mutually-orthogonal blocks (True) vs independent random
    # subspaces that overlap (False, the synthetic_recovery regime).
    orthogonal: bool = True
    # Eval.
    t_grid_size: int = 128
    chamfer_threshold: float = 0.3   # mean chamfer above -> nonzero exit
    output_dir: str = "runs/orthogonal_manifolds"
    plot: bool = True


DEFAULT_CONFIG = Config()


# ---------------------------------------------------------------------------
# Loaders (curriculum on GT active-count, same idea as synthetic_recovery)
# ---------------------------------------------------------------------------


def _full_loader(ds: SyntheticDataset, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed + 1)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)


def _curriculum_loader(ds: SyntheticDataset, batch_size: int, max_active: int, seed: int) -> DataLoader:
    active = ds.ground_truth["active"]
    counts = active.to(torch.int64).sum(dim=1)
    keep = (counts <= max_active) & (counts >= 1)
    idx = torch.nonzero(keep, as_tuple=True)[0].tolist()
    if len(idx) < batch_size:
        return _full_loader(ds, batch_size, seed)
    g = torch.Generator().manual_seed(seed + 1)
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=True, drop_last=True, generator=g)


# ---------------------------------------------------------------------------
# Matching + geometry metrics
# ---------------------------------------------------------------------------


def _match_curves(gt_points: np.ndarray, learned_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy min-Chamfer bipartite match: GT feature -> learned atom index."""
    F_gt, F_sae = gt_points.shape[0], learned_points.shape[0]
    cost = np.empty((F_gt, F_sae))
    for i in range(F_gt):
        for j in range(F_sae):
            cost[i, j] = chamfer_distance(gt_points[i], learned_points[j])
    matches = np.full(F_gt, -1, dtype=np.int64)
    costs = np.full(F_gt, np.inf)
    used = np.zeros(F_sae, dtype=bool)
    for flat in np.argsort(cost, axis=None):
        i, j = divmod(int(flat), F_sae)
        if matches[i] != -1 or used[j]:
            continue
        matches[i], costs[i], used[j] = j, cost[i, j], True
        if (matches != -1).all():
            break
    return matches, costs


def _principal_angle_cos(basis_a: np.ndarray, cloud_b: np.ndarray) -> float:
    """Mean cosine of principal angles between span(basis_a) and the learned
    atom's realized subspace.

    ``basis_a`` is the planted (d, D) row-orthonormal block. The learned subspace
    is the top-d right singular directions of the centered learned curve cloud
    ``cloud_b`` (T, D) — i.e. the directions the atom actually exercises over its
    parameter range. Mean cos in [0, 1]; 1.0 == perfectly aligned subspaces.
    """
    d = basis_a.shape[0]
    c = cloud_b - cloud_b.mean(axis=0, keepdims=True)
    if np.linalg.norm(c) < 1e-12:
        return 0.0
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    learned_basis = vt[:d]  # (d, D) top directions
    # Principal-angle cosines are the singular values of A Vᵀ (both orthonormal).
    s = np.linalg.svd(basis_a @ learned_basis.T, compute_uv=False)
    return float(np.clip(s, 0.0, 1.0).mean())


def _leakage(basis_a: np.ndarray, cloud_b: np.ndarray) -> float:
    """Fraction of the learned atom's curve energy OUTSIDE the planted block.

    ``P = basis_aᵀ basis_a`` projects onto the planted subspace. Leakage =
    1 - ||cloud_b P||² / ||cloud_b||². In orthogonal_subspaces mode any nonzero
    leakage is energy bleeding into *other features'* blocks (the only place it
    can go), so this directly measures whether the SAE respects orthogonality.
    """
    c = cloud_b - cloud_b.mean(axis=0, keepdims=True)
    total = float((c * c).sum())
    if total < 1e-18:
        return 0.0
    proj = c @ basis_a.T @ basis_a  # (T, D) component inside the block
    inside = float((proj * proj).sum())
    return float(np.clip(1.0 - inside / total, 0.0, 1.0))


def _nonlinearity(cloud: np.ndarray) -> float:
    """Curvature score that a straight line CANNOT fake: sigma_2 / sigma_1 of the
    centered point cloud. A straight segment puts all variance on its first
    singular direction -> ~0; a parabola ~0.2-0.4; a circle ~1.0. This is the
    line-proof companion to Chamfer (which scores a diameter through a circle
    almost as well as the circle itself)."""
    c = cloud - cloud.mean(axis=0, keepdims=True)
    s = np.linalg.svd(c, compute_uv=False)
    if s[0] < 1e-12:
        return 0.0
    return float(s[1] / s[0]) if len(s) > 1 else 0.0


def _fit_open_curve(t, y, by, grid_size):
    """Open-curve fit via Gaussian-REML Duchon spline. Returns (grid_curve, resid)."""
    import gamfit
    res = gamfit.gaussian_reml_fit_positions(
        torch.from_numpy(t), torch.from_numpy(y),
        basis="duchon", basis_order=2, by=torch.from_numpy(by),
    )
    coef = np.asarray(res["coefficients"])
    knots = np.asarray(res["knots_or_centers"]).reshape(-1, 1)
    fitted = np.asarray(res["fitted"])                       # by * curve at t
    resid = float(((y - fitted) ** 2).sum())
    lo, hi = float(t.min()), float(t.max())
    grid = np.linspace(lo, hi, grid_size).reshape(-1, 1)
    Bg = np.asarray(gamfit.duchon_basis(torch.from_numpy(grid), torch.from_numpy(knots), m=2))
    return Bg @ coef, resid


def _fit_closed_curve(t, y, by, grid_size, n_knots=12):
    """Closed-curve fit via a periodic cyclic B-spline (gam#580 workaround: the
    periodic Duchon Gram is indefinite, but the periodic B-spline penalty is PSD
    and recovers closed curves at R²≈0.998). Returns (grid_curve, resid)."""
    import gamfit
    # Positions are the manifold coordinate; normalize into [0, 1) for the cyclic
    # basis so the wrap is well defined.
    lo, hi = float(t.min()), float(t.max())
    span = max(hi - lo, 1e-9)
    tn = (t - lo) / span
    tn = np.clip(tn, 0.0, 1.0 - 1e-9)
    B, P = gamfit.periodic_spline_curve_basis(torch.from_numpy(tn), n_knots=n_knots,
                                              degree=3, penalty_order=2)
    B = np.asarray(B)
    X = by[:, None] * B                                      # model y = by * (B @ coef)
    beta, _ = gamfit.gaussian_weighted_ridge(
        torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(np.asarray(P)),
        torch.from_numpy(np.ones(len(t))), ridge_lambda=1e-3,
    )
    beta = np.asarray(beta)
    fitted = X @ beta
    resid = float(((y - fitted) ** 2).sum())
    grid_n = np.linspace(0.0, 1.0, grid_size, endpoint=False)
    Bg, _ = gamfit.periodic_spline_curve_basis(torch.from_numpy(grid_n), n_knots=n_knots,
                                               degree=3, penalty_order=2)
    return np.asarray(Bg) @ beta, resid


def _fit_atom_curves(sae, X, n_sae, grid_size, device, amp_thresh=1e-2):
    """Recover each atom's ambient curve from the encoder's (position, amplitude)
    for the tokens that atom fires on, trying BOTH topologies and keeping the
    better fit:

    * open  — Gaussian-REML Duchon spline (``gaussian_reml_fit_positions``)
    * closed — periodic cyclic B-spline (``periodic_spline_curve_basis`` +
      ``gaussian_weighted_ridge``), the working path for circles/cyclic atoms.

    Per atom we fit both, compare the reconstruction residual on the atom's
    active tokens, and emit the lower-residual curve. This auto-detects topology
    and lets closed curves recover despite the broken periodic Duchon (gam#580).

    Replaces the broken closed-form SAE solve (gam#577/#578) and the collapsed
    backprop decoder blocks. Returns ``(n_sae, grid_size, D)``; degenerate atoms
    come back as zeros.
    """
    with torch.no_grad():
        out = sae(X.to(device=device, dtype=sae.cfg.dtype))
    pos = out.positions[..., 0].detach().cpu().numpy()   # (N, F) first intrinsic coord
    amp = out.amplitudes.detach().cpu().numpy()           # (N, F)
    Xn = X.detach().cpu().numpy()                          # (N, D)
    D = Xn.shape[1]
    curves = np.zeros((n_sae, grid_size, D), dtype=np.float64)
    for k in range(n_sae):
        active = amp[:, k] > amp_thresh
        if active.sum() < 32:
            continue
        t = pos[active, k].astype(np.float64)
        if float(t.max() - t.min()) < 1e-3:
            continue
        y = Xn[active].astype(np.float64)
        by = amp[active, k].astype(np.float64)
        best, best_resid = None, np.inf
        for fit_fn in (_fit_open_curve, _fit_closed_curve):
            try:
                curve, resid = fit_fn(t, y, by, grid_size)
            except Exception:
                continue
            if resid < best_resid:
                best, best_resid = curve, resid
        if best is not None:
            curves[k] = best
    return curves


# ---------------------------------------------------------------------------
# Plot (reuse synthetic_recovery's Procrustes overlay)
# ---------------------------------------------------------------------------


def _plot(output_dir, gt_points, learned_points, matches, names):
    from experiments.synthetic_recovery import _plot_curves
    _plot_curves(output_dir, gt_points, learned_points, matches, names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config = DEFAULT_CONFIG) -> int:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_indices = list(cfg.curve_indices)
    n_features = len(curve_indices)
    mode = "mutually-orthogonal" if cfg.orthogonal else "overlapping-random"
    print(f"[setup] planting {n_features} {mode} manifolds in R^{cfg.d_ambient}: "
          f"{[CURVE_TYPES[i][0] for i in curve_indices]}")

    dataset = SyntheticDataset(
        d_ambient=cfg.d_ambient,
        n_features=n_features,
        n_samples=cfg.n_samples,
        sparsity=cfg.sparsity,
        noise=cfg.noise,
        seed=cfg.seed,
        t_grid_size=cfg.t_grid_size,
        orthogonal_subspaces=cfg.orthogonal,
        curve_indices=curve_indices,
    )

    from manifold_sae.sae import (
        ManifoldSAE, ManifoldSAEConfig, DecoderConfig, SparsityConfig, RemlConfig,
    )
    from manifold_sae.train import build_optimizer, train
    from manifold_sae.diagnostics import dead_feature_mask, position_variance

    n_sae = n_features + cfg.slack_features
    sae_cfg = ManifoldSAEConfig(
        input_dim=cfg.d_ambient,
        n_atoms=n_sae,
        intrinsic_rank=cfg.intrinsic_rank,
        atom_manifold="product",
        n_basis_per_atom=cfg.n_basis,
        sparsity=SparsityConfig(kind="softmax_topk", target_k=cfg.top_k),
        decoder=DecoderConfig(ortho_weight=cfg.ortho_weight),
        reml=RemlConfig(),
    )
    sae = ManifoldSAE(sae_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, n_sae_atoms={n_sae}")
    sae.to(device)
    optimizer = build_optimizer(sae, lr=cfg.lr)

    # Curriculum over GT active-count.
    t0 = time.time()
    history: dict = {}
    n_phases = n_features - cfg.curriculum_start_active + 1
    phase_size = max(cfg.curriculum_steps // max(n_phases, 1), 1)
    step_done = 0
    print(f"[train] {cfg.n_steps} steps, curriculum {cfg.curriculum_start_active}->{n_features} "
          f"active over {cfg.curriculum_steps} steps")
    for phase in range(n_phases):
        max_active = cfg.curriculum_start_active + phase
        steps_this = (cfg.n_steps - step_done) if phase == n_phases - 1 else phase_size
        if steps_this <= 0:
            continue
        loader = _curriculum_loader(dataset, cfg.batch_size, max_active, cfg.seed + phase)
        h = train(sae, loader, optimizer, n_steps=steps_this, log_every=max(steps_this // 4, 1))
        for k, v in h.items():
            history.setdefault(k, []).extend(v)
        step_done += steps_this
    train_time = time.time() - t0

    # --- Probe learned curves via the working per-atom REML fit ---
    print("[eval] fitting per-atom curves (gaussian_reml_fit_positions)")
    t_grid = dataset.ground_truth["t_grid"]
    gt_points = dataset.ground_truth["curve_points"]             # (F_gt, T, D)
    bases = dataset.ground_truth["subspace_bases"]               # list of (d, D)
    # Fit each atom's curve from the encoder's (position, amplitude) using the
    # fast, working low-level path -- NOT sae.fit() (broken: gam #577/#578) and
    # NOT the collapsed backprop decoder blocks. See _fit_atom_curves.
    T = int(t_grid.shape[0])
    learned_points = _fit_atom_curves(sae, dataset.x, n_sae, T, device)  # (F_sae, T, D)

    matches, chamfer = _match_curves(gt_points, learned_points)

    # --- Per-feature geometry ---
    per_feature = []
    for i in range(n_features):
        j = int(matches[i])
        cloud = learned_points[j]
        gt_nl = _nonlinearity(gt_points[i])
        learned_nl = _nonlinearity(cloud)
        per_feature.append({
            "gt_index": i,
            "name": dataset.features[i].name,
            "periodic": dataset.features[i].periodic,
            "d_proj": int(bases[i].shape[0]),
            "sae_match_index": j,
            "chamfer": float(chamfer[i]),
            "subspace_cos": _principal_angle_cos(bases[i], cloud),
            "leakage": _leakage(bases[i], cloud),
            "gt_nonlinearity": gt_nl,
            "learned_nonlinearity": learned_nl,
            # 1.0 == atom reproduces the GT's curvature; ~0 == collapsed to a line
            "curvature_recovery": float(min(learned_nl, gt_nl) / (gt_nl + 1e-9)),
        })

    # --- Overall reconstruction EV ---
    sae.eval()
    with torch.no_grad():
        xb = dataset.x[: min(4096, len(dataset))].to(device=device, dtype=sae.cfg.dtype)
        out = sae(xb)
        mse = float(torch.mean((out.x_hat - xb) ** 2).item())
        var = float(torch.mean((xb - xb.mean(0, keepdim=True)) ** 2).item())
        ev = 1.0 - mse / var if var > 0 else 0.0
        pos_var = position_variance(out).reshape(-1).cpu().numpy().tolist()
        dead = dead_feature_mask(out).cpu().numpy().tolist()
    sae.train()

    report = {
        "status": "ok",
        "config": asdict(cfg),
        "train_seconds": train_time,
        "n_features": n_features,
        "n_sae_atoms": n_sae,
        "mse_eval": mse,
        "explained_variance": ev,
        "chamfer_mean": float(np.mean(chamfer)),
        "chamfer_max": float(np.max(chamfer)),
        "subspace_cos_mean": float(np.mean([p["subspace_cos"] for p in per_feature])),
        "leakage_mean": float(np.mean([p["leakage"] for p in per_feature])),
        "leakage_max": float(np.max([p["leakage"] for p in per_feature])),
        "curvature_recovery_mean": float(np.mean([p["curvature_recovery"] for p in per_feature])),
        # curvature recovery on the genuinely-curved GTs only (gt_nonlinearity>0.05)
        "curvature_recovery_curved": float(np.mean(
            [p["curvature_recovery"] for p in per_feature if p["gt_nonlinearity"] > 0.05] or [0.0]
        )),
        "dead_feature_count": int(sum(dead)),
        "per_feature": per_feature,
        "history": history,
    }
    (out_dir / "orthogonal_manifolds_results.json").write_text(
        json.dumps(report, indent=2, default=float)
    )

    # Console summary table.
    print(f"\n[eval] EV={ev:.3f}  chamfer mean={report['chamfer_mean']:.3f}  "
          f"subspace_cos mean={report['subspace_cos_mean']:.3f}  "
          f"leakage mean={report['leakage_mean']:.3f}  "
          f"curvature_recovery(curved)={report['curvature_recovery_curved']:.3f}  "
          f"dead={report['dead_feature_count']}/{n_sae}")
    print(f"\n  {'manifold':<10} {'chamfer':>8} {'subsp_cos':>10} {'leak':>6} "
          f"{'gt_nl':>6} {'lrn_nl':>6} {'curv_rec':>8}  match")
    print("  " + "-" * 72)
    for p in per_feature:
        # line-proof flag: GT is curved but the atom didn't reproduce the curvature
        flag = "  <-- collapsed-to-line" if (p["gt_nonlinearity"] > 0.05 and p["curvature_recovery"] < 0.5) else ""
        print(f"  {p['name']:<10} {p['chamfer']:>8.3f} {p['subspace_cos']:>10.3f} "
              f"{p['leakage']:>6.3f} {p['gt_nonlinearity']:>6.3f} {p['learned_nonlinearity']:>6.3f} "
              f"{p['curvature_recovery']:>8.3f}{flag}")

    if cfg.plot:
        try:
            _plot(out_dir, gt_points, learned_points, matches, [f.name for f in dataset.features])
        except Exception as e:  # plotting is best-effort
            print(f"[plot] skipped: {e}")

    if report["chamfer_mean"] > cfg.chamfer_threshold:
        print(f"\n[fail] chamfer mean {report['chamfer_mean']:.3f} > {cfg.chamfer_threshold}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
