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

    # --- Probe learned curves (drives REML solve + locks snapshot) ---
    print("[eval] probing learned curves")
    t_grid = dataset.ground_truth["t_grid"]
    gt_points = dataset.ground_truth["curve_points"]             # (F_gt, T, D)
    bases = dataset.ground_truth["subspace_bases"]               # list of (d, D)
    # Read curves straight from the backprop-trained decoder blocks. We do NOT
    # call the closed-form REML probe (sae.fit): with fit_every=0 the decoder is
    # trained purely by Adam, so the blocks already encode the curves, and the
    # gamfit SAE-manifold solve is both slow and numerically brittle at this
    # over-parameterized toy scale (rank-deficient designs -> ill-conditioned
    # arrow-Schur; see issue #1). The primitive's extract_feature_curves reads
    # decoder_blocks directly and needs no fit/lock.
    T = int(t_grid.shape[0])
    curve_dict = sae.extract_feature_curves(grid_size=T)
    learned_points = torch.stack(
        [curve_dict[j] for j in range(n_sae)], dim=0
    ).cpu().numpy()                                              # (F_sae, T, D)

    matches, chamfer = _match_curves(gt_points, learned_points)

    # --- Per-feature geometry ---
    per_feature = []
    for i in range(n_features):
        j = int(matches[i])
        cloud = learned_points[j]
        per_feature.append({
            "gt_index": i,
            "name": dataset.features[i].name,
            "periodic": dataset.features[i].periodic,
            "d_proj": int(bases[i].shape[0]),
            "sae_match_index": j,
            "chamfer": float(chamfer[i]),
            "subspace_cos": _principal_angle_cos(bases[i], cloud),
            "leakage": _leakage(bases[i], cloud),
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
          f"leakage mean={report['leakage_mean']:.3f}  dead={report['dead_feature_count']}/{n_sae}")
    print(f"\n  {'manifold':<10} {'chamfer':>8} {'subsp_cos':>10} {'leakage':>8}  match")
    print("  " + "-" * 50)
    for p in per_feature:
        flag = "" if (p["chamfer"] < 0.3 and p["subspace_cos"] > 0.9) else "  <-- weak"
        print(f"  {p['name']:<10} {p['chamfer']:>8.3f} {p['subspace_cos']:>10.3f} "
              f"{p['leakage']:>8.3f}  #{p['sae_match_index']}{flag}")

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
