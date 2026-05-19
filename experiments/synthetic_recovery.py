"""Synthetic-recovery experiment for Manifold-SAE.

End-to-end validation that the SAE recovers planted 1D feature manifolds:

1. Build a :class:`SyntheticDataset` with N known ground-truth curves embedded in
   ``R^d_ambient`` via random orthogonal projections.
2. Build a :class:`ManifoldSAE` with a small dictionary surplus (``n_features + 2``)
   so the SAE has a small amount of slack but must still genuinely use the planted
   curves to drive reconstruction error down.
3. Train via :func:`manifold_sae.train.train`.
4. Probe the trained decoder on a unit-amplitude, single-feature design at a uniform
   t-grid; the resulting reconstructions form *learned curves* in ambient space.
5. Match each ground-truth curve to its best-fitting learned curve by symmetric
   Chamfer distance (gauge-free over reparameterization / sign flips), and report
   per-feature and mean recovery quality.

The script is designed to run end-to-end against the contract documented in the
README and the swarm's interface notes. If sibling modules aren't yet importable,
the script reports the import error and exits nonzero rather than silently
falling back.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from manifold_sae.data_synthetic import SyntheticDataset, chamfer_distance


@dataclass(frozen=True)
class Config:
    """Synthetic-recovery experiment configuration.

    All knobs live here; edit at the bottom of the file (or import this module
    and call ``main(Config(...))``) to override. No CLI.
    """

    d_ambient: int = 64
    n_features: int = 5
    n_basis: int = 12
    n_samples: int = 8192
    sparsity: float = 0.3
    noise: float = 0.05
    n_steps: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0
    sparsity_weight: float = 1e-3
    top_k: int = 2
    intrinsic_rank: int = 3
    ortho_weight: float = 1e-2
    reml_weight: float = 1.0
    continuous_amp: bool = False
    # Curriculum: train first on samples with only `curriculum_start_active`
    # GT features active, ramping linearly to all-active over
    # `curriculum_steps` steps. Forces each SAE feature to first specialize
    # on a single GT direction before being asked to handle superpositions.
    curriculum_start_active: int = 1
    curriculum_steps: int = 2000
    # Extra learned features beyond planted ground truth, acts as slack.
    slack_features: int = 2
    # Number of t samples used to probe each learned curve.
    t_grid_size: int = 128
    # Mean chamfer above this -> nonzero exit code.
    chamfer_threshold: float = 0.3
    output_dir: str = "runs/synthetic_recovery"
    plot: bool = True


DEFAULT_CONFIG = Config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_loader(dataset: SyntheticDataset, batch_size: int, seed: int) -> Iterable[torch.Tensor]:
    """A simple infinite shuffled minibatch iterator over ``dataset.x``."""
    g = torch.Generator().manual_seed(seed + 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )


def _curriculum_loader(
    dataset: SyntheticDataset, batch_size: int, max_active: int, seed: int,
) -> Iterable[torch.Tensor]:
    """Yield batches of samples whose GT active-feature count is <= max_active."""
    active = dataset.ground_truth["active"]  # (N, F)
    counts = active.to(torch.int64).sum(dim=1)
    keep = (counts <= max_active) & (counts >= 1)
    idx = torch.nonzero(keep, as_tuple=True)[0].numpy()
    if len(idx) < batch_size:
        # Not enough samples — fall back to all data.
        return _build_loader(dataset, batch_size, seed)
    subset = torch.utils.data.Subset(dataset, idx.tolist())
    g = torch.Generator().manual_seed(seed + 1)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
    )


def _probe_learned_curves(
    sae,
    t_grid: np.ndarray,
    n_features_sae: int,
    device: torch.device,
    activations: torch.Tensor,
) -> np.ndarray:
    """Probe the decoder with a single-feature unit-amplitude design at each t in the grid.

    For each learned feature ``k`` and each ``t`` in ``t_grid``:

    - Build a batch where positions are uninformative for the inactive features
      (set to the constant ``0.5``) and equal to ``t`` for feature ``k``.
    - Set amplitudes to a one-hot at ``k``.
    - Read the SAE's reconstruction; that vector is the learned curve at ``t``.

    Because gamfit fits coefficients to the *batch's* targets, we pass the
    ground-truth-free dataset activation as the regression target. The decoder
    has been trained jointly with the encoder, so the resulting coefficients are
    determined primarily by the planted activation manifold, not by this specific
    probe batch. We average over a small batch dimension to stabilize.

    This is intentionally a side-channel evaluator: it doesn't go through the
    encoder at all (per the experiment contract), but it does require that the
    fitted coefficients be sensible for the probe positions. See the experiment
    description in the swarm brief.

    Returns
    -------
    ndarray
        Shape ``(F_sae, T_grid, d_ambient)``.
    """
    from manifold_sae.sae import extract_feature_curves

    # The probe needs real activations as fit targets — passing zeros makes
    # gamfit's inner solve collapse every coefficient to zero and the probed
    # curves come out identically zero (a real bug we tripped over earlier).
    # extract_feature_curves uses the canonical pattern: one feature at a
    # time, unit amplitude, real activations as target.
    t_grid_t = torch.from_numpy(t_grid.astype(np.float64))
    curves = extract_feature_curves(sae, activations.to(device), t_grid_t)
    return curves.cpu().numpy()


def _match_curves(
    gt_points: np.ndarray, learned_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bipartite-style greedy match by Chamfer distance.

    ``gt_points`` is ``(F_gt, T, D)`` and ``learned_points`` is ``(F_sae, T, D)``.
    Returns:

    - ``matches``: int array of length ``F_gt`` indexing into learned features.
    - ``costs``: chamfer distance per ground-truth feature.
    """
    F_gt = gt_points.shape[0]
    F_sae = learned_points.shape[0]
    cost = np.zeros((F_gt, F_sae), dtype=np.float64)
    for i in range(F_gt):
        for j in range(F_sae):
            cost[i, j] = chamfer_distance(gt_points[i], learned_points[j])

    matches = np.full(F_gt, -1, dtype=np.int64)
    costs = np.full(F_gt, np.inf, dtype=np.float64)
    used = np.zeros(F_sae, dtype=bool)

    # Greedy: pick the cheapest unassigned (i, j) pair until all gt assigned.
    order = np.argsort(cost.ravel())
    for flat in order:
        i, j = divmod(int(flat), F_sae)
        if matches[i] != -1 or used[j]:
            continue
        matches[i] = j
        costs[i] = cost[i, j]
        used[j] = True
        if (matches != -1).all():
            break

    return matches, costs


def _plot_curves(
    output_dir: Path,
    gt_points: np.ndarray,
    learned_points: np.ndarray,
    matches: np.ndarray,
    feature_names: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = gt_points.shape[0]
    cols = min(F, 3)
    rows = int(np.ceil(F / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    # Use the planted projection matrix per feature as the 2D plotting basis.
    # This is the "natural" view of each feature: its planted ambient subspace.
    # When the SAE recovers the right subspace, learned and GT curves end up
    # in the same plane here and the visual is faithful.
    from manifold_sae.data_synthetic import SyntheticDataset  # noqa: F401
    for i in range(F):
        ax = axes[i // cols, i % cols]
        m = int(matches[i])
        gt_i = gt_points[i]
        learned_i = learned_points[m] if m >= 0 else np.zeros_like(gt_i)
        # Joint-PCA + Procrustes-aligned plot: builds a 2D basis on the
        # combined point cloud so neither curve is privileged, then
        # solves the optimal rotation+scale to overlay the learned onto
        # GT. This is the visualization that honestly tracks the
        # chamfer (which itself centers + Frob-normalizes both curves).
        gt_c = gt_i - gt_i.mean(axis=0, keepdims=True)
        lp_c = learned_i - learned_i.mean(axis=0, keepdims=True)
        # Frobenius-normalize each independently (matches chamfer's gauge).
        gt_n = gt_c / max(np.linalg.norm(gt_c), 1e-12)
        lp_n = lp_c / max(np.linalg.norm(lp_c), 1e-12)
        # Joint-PCA basis.
        joint = np.concatenate([gt_n, lp_n], axis=0)
        _, _, vt = np.linalg.svd(joint, full_matrices=False)
        pcs = vt[:2]
        gt2 = gt_n @ pcs.T
        lp2 = lp_n @ pcs.T
        # Procrustes: optimal 2x2 rotation+reflection (Frobenius norm fixed).
        M = lp2.T @ gt2
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
        lp2_aligned = lp2 @ Q
        ax.plot(gt2[:, 0], gt2[:, 1], "o-", color="C0", markersize=2, label="ground truth")
        ax.plot(lp2_aligned[:, 0], lp2_aligned[:, 1], "x-", color="C1", markersize=3, label="learned")
        ax.set_title(f"{feature_names[i]} (sae idx {m})")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    for j in range(F, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout()
    out = output_dir / "synthetic_recovery_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config = DEFAULT_CONFIG) -> int:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] SyntheticDataset(d_ambient={cfg.d_ambient}, "
          f"n_features={cfg.n_features}, n_samples={cfg.n_samples})")
    dataset = SyntheticDataset(
        d_ambient=cfg.d_ambient,
        n_features=cfg.n_features,
        n_samples=cfg.n_samples,
        sparsity=cfg.sparsity,
        noise=cfg.noise,
        seed=cfg.seed,
        t_grid_size=cfg.t_grid_size,
    )
    loader = _build_loader(dataset, cfg.batch_size, cfg.seed)

    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    from manifold_sae.train import build_optimizer, train
    from manifold_sae.diagnostics import dead_feature_mask, position_variance

    n_sae_features = cfg.n_features + cfg.slack_features
    sae_config = ManifoldSAEConfig(
        input_dim=cfg.d_ambient,
        n_features=n_sae_features,
        n_basis=cfg.n_basis,
        sparsity_weight=cfg.sparsity_weight,
        top_k=cfg.top_k,
        intrinsic_rank=cfg.intrinsic_rank,
        ortho_weight=cfg.ortho_weight,
        reml_weight=cfg.reml_weight,
        continuous_amp=cfg.continuous_amp,
    )
    sae = ManifoldSAE(sae_config)

    # CPU is faster than MPS at this small scale (D=64, F~10); MPS
    # overhead dominates per-kernel launch. Real LLM-scale runs should
    # use CUDA explicitly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    sae.to(device)

    optimizer = build_optimizer(sae, lr=cfg.lr)

    print(f"[train] {cfg.n_steps} steps, batch={cfg.batch_size}, lr={cfg.lr}, "
          f"n_sae_features={n_sae_features}, n_basis={cfg.n_basis}, "
          f"curriculum: {cfg.curriculum_start_active}->{cfg.n_features} active over {cfg.curriculum_steps} steps")
    t0 = time.time()
    history = {"step": [], "mse": [], "sparsity": [], "position_variance_mean": [], "dead_feature_count": [], "grad_ratio_mean": []}
    # Curriculum schedule: ramp max_active from start to full over curriculum_steps.
    n_phases = cfg.n_features - cfg.curriculum_start_active + 1
    phase_size = max(cfg.curriculum_steps // max(n_phases, 1), 1)
    step_done = 0
    for phase in range(n_phases):
        max_active = cfg.curriculum_start_active + phase
        if phase == n_phases - 1:
            steps_this = cfg.n_steps - step_done
        else:
            steps_this = phase_size
        if steps_this <= 0:
            continue
        loader_phase = _curriculum_loader(dataset, cfg.batch_size, max_active, cfg.seed + phase)
        print(f"[curriculum] phase {phase}: max_active={max_active}, {steps_this} steps")
        h = train(sae, loader_phase, optimizer, n_steps=steps_this, log_every=max(steps_this // 5, 1))
        for k, v in h.items():
            history.setdefault(k, []).extend(v)
        step_done += steps_this
    train_time = time.time() - t0

    print("[eval] probing learned curves")
    t_grid = dataset.ground_truth["t_grid"]
    gt_points = dataset.ground_truth["curve_points"]  # (F_gt, T_grid, D)
    learned_points = _probe_learned_curves(
        sae, t_grid, n_sae_features, device=device, activations=dataset.x
    )

    matches, costs = _match_curves(gt_points, learned_points)

    sae.eval()
    with torch.no_grad():
        eval_batch = dataset.x[: min(1024, len(dataset))].to(device)
        out = sae(eval_batch)
        pos_var = position_variance(out).cpu().numpy().tolist()
        dead = dead_feature_mask(out).cpu().numpy().tolist()
        mse = float(torch.mean((out.reconstruction - eval_batch) ** 2).item())
    sae.train()

    feature_names = [f.name for f in dataset.features]
    per_feature = [
        {
            "gt_index": i,
            "name": feature_names[i],
            "periodic": dataset.features[i].periodic,
            "sae_match_index": int(matches[i]),
            "chamfer": float(costs[i]),
        }
        for i in range(cfg.n_features)
    ]
    report = {
        "status": "ok",
        "config": asdict(cfg),
        "train_seconds": train_time,
        "mse_eval": mse,
        "chamfer_per_feature": per_feature,
        "chamfer_mean": float(np.mean(costs)),
        "chamfer_max": float(np.max(costs)),
        "position_variance": pos_var,
        "dead_feature_mask": dead,
        "dead_feature_count": int(sum(dead)),
        "history": history,
    }
    out_path = output_dir / "synthetic_recovery_results.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"[eval] wrote {out_path}")
    print(f"[eval] chamfer mean={report['chamfer_mean']:.4f} max={report['chamfer_max']:.4f} "
          f"dead={report['dead_feature_count']}")

    if cfg.plot:
        _plot_curves(output_dir, gt_points, learned_points, matches, feature_names)

    if report["chamfer_mean"] > cfg.chamfer_threshold:
        print(f"[fail] chamfer mean {report['chamfer_mean']:.4f} > threshold "
              f"{cfg.chamfer_threshold}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
