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

import argparse
import json
import sys
import time
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from manifold_sae.data_synthetic import SyntheticDataset, chamfer_distance


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manifold-SAE synthetic recovery experiment")
    p.add_argument("--d-ambient", type=int, default=64)
    p.add_argument("--n-features", type=int, default=5)
    p.add_argument("--n-basis", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    # v1: every feature uses 1D Duchon on [0, 1]. Cyclic concepts (if any) are
    # fit as approximately-closed open curves per the spec — no per-run flag.
    p.add_argument("--sparsity-weight", type=float, default=1e-3)
    p.add_argument("--reml-weight", type=float, default=1e-2)
    p.add_argument("--position-spread-weight", type=float, default=1e-3)
    p.add_argument("--slack-features", type=int, default=2,
                   help="Extra learned features beyond planted ground truth (acts as slack).")
    p.add_argument("--t-grid-size", type=int, default=128,
                   help="Number of t samples used to probe each learned curve.")
    p.add_argument("--chamfer-threshold", type=float, default=0.3,
                   help="Mean chamfer above this -> nonzero exit code.")
    p.add_argument("--output-dir", type=str, default="runs/synthetic_recovery")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


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
    from manifold_sae.decoder import extract_feature_curves  # noqa: WPS433

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


def _maybe_plot(
    output_dir: Path,
    gt_points: np.ndarray,
    learned_points: np.ndarray,
    matches: np.ndarray,
    feature_names: list[str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping plots", file=sys.stderr)
        return

    F = gt_points.shape[0]
    # PCA on the union of GT and matched learned points so the projection is shared.
    union = np.concatenate(
        [gt_points.reshape(-1, gt_points.shape[-1])]
        + [learned_points[m].reshape(-1, learned_points.shape[-1]) for m in matches if m >= 0],
        axis=0,
    )
    union -= union.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(union, full_matrices=False)
    pcs = vt[:2]  # (2, D)

    cols = min(F, 3)
    rows = int(np.ceil(F / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i in range(F):
        ax = axes[i // cols, i % cols]
        gt2 = gt_points[i] @ pcs.T
        ax.plot(gt2[:, 0], gt2[:, 1], "o-", color="C0", markersize=2, label="ground truth")
        m = int(matches[i])
        if m >= 0:
            lp2 = learned_points[m] @ pcs.T
            ax.plot(lp2[:, 0], lp2[:, 1], "x-", color="C1", markersize=3, label="learned")
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] building SyntheticDataset (d_ambient={args.d_ambient}, "
          f"n_features={args.n_features}, n_samples={args.n_samples})")
    dataset = SyntheticDataset(
        d_ambient=args.d_ambient,
        n_features=args.n_features,
        n_samples=args.n_samples,
        sparsity=args.sparsity,
        noise=args.noise,
        seed=args.seed,
        t_grid_size=args.t_grid_size,
    )
    loader = _build_loader(dataset, args.batch_size, args.seed)

    # Import the swarm-built modules late so we can give a helpful error if a
    # sibling module isn't yet present (gamfit_glue / sae / train / losses).
    try:
        from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
        from manifold_sae.train import build_optimizer, train
        from manifold_sae.diagnostics import dead_feature_mask, position_variance
    except ImportError as e:
        print(f"[fatal] swarm module not importable yet: {e}", file=sys.stderr)
        report = {
            "status": "import_error",
            "error": str(e),
            "args": vars(args),
        }
        (output_dir / "synthetic_recovery_results.json").write_text(json.dumps(report, indent=2))
        return 2

    n_sae_features = args.n_features + args.slack_features
    config = ManifoldSAEConfig(
        input_dim=args.d_ambient,
        n_features=n_sae_features,
        n_basis=args.n_basis,
        sparsity_weight=args.sparsity_weight,
        reml_weight=args.reml_weight,
        position_spread_weight=args.position_spread_weight,
    )
    sae = ManifoldSAE(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae.to(device)

    optimizer = build_optimizer(sae, lr=args.lr)

    print(f"[train] {args.n_steps} steps, batch={args.batch_size}, lr={args.lr}, "
          f"n_sae_features={n_sae_features}, n_basis={args.n_basis}")
    t0 = time.time()
    history = train(sae, loader, optimizer, n_steps=args.n_steps, log_every=max(args.n_steps // 20, 1))
    train_time = time.time() - t0

    # Evaluation -------------------------------------------------------------
    print("[eval] probing learned curves")
    t_grid = dataset.ground_truth["t_grid"]
    gt_points = dataset.ground_truth["curve_points"]  # (F_gt, T_grid, D)
    learned_points = _probe_learned_curves(
        sae, t_grid, n_sae_features, device=device, activations=dataset.x
    )

    matches, costs = _match_curves(gt_points, learned_points)

    # Diagnostics over a final batch.
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
        for i in range(args.n_features)
    ]
    report = {
        "status": "ok",
        "args": vars(args),
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

    if not args.no_plot:
        _maybe_plot(output_dir, gt_points, learned_points, matches, feature_names)

    if report["chamfer_mean"] > args.chamfer_threshold:
        print(f"[fail] chamfer mean {report['chamfer_mean']:.4f} > threshold "
              f"{args.chamfer_threshold}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
