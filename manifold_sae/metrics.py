"""Honest recovery metrics for the manifold-SAE synthetic experiment.

Solves the optimal one-to-one assignment between SAE features and ground-truth
features (Hungarian algorithm), then reports recovery quality on matched pairs:

  - Position Spearman: rank correlation of encoder positions with planted t,
    averaged over matched pairs. Reparameterization-invariant; the right
    "did the encoder learn where each token sits on the curve?" metric.
  - Activation precision/recall: agreement between SAE binary firing and GT
    binary active flags, on matched pairs.
  - Ambient direction cosine: cosine similarity between the planted projection
    matrix's column span and the SAE feature's W_k column span, per matched
    pair. Reports whether the SAE found the right ambient subspace.

Chamfer remains useful as a coarse curve-shape diagnostic but is not a
success criterion.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from .data_synthetic import chamfer_distance, SyntheticDataset
from .sae import ManifoldSAE


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Returns |ρ| so sign-flips count as success."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    rho = spearmanr(a, b).correlation
    if not np.isfinite(rho):
        return 0.0
    return float(abs(rho))


def _subspace_cosine(W_sae: np.ndarray, P_gt: np.ndarray) -> float:
    """Largest principal-angle cosine between two subspaces.

    ``W_sae`` is (D, R_sae), ``P_gt`` is (D, R_gt) representing the planted
    projection matrix's columns. Returns cos(min principal angle) ∈ [0, 1].
    """
    return float(np.cos(subspace_angles(W_sae, P_gt).min()))


@torch.no_grad()
def hungarian_matched_recovery(
    sae: ManifoldSAE,
    dataset: SyntheticDataset,
    chamfer_curves_sae: np.ndarray,
    chamfer_curves_gt: np.ndarray,
) -> dict:
    """Compute Hungarian-matched recovery metrics.

    ``chamfer_curves_sae``: (F_sae, T, D) probed SAE curves at uniform t-grid.
    ``chamfer_curves_gt``:  (F_gt,  T, D) planted GT curve point clouds.

    Returns dict with per-feature matched metrics and the cost matrix used.
    """
    device = next(sae.parameters()).device
    out = sae(dataset.x.to(device=device, dtype=sae.cfg.dtype))
    # positions is now (N, F_sae, d); take the first manifold coordinate.
    sae_pos = out.positions[..., 0].cpu().numpy()        # (N, F_sae)
    sae_amp = out.amplitudes.cpu().numpy()               # (N, F_sae) binary
    # The old per-atom ambient ``directions`` field (F, D, R) is gone. The
    # gamfit decoder block ``decoder_blocks[k]`` is (K, D) and already lives
    # in ambient R^D; the atom's ambient subspace is spanned by the top-R
    # right-singular vectors of that block. We build (F_sae, D, R) to match
    # the downstream ``_subspace_cosine`` contract.
    blocks = sae.decoder_blocks.detach().cpu().numpy()   # (F_sae, K, D)
    R = int(sae.cfg.intrinsic_rank)
    F_sae_blocks = blocks.shape[0]
    D = blocks.shape[2]
    sae_W = np.zeros((F_sae_blocks, D, R), dtype=np.float64)
    for k in range(F_sae_blocks):
        # right-singular vectors Vt: (min(K,D), D); rows span the ambient image
        _, _, Vt = np.linalg.svd(blocks[k], full_matrices=False)
        r = min(R, Vt.shape[0])
        sae_W[k, :, :r] = Vt[:r].T                        # (D, r)

    gt_active = dataset.ground_truth["active"].numpy().astype(bool)  # (N, F_gt)
    gt_ts = dataset.ground_truth["ts"].numpy()                       # (N, F_gt)
    # Each GT feature has its own projection of shape (d_intrinsic_k, D); they
    # have different d_intrinsic so we keep them as a list.
    gt_projections = [f.projection for f in dataset.features]  # list of (R_int_k, D)

    F_gt = gt_active.shape[1]
    F_sae = sae_amp.shape[1]

    # Cost matrix for assignment: 1 - max(0, position-Spearman) on co-firing
    # tokens. Captures "does this SAE feature track this GT feature's t?".
    cost = np.full((F_gt, F_sae), 1.0, dtype=np.float64)
    for j in range(F_gt):
        for k in range(F_sae):
            both = gt_active[:, j] & (sae_amp[:, k] > 0.5)
            if both.sum() < 5:
                continue
            sp = _spearman(sae_pos[both, k], gt_ts[both, j])
            cost[j, k] = 1.0 - sp

    row_ind, col_ind = linear_sum_assignment(cost)

    per_pair = []
    for j, k in zip(row_ind, col_ind):
        both = gt_active[:, j] & (sae_amp[:, k] > 0.5)
        n_both = int(both.sum())
        sp = _spearman(sae_pos[both, k], gt_ts[both, j]) if n_both >= 5 else 0.0
        # Activation agreement: precision/recall of SAE firing vs GT active.
        tp = int(((sae_amp[:, k] > 0.5) & gt_active[:, j]).sum())
        fp = int(((sae_amp[:, k] > 0.5) & ~gt_active[:, j]).sum())
        fn = int((~(sae_amp[:, k] > 0.5) & gt_active[:, j]).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        P = gt_projections[j].T  # (D, R_int_j) — varies by feature
        Wk = sae_W[k]            # (D, R)
        cos = _subspace_cosine(Wk, P)
        # Chamfer of matched curves
        cham = chamfer_distance(chamfer_curves_gt[j], chamfer_curves_sae[k])
        per_pair.append({
            "gt_index": int(j),
            "gt_name": dataset.features[j].name,
            "sae_index": int(k),
            "position_spearman": sp,
            "activation_precision": precision,
            "activation_recall": recall,
            "subspace_cosine": cos,
            "chamfer": cham,
            "n_co_firing": n_both,
        })

    return {
        "per_pair": per_pair,
        "mean_position_spearman": float(np.mean([p["position_spearman"] for p in per_pair])),
        "mean_subspace_cosine": float(np.mean([p["subspace_cosine"] for p in per_pair])),
        "mean_activation_f1": float(np.mean([
            2 * p["activation_precision"] * p["activation_recall"]
            / max(p["activation_precision"] + p["activation_recall"], 1e-12)
            for p in per_pair
        ])),
        "mean_chamfer": float(np.mean([p["chamfer"] for p in per_pair])),
        "cost_matrix": cost.tolist(),
    }


def print_recovery_summary(metrics: dict) -> None:
    """Pretty-print Hungarian-matched recovery metrics."""
    print("Hungarian-matched recovery:")
    print(f"  mean position Spearman: {metrics['mean_position_spearman']:.3f}  (1.0 = perfect)")
    print(f"  mean subspace cosine:   {metrics['mean_subspace_cosine']:.3f}  (1.0 = exact subspace)")
    print(f"  mean activation F1:     {metrics['mean_activation_f1']:.3f}  (1.0 = perfect firing pattern)")
    print(f"  mean chamfer:           {metrics['mean_chamfer']:.4f}  (diagnostic, lower better)")
    print()
    print(f"  {'GT':12s} {'SAE':>5s} {'Spearman':>10s} {'subspace':>10s} {'prec':>6s} {'rec':>6s} {'cham':>7s}")
    for p in metrics["per_pair"]:
        print(f"  {p['gt_name']:12s} {p['sae_index']:>5d} "
              f"{p['position_spearman']:>10.3f} {p['subspace_cosine']:>10.3f} "
              f"{p['activation_precision']:>6.2f} {p['activation_recall']:>6.2f} "
              f"{p['chamfer']:>7.4f}")
