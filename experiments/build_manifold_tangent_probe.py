"""Build a TRUE-manifold-tangent probe for cogito-probed.

A "manifold direction" is a tangent vector to the fitted color manifold,
not a linear contrast of centroids. We fit the U_3d alternating-Duchon
nonlinear manifold f: ℝ³ → ℝ⁷¹⁶⁸ on the 886 filtered xkcd centroids
(top-6-template averages), then at several base points compute the
Jacobian columns ∂f/∂tᵢ. Each column is a 7168-vector tangent to the
manifold at that base point — the *local axis* of the manifold there.

Steering along these directions moves cogito along the manifold's
natural geometry, not along an arbitrary linear contrast.

Saved probes:
  • t1_at_center, t2_at_center, t3_at_center  — tangents at the latent
    midpoint (0.5, 0.5, 0.5), the "average color" basepoint
  • t1_at_{red,green,blue,yellow,black,white}_etc. — tangents at the
    nearest-on-manifold projection of each canonical color
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors, load_harvest
from color_filter_list import filter_colors


N_T = 28
LAYER = 40
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
D_LATENT = 3            # U_3d
EPS = 1e-3              # finite-diff step for Jacobian


def main() -> int:
    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X_full = load_harvest(cache)
    n_colors = X_full.shape[0] // N_T
    X_full = X_full[: n_colors * N_T]

    centroids_all = np.zeros((n_colors, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        centroids_all[ci] = X_full[rows].mean(axis=0)

    colors_all = load_xkcd_colors()[:n_colors]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids_all[kept_idx]
    names = [c[0] for c in kept]
    name_to_idx = {n: i for i, n in enumerate(names)}
    print(f"[probe] fitting U_3d on {len(kept)} centroids (each 7168-D)", flush=True)

    # Fit U_3d via the existing alternating-Duchon helper. Uses gamfit's
    # REML for λ selection; returns dict with T (per-color latent, 0..1
    # in each of 3 dims), B (basis_cols × 7168 coefficients), centers
    # (n_centers × 3 lattice).
    from color_manifold_gam import (
        fit_unsupervised_manifold, duchon_basis_radial, Config,
    )
    cfg = Config()
    fit = fit_unsupervised_manifold(
        centroids, d=D_LATENT, cfg=cfg, n_iters=20, verbose=False,
    )
    T = fit["T"]                                  # (n_colors, 3) latent
    B = fit["B"]                                  # (basis_cols, 7168)
    centers = fit["centers"]                      # (K, 3) Duchon centers
    print(f"[probe] U_3d fit done. T range={T.min(0)}..{T.max(0)}, "
          f"basis cols={B.shape[0]}, K_centers={centers.shape[0]}, "
          f"log_lambda={fit['log_lambda']:+.2f}", flush=True)

    def smooth_at(t: np.ndarray) -> np.ndarray:
        """Evaluate the smooth f(t) ∈ ℝ⁷¹⁶⁸ at the latent point t ∈ ℝ³."""
        t2 = np.atleast_2d(t).astype(np.float64)
        Phi, _ = duchon_basis_radial(t2, centers)        # (1, basis_cols)
        return (Phi @ B).ravel()                          # (7168,)

    def jacobian_at(t: np.ndarray) -> np.ndarray:
        """Central finite-difference Jacobian ∂f/∂t ∈ ℝ^(3 × 7168) at t."""
        J = np.zeros((D_LATENT, B.shape[1]), dtype=np.float32)
        for i in range(D_LATENT):
            e = np.zeros(D_LATENT); e[i] = EPS
            f_plus  = smooth_at(np.clip(t + e, 0.001, 0.999))
            f_minus = smooth_at(np.clip(t - e, 0.001, 0.999))
            J[i] = ((f_plus - f_minus) / (2 * EPS)).astype(np.float32)
        return J

    def normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-12)

    # Pick base points on the manifold and compute tangent triples
    base_points = {"center": np.array([0.5, 0.5, 0.5])}

    # For each named anchor color, use its discovered latent t as the base
    canonical_names = ["red", "orange", "yellow", "green", "blue",
                        "purple", "pink", "black", "white", "grey"]
    for nm in canonical_names:
        if nm in name_to_idx:
            base_points[nm] = T[name_to_idx[nm]].copy()
        else:
            print(f"  ! {nm} not in filtered colors, skipping anchor")

    all_dirs = []
    all_labels = []
    for base_name, t_base in base_points.items():
        J = jacobian_at(t_base)
        for i in range(D_LATENT):
            v = normalize(J[i])
            all_dirs.append(v)
            all_labels.append(f"t{i+1}_at_{base_name}")
        print(f"  base={base_name:8s}  t={t_base.round(3).tolist()}  "
              f"tangent norms={[float(np.linalg.norm(J[i])) for i in range(D_LATENT)]}",
              flush=True)

    all_dirs = np.stack(all_dirs).astype(np.float32)
    assert all_dirs.shape[1] == 7168, all_dirs.shape

    out_path = cache.parent / "color_manifold_probes_layer40.npz"
    desc = (
        f"True-manifold tangent probes: at {len(base_points)} base points on "
        f"the U_3d nonlinear color manifold (alternating-Duchon fit, REML λ), "
        f"the 3 Jacobian columns ∂f/∂t_i (i=1,2,3) are normalized to unit "
        f"vectors in ℝ⁷¹⁶⁸. Each base point = (center) midpoint of the "
        f"latent unit cube, or the discovered latent of a named anchor color "
        f"(red/orange/yellow/green/blue/purple/pink/black/white/grey). "
        f"Fitted on {len(kept)} xkcd centroids at layer 40 using the 6 "
        f"color-focused templates."
    )
    np.savez(out_path,
              directions=all_dirs,
              labels=np.array(all_labels),
              description=desc)
    print(f"\n[saved] {out_path}")
    print(f"  total directions: {len(all_labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())