"""auto_68.py — Empirical test of the gauge-invariance critique of U_3d.

CLAIM: the alternating loss Z ≈ Φ(T) · B is invariant under T ↦ T∘φ for any
diffeomorphism φ. So multi-seed U_3d fits should give the same embedded
manifold f([0,1]^3) ⊂ Z, viewed through different charts.

We fit U_3d 8 times with different random-init T ∼ Uniform[0,1]^3, then
compare:
  (1) Procrustes T-distance (gauge-DEPENDENT, expected: LARGE)
  (2) Hausdorff/mean-NN distance in Z between f_i(grid) and f_j(grid)
      (gauge-INVARIANT, expected: ~0 if claim holds)
  (3) Train R² per seed (gauge-INVARIANT, expected: ≈identical)
  (4) Per-color Z disagreement between f_i(T_i_color) and f_j(T_j_color)
      (gauge-INVARIANT, expected: small).

No Gaussian RBF; no Duchon length_scale; no B-splines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors
from color_filter_list import filter_colors
from _pca_basis import load_pc_basis, project
from color_manifold_gam import (
    fit_unsupervised_manifold,
    Config,
    duchon_basis_radial,
)

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
K_PC = 64
N_SEEDS = 8
N_ITERS = 15
GRID_PER_AXIS = 10  # 10^3 = 1000 dense grid points in [0,1]^3


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized orthogonal-Procrustes distance allowing scaling+rotation+
    translation. Returns sqrt(1 - sum(s)^2 / (||Ac||·||Bc||)) ∈ [0, 1],
    where Ac,Bc are centered. 0 = identical up to rigid+scale; 1 = orthogonal.
    Permutation/labelling assumed shared (row i in A and B are the same color).
    """
    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    nA = np.linalg.norm(Ac)
    nB = np.linalg.norm(Bc)
    if nA == 0 or nB == 0:
        return 1.0
    Ac = Ac / nA
    Bc = Bc / nB
    M = Bc.T @ Ac
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return float(np.sqrt(max(0.0, 1.0 - (s.sum()) ** 2)))


def cloud_mean_nn_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance (Chamfer-like)."""
    tP = cKDTree(P)
    tQ = cKDTree(Q)
    dPQ, _ = tQ.query(P, k=1)
    dQP, _ = tP.query(Q, k=1)
    return float(0.5 * (dPQ.mean() + dQP.mean()))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load X via mmap, build centroids
    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X = np.load(cache, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    print(f"[auto_68] X shape={X.shape}, n_raw={n_raw}")

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    N = len(kept)

    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    z_norm_typical = float(np.linalg.norm(Z, axis=1).mean())
    print(f"[auto_68] N={N}, Z.shape={Z.shape}, typical ||Z||={z_norm_typical:.3f}")

    cfg = Config()

    # Build dense grid in [0,1]^3
    g = np.linspace(0.0, 1.0, GRID_PER_AXIS)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    print(f"[auto_68] dense grid: {grid.shape}")

    fits = []
    for seed in range(N_SEEDS):
        rs = np.random.RandomState(seed)
        init_T = rs.uniform(size=(N, 3))
        print(f"[auto_68] seed {seed}: fitting U_3d (n_iters={N_ITERS})...")
        fit = fit_unsupervised_manifold(
            Z, d=3, cfg=cfg, n_iters=N_ITERS, init_T=init_T, verbose=False,
        )
        T = fit["T"]
        centers = fit["centers"]
        B = fit["B"]

        # train R^2
        Phi_t, _ = duchon_basis_radial(T, centers)
        Z_hat = Phi_t @ B
        train_r2 = r2_macro(Z, Z_hat)

        # f(grid) -> embedded image in Z
        Phi_g, _ = duchon_basis_radial(grid, centers)
        Z_grid = Phi_g @ B
        print(f"   seed {seed}: train R² = {train_r2:+.4f}, "
              f"Z_grid norm med = {np.linalg.norm(Z_grid, axis=1).mean():.3f}")
        fits.append({
            "seed": seed, "T": T, "centers": centers, "B": B,
            "train_r2": float(train_r2), "Z_grid": Z_grid,
            "Z_at_colors": Z_hat,
        })

    # --- Pairwise comparisons ---
    proc_mat = np.zeros((N_SEEDS, N_SEEDS))
    haus_mat = np.zeros((N_SEEDS, N_SEEDS))
    color_disagree_01 = None
    for i in range(N_SEEDS):
        for j in range(N_SEEDS):
            if i == j:
                continue
            proc_mat[i, j] = procrustes_distance(fits[i]["T"], fits[j]["T"])
            haus_mat[i, j] = cloud_mean_nn_distance(
                fits[i]["Z_grid"], fits[j]["Z_grid"]
            )
    # per-color Z disagreement: ||f_i(T_i_color) - f_j(T_j_color)||
    # Z_at_colors already gives f(T_color) for each seed since T are the
    # in-sample latents and B is fit at them. Compare seed 0 vs seed 1.
    color_disagree_01 = np.linalg.norm(
        fits[0]["Z_at_colors"] - fits[1]["Z_at_colors"], axis=1,
    )

    median_color_disagree = float(np.median(color_disagree_01))
    median_haus = float(np.median(haus_mat[haus_mat > 0]))
    median_proc = float(np.median(proc_mat[proc_mat > 0]))
    print(f"\n[auto_68] median Procrustes(T) = {median_proc:.4f}")
    print(f"[auto_68] median Hausdorff(Z-grid) = {median_haus:.4f}")
    print(f"[auto_68] median per-color disagree (Z) = {median_color_disagree:.4f}")
    print(f"[auto_68] typical ||Z|| = {z_norm_typical:.4f}")
    print(f"[auto_68] disagree / ||Z|| = {median_color_disagree / z_norm_typical:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    im0 = axes[0, 0].imshow(proc_mat, cmap="viridis", vmin=0, vmax=max(proc_mat.max(), 1e-3))
    axes[0, 0].set_title(
        f"(1) Procrustes T-distance [gauge-DEPENDENT]\n"
        f"median off-diag = {median_proc:.3f}"
    )
    axes[0, 0].set_xlabel("seed j"); axes[0, 0].set_ylabel("seed i")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(haus_mat, cmap="viridis", vmin=0, vmax=max(haus_mat.max(), 1e-3))
    axes[0, 1].set_title(
        f"(2) Mean-NN distance of f(grid) in Z [gauge-INVARIANT]\n"
        f"median off-diag = {median_haus:.3f}  ({100*median_haus/z_norm_typical:.1f}% of typical ||Z||)"
    )
    axes[0, 1].set_xlabel("seed j"); axes[0, 1].set_ylabel("seed i")
    plt.colorbar(im1, ax=axes[0, 1])

    r2s = [f["train_r2"] for f in fits]
    axes[1, 0].bar(range(N_SEEDS), r2s, color="steelblue", edgecolor="black")
    axes[1, 0].set_xlabel("seed"); axes[1, 0].set_ylabel("train R²")
    axes[1, 0].set_title(
        f"(3) Train R² per seed [gauge-INVARIANT]\n"
        f"mean = {np.mean(r2s):+.4f}, std = {np.std(r2s):.2e}"
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(r2s):
        axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    axes[1, 1].hist(color_disagree_01, bins=40, color="indianred", edgecolor="black")
    axes[1, 1].axvline(z_norm_typical, color="black", ls="--", lw=1.5,
                       label=f"typical ||Z|| = {z_norm_typical:.2f}")
    axes[1, 1].axvline(median_color_disagree, color="navy", ls=":", lw=1.5,
                       label=f"median disagree = {median_color_disagree:.2f}")
    axes[1, 1].set_xlabel("‖f_0(T_0_color) − f_1(T_1_color)‖  in Z")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title(
        f"(4) Per-color Z-disagreement, seed 0 vs seed 1 [gauge-INVARIANT]\n"
        f"median = {median_color_disagree:.3f} = "
        f"{100*median_color_disagree/z_norm_typical:.1f}% of typical ||Z||"
    )
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"U_3d gauge test · 8 seeds × random-uniform init · cogito L40 · K={K_PC}",
        fontsize=13,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "auto_68.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[auto_68] saved {out_png}")

    out_json = OUT_DIR / "auto_68.json"
    out_json.write_text(json.dumps({
        "n_colors": int(N),
        "K_PC": K_PC,
        "n_seeds": N_SEEDS,
        "n_iters": N_ITERS,
        "grid_per_axis": GRID_PER_AXIS,
        "typical_Z_norm": z_norm_typical,
        "train_r2_per_seed": r2s,
        "train_r2_mean": float(np.mean(r2s)),
        "train_r2_std": float(np.std(r2s)),
        "procrustes_T_offdiag_median": median_proc,
        "procrustes_T_offdiag_max": float(proc_mat.max()),
        "hausdorff_Z_grid_offdiag_median": median_haus,
        "hausdorff_Z_grid_offdiag_max": float(haus_mat.max()),
        "color_disagree_seed0_seed1_median": median_color_disagree,
        "color_disagree_seed0_seed1_mean": float(color_disagree_01.mean()),
        "color_disagree_over_typical_Z": median_color_disagree / z_norm_typical,
        "procrustes_matrix": proc_mat.tolist(),
        "hausdorff_matrix": haus_mat.tolist(),
    }, indent=2))
    print(f"[auto_68] saved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
