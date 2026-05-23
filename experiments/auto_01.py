"""auto_01: U_3d latent axes vs perceptual color variables.

For each unsupervised d=3 latent embedding (PCA, UMAP, Isomap) of the
per-color centroids, we:
  1. Scatter the 3D latent, coloring each point by the xkcd RGB.
  2. Quantify how much each latent axis aligns with each perceptual
     variable (R, G, B, H_cos, H_sin, S, V, L) via per-axis R² of an
     OLS fit y_axis ~ var.

Question: do the unsupervised "intrinsic" axes line up with anything
human-interpretable, or are they rotated by the model?

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_01.png
"""
from __future__ import annotations

import colorsys
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from plot_color_geometry import load_xkcd_colors, load_harvest  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402

N_T = 28
OUT = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_01.png")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")


def pca3(X):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T


def umap3(X):
    import umap
    return np.asarray(umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                                random_state=0).fit_transform(X))


def isomap3(X):
    from sklearn.manifold import Isomap
    return np.asarray(Isomap(n_components=3, n_neighbors=15).fit_transform(X))


def r2_axis(y, x):
    # univariate OLS R²: 1 - SSR/SST
    x = np.asarray(x); y = np.asarray(y)
    xb = x - x.mean(); yb = y - y.mean()
    denom = (xb * xb).sum()
    if denom < 1e-12:
        return 0.0
    beta = (xb * yb).sum() / denom
    pred = beta * xb
    ss_res = ((yb - pred) ** 2).sum()
    ss_tot = (yb * yb).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main():
    X_full = load_harvest(HARVEST)
    n_raw = X_full.shape[0] // N_T
    X_full = X_full[: n_raw * N_T]
    centroids_all = X_full.reshape(n_raw, N_T, -1).mean(1)
    colors_all = load_xkcd_colors()[:n_raw]
    _, kept = filter_colors(colors_all)
    C = centroids_all[kept]
    rgb = np.array([(r, g, b) for _, r, g, b in [colors_all[i] for i in kept]],
                   dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    L = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    feats = {
        "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
        "H_cos": np.cos(2 * np.pi * H), "H_sin": np.sin(2 * np.pi * H),
        "S": S, "V": V, "L": L,
    }
    print(f"[auto_01] kept n_colors={len(C)} D={C.shape[1]}", flush=True)

    candidates = [("PCA-3", pca3), ("UMAP-3", umap3), ("Isomap-3", isomap3)]
    latents = {}
    methods = []
    for name, fn in candidates:
        print(f"  fitting {name} ...", flush=True)
        try:
            latents[name] = fn(C)
            methods.append((name, fn))
        except Exception as exc:
            print(f"    skipping {name}: {type(exc).__name__}: {exc}", flush=True)

    ncol = len(methods)
    fig = plt.figure(figsize=(5.7 * ncol, 11))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.85], hspace=0.30, wspace=0.25)

    for col, (name, _) in enumerate(methods):
        T = latents[name]
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.scatter(T[:, 0], T[:, 1], T[:, 2], c=rgb, s=10, alpha=0.85,
                   edgecolors="none")
        ax.set_title(f"{name}  (color = xkcd RGB)", fontsize=11)
        ax.set_xlabel("t1"); ax.set_ylabel("t2"); ax.set_zlabel("t3")
        ax.tick_params(labelsize=7)

        # heatmap: 3 axes × 8 features
        axh = fig.add_subplot(gs[1, col])
        feat_names = list(feats.keys())
        H_mat = np.zeros((3, len(feat_names)))
        for i in range(3):
            for j, fn_ in enumerate(feat_names):
                H_mat[i, j] = r2_axis(T[:, i], feats[fn_])
        im = axh.imshow(H_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axh.set_xticks(range(len(feat_names)))
        axh.set_xticklabels(feat_names, rotation=40, ha="right", fontsize=9)
        axh.set_yticks([0, 1, 2]); axh.set_yticklabels(["t1", "t2", "t3"], fontsize=9)
        axh.set_title(f"{name}: per-axis R² vs feature", fontsize=10)
        for i in range(3):
            for j in range(len(feat_names)):
                v = H_mat[i, j]
                axh.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="white" if v < 0.5 else "black")
        plt.colorbar(im, ax=axh, shrink=0.8, label="R²")

    fig.suptitle(
        "Do the discovered latent axes line up with perceptual color variables?\n"
        f"cogito L40 centroids · n_colors={len(C)} (filtered) · d=3",
        fontsize=13, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
