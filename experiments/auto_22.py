"""auto_22: centroid distance vs RGB extremity (idea uu).

Question
--------
For each xkcd color, compute its per-color centroid in cogito L40 PCA
space (mean across 28 templates, after the same per-dim standardization
the gam pipeline uses). Then ask: does a color's distance from the
grand centroid track its "extremity" in RGB space (saturation, chroma,
distance-from-grey, distance-from-nearest-axis-extreme)?

If yes, the manifold direction simply reflects "boring grey vs vivid",
i.e. residual norm scales with saturation. If no, the geometry is doing
something perceptually richer than that.

Diagnostics
-----------
1. Scatter: centroid L2 distance (full 64-PC subspace) vs RGB chroma
   (max(R,G,B) - min(R,G,B)) — Spearman ρ.
2. Same vs HSV saturation, vs distance-from-grey [0.5,0.5,0.5], and
   vs luminance distance from 0.5 (extremity along L).
3. Top-PC version: distance along PC1+PC2+PC3 only.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_22.{png,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path("/Users/user/Manifold-SAE")
RESULTS = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
HARVEST = ROOT / "runs/COLOR_COGITO_L40/X_L40.npy"
XKCD = ROOT / "experiments/xkcd_colors.txt"
OUT_PNG = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_22.png"
OUT_JSON = ROOT / "runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_22.json"

N_COLORS = 949
N_TEMPLATES = 28


def load_xkcd_rgb(n: int) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rgb: list[tuple[float, float, float]] = []
    for line in XKCD.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        h = parts[1].lstrip("#")
        names.append(parts[0])
        rgb.append((int(h[0:2], 16) / 255., int(h[2:4], 16) / 255., int(h[4:6], 16) / 255.))
        if len(names) >= n:
            break
    return names, np.array(rgb, dtype=np.float64)


def main() -> None:
    d = json.loads(RESULTS.read_text())
    pl = d["per_layer"]["L40"]
    mu = np.array(pl["mu"], dtype=np.float32)         # (D,)
    sigma = np.array(pl["sigma"], dtype=np.float32)   # (D,)
    Vt = np.array(pl["Vt_topK"], dtype=np.float32)    # (K, D)
    K, D = Vt.shape
    print(f"[load] mu/sigma D={D}  PCs K={K}")

    X = np.load(HARVEST, mmap_mode="r")
    N = X.shape[0]
    assert N == N_COLORS * N_TEMPLATES, (N, N_COLORS, N_TEMPLATES)

    # Per-color centroid in raw residual space
    centroids = np.zeros((N_COLORS, D), dtype=np.float32)
    for ci in range(N_COLORS):
        s = ci * N_TEMPLATES
        centroids[ci] = X[s : s + N_TEMPLATES].mean(0)

    # Same pipeline as gam: standardize across colors, then center, then PCA project.
    Xn = (centroids - mu) / sigma                       # (n_c, D)
    Xn_c = Xn - Xn.mean(0, keepdims=True)
    Z = Xn_c @ Vt.T                                     # (n_c, K)

    # Distance from grand centroid (which is now 0 by construction in PCA space)
    dist_full = np.linalg.norm(Z, axis=1)               # (n_c,)
    dist_pc1_3 = np.linalg.norm(Z[:, :3], axis=1)
    dist_pc1 = np.abs(Z[:, 0])

    # RGB extremity metrics
    names, rgb = load_xkcd_rgb(N_COLORS)
    chroma = rgb.max(1) - rgb.min(1)
    sat_hsv = np.where(rgb.max(1) > 0, chroma / np.maximum(rgb.max(1), 1e-12), 0.0)
    val = rgb.max(1)
    dist_grey = np.linalg.norm(rgb - 0.5, axis=1)
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    lum_extreme = np.abs(lum - 0.5)

    rho = {}
    rho["full_vs_chroma"]    = float(spearmanr(dist_full, chroma).correlation)
    rho["full_vs_sat_hsv"]   = float(spearmanr(dist_full, sat_hsv).correlation)
    rho["full_vs_dist_grey"] = float(spearmanr(dist_full, dist_grey).correlation)
    rho["full_vs_lum_extreme"] = float(spearmanr(dist_full, lum_extreme).correlation)
    rho["full_vs_value"]     = float(spearmanr(dist_full, val).correlation)
    rho["pc1_3_vs_chroma"]   = float(spearmanr(dist_pc1_3, chroma).correlation)
    rho["pc1_vs_chroma"]     = float(spearmanr(dist_pc1, chroma).correlation)
    rho["pc1_vs_dist_grey"]  = float(spearmanr(dist_pc1, dist_grey).correlation)
    for k, v in rho.items():
        print(f"  spearman {k:30s} = {v:+.3f}")

    # ---- plot ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def scatter(ax, x, y, title, xlabel, ylabel, key):
        ax.scatter(x, y, c=rgb, s=14, edgecolor="k", linewidth=0.2, alpha=0.85)
        r = rho[key]
        ax.set_title(f"{title}\nSpearman ρ = {r:+.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    scatter(axes[0, 0], chroma, dist_full,
            "centroid ‖·‖ (all 64 PCs) vs RGB chroma",
            "RGB chroma = max-min", "‖centroid‖₂  (PCA, 64-D)", "full_vs_chroma")
    scatter(axes[0, 1], sat_hsv, dist_full,
            "centroid ‖·‖ (full) vs HSV saturation",
            "HSV saturation", "‖centroid‖₂  (PCA, 64-D)", "full_vs_sat_hsv")
    scatter(axes[0, 2], dist_grey, dist_full,
            "centroid ‖·‖ (full) vs distance from grey",
            "‖RGB − [0.5,0.5,0.5]‖₂", "‖centroid‖₂  (PCA, 64-D)", "full_vs_dist_grey")

    scatter(axes[1, 0], lum_extreme, dist_full,
            "centroid ‖·‖ (full) vs |luminance − 0.5|",
            "|luminance − 0.5|", "‖centroid‖₂  (PCA, 64-D)", "full_vs_lum_extreme")
    scatter(axes[1, 1], chroma, dist_pc1_3,
            "centroid ‖·‖ (top-3 PCs) vs RGB chroma",
            "RGB chroma", "‖centroid‖₂  (PC1..PC3)", "pc1_3_vs_chroma")
    scatter(axes[1, 2], chroma, dist_pc1,
            "|PC1| vs RGB chroma",
            "RGB chroma", "|PC1|", "pc1_vs_chroma")

    fig.suptitle("auto_22 — does residual-centroid magnitude track RGB extremity?  "
                 "(cogito L40, 949 xkcd colors)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[save] {OUT_PNG}")

    OUT_JSON.write_text(json.dumps({
        "spearman": rho,
        "n_colors": N_COLORS,
        "n_pcs_full": K,
        "median_centroid_norm_full": float(np.median(dist_full)),
        "median_centroid_norm_pc1_3": float(np.median(dist_pc1_3)),
        # top-10 highest and lowest centroid norms
        "top10_largest_centroid": [
            {"name": names[i], "norm": float(dist_full[i]), "chroma": float(chroma[i])}
            for i in np.argsort(-dist_full)[:10]
        ],
        "top10_smallest_centroid": [
            {"name": names[i], "norm": float(dist_full[i]), "chroma": float(chroma[i])}
            for i in np.argsort(dist_full)[:10]
        ],
    }, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
