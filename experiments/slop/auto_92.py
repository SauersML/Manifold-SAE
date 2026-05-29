"""auto_92: per-color residual norm binned in HSV saturation x luminance space.

Motivation: auto_88 computed per-color residuals ||T0[c, K:]||_2 from PCA
truncation of the standardized centroids, sorted colors by that residual,
and named the top-30 universal outliers. It never asked the perceptual
question: do GREY colors (low saturation) and VIVID colors (high saturation)
sit in the centroid manifold equally well? Likewise for dark vs bright
(luminance). All other 14 prior plots focus on PCs, families, templates,
or name-token counts — none stratify the 949 colors by chromatic content
of the RGB reference.

This script:
  1. Reuses per_color_stats_mmap to get the K=64 standardized centroid
     scores T0 (949 x 64).
  2. Computes per-color residual norm at three truncation cutoffs
     K_keep ∈ {4, 8, 16}: e_K(c) = ||T0[c, K:]||_2.
  3. Computes per-color HSV from xkcd RGB and bins into a 6x6 grid in
     (saturation, value) space. Reports mean residual per bin and bin
     count.
  4. Plots:
        (a) 1x3 row of 6x6 heatmaps (one per K_keep), bin = mean residual
        (b) bottom row: marginal residual vs saturation (line) and vs
            value/luminance (line), with bootstrap CI bands
        (c) annotates each bin cell with N (count).

Headline numbers printed: Pearson r between per-color residual and
saturation, and between residual and value, for each K_keep.

If saturation correlates negatively with residual: vivid colors fit
better (LLM color-manifold is hue-organized, greys are off-manifold).
If positively: greys fit better (LLM aligns achromatic dimension first).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_92.png"

K_TOTAL = 64
K_LIST = [4, 8, 16]
N_BINS = 6
N_BOOT = 200
RNG = np.random.default_rng(0)


def bin_mean(x, y, n_bins, lo=0.0, hi=1.0):
    """Return (centers, means, counts) of y binned by x."""
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1 else x <= edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] > 0:
            means[i] = float(y[m].mean())
    return centers, means, counts, edges


def bootstrap_band(x, y, n_bins, n_boot=200, lo=0.0, hi=1.0):
    """Return (centers, mean, lo_band, hi_band) using bootstrap over colors."""
    n = len(x)
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bb = np.full((n_boot, n_bins), np.nan)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        for i in range(n_bins):
            m = (xb >= edges[i]) & (xb < edges[i + 1] if i < n_bins - 1 else xb <= edges[i + 1])
            if m.any():
                bb[b, i] = yb[m].mean()
    mean = np.nanmean(bb, axis=0)
    lo_b = np.nanpercentile(bb, 16, axis=0)
    hi_b = np.nanpercentile(bb, 84, axis=0)
    return centers, mean, lo_b, hi_b


def main():
    t0 = time.time()
    print("[auto_92] residual norm vs HSV saturation x value")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=K_TOTAL)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_TOTAL)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n)
    hsv = hsv_from_rgb(rgb)
    sat = hsv[:, 1]
    val = hsv[:, 2]
    # luminance for reporting
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    # Per-color residuals at each K_keep
    residuals = {}
    for K in K_LIST:
        e = np.linalg.norm(T0[:, K:], axis=1)
        residuals[K] = e
        r_sat = float(np.corrcoef(sat, e)[0, 1])
        r_val = float(np.corrcoef(val, e)[0, 1])
        r_lum = float(np.corrcoef(lum, e)[0, 1])
        print(f"\n[K_keep={K:>2}]  residual range=[{e.min():.3f}, {e.max():.3f}]")
        print(f"   Pearson r(residual, saturation) = {r_sat:+.3f}")
        print(f"   Pearson r(residual, value)      = {r_val:+.3f}")
        print(f"   Pearson r(residual, luminance)  = {r_lum:+.3f}")

    # 2D bin grid: rows = value (low->high), cols = saturation (low->high)
    fig = plt.figure(figsize=(15.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])

    # Top row: 2D heatmaps
    for ci, K in enumerate(K_LIST):
        ax = fig.add_subplot(gs[0, ci])
        e = residuals[K]
        edges = np.linspace(0.0, 1.0, N_BINS + 1)
        grid = np.full((N_BINS, N_BINS), np.nan)
        counts = np.zeros((N_BINS, N_BINS), dtype=int)
        for i in range(N_BINS):  # value bin (row)
            for j in range(N_BINS):  # saturation bin (col)
                mv = (val >= edges[i]) & (val < edges[i + 1] if i < N_BINS - 1 else val <= edges[i + 1])
                ms = (sat >= edges[j]) & (sat < edges[j + 1] if j < N_BINS - 1 else sat <= edges[j + 1])
                m = mv & ms
                counts[i, j] = int(m.sum())
                if counts[i, j] > 0:
                    grid[i, j] = float(e[m].mean())
        im = ax.imshow(grid, origin="lower", cmap="magma_r", aspect="auto",
                       extent=[0, 1, 0, 1])
        ax.set_xlabel("HSV saturation  (grey <-- --> vivid)")
        ax.set_ylabel("HSV value  (dark <-- --> bright)")
        ax.set_title(f"K_keep = {K}\nmean residual ‖T0[c, K:]‖₂ per bin")
        # annotate counts
        for i in range(N_BINS):
            for j in range(N_BINS):
                if counts[i, j] > 0:
                    txt_color = "white" if (np.nan_to_num(grid[i, j]) >
                                            np.nanmean(grid)) else "black"
                    ax.text((j + 0.5) / N_BINS, (i + 0.5) / N_BINS,
                            f"{counts[i, j]}", ha="center", va="center",
                            color=txt_color, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85, label="mean residual")

    # Bottom row: marginals
    ax_s = fig.add_subplot(gs[1, 0])
    ax_v = fig.add_subplot(gs[1, 1])
    ax_scatter = fig.add_subplot(gs[1, 2])

    colors_K = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c"}
    for K in K_LIST:
        e = residuals[K]
        c, m, lo_b, hi_b = bootstrap_band(sat, e, N_BINS, N_BOOT)
        ax_s.plot(c, m, "-o", color=colors_K[K], label=f"K_keep={K}")
        ax_s.fill_between(c, lo_b, hi_b, color=colors_K[K], alpha=0.15)
        c, m, lo_b, hi_b = bootstrap_band(val, e, N_BINS, N_BOOT)
        ax_v.plot(c, m, "-o", color=colors_K[K], label=f"K_keep={K}")
        ax_v.fill_between(c, lo_b, hi_b, color=colors_K[K], alpha=0.15)
    ax_s.set_xlabel("HSV saturation"); ax_s.set_ylabel("mean residual ‖T0[c, K:]‖₂")
    ax_s.set_title("residual vs saturation\n(±1σ bootstrap, 200 reps)")
    ax_s.legend(loc="best", fontsize=9); ax_s.grid(alpha=0.3)
    ax_v.set_xlabel("HSV value"); ax_v.set_ylabel("mean residual ‖T0[c, K:]‖₂")
    ax_v.set_title("residual vs value\n(±1σ bootstrap, 200 reps)")
    ax_v.legend(loc="best", fontsize=9); ax_v.grid(alpha=0.3)

    # Scatter of all 949 colors at K_keep=8, painted with their RGB
    e = residuals[8]
    sc = ax_scatter.scatter(sat, e, c=rgb, s=18, edgecolor="black",
                            linewidth=0.2, alpha=0.85)
    ax_scatter.set_xlabel("HSV saturation")
    ax_scatter.set_ylabel("residual ‖T0[c, 8:]‖₂")
    ax_scatter.set_title(f"per-color scatter (K_keep=8)\n"
                         f"r={np.corrcoef(sat, e)[0,1]:+.3f}; points colored by their xkcd RGB")
    ax_scatter.grid(alpha=0.3)

    fig.suptitle(
        f"auto_92 — per-color centroid residual vs HSV (saturation x value)   "
        f"N={n} xkcd colors, cogito-L40 standardized centroids, K_total={K_TOTAL}",
        fontsize=13,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\n[saved] {OUT_PNG}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
