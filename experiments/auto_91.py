"""auto_91: PC4-centric correlation atlas.

Motivation: auto_82 (J-S circ-corr heatmap), auto_85 (variance vs
hue-circularity per PC), and auto_86 (direct PC2xPC4 hue-ring scatter)
together identified PC4 as a universal hue-axis partner in the L40
standardized centroid PCA. But none of those tested whether PC4 ALSO
carries non-perceptual signal (token count, monoword, template sigma)
or other perceptual dims (saturation, luminance, raw RGB).

This script answers "what else does PC4 encode?" by computing, for each
of the top-16 PCs, its correlation with eight target signals:

  Linear Pearson  : R, G, B, value (V), luminance (0.299R+0.587G+0.114B),
                    saturation (S), token_count, monoword
  Circular J-S    : hue (HSV H, 2*pi*H)

Display: a 16-row x 9-col heatmap of |corr|, with the PC4 row boxed,
and a companion bar chart focusing on PC4 alone, sorted by |corr|.
This is auto_82 (PC-pair circ-corr with hue) extended to single-PC
correlation with EIGHT scalar targets — covering perceptual, raw-color,
and compositional axes simultaneously.

Headline numbers printed: PC4's top-3 absolute correlations across all
9 targets.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_91.png"

K_PCS = 16
PC_FOCUS = 4


def circ_lin_corr(theta, x):
    """Circular-linear correlation (Mardia 1976): sqrt((r_xc^2 + r_xs^2 - 2*r_xc*r_xs*r_cs) / (1 - r_cs^2))."""
    c = np.cos(theta); s = np.sin(theta)
    r_xc = np.corrcoef(x, c)[0, 1]
    r_xs = np.corrcoef(x, s)[0, 1]
    r_cs = np.corrcoef(c, s)[0, 1]
    denom = 1.0 - r_cs * r_cs
    if denom <= 0:
        return float("nan")
    val = (r_xc**2 + r_xs**2 - 2 * r_xc * r_xs * r_cs) / denom
    return float(np.sqrt(max(0.0, val)))


def main():
    t0 = time.time()
    print("[auto_91] PC4-centric correlation atlas")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n)
    hsv = hsv_from_rgb(rgb)
    hue = hsv[:, 0]; sat = hsv[:, 1]; val = hsv[:, 2]
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    tok = np.array([len(nm.split()) for nm in names], dtype=np.float64)
    mono = (tok == 1).astype(np.float64)

    # PCA on centered centroids (same as auto_82/86 convention)
    Tc = T0 - T0.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    Z = Tc @ Vt.T  # n x K_PCS scores
    print(f"[pca] top singular values: {S[:6].round(3)}")

    # Linear targets
    lin_targets = {
        "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
        "luminance": lum, "saturation (S)": sat, "value (V)": val,
        "token_count": tok, "monoword": mono,
    }
    target_order = ["R", "G", "B", "luminance", "saturation (S)", "value (V)",
                    "token_count", "monoword", "hue (circ)"]

    M = np.zeros((K_PCS, len(target_order)), dtype=np.float64)
    hue_rad = 2 * np.pi * hue
    for j, name in enumerate(target_order):
        if name == "hue (circ)":
            for k in range(K_PCS):
                M[k, j] = circ_lin_corr(hue_rad, Z[:, k])
        else:
            y = lin_targets[name]
            for k in range(K_PCS):
                M[k, j] = abs(np.corrcoef(Z[:, k], y)[0, 1])

    print("\n[PC4 row]  |corr| with each target:")
    row = M[PC_FOCUS]
    order = np.argsort(-row)
    for rk, j in enumerate(order):
        print(f"   {rk+1:>2}. {target_order[j]:>18s}  |corr| = {row[j]:.3f}")

    # Plot: heatmap + PC4 bar
    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])
    ax_h = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    im = ax_h.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_h.set_xticks(range(len(target_order)))
    ax_h.set_xticklabels(target_order, rotation=40, ha="right")
    ax_h.set_yticks(range(K_PCS))
    ax_h.set_yticklabels([f"PC{k}" for k in range(K_PCS)])
    ax_h.set_title("|corr| of each PC with each target  (Pearson; hue=circ-lin)")
    # box PC4 row
    rect = mpatches.Rectangle((-0.5, PC_FOCUS - 0.5),
                              len(target_order), 1,
                              fill=False, edgecolor="red", lw=2.0)
    ax_h.add_patch(rect)
    # annotate cells
    for i in range(K_PCS):
        for j in range(len(target_order)):
            v = M[i, j]
            ax_h.text(j, i, f"{v:.2f}", ha="center", va="center",
                      color="white" if v < 0.55 else "black", fontsize=6.5)
    fig.colorbar(im, ax=ax_h, shrink=0.85, label="|correlation|")

    # PC4 bar (sorted)
    row_sorted_idx = np.argsort(-row)
    labels_sorted = [target_order[i] for i in row_sorted_idx]
    vals_sorted = row[row_sorted_idx]
    bars = ax_b.barh(range(len(labels_sorted))[::-1], vals_sorted,
                     color="crimson", edgecolor="black")
    ax_b.set_yticks(range(len(labels_sorted))[::-1])
    ax_b.set_yticklabels(labels_sorted)
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("|correlation|")
    ax_b.set_title(f"PC{PC_FOCUS}'s correlations, ranked\n"
                   f"(red row in heatmap)")
    for b, v in zip(bars, vals_sorted):
        ax_b.text(v + 0.01, b.get_y() + b.get_height() / 2,
                  f"{v:.2f}", va="center", fontsize=8)

    fig.suptitle(
        f"auto_91 — PC{PC_FOCUS}-centric correlation atlas: "
        f"what else does the hue-partner PC carry?   N={n} centroids",
        fontsize=12)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\n[saved] {OUT_PNG}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
