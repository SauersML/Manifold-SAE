"""auto_85: PC variance vs hue-circularity per PC — why ARD-alpha failed.

Motivated by auto_exp_47/49: ARD over the top-K PC basis prunes by
variance/magnitude, but auto_exp_47 found PC2+PC4 carries the hue circle and
auto_exp_49 saw ARD fail to keep those informative PCs because their *variance*
is much smaller than PC0/PC1 (which carry brightness/saturation, not hue
identity).

This plot makes that inversion visually obvious in one figure:
  - Left:  twin-axis bar chart per PC (0..15) of
             (a) variance(PC_k) = singular value^2 / (n-1)  [log y-axis, blue]
             (b) max |J-S circ-corr| between angle(PC_k, PC_j) and true hue
                 across all j != k                          [red dots, right y]
  - Right: scatter of variance vs hue-circularity, log-x, annotated by PC index.
           A negative trend (high-variance PCs are LOW circularity) directly
           shows why ARD-alpha (which favors high-variance dims) would prune
           the hue-carrying PCs.
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
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_85.png"

K_PCS = 16


def circ_corr_js(a_rad, b_rad):
    a_bar = np.angle(np.mean(np.exp(1j * a_rad)))
    b_bar = np.angle(np.mean(np.exp(1j * b_rad)))
    num = np.sum(np.sin(a_rad - a_bar) * np.sin(b_rad - b_bar))
    den = np.sqrt(np.sum(np.sin(a_rad - a_bar) ** 2)
                  * np.sum(np.sin(b_rad - b_bar) ** 2))
    return float(num / den) if den > 0 else float("nan")


def main():
    t_start = time.time()
    print("[auto_85] PC variance vs hue-circularity per PC")

    X = np.load(X_PATH, mmap_mode="r")
    basis = load_pc_basis(K=64)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    _, rgb = load_xkcd_rgb(n)
    hue_rad = 2 * np.pi * hsv_from_rgb(rgb)[:, 0]

    Tc = T0 - T0.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    var_per_pc = (S[:K_PCS] ** 2) / (n - 1)
    print(f"[pca] var top-8: {var_per_pc[:8].round(3)}")

    # For each PC k, find the strongest |circ-corr| with hue across all (k, j) planes.
    best_cc = np.zeros(K_PCS)
    best_partner = np.full(K_PCS, -1, dtype=int)
    cc_matrix = np.full((K_PCS, K_PCS), np.nan)
    for i in range(K_PCS):
        for j in range(K_PCS):
            if i == j:
                continue
            plane = Tc @ Vt[[i, j]].T
            th = np.arctan2(plane[:, 1], plane[:, 0])
            cc = circ_corr_js(th, hue_rad)
            cc_matrix[i, j] = cc
            if abs(cc) > abs(best_cc[i]):
                best_cc[i] = cc
                best_partner[i] = j

    print("[per-PC best partner / |circ-corr|]")
    for k in range(K_PCS):
        print(f"  PC{k:2d}: var={var_per_pc[k]:.3f}  best |cc|={abs(best_cc[k]):.3f} (with PC{best_partner[k]})")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Left: bars + overlay
    ax = axes[0]
    pcs = np.arange(K_PCS)
    bars = ax.bar(pcs, var_per_pc, color="steelblue", alpha=0.7,
                  label="variance(PC_k)")
    ax.set_yscale("log")
    ax.set_xlabel("PC index")
    ax.set_ylabel("variance(PC_k) = S^2/(n-1)  [log]", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xticks(pcs)

    ax2 = ax.twinx()
    ax2.plot(pcs, np.abs(best_cc), "o-", color="crimson",
             markersize=8, linewidth=1.5,
             label="max |circ-corr(angle(PC_k, PC_j), hue)|")
    ax2.set_ylabel("max |J-S circ-corr| with hue", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, max(0.85, np.abs(best_cc).max() * 1.1))
    for k in pcs:
        ax2.annotate(f"PC{best_partner[k]}", (k, abs(best_cc[k])),
                     textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7, color="crimson")
    ax.set_title("Per-PC: variance (blue bars, log) vs hue-circularity (red dots)\n"
                 "labels = best partner PC; ARD prunes by variance, not by hue role")

    # Right: variance vs circularity scatter
    ax = axes[1]
    ax.scatter(var_per_pc, np.abs(best_cc), s=60, c=pcs, cmap="viridis",
               edgecolors="k", linewidths=0.5)
    for k in pcs:
        ax.annotate(f"PC{k}", (var_per_pc[k], abs(best_cc[k])),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("variance(PC_k)  [log]")
    ax.set_ylabel("max |circ-corr| with hue")
    # Spearman trend
    from scipy.stats import spearmanr  # type: ignore
    rho, p = spearmanr(var_per_pc, np.abs(best_cc))
    ax.set_title(f"variance vs hue-circularity per PC\n"
                 f"Spearman rho = {rho:+.3f} (p={p:.3g}) over K={K_PCS} PCs")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "auto_85: ARD-alpha failure mechanism on cogito-L40 — "
        "high-variance PCs do NOT carry the hue circle",
        fontsize=12)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
