"""auto_82: All-pairs (PC_i, PC_j) circular-hue structure heatmap.

Validates / extends auto_exp_47's finding that PC2+PC4 has circ-corr ~ -0.72.
auto_exp_47 swept top-8 pairs but only reported the single best. Here we
visualize the FULL pair table so we can see:
  (a) is PC2+PC4 really the strongest, or are there other comparable pairs?
  (b) is the hue-circle structure concentrated in a few pairs or spread across?
  (c) which PCs (rows/cols) "carry" hue most reliably?

Plot:
  - Left:  8x8 heatmap of |Jammalamadaka-Sarma circ-corr| between
            angle(PC_i, PC_j) and true hue. Annotated with signed values.
  - Right: 2x2 mini-scatters of the top-4 pairs, scattered as (PC_i, PC_j)
           coordinates colored by xkcd-RGB, so the hue ring is visible.
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
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_82.png"

K_PCS = 16
N_PAIRS = 8  # sweep PC_0..PC_7


def circ_corr_js(a_rad, b_rad):
    """Jammalamadaka-Sarma circular correlation (symmetric, [-1,1])."""
    a_bar = np.angle(np.mean(np.exp(1j * a_rad)))
    b_bar = np.angle(np.mean(np.exp(1j * b_rad)))
    num = np.sum(np.sin(a_rad - a_bar) * np.sin(b_rad - b_bar))
    den = np.sqrt(np.sum(np.sin(a_rad - a_bar) ** 2)
                  * np.sum(np.sin(b_rad - b_bar) ** 2))
    return float(num / den) if den > 0 else float("nan")


def main():
    t_start = time.time()
    print("[auto_82] All-pairs PC_i x PC_j circular-hue structure sweep")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, _tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n)
    hsv = hsv_from_rgb(rgb)
    hue = hsv[:, 0]
    hue_rad = 2 * np.pi * hue
    rgb_clip = np.clip(rgb, 0, 1)

    # PCA on centroids
    Tc = T0 - T0.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    print(f"[pca] singular values top-8: {S[:8].round(3)}")

    # All-pairs sweep over PCs 0..N_PAIRS-1
    M = np.full((N_PAIRS, N_PAIRS), np.nan)
    pair_records = []
    for i in range(N_PAIRS):
        for j in range(i + 1, N_PAIRS):
            plane = Tc @ Vt[[i, j]].T
            th = np.arctan2(plane[:, 1], plane[:, 0])
            cc = circ_corr_js(th, hue_rad)
            M[i, j] = cc
            M[j, i] = cc
            pair_records.append((abs(cc), cc, i, j, plane))

    pair_records.sort(reverse=True, key=lambda r: r[0])
    print("[top-8 pairs by |circ-corr|]:")
    for ac, cc, i, j, _ in pair_records[:8]:
        print(f"  PC{i}+PC{j}: circ-corr = {cc:+.3f}")

    # Plot
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.4, 1, 1, 1])

    # Heatmap (signed values; color by sign+magnitude)
    ax = fig.add_subplot(gs[:, 0])
    M_show = np.where(np.isnan(M), 0.0, M)
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M_show, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="upper")
    ax.set_xticks(range(N_PAIRS))
    ax.set_yticks(range(N_PAIRS))
    ax.set_xticklabels([f"PC{k}" for k in range(N_PAIRS)])
    ax.set_yticklabels([f"PC{k}" for k in range(N_PAIRS)])
    ax.set_title("All-pairs circular-corr\nangle(PC_i, PC_j) vs true hue")
    for i in range(N_PAIRS):
        for j in range(N_PAIRS):
            if np.isnan(M[i, j]):
                txt = "-"
                color = "black"
            else:
                txt = f"{M[i, j]:+.2f}"
                color = "white" if abs(M[i, j]) > 0.45 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.7, label="J-S circular corr")

    # Top-4 pair scatters colored by xkcd-RGB
    for idx, (ac, cc, i, j, plane) in enumerate(pair_records[:4]):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs[r, 1 + c])
        ax.scatter(plane[:, 0], plane[:, 1], c=rgb_clip,
                   s=8, edgecolors="k", linewidths=0.1)
        ax.set_xlabel(f"PC{i}")
        ax.set_ylabel(f"PC{j}")
        ax.set_title(f"PC{i} x PC{j}: circ-corr = {cc:+.3f}",
                     fontsize=10)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        ax.axvline(0, color="k", lw=0.3, alpha=0.4)

    fig.suptitle(
        "auto_82: All-pairs (PC_i, PC_j) circular hue structure on cogito-L40 centroids (n=949)\n"
        f"top pair = PC{pair_records[0][2]}+PC{pair_records[0][3]} "
        f"(circ-corr {pair_records[0][1]:+.3f}); "
        f"validating auto_exp_47's PC2+PC4 = {M[2, 4]:+.3f}",
        fontsize=11)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
