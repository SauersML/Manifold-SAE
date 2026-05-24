"""auto_88: Per-color residual heatmap across top PCA-truncation specs.

Fresh angle — NOT covered by auto_77..87. Asks: ACROSS the top
unsupervised reconstruction specs (which dominate the leaderboard in
results.json), which xkcd colors are UNIVERSAL outliers? A color that
stays bright across every column of the heatmap resists every spec we
tried — a candidate target for the next probe / SAE atom hunt.

Method (uses only on-disk artifacts — no harvest, no GAM refits):
  1. Reuse per_color_stats_mmap from auto_exp_38 with K=128 PCs.
     T0 ∈ R^{949 x 128} is the per-color centroid in standardized
     PC coordinates (μ, σ, Vt all on disk inside results.json /
     pca_basis_K128).
  2. For each spec U_pca_Kd in K ∈ {2,3,4,6,8,12,16,24,32,48,64,96},
     the spec's reconstruction in PC-space is exactly T0[:, :K]
     padded with zeros. Per-color residual:
        e_{c,K} = || T0[c, K:] ||_2   (rest of the spectrum)
     This is the EXACT residual the spec achieves on T0 (since
     post-K coords are not modelled at all).
  3. Sort rows (colors) by residual-at-K=8 (mid-spec), so the
     heatmap reveals the universal-hard band at the top.
  4. Plot:
        (a) 949 x 12 residual heatmap (log-norm)
        (b) RGB swatch column on the left
        (c) bar of top-30 colors with mean residual across K
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_PNG = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_88.png"

K_FULL = 128                 # widest PCA basis on disk
K_LIST = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
SORT_K = 8                   # row-sort key (mid spec)


def main():
    t0 = time.time()
    print(f"[auto_88] per-color residual heatmap, K_list={K_LIST}")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")

    # We need K_FULL PCs; try the cached basis at that K, else fall back.
    try:
        basis = load_pc_basis(K=K_FULL)
        K_used = K_FULL
    except Exception as e:
        print(f"[basis] K={K_FULL} not cached ({e}); falling back to K=64")
        basis = load_pc_basis(K=64)
        K_used = 64

    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_used)
    n_c, K = T0.shape
    print(f"[centroids] T0={T0.shape}  total energy/row²={(T0**2).sum(1).mean():.2f}")

    # Filter K_LIST to only those < K_used (so residual is nontrivial)
    k_list = [k for k in K_LIST if k < K_used]
    print(f"[specs] using k_list={k_list}  (spec U_pca_Kd → keep top K PCs)")

    # Per-color residual norm: e[c, j] = || T0[c, k_list[j]:] ||
    E = np.zeros((n_c, len(k_list)))
    for j, k in enumerate(k_list):
        E[:, j] = np.linalg.norm(T0[:, k:], axis=1)
    print(f"[residual] E={E.shape}  range=[{E.min():.3f}, {E.max():.3f}]")

    # Mean across specs as a single "universally hard" score
    mean_e = E.mean(axis=1)

    # Sort rows by SORT_K residual (descending: hardest on top)
    if SORT_K in k_list:
        sort_col = k_list.index(SORT_K)
    else:
        sort_col = len(k_list) // 2
    order = np.argsort(-E[:, sort_col])
    E_sorted = E[order]

    names, rgb = load_xkcd_rgb(n_c)
    rgb_sorted = rgb[order]
    rgb_strip = np.clip(rgb_sorted, 0, 1)[:, None, :]   # (n_c, 1, 3)

    # ---- plotting ----
    fig = plt.figure(figsize=(14, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.08, 1.0, 0.55])

    # (a) RGB swatch strip
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_strip, aspect="auto", interpolation="nearest")
    ax0.set_xticks([])
    ax0.set_yticks([0, n_c // 2, n_c - 1])
    ax0.set_yticklabels([f"#{order[0]}", f"#{order[n_c//2]}", f"#{order[-1]}"],
                        fontsize=8)
    ax0.set_ylabel(f"color (sorted by residual @ U_pca_{SORT_K}d, hardest top)")
    ax0.set_title("xkcd\nRGB", fontsize=9)

    # (b) residual heatmap (log-norm so the dynamic range is visible)
    ax1 = fig.add_subplot(gs[0, 1])
    vmin = max(E_sorted.min(), 1e-3)
    vmax = E_sorted.max()
    im = ax1.imshow(E_sorted, aspect="auto", interpolation="nearest",
                    cmap="magma_r", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax1.set_xticks(range(len(k_list)))
    ax1.set_xticklabels([f"K={k}" for k in k_list], rotation=0, fontsize=9)
    ax1.set_xlabel("spec U_pca_Kd  (top-K PCs kept)")
    ax1.set_yticks([])
    ax1.set_title(f"Per-color residual ‖T0[c, K:]‖₂   "
                  f"(N={n_c} xkcd colors × {len(k_list)} specs)")
    cb = fig.colorbar(im, ax=ax1, shrink=0.7, pad=0.01)
    cb.set_label("residual norm (log)")

    # (c) top-30 universally hardest colors as labelled swatches
    ax2 = fig.add_subplot(gs[0, 2])
    rank = np.argsort(-mean_e)
    top = rank[:30]
    for r_, i in enumerate(top):
        y = -r_
        ax2.add_patch(plt.Rectangle((0, y - 0.4), 0.8, 0.8,
                                    facecolor=np.clip(rgb[i], 0, 1),
                                    edgecolor="black", lw=0.4))
        ax2.text(0.95, y, names[i], va="center", fontsize=7.5)
        ax2.text(5.3, y, f"ē={mean_e[i]:.2f}",
                 va="center", fontsize=7.5, family="monospace")
    ax2.set_xlim(-0.2, 7.0)
    ax2.set_ylim(-30.5, 0.7)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Top-30 UNIVERSAL outliers\n(mean residual across all K)",
                  fontsize=10)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"auto_88 — per-color residual heatmap across top PCA-truncation specs  "
        f"(L40 cogito, K_PCs={K_used})",
        fontsize=12)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[saved] {OUT_PNG}  ({time.time()-t0:.1f}s)")

    # Print summary stats
    print("\n[summary] top-15 universally hardest colors:")
    for r_, i in enumerate(rank[:15]):
        print(f"  {r_:2d}  ē={mean_e[i]:.3f}  {names[i]}")
    print("\n[summary] mean residual at each K:")
    for j, k in enumerate(k_list):
        print(f"  K={k:3d}  mean_e={E[:, j].mean():.3f}  "
              f"max={E[:, j].max():.3f}  p99={np.quantile(E[:, j], 0.99):.3f}")


if __name__ == "__main__":
    main()
