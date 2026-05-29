"""auto_05: which xkcd colors are hardest to predict from RGB?

The GAM zoo only reports macro held-out R² per spec — a single
number aggregated over all 64 PCs and all 949 colors. Hidden inside
that number is a per-color story: some xkcd colors live exactly where
a linear-RGB fit predicts them in PC-space, others sit far off the
manifold the linear model implies.

Procedure:
  1. Reload the per-color centroid matrix (load_harvest → average
     over 28 templates → filtered to 949 colors, matching the
     GAM-zoo's n_c).
  2. Project to the 64 PCs via the *exact* Vt_topK / mu / sigma
     stored in results.json, so Z is the same Z that L_lin_rgb was
     fit on.
  3. Do 5-fold color-grouped CV with the same ridge-style linear
     fit Z ~ [R, G, B, 1] used by the zoo's L_lin_rgb. Collect
     held-out predictions Z_hat for every color.
  4. Per-color residual norm  rho_c = ||Z_c - Z_hat_c||_2 / ||Z_c||_2
     (relative; Z is standardized so ||Z|| ≈ same scale across
     colors but the relative form is the cleanest ranking).
  5. Plot: (top-left) hardest 24 colors as a swatch grid with rho
     printed; (top-right) easiest 24 colors; (bottom) rho vs HSV-S
     and rho vs HSV-V scatter to ask whether the failures cluster
     in saturation/lightness space (highly-saturated rare colors?
     near-black / near-white?).

What we are asking: is residual hardness random, or does it have a
perceptual signature? If hard colors are predominantly low-S
('greys/browns') or extreme-V ('blacks/whites'), the linear-RGB
parameterisation systematically underfits the achromatic axis.
"""
from __future__ import annotations

import colorsys
import sys
import warnings
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from plot_color_geometry import load_xkcd_colors, load_harvest  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402

N_T = 28
RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN / "auto_05.png"
RESULTS = RUN / "results.json"
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
N_FOLDS = 5


def kfold_color_indices(n_colors, n_folds, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_colors)
    fold_of = np.empty(n_colors, dtype=int)
    fold_of[perm] = np.arange(n_colors) % n_folds
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0])
            for k in range(n_folds)]


def main():
    # ---- load saved PCA basis ----
    with open(RESULTS) as f:
        res = json.load(f)
    L = res["per_layer"]["L40"]
    Vt = np.asarray(L["Vt_topK"], dtype=np.float64)        # (K, D)
    mu = np.asarray(L["mu"], dtype=np.float64)             # (D,)
    sigma = np.asarray(L["sigma"], dtype=np.float64)       # (D,)
    K = Vt.shape[0]
    D = Vt.shape[1]
    print(f"[auto_05] PCA basis K={K} D={D}", flush=True)

    # ---- load centroids and filter to match GAM zoo ----
    X_full = load_harvest(HARVEST)
    n_raw = X_full.shape[0] // N_T
    X_full = X_full[: n_raw * N_T]
    centroids_all = X_full.reshape(n_raw, N_T, -1).mean(1)
    colors_all = load_xkcd_colors()[:n_raw]
    _, kept = filter_colors(colors_all)
    centroids = centroids_all[kept]
    colors = [colors_all[i] for i in kept]
    n_c = len(colors)
    print(f"[auto_05] filtered n_colors={n_c}", flush=True)
    assert centroids.shape[1] == D, f"D mismatch {centroids.shape[1]} vs {D}"

    # ---- standardize, project to top-K PCs ----
    Xn = (centroids - mu[None, :]) / np.maximum(sigma[None, :], 1e-8)
    Z = Xn @ Vt.T                                          # (n_c, K)
    print(f"[auto_05] Z shape {Z.shape}", flush=True)

    # ---- per-color RGB features (0..1) ----
    rgb = np.array([(r, g, b) for _, r, g, b in colors],
                   dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    # ---- 5-fold ridge: Z ~ [R G B 1] ----
    Phi = np.concatenate([rgb, np.ones((n_c, 1))], axis=1)  # (n_c, 4)
    Z_hat = np.zeros_like(Z)
    ridge_lam = 1e-6   # numerical stabilizer; matches typical OLS for n>>p
    folds = kfold_color_indices(n_c, N_FOLDS)
    for k, (tr, te) in enumerate(folds):
        A = Phi[tr].T @ Phi[tr] + ridge_lam * np.eye(4)
        B = Phi[tr].T @ Z[tr]
        W = np.linalg.solve(A, B)                          # (4, K)
        Z_hat[te] = Phi[te] @ W

    # per-color L2 residual, normalized by ||Z_c||
    res_l2 = np.linalg.norm(Z - Z_hat, axis=1)
    z_l2 = np.linalg.norm(Z, axis=1)
    rho = res_l2 / np.maximum(z_l2, 1e-8)

    # sanity macro R²
    ss_res = ((Z - Z_hat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    macro_r2 = 1 - ss_res / ss_tot
    print(f"[auto_05] CV macro R² (linear RGB) = {macro_r2:.4f}", flush=True)
    print(f"[auto_05] rho median={np.median(rho):.3f}  "
          f"min={rho.min():.3f} max={rho.max():.3f}", flush=True)

    order = np.argsort(rho)          # easy → hard
    n_show = 24

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 1.0],
                          hspace=0.55, wspace=0.18)

    def swatch_panel(ax, idxs, title):
        cols, rows = 6, 4
        ax.set_xlim(0, cols); ax.set_ylim(0, rows)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=12, pad=6)
        for i, ci in enumerate(idxs):
            r_, c_ = divmod(i, cols)
            x = c_ + 0.05; y = r_ + 0.05
            ax.add_patch(mpatches.Rectangle((x, y), 0.9, 0.6,
                                            facecolor=rgb[ci],
                                            edgecolor="black",
                                            linewidth=0.4))
            name = colors[ci][0]
            ax.text(x + 0.45, y + 0.78, name, ha="center", va="top",
                    fontsize=7.2,
                    color="black")
            ax.text(x + 0.45, y + 0.92, f"ρ={rho[ci]:.2f}",
                    ha="center", va="top", fontsize=7,
                    color="dimgray")

    ax_easy = fig.add_subplot(gs[0, 0])
    swatch_panel(ax_easy, order[:n_show],
                 f"24 easiest colors (smallest residual norm)\n"
                 f"linear RGB explains them almost exactly")
    ax_hard = fig.add_subplot(gs[0, 1])
    swatch_panel(ax_hard, order[-n_show:][::-1],
                 f"24 hardest colors (largest residual norm)\n"
                 f"linear RGB systematically misses these")

    # full histogram + ECDF on row 1
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.hist(rho, bins=50, color="steelblue", edgecolor="white")
    ax_hist.axvline(np.median(rho), color="crimson", lw=1.2,
                    label=f"median = {np.median(rho):.2f}")
    ax_hist.set_xlabel("relative residual ρ = ||Z−Ẑ|| / ||Z||")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("distribution of per-color residual norm", fontsize=11)
    ax_hist.legend(fontsize=9)

    ax_top = fig.add_subplot(gs[1, 1])
    top = order[-30:][::-1]
    y_pos = np.arange(len(top))
    bar_colors = [rgb[i] for i in top]
    ax_top.barh(y_pos, rho[top], color=bar_colors,
                edgecolor="black", linewidth=0.3)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels([colors[i][0] for i in top], fontsize=7.5)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("ρ")
    ax_top.set_title("top-30 hardest colors (bars colored by RGB)", fontsize=11)

    # bottom row: rho vs S and rho vs V
    ax_s = fig.add_subplot(gs[2, 0])
    ax_s.scatter(S, rho, c=rgb, s=22, edgecolors="black", linewidths=0.2)
    # binned mean overlay
    sb = np.linspace(0, 1, 11)
    bm = [rho[(S >= sb[i]) & (S < sb[i + 1])].mean() if
          ((S >= sb[i]) & (S < sb[i + 1])).any() else np.nan
          for i in range(len(sb) - 1)]
    ax_s.plot(0.5 * (sb[:-1] + sb[1:]), bm, "k--", lw=1.4,
              label="bin mean")
    ax_s.set_xlabel("HSV saturation S")
    ax_s.set_ylabel("ρ")
    ax_s.set_title("residual vs saturation", fontsize=11)
    ax_s.legend(fontsize=8)

    ax_v = fig.add_subplot(gs[2, 1])
    ax_v.scatter(V, rho, c=rgb, s=22, edgecolors="black", linewidths=0.2)
    vb = np.linspace(0, 1, 11)
    bmv = [rho[(V >= vb[i]) & (V < vb[i + 1])].mean() if
           ((V >= vb[i]) & (V < vb[i + 1])).any() else np.nan
           for i in range(len(vb) - 1)]
    ax_v.plot(0.5 * (vb[:-1] + vb[1:]), bmv, "k--", lw=1.4,
              label="bin mean")
    ax_v.set_xlabel("HSV value V")
    ax_v.set_ylabel("ρ")
    ax_v.set_title("residual vs value/lightness", fontsize=11)
    ax_v.legend(fontsize=8)

    fig.suptitle(
        "auto_05 · per-color residual norm after L_lin_rgb fit  ·  "
        f"cogito L40 · n_colors={n_c} · 5-fold color-grouped CV  "
        f"(macro R² = {macro_r2:.3f})",
        fontsize=13, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT}", flush=True)

    # quick correlation summary in stdout
    from scipy.stats import spearmanr
    for name, x in [("S", S), ("V", V), ("R", rgb[:, 0]),
                    ("G", rgb[:, 1]), ("B", rgb[:, 2])]:
        rho_s, p = spearmanr(x, rho)
        print(f"  spearman(ρ, {name}) = {rho_s:+.3f}  p={p:.2g}", flush=True)


if __name__ == "__main__":
    main()
