"""auto_41: hue-binned held-out R² (12 bins) across 5 supervised specs.

results.json reports a single macro R² per spec, averaged over all 949
colors and 64 PCs. Idea (bb): does each spec's error concentrate in
particular regions of the hue wheel? If e.g. all linear-color-space
fits fail in the same hue bins, that's where the language model's
color manifold deviates from low-dim color geometry.

Procedure
---------
1. Reload centroid matrix exactly as auto_05 does (matching n_c=949).
2. Project to top-K PCs with the saved Vt_topK / mu / sigma.
3. For each of 5 supervised specs, do 5-fold color-grouped CV and
   collect held-out Z_hat for every color.
4. Bin colors by HSV hue into 12 equal-width bins (each bin = 30°).
5. Per-(spec, bin) compute macro R² across PCs:
       R²_bin = 1 - SS_res_bin / SS_tot_bin
   where SS_tot_bin uses the *global* Z mean (not the bin mean), so
   each bin's R² is comparable to the global macro R².
6. Plot a heatmap (specs × hue bins) with the hue bin colored along
   the x-axis, and a per-spec line plot beneath it. Annotate macro R².

Specs (no Gaussian RBF; no Duchon length_scale used anywhere):
  - L_lin_rgb            (linear in R,G,B)
  - L_poly3_rgb          (deg-3 polynomial of R,G,B with cross-terms)
  - L_poly3_lab          (deg-3 polynomial of L*,a*,b*)
  - L_cyc_hue_polysv     (sin/cos hue + deg-3 poly of S,V)
  - N_knn_lab_k10        (k-NN regression in LAB with k=10)
"""
from __future__ import annotations

import colorsys
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from plot_color_geometry import load_xkcd_colors, load_harvest  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402

N_T = 28
RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN / "auto_41.png"
RESULTS = RUN / "results.json"
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
N_FOLDS = 5
N_BINS = 12


def kfold_color_indices(n_colors, n_folds, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_colors)
    fold_of = np.empty(n_colors, dtype=int)
    fold_of[perm] = np.arange(n_colors) % n_folds
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0])
            for k in range(n_folds)]


def rgb_to_lab(rgb):
    """rgb in [0,1] -> CIE L*a*b* via sRGB->XYZ->LAB (D65). Vectorised."""
    def f_inv(t):
        return np.where(t > 0.04045, ((t + 0.055) / 1.055) ** 2.4, t / 12.92)

    rgb_lin = f_inv(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb_lin @ M.T
    # D65 white
    xn, yn, zn = 0.95047, 1.0, 1.08883
    xyz_n = xyz / np.array([xn, yn, zn])

    def f(t):
        delta = 6 / 29
        return np.where(t > delta ** 3, np.cbrt(t),
                        t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(xyz_n[:, 0]), f(xyz_n[:, 1]), f(xyz_n[:, 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def ridge_fit_predict(Phi_tr, Z_tr, Phi_te, lam=1e-6):
    A = Phi_tr.T @ Phi_tr + lam * np.eye(Phi_tr.shape[1])
    B = Phi_tr.T @ Z_tr
    W = np.linalg.solve(A, B)
    return Phi_te @ W


def cv_predict_linear(Phi, Z, folds, lam=1e-6):
    Z_hat = np.zeros_like(Z)
    for tr, te in folds:
        Z_hat[te] = ridge_fit_predict(Phi[tr], Z[tr], Phi[te], lam=lam)
    return Z_hat


def cv_predict_knn(X, Z, folds, k=10):
    Z_hat = np.zeros_like(Z)
    for tr, te in folds:
        m = KNeighborsRegressor(n_neighbors=k, weights="distance")
        m.fit(X[tr], Z[tr])
        Z_hat[te] = m.predict(X[te])
    return Z_hat


def macro_r2(Z, Z_hat, idx=None):
    if idx is None:
        idx = np.arange(len(Z))
    Zm = Z.mean(0, keepdims=True)  # global mean as baseline
    ss_res = ((Z[idx] - Z_hat[idx]) ** 2).sum()
    ss_tot = ((Z[idx] - Zm) ** 2).sum()
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def main():
    with open(RESULTS) as f:
        res = json.load(f)
    L = res["per_layer"]["L40"]
    Vt = np.asarray(L["Vt_topK"], dtype=np.float64)
    mu = np.asarray(L["mu"], dtype=np.float64)
    sigma = np.asarray(L["sigma"], dtype=np.float64)
    D = Vt.shape[1]

    X_full = load_harvest(HARVEST)
    n_raw = X_full.shape[0] // N_T
    X_full = X_full[: n_raw * N_T]
    centroids_all = X_full.reshape(n_raw, N_T, -1).mean(1)
    colors_all = load_xkcd_colors()[:n_raw]
    _, kept = filter_colors(colors_all)
    centroids = centroids_all[kept]
    colors = [colors_all[i] for i in kept]
    n_c = len(colors)
    assert centroids.shape[1] == D

    Xn = (centroids - mu[None, :]) / np.maximum(sigma[None, :], 1e-8)
    Z = Xn @ Vt.T
    print(f"[auto_41] n_c={n_c} Z={Z.shape}", flush=True)

    rgb = np.array([(r, g, b) for _, r, g, b in colors],
                   dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lab = rgb_to_lab(rgb)
    lab_std = (lab - lab.mean(0)) / lab.std(0)
    rgb_std = (rgb - rgb.mean(0)) / rgb.std(0)

    folds = kfold_color_indices(n_c, N_FOLDS)

    # --- feature builders ---
    def phi_lin_rgb():
        return np.concatenate([rgb, np.ones((n_c, 1))], 1)

    def phi_poly3(X):
        # full deg-3 with bias
        pf = PolynomialFeatures(degree=3, include_bias=True)
        return pf.fit_transform(X)

    def phi_cyc_polysv():
        # sin/cos of 2πH, plus poly3 of (S,V)
        cyc = np.stack([np.cos(2 * np.pi * H), np.sin(2 * np.pi * H)], 1)
        pf = PolynomialFeatures(degree=3, include_bias=True)
        sv = pf.fit_transform(np.stack([S, V], 1))
        return np.concatenate([cyc, sv], 1)

    Phi_specs = {
        "L_lin_rgb":        (phi_lin_rgb(),                  1e-4),
        "L_poly3_rgb":      (phi_poly3(rgb_std),              1e-2),
        "L_poly3_lab":      (phi_poly3(lab_std),              1e-2),
        "L_cyc_hue_polysv": (phi_cyc_polysv(),                1e-3),
    }
    print(f"[auto_41] Phi sizes:", flush=True)
    for k, (p, _) in Phi_specs.items():
        print(f"  {k:20s} {p.shape}", flush=True)

    Z_hats = {}
    for name, (Phi, lam) in Phi_specs.items():
        Z_hats[name] = cv_predict_linear(Phi, Z, folds, lam=lam)
    # k-NN in standardized LAB
    Z_hats["N_knn_lab_k10"] = cv_predict_knn(lab_std, Z, folds, k=10)

    spec_names = ["L_lin_rgb", "L_poly3_rgb", "L_poly3_lab",
                  "L_cyc_hue_polysv", "N_knn_lab_k10"]

    macro = {n: macro_r2(Z, Z_hats[n]) for n in spec_names}
    for n in spec_names:
        print(f"  macro R² {n:20s} = {macro[n]:+.4f}", flush=True)

    # --- bin by hue ---
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    bin_idx = np.clip(np.digitize(H, bin_edges) - 1, 0, N_BINS - 1)
    bin_counts = np.bincount(bin_idx, minlength=N_BINS)
    print(f"[auto_41] bin counts: {bin_counts.tolist()}", flush=True)

    r2_grid = np.full((len(spec_names), N_BINS), np.nan)
    for i, n in enumerate(spec_names):
        for b in range(N_BINS):
            idx = np.where(bin_idx == b)[0]
            if len(idx) >= 5:
                r2_grid[i, b] = macro_r2(Z, Z_hats[n], idx=idx)

    # --- plot ---
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_swatch = hsv_to_rgb(np.stack(
        [bin_centers, np.ones(N_BINS), np.ones(N_BINS)], 1))

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.25, 2.0, 2.0],
                          hspace=0.35)

    # row 0: hue swatch strip
    ax_strip = fig.add_subplot(gs[0])
    for b in range(N_BINS):
        ax_strip.add_patch(plt.Rectangle((b, 0), 1, 1,
                                         facecolor=bin_swatch[b],
                                         edgecolor="black", lw=0.3))
        ax_strip.text(b + 0.5, -0.35,
                      f"{int(bin_edges[b]*360):d}–{int(bin_edges[b+1]*360):d}°\n"
                      f"n={bin_counts[b]}",
                      ha="center", va="top", fontsize=8)
    ax_strip.set_xlim(0, N_BINS); ax_strip.set_ylim(0, 1)
    ax_strip.set_aspect("equal"); ax_strip.axis("off")
    ax_strip.set_title("hue bins (12 × 30°)", fontsize=11, pad=4)

    # row 1: heatmap
    ax_h = fig.add_subplot(gs[1])
    vmin = np.nanmin(r2_grid)
    vmax = np.nanmax(r2_grid)
    span = max(abs(vmin), abs(vmax))
    im = ax_h.imshow(r2_grid, aspect="auto", cmap="RdBu_r",
                     vmin=-span, vmax=span)
    ax_h.set_yticks(range(len(spec_names)))
    ax_h.set_yticklabels(
        [f"{n}  (macro={macro[n]:+.3f})" for n in spec_names], fontsize=10)
    ax_h.set_xticks(range(N_BINS))
    ax_h.set_xticklabels([f"{int(c*360)}°" for c in bin_centers], fontsize=9)
    ax_h.set_xlabel("hue bin centre")
    ax_h.set_title("held-out R² per hue bin (red = poor, blue = good)",
                   fontsize=11)
    for i in range(len(spec_names)):
        for b in range(N_BINS):
            v = r2_grid[i, b]
            if np.isnan(v):
                continue
            ax_h.text(b, i, f"{v:+.2f}",
                      ha="center", va="center", fontsize=7.5,
                      color="white" if abs(v) > 0.5 * span else "black")
    fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.01, label="R²")

    # row 2: line plot per spec, with hue-bin swatches on x-axis ticks
    ax_l = fig.add_subplot(gs[2])
    colors_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, n in enumerate(spec_names):
        ax_l.plot(np.arange(N_BINS), r2_grid[i], "-o",
                  color=colors_cycle[i],
                  label=f"{n}  (macro={macro[n]:+.3f})", lw=1.6, ms=5)
    ax_l.axhline(0, color="gray", lw=0.6, ls="--")
    ax_l.set_xticks(range(N_BINS))
    ax_l.set_xticklabels([f"{int(c*360)}°" for c in bin_centers], fontsize=9)
    ax_l.set_xlabel("hue bin centre")
    ax_l.set_ylabel("R² (vs global mean baseline)")
    ax_l.set_title("per-spec R² vs hue bin", fontsize=11)
    ax_l.legend(fontsize=8, loc="lower center", ncol=3)
    ax_l.grid(True, alpha=0.3)

    fig.suptitle(
        f"auto_41 · hue-binned R² (12 × 30°) · cogito L40 · n_c={n_c} · "
        f"5-fold color-grouped CV  [idea bb]",
        fontsize=12, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
