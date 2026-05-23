"""auto_13: angle (ff) — per-color centroid magnitude vs prediction R²:
are large-norm colors easier to predict from RGB?

Hypothesis: a centroid that "sticks out" of the activation cloud (large
||Z_c||) might be (a) easier — its identity is strongly encoded so a
linear-RGB readout has obvious signal — or (b) harder — large-norm
points often live in idiosyncratic regions the linear model can't reach
(very-saturated reds, near-blacks, fluorescents).

Procedure:
  1. Reload PCA basis (Vt_topK, mu, sigma) from results.json and the
     same filtered xkcd centroids used by the GAM zoo (matching n_c).
  2. Compute per-color Z = standardized centroid in top-K PCs and
     ||Z_c||_2 (the centroid magnitude in the projected space).
  3. 5-fold color-grouped CV ridge of Z ~ [R G B 1] (matches L_lin_rgb)
     to get per-color held-out Z_hat. Per-color R²:
        R²_c = 1 - ||Z_c - Z_hat_c||² / ||Z_c||²
     (this is the per-row coefficient of determination versus the
     null prediction Z=0, which equals predicting the global mean since
     Z is centered).
  4. Plot: scatter R²_c vs ||Z_c||, points colored by their xkcd RGB.
     Overlay (a) Spearman & Pearson correlation, (b) binned mean ± SE
     per magnitude quantile, and (c) annotate the 5 largest-norm and
     5 smallest-norm colors by name to make the story concrete.

Reads:  runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json
        runs/COLOR_COGITO_L40/X_L40.npy
Writes: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_13.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from plot_color_geometry import load_xkcd_colors, load_harvest  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402

N_T = 28
RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN / "auto_13.png"
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


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))


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
    print(f"[auto_13] n_c={n_c} D={D} K={Vt.shape[0]}", flush=True)

    Xn = (centroids - mu[None, :]) / np.maximum(sigma[None, :], 1e-8)
    Z = Xn @ Vt.T  # (n_c, K)

    rgb = np.array([(r, g, b) for _, r, g, b in colors],
                   dtype=np.float64) / 255.0
    Phi = np.concatenate([rgb, np.ones((n_c, 1))], axis=1)

    Z_hat = np.zeros_like(Z)
    ridge = 1e-6
    for tr, te in kfold_color_indices(n_c, N_FOLDS):
        A = Phi[tr].T @ Phi[tr] + ridge * np.eye(4)
        B = Phi[tr].T @ Z[tr]
        W = np.linalg.solve(A, B)
        Z_hat[te] = Phi[te] @ W

    # Per-color magnitude in PC space (Z is standardized → fair across PCs)
    norm_z = np.linalg.norm(Z, axis=1)
    sse = np.sum((Z - Z_hat) ** 2, axis=1)
    sst = np.sum(Z ** 2, axis=1)  # Z is already centered per-PC after standardization
    r2_c = 1.0 - sse / np.maximum(sst, 1e-12)

    macro = float(np.mean(r2_c))
    sp = spearman(norm_z, r2_c)
    pr = pearson(norm_z, r2_c)
    print(f"[auto_13] macro R² mean={macro:.3f}  spearman={sp:+.3f}  pearson={pr:+.3f}",
          flush=True)

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.5))

    pt_colors = np.clip(rgb, 0, 1)
    ax.scatter(norm_z, r2_c, c=pt_colors, s=22, edgecolor="black",
               linewidth=0.3, alpha=0.9)

    # binned mean ± SE
    n_bins = 8
    q = np.quantile(norm_z, np.linspace(0, 1, n_bins + 1))
    centers, means, ses = [], [], []
    for i in range(n_bins):
        mask = (norm_z >= q[i]) & (norm_z <= q[i + 1] if i == n_bins - 1
                                    else norm_z < q[i + 1])
        if mask.sum() < 3:
            continue
        centers.append(0.5 * (q[i] + q[i + 1]))
        means.append(float(np.mean(r2_c[mask])))
        ses.append(float(np.std(r2_c[mask]) / np.sqrt(mask.sum())))
    ax.errorbar(centers, means, yerr=ses, fmt="o-", color="black",
                lw=2, ms=7, capsize=4, label="binned mean ± SE", zorder=5)

    # annotate top-5 largest & smallest norm
    order = np.argsort(norm_z)
    pick = list(order[:5]) + list(order[-5:])
    for idx in pick:
        name = colors[idx][0]
        ax.annotate(name, (norm_z[idx], r2_c[idx]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="0.2")

    ax.set_xlabel("centroid magnitude  ||Z_c||  in top-64 PC space")
    ax.set_ylabel("per-color held-out R²   (L_lin_rgb, 5-fold)")
    ax.set_title(
        f"auto_13 · cogito L40 · n_c={n_c} · per-color centroid magnitude vs prediction R²\n"
        f"macro mean R² = {macro:.3f}   "
        f"Spearman ρ = {sp:+.3f}   Pearson r = {pr:+.3f}"
    )
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"[auto_13] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
