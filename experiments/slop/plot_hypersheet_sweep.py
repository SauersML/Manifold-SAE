"""Single-term joint Duchon sweep — the 'hypersheet' fit.

For each d in {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32}, fit ONE joint
Duchon smooth (single term, no additive split, no pairs/triples) in the
top-d PCA-latent of cogito's per-color centroids. Effective flexibility
is selected by REML's smoothing-parameter λ — we don't vary the center
count (that only sets representational capacity; REML regularizes).
Centers fixed at a large-enough value so the basis is expressive and
REML's λ is the bottleneck.

Outputs:
  hypersheet_sweep.png  — single bar series: x=d, y=held-out R²

The 'hypersheet' framing: a single joint Duchon spline IS a d-dimensional
manifold (hypersheet) in cogito's residual space. We're asking how many
dims that hypersheet needs to capture the color manifold.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


N_T = 28


def load_xkcd_colors():
    from plot_color_geometry import load_xkcd_colors as _f
    return _f()


def load_harvest(p: Path) -> np.ndarray:
    from plot_color_geometry import load_harvest as _f
    return _f(p)


def kfold_color_indices(n_colors: int, n_folds: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_colors)
    fold_of = np.empty(n_colors, dtype=int)
    fold_of[perm] = np.arange(n_colors) % n_folds
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0])
            for k in range(n_folds)]


def r2(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def kmeans_centers(X: np.ndarray, K: int, seed: int = 0, iters: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    K = min(K, X.shape[0])
    idx = rng.choice(X.shape[0], K, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        a = d2.argmin(axis=1)
        for ki in range(K):
            m = (a == ki)
            if m.any():
                C[ki] = X[m].mean(0)
    return C


def fit_paired(train_centroids: np.ndarray, test_centroids: np.ndarray,
                d: int, n_centers: int) -> tuple[float, float, str]:
    """One CV fold: project to top-d PCA, return both linear-baseline R²
    (PCA-d reconstruction) and joint d-D HYBRID Duchon smooth R² (REML λ).

    Defaults only — gamfit picks nullspace_order, power, kernel.
    """
    import gamfit
    from color_manifold_gam import reml_fit
    from _pca_basis import fit_top_pcs
    mu = train_centroids.mean(0, keepdims=True)
    Xc = train_centroids - mu
    # sklearn PCA (center-only; matches prior np.linalg.svd behavior)
    _, Vt = fit_top_pcs(Xc, d=d, standardize=False)
    proj = Vt[:d]
    T_tr = Xc @ proj.T
    test_Xc = test_centroids - mu
    T_te = test_Xc @ proj.T
    # Linear baseline: rank-d reconstruction
    test_pred_lin = T_te @ proj + mu
    r2_lin = r2(test_centroids, test_pred_lin)
    # Joint hybrid Duchon: normalize latent, k-means centers, REML fit
    t_min = T_tr.min(0); t_max = T_tr.max(0)
    T_tr_n = np.clip((T_tr - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
    T_te_n = np.clip((T_te - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
    centers = kmeans_centers(T_tr_n, n_centers)
    # Defaults only — let gamfit choose nullspace_order/power.
    try:
        Phi_tr = np.asarray(gamfit.duchon_basis(T_tr_n, centers, m=2))
        Phi_te = np.asarray(gamfit.duchon_basis(T_te_n, centers, m=2))
        P = np.asarray(gamfit.duchon_function_norm_penalty(centers, m=2))
        B, _ = reml_fit(Phi_tr, train_centroids, P, init_log_lambda=0.0)
        pred_te = Phi_te @ B
        return r2_lin, r2(test_centroids, pred_te), "ok"
    except Exception as exc:
        return r2_lin, float("nan"), f"{type(exc).__name__}: {str(exc)[:80]}"


def _centers_for_d(d: int) -> int:
    return 200


def main() -> int:
    cache_path = Path(os.environ.get(
        "HARVEST_PATH",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy",
    ))
    out_dir = Path(
        "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full = load_harvest(cache_path)
    n_colors_raw = X_full.shape[0] // N_T
    X_full = X_full[: n_colors_raw * N_T]
    centroids_all = np.zeros((n_colors_raw, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors_raw):
        centroids_all[ci] = X_full[ci * N_T:(ci + 1) * N_T].mean(0)

    # Drop the 63 names with strong non-color connotations
    from color_filter_list import filter_colors
    colors_all = load_xkcd_colors()[:n_colors_raw]
    kept_colors, kept_idx = filter_colors(colors_all)
    centroids = centroids_all[kept_idx]
    n_colors = len(kept_colors)
    n_dropped = n_colors_raw - n_colors
    print(f"[hypersheet] {n_colors_raw} raw → dropped {n_dropped} bad-name → "
          f"using {n_colors} clean colors  ·  D={centroids.shape[1]}",
          flush=True)

    folds = kfold_color_indices(n_colors, 5)
    # Hybrid Duchon scales to any d. Sweep wide.
    dims = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]

    results = {}
    for d in dims:
        n_centers = _centers_for_d(d)
        lin_fold, du_fold = [], []
        for tr, te in folds:
            r2_lin, r2_du, status = fit_paired(
                centroids[tr], centroids[te], d=d, n_centers=n_centers,
            )
            if np.isfinite(r2_lin):  lin_fold.append(r2_lin)
            if np.isfinite(r2_du):   du_fold.append(r2_du)
        lin = (float(np.mean(lin_fold)) if lin_fold else float("nan"),
               float(np.std(lin_fold)) if len(lin_fold) > 1 else 0.0)
        du = (float(np.mean(du_fold)) if du_fold else float("nan"),
              float(np.std(du_fold)) if len(du_fold) > 1 else 0.0)
        results[d] = {"linear": lin, "duchon": du, "n_centers": n_centers}
        print(f"  d={d:2d}  K={n_centers:5d}  "
              f"linear R²={lin[0]:+.3f}±{lin[1]:.3f}    "
              f"Duchon R²={du[0]:+.3f}±{du[1]:.3f}    "
              f"Δ={du[0] - lin[0]:+.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    xs = np.arange(len(dims))
    bar_w = 0.4
    lin_ys = [results[d]["linear"][0] for d in dims]
    lin_es = [results[d]["linear"][1] for d in dims]
    du_ys  = [results[d]["duchon"][0]  for d in dims]
    du_es  = [results[d]["duchon"][1]  for d in dims]
    ax.bar(xs - bar_w/2, lin_ys, bar_w, yerr=lin_es,
            color="#cfdee9", edgecolor="black", linewidth=0.5, capsize=3,
            label="linear PCA-d reconstruction (no smooth)")
    ax.bar(xs + bar_w/2, du_ys, bar_w, yerr=du_es,
            color="#356d96", edgecolor="black", linewidth=0.5, capsize=3,
            label="joint d-D Duchon smooth (REML λ)")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d={d}\n(K={results[d]['n_centers']})" for d in dims],
                        fontsize=10)
    ax.set_ylabel("held-out R²_macro  (5-fold CV by color)", fontsize=11)
    ax.set_xlabel("dimensionality of top-d PCA latent  (K = Duchon centers)", fontsize=11)
    ax.set_title(
        f"Matched comparison — linear vs joint Duchon at each d\n"
        f"n_colors = {n_colors} (filtered)  ·  cogito L40", fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotate bars and Δ
    for xi, d in enumerate(dims):
        if np.isfinite(lin_ys[xi]):
            ax.annotate(f"{lin_ys[xi]:+.3f}", (xi - bar_w/2, lin_ys[xi]),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=8)
        if np.isfinite(du_ys[xi]):
            ax.annotate(f"{du_ys[xi]:+.3f}", (xi + bar_w/2, du_ys[xi]),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=8)
        delta = du_ys[xi] - lin_ys[xi]
        if np.isfinite(delta):
            ax.annotate(f"Δ={delta:+.3f}", (xi, max(lin_ys[xi], du_ys[xi])),
                         xytext=(0, 18), textcoords="offset points",
                         ha="center", fontsize=9, color="#aa3333",
                         weight="bold")
    plt.tight_layout()
    out = out_dir / "hypersheet_sweep.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
