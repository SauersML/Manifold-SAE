"""Latent-discovery method comparison.

For each d in {2, 3, 4, 5}, find d-D latent coordinates for cogito's
per-color centroids using four different methods:

  • PCA          — linear-variance optimal
  • UMAP         — preserves local-neighborhood structure (nonlinear)
  • Isomap       — preserves geodesic distances (nonlinear)
  • Random       — projection to d random directions (baseline)

Then fit ONE joint d-D Duchon smooth on top of each latent. The R²
difference tells us how much "find better directions" matters when
the smooth is fixed.

Held-out CV: for each fold, fit the latent-discovery on training colors
only, then embed test colors via the method's transform() (Nyström for
Isomap, UMAP's built-in for UMAP, projection for PCA/random).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent))


N_T = 28


def load_xkcd_colors():
    from plot_color_geometry import load_xkcd_colors as _f
    return _f()


def load_harvest(p: Path) -> np.ndarray:
    from plot_color_geometry import load_harvest as _f
    return _f(p)


def kfold_color_indices(n_colors, n_folds=5, seed=0):
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


def kmeans_centers(X, K, seed=0, iters=12):
    rng = np.random.default_rng(seed)
    K = min(K, X.shape[0])
    idx = rng.choice(X.shape[0], K, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        a = d2.argmin(axis=1)
        for ki in range(K):
            m = (a == ki)
            if m.any(): C[ki] = X[m].mean(0)
    return C


# --------------- Latent discovery methods ---------------


def discover_pca(train_X, test_X, d):
    # sklearn PCA via _pca_basis.fit_top_pcs (center-only; same as before).
    from _pca_basis import fit_top_pcs
    mu = train_X.mean(0, keepdims=True)
    Xc = train_X - mu
    _, Vt = fit_top_pcs(Xc, d=d, standardize=False)
    proj = Vt[:d]
    return Xc @ proj.T, (test_X - mu) @ proj.T


def discover_random(train_X, test_X, d, seed=0):
    rng = np.random.default_rng(seed)
    D = train_X.shape[1]
    R = rng.standard_normal((D, d)) / np.sqrt(D)
    return train_X @ R, test_X @ R


def discover_umap(train_X, test_X, d):
    import umap
    reducer = umap.UMAP(n_components=d, n_neighbors=15, min_dist=0.1,
                         random_state=0, metric="euclidean")
    T_tr = reducer.fit_transform(train_X)
    T_te = reducer.transform(test_X)
    return np.asarray(T_tr), np.asarray(T_te)


def discover_isomap(train_X, test_X, d):
    from sklearn.manifold import Isomap
    iso = Isomap(n_components=d, n_neighbors=15, eigen_solver="auto")
    T_tr = iso.fit_transform(train_X)
    T_te = iso.transform(test_X)
    return np.asarray(T_tr), np.asarray(T_te)


METHODS = {
    "PCA":    discover_pca,
    "Random": discover_random,
    "UMAP":   discover_umap,
    "Isomap": discover_isomap,
}


# --------------- Duchon-on-latent helpers ---------------


def fit_duchon_on_latent(T_tr, T_te, train_Y, n_centers):
    from color_manifold_gam import duchon_basis_radial, reml_fit
    t_min = T_tr.min(0); t_max = T_tr.max(0)
    T_tr_n = np.clip((T_tr - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
    T_te_n = np.clip((T_te - t_min) / (t_max - t_min + 1e-9), 0.001, 0.999)
    centers = kmeans_centers(T_tr_n, n_centers)
    try:
        Phi_tr, P = duchon_basis_radial(T_tr_n, centers)
        Phi_te, _ = duchon_basis_radial(T_te_n, centers)
        B, _ = reml_fit(Phi_tr, train_Y, P, init_log_lambda=0.0)
        return Phi_te @ B
    except Exception as exc:
        return None


def main() -> int:
    cache_path = Path(os.environ.get(
        "HARVEST_PATH",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy",
    ))
    out_dir = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full = load_harvest(cache_path)
    n_colors_raw = X_full.shape[0] // N_T
    X_full = X_full[: n_colors_raw * N_T]
    centroids_all = np.zeros((n_colors_raw, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors_raw):
        centroids_all[ci] = X_full[ci * N_T:(ci + 1) * N_T].mean(0)
    from color_filter_list import filter_colors
    colors_all = load_xkcd_colors()[:n_colors_raw]
    _, kept_idx = filter_colors(colors_all)
    centroids = centroids_all[kept_idx]
    n_colors = len(kept_idx)
    print(f"[latent-discovery] n_colors={n_colors}", flush=True)

    folds = kfold_color_indices(n_colors, 5)
    dims = [2, 3, 4, 5]
    n_centers = 400

    results = {}
    for d in dims:
        results[d] = {}
        for method_name, method_fn in METHODS.items():
            fold_r2 = []
            for tr, te in folds:
                try:
                    T_tr, T_te = method_fn(centroids[tr], centroids[te], d)
                except Exception as exc:
                    print(f"  d={d} {method_name}: discovery failed: "
                          f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                    break
                pred = fit_duchon_on_latent(T_tr, T_te, centroids[tr], n_centers)
                if pred is None:
                    continue
                fold_r2.append(r2(centroids[te], pred))
            mean_r2 = float(np.mean(fold_r2)) if fold_r2 else float("nan")
            std_r2 = float(np.std(fold_r2)) if len(fold_r2) > 1 else 0.0
            results[d][method_name] = (mean_r2, std_r2, len(fold_r2))
            print(f"  d={d}  {method_name:8s}  R²={mean_r2:+.3f} ± {std_r2:.3f}  "
                  f"({len(fold_r2)}/5)", flush=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    method_colors = {"PCA": "#cfdee9", "Random": "#b8a989",
                      "UMAP": "#4f93bf", "Isomap": "#356d96"}
    method_order = ["Random", "PCA", "UMAP", "Isomap"]
    xs = np.arange(len(dims))
    bar_w = 0.2
    for i, m in enumerate(method_order):
        ys = [results[d][m][0] for d in dims]
        es = [results[d][m][1] for d in dims]
        ax.bar(xs + (i - 1.5) * bar_w, ys, bar_w, yerr=es,
                color=method_colors[m], edgecolor="black", linewidth=0.4,
                label=m, capsize=2)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d={d}" for d in dims], fontsize=11)
    ax.set_ylabel("held-out R²_macro  (5-fold CV by color)", fontsize=11)
    ax.set_xlabel("dimensionality of latent + joint d-D Duchon smooth", fontsize=11)
    ax.set_title(
        f"Latent-discovery comparison — same joint Duchon smooth, different ways to find the d directions\n"
        f"n_colors = {n_colors} (filtered)  ·  cogito L40", fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, m in enumerate(method_order):
        for xi, d in enumerate(dims):
            v = results[d][m][0]
            if np.isfinite(v):
                ax.annotate(f"{v:+.2f}", (xi + (i - 1.5) * bar_w, v),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=7)
    plt.tight_layout()
    out = out_dir / "latent_discovery_comparison.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
