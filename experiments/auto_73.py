"""auto_73.py — Triangulate the intrinsic dimension of cogito's per-color
centroids using three independent, parameter-light estimators:

  (1) TwoNN  (Facco et al., Sci Rep 2017)
  (2) MLE / Levina-Bickel  (kNN log-distance ratios, swept over k)
  (3) Correlation dimension  (Grassberger-Procaccia)

We sweep across K_PC ∈ {3, 8, 16, 32, 64} to see whether the estimated
dimension grows with the size of the PCA envelope we project into. Combined
with the three prior estimators (score-Jacobian d≈1, local-PCA d≈5-7, U_3d
alternating-fit elbow d≈3), this should narrow down the geometric picture.

No Gaussian RBF, no Duchon length_scale, no B-splines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors
from color_filter_list import filter_colors
from _pca_basis import load_pc_basis, project

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
K_PC_LIST = [3, 8, 16, 32, 64]
K_MLE_LIST = [5, 10, 20, 50, 100]
TWONN_TOP_FRAC = 0.10  # discard top 10% of μ values (Facco's heuristic)
CORRDIM_N_EPS = 40


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def twoNN(Z: np.ndarray, top_frac: float = TWONN_TOP_FRAC):
    """Facco et al. TwoNN.

    Returns: d_hat, (log_mu_used, log1mF_used, slope, intercept, mu_all).
    """
    N = Z.shape[0]
    tree = cKDTree(Z)
    # k=3 because tree includes self at distance 0 → r1=dists[:,1], r2=dists[:,2]
    dists, _ = tree.query(Z, k=3)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    eps = 1e-300
    # Drop points with r1==0 (duplicates) — pathological
    mask = (r1 > 0) & (r2 > 0)
    mu = r2[mask] / np.maximum(r1[mask], eps)
    # Sort ascending → empirical CDF F_i = i/n
    mu_sorted = np.sort(mu)
    n = mu_sorted.size
    F = (np.arange(1, n + 1) - 0.5) / n  # mid-rank, avoids log(0) at top
    log_mu = np.log(mu_sorted)
    log1mF = np.log(1.0 - F)
    # Discard top fraction (Facco recommends to avoid edge effects)
    cutoff = int(n * (1.0 - top_frac))
    cutoff = max(cutoff, int(0.5 * n))
    x = log_mu[:cutoff]
    y = log1mF[:cutoff]
    # Linear regression through the origin gives the cleanest TwoNN estimate
    # (theory: log(1-F) = -d * log(mu), no intercept).
    d_hat_origin = -float((x * y).sum() / (x * x).sum())
    # Unconstrained linear fit for diagnostic plotting
    slope, intercept = np.polyfit(x, y, 1)
    return d_hat_origin, {
        "log_mu_used": x,
        "log1mF_used": y,
        "slope_unconstr": float(slope),
        "intercept_unconstr": float(intercept),
        "mu_sorted": mu_sorted,
        "F": F,
        "log_mu_all": log_mu,
        "log1mF_all": log1mF,
    }


def mle_levina_bickel(Z: np.ndarray, k: int) -> float:
    """Levina-Bickel MLE intrinsic-dimension estimator for a single k.

    d̂_k(i) = (1/(k-1)) * Σ_{j=1..k-1} log(r_k(i) / r_j(i))
    d̂_k = 1 / mean_i d̂_k(i)   (this is the standard 'inverse-mean' form;
    averaging the per-point inverses is biased).
    """
    N = Z.shape[0]
    tree = cKDTree(Z)
    # Need k+1 neighbors to skip self
    dists, _ = tree.query(Z, k=k + 1)
    # dists[:, 0] is self (==0); use indices 1..k
    r = dists[:, 1:]  # shape (N, k)
    rk = r[:, -1:]    # shape (N, 1) – kth neighbor
    rj = r[:, :-1]    # shape (N, k-1) – 1st..(k-1)th neighbors
    eps = 1e-300
    # Per-point sum of log ratios
    log_ratios = np.log(np.maximum(rk, eps) / np.maximum(rj, eps))
    per_point = log_ratios.sum(axis=1) / (k - 1)
    # Drop any nonpositive (duplicates)
    per_point = per_point[per_point > 0]
    if per_point.size == 0:
        return float("nan")
    d_hat = 1.0 / per_point.mean()
    return float(d_hat)


def correlation_dimension(Z: np.ndarray, n_eps: int = CORRDIM_N_EPS):
    """Grassberger-Procaccia correlation-dimension estimator.

    Returns: d_hat (slope of best linear segment), (log_eps, log_C, fit_mask).
    """
    from scipy.spatial.distance import pdist
    d_pair = pdist(Z)
    eps_lo = np.percentile(d_pair, 5.0)
    eps_hi = np.percentile(d_pair, 95.0)
    eps_grid = np.exp(np.linspace(np.log(eps_lo), np.log(eps_hi), n_eps))
    # C(ε) = #{pairs i<j : d_ij <= ε} / (N choose 2)
    d_sorted = np.sort(d_pair)
    n_pairs = d_pair.size
    counts = np.searchsorted(d_sorted, eps_grid, side="right")
    C = counts / n_pairs
    log_eps = np.log(eps_grid)
    log_C = np.log(np.maximum(C, 1e-300))
    # Find the linear regime: scan all windows of length >= 8, pick the one
    # with smallest residual where C is nontrivial (C in [1e-3, 0.5]).
    ok = (C > 1e-3) & (C < 0.5)
    if ok.sum() < 8:
        ok = (C > 0) & (C < 1)
    idx = np.where(ok)[0]
    if idx.size < 4:
        return float("nan"), {"log_eps": log_eps, "log_C": log_C,
                              "fit_mask": ok, "slope": float("nan"),
                              "intercept": float("nan")}
    best = (np.inf, None, None, None)
    for L in range(max(8, idx.size // 3), idx.size + 1):
        for start in range(0, idx.size - L + 1):
            sub = idx[start:start + L]
            x = log_eps[sub]
            y = log_C[sub]
            slope, intercept = np.polyfit(x, y, 1)
            resid = float(((y - (slope * x + intercept)) ** 2).mean())
            if resid < best[0]:
                best = (resid, slope, intercept, sub)
    _, slope, intercept, fit_idx = best
    fit_mask = np.zeros_like(C, dtype=bool)
    fit_mask[fit_idx] = True
    return float(slope), {
        "log_eps": log_eps,
        "log_C": log_C,
        "fit_mask": fit_mask,
        "slope": float(slope),
        "intercept": float(intercept),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X = np.load(cache, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    print(f"[load] X mmap shape={X.shape}, n_raw={n_raw}")

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    N = len(kept)
    print(f"[load] N={N} filtered colors")

    # Build the K=64 basis once; restrict to the first K_pc cols for sweeps
    basis = load_pc_basis(K=max(K_PC_LIST))
    Z_full = project(centroids, basis)
    print(f"[load] Z_full shape={Z_full.shape}  "
          f"EVR_top{max(K_PC_LIST)}={float(basis['evr'].sum()):.3f}")

    results = {}
    diag_twoNN_64 = None
    diag_corrdim_64 = None
    for K in K_PC_LIST:
        Z = np.ascontiguousarray(Z_full[:, :K], dtype=np.float64)
        # TwoNN
        d_two, diag_two = twoNN(Z)
        # MLE sweep
        d_mle = {k: mle_levina_bickel(Z, k) for k in K_MLE_LIST}
        # Correlation dimension
        d_corr, diag_corr = correlation_dimension(Z)
        results[K] = {
            "twoNN": float(d_two),
            "MLE": {int(k): float(v) for k, v in d_mle.items()},
            "corrdim": float(d_corr),
        }
        print(f"[K_PC={K:>2}]  TwoNN={d_two:5.2f}   "
              f"MLE(k=5,10,20,50,100)="
              f"{[f'{d_mle[k]:.2f}' for k in K_MLE_LIST]}   "
              f"corrdim={d_corr:5.2f}")
        if K == 64:
            diag_twoNN_64 = diag_two
            diag_corrdim_64 = diag_corr

    # ----- 3-panel figure -----
    fig = plt.figure(figsize=(17, 5.5))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    # (1) TwoNN regression for K_PC=64
    ax = fig.add_subplot(gs[0, 0])
    d_two_64 = results[64]["twoNN"]
    x_all = diag_twoNN_64["log_mu_all"]
    y_all = diag_twoNN_64["log1mF_all"]
    x_used = diag_twoNN_64["log_mu_used"]
    y_used = diag_twoNN_64["log1mF_used"]
    ax.scatter(x_all, y_all, s=8, c="lightgrey",
               label="all points", zorder=1)
    ax.scatter(x_used, y_used, s=10, c="#1f77b4",
               label=f"used (lowest {int(100 * (1 - TWONN_TOP_FRAC))}%)",
               zorder=2)
    xs = np.linspace(x_used.min(), x_used.max(), 100)
    ax.plot(xs, -d_two_64 * xs, "r-", lw=2,
            label=f"fit slope = −{d_two_64:.2f}")
    ax.set_xlabel("log μ  (μ = r₂/r₁)")
    ax.set_ylabel("log(1 − F(μ))")
    ax.set_title(f"(1) TwoNN regression  K_PC=64\n"
                 f"d̂_TwoNN = {d_two_64:.2f}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # (2) MLE d̂ vs k for each K_PC
    ax = fig.add_subplot(gs[0, 1])
    colors_k = plt.cm.viridis(np.linspace(0.15, 0.9, len(K_PC_LIST)))
    for K, c in zip(K_PC_LIST, colors_k):
        ys = [results[K]["MLE"][k] for k in K_MLE_LIST]
        ax.plot(K_MLE_LIST, ys, "o-", c=c, lw=1.8,
                label=f"K_PC={K}")
    ax.set_xscale("log")
    ax.set_xlabel("k  (#neighbors)")
    ax.set_ylabel("MLE d̂  (Levina-Bickel)")
    ax.set_title("(2) MLE intrinsic-dim vs k\n"
                 "(plateau ⇒ signal-dominated)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, which="both")

    # (3) Correlation dimension log C vs log ε
    ax = fig.add_subplot(gs[0, 2])
    le = diag_corrdim_64["log_eps"]
    lc = diag_corrdim_64["log_C"]
    fm = diag_corrdim_64["fit_mask"]
    d_corr_64 = results[64]["corrdim"]
    intercept = diag_corrdim_64["intercept"]
    ax.plot(le, lc, "o-", c="lightgrey", ms=4, lw=1, label="all ε")
    ax.plot(le[fm], lc[fm], "o", c="#1f77b4", ms=6,
            label="linear regime")
    if np.isfinite(d_corr_64):
        xs = np.linspace(le[fm].min(), le[fm].max(), 100)
        ax.plot(xs, d_corr_64 * xs + intercept, "r-", lw=2,
                label=f"slope = {d_corr_64:.2f}")
    ax.set_xlabel("log ε")
    ax.set_ylabel("log C(ε)")
    ax.set_title(f"(3) Correlation dimension  K_PC=64\n"
                 f"d̂_corrdim = {d_corr_64:.2f}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "auto_73 · intrinsic-dim triangulation of cogito L40 per-color "
        f"centroids · N={N} colors",
        fontsize=13,
    )
    out_png = OUT_DIR / "auto_73.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_png}")

    payload = {
        "n_colors": int(N),
        "K_PC_list": K_PC_LIST,
        "K_MLE_list": K_MLE_LIST,
        "twoNN_top_frac_discarded": TWONN_TOP_FRAC,
        "corrdim_n_eps": CORRDIM_N_EPS,
        "estimates_by_K_PC": {
            str(K): {
                "twoNN": results[K]["twoNN"],
                "MLE": results[K]["MLE"],
                "corrdim": results[K]["corrdim"],
            }
            for K in K_PC_LIST
        },
        "prior_estimates": {
            "score_jacobian_auto_exp_16_median_d_tan": 1.0,
            "local_pca_auto_exp_07_participation_ratio_median": [5.5, 7.4],
            "local_pca_auto_exp_07_d90_median": [7.0, 10.0],
            "U3d_alt_fit_auto_exp_06_cv_elbow": 3.0,
        },
        "notes": (
            "TwoNN: Facco et al., Sci Rep 2017. d̂ from origin-constrained "
            "linear fit of log(1-F) on log(μ=r2/r1), discarding the top "
            f"{int(100*TWONN_TOP_FRAC)}% of μ. "
            "MLE: Levina-Bickel inverse-mean-of-log-ratio over kNN, k swept "
            "over {5,10,20,50,100}. "
            "Correlation dimension: Grassberger-Procaccia slope over best "
            "linear window of log C(ε) vs log ε in [p5, p95]. "
            "All three are computed on Z = top-K_PC PCA projections of "
            "cogito L40 per-color centroids (TOP_TEMPLATES averaged). "
            "No Gaussian RBF, no Duchon length_scale, no B-splines."
        ),
    }
    (OUT_DIR / "auto_73.json").write_text(json.dumps(payload, indent=2))
    print(f"[saved] {OUT_DIR / 'auto_73.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
