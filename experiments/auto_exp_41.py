"""auto_exp_41: TOPOLOGY of the free 2D name-semantic block from auto_exp_40.

Question: after HSV-supervised gauge-fix on axes {0,1,2}, is the residual free
2D block (auto_exp_40's axes 3-4, the 2 ARD-dominant ones; axis-5 was noise)
best modeled as Euclidean R^2, Circle S^1 (× R amplitude), Sphere S^2, or
Cylinder S^1 × R?

Reuses auto_exp_38 pipeline (mmap='r', cached PCA basis) and auto_exp_40's
PPCA-ARD fit (w_ard=0.01 — lowest alpha, dominant variance config).
We pick the two free axes with largest free_var.

GAMFIT PATH: production gamfit 0.1.112 has no TopologyAutoSelector. FALLBACK:
manual per-topology KDE / von Mises / spherical KDE / product kernel, scored by
BIC (-2 log_lik + k log n) ranked alongside a Tierney-Kadane-style normalized
score (log_marginal / sqrt(n) * effective_dim, per auto_76 convention).

Held-out generalization: 5-fold CV by color (rows are color centroids).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.stats import vonmises, gaussian_kde

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (
    X_PATH, N_TEMPLATES, K_PCS,
    per_color_stats_mmap, load_xkcd_rgb, hsv_from_rgb,
    fit_aux_supervised_hsv,
)
from auto_exp_40 import fit_free_ard_ppca, D_FREE

ROOT = Path("/Users/user/Manifold-SAE")
OUT_NPZ = ROOT / "runs" / "auto_exp_41_results.npz"
MEMO = Path("/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/"
            "project_cogito_recovery_at_d_aux_3.md")

W_ARD_PICK = 0.01
N_FOLDS = 5
RNG = np.random.default_rng(41)


# ---------- topology scorers ----------------------------------------------
def fit_score_euclidean(train, test):
    """Plane R^2: full Gaussian (mean + 2x2 cov). 5 params."""
    mu = train.mean(0)
    cov = np.cov(train.T) + 1e-6 * np.eye(2)
    L = np.linalg.cholesky(cov)
    inv = np.linalg.inv(cov)
    logdet = 2 * np.log(np.diag(L)).sum()
    d = test - mu
    ll = -0.5 * (np.einsum("ni,ij,nj->n", d, inv, d)
                 + logdet + 2 * np.log(2 * np.pi))
    return float(ll.sum()), 5  # 2 mean + 3 cov


def fit_score_circle(train, test):
    """S^1 × R: von Mises on angle theta = atan2(y, x); Gaussian on radius r."""
    th_tr = np.arctan2(train[:, 1], train[:, 0])
    r_tr = np.linalg.norm(train, axis=1)
    # von Mises MLE
    C, S = np.cos(th_tr).mean(), np.sin(th_tr).mean()
    R = np.hypot(C, S)
    mu = np.arctan2(S, C)
    # kappa: Banerjee 2005 approx
    if R < 0.53:
        kappa = 2 * R + R ** 3 + 5 * R ** 5 / 6
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        kappa = 1 / (R ** 3 - 4 * R ** 2 + 3 * R)
    kappa = max(kappa, 1e-6)
    # Gaussian on r
    r_mean = r_tr.mean()
    r_var = r_tr.var() + 1e-6
    th_te = np.arctan2(test[:, 1], test[:, 0])
    r_te = np.linalg.norm(test, axis=1)
    ll_th = vonmises.logpdf(th_te, kappa, loc=mu).sum()
    ll_r = (-0.5 * np.log(2 * np.pi * r_var)
            - (r_te - r_mean) ** 2 / (2 * r_var)).sum()
    return float(ll_th + ll_r), 4  # mu, kappa, r_mean, r_var


def fit_score_sphere(train, test):
    """S^2: lift (x,y) to unit 3-vector via stereographic projection,
    fit a Kent-lite (von Mises-Fisher) distribution, score on test.

    For an intrinsic-2D check on a sphere: standardize then project to
    the unit sphere via z = (2x, 2y, |p|^2-1) / (|p|^2+1), score with vMF.
    """
    def to_sphere(p):
        sc = p / (np.abs(p).max() + 1e-9)  # scale into reasonable range
        n2 = (sc ** 2).sum(1, keepdims=True)
        return np.concatenate([2 * sc, n2 - 1], axis=1) / (n2 + 1)
    z_tr = to_sphere(train)
    z_te = to_sphere(test)
    # vMF MLE: mu = mean / |mean|, kappa via Banerjee approx
    m = z_tr.mean(0)
    R = np.linalg.norm(m)
    mu_v = m / max(R, 1e-9)
    p_dim = 3
    kappa = R * (p_dim - R ** 2) / max(1 - R ** 2, 1e-6)
    kappa = max(kappa, 1e-6)
    # vMF density on S^{p-1}: C_p(k) exp(k mu·z), with
    # C_3(k) = k / (4 pi sinh(k))
    log_C = np.log(kappa) - np.log(4 * np.pi) - kappa - np.log1p(
        -np.exp(-2 * kappa) + 1e-30) + np.log(2)
    ll = (log_C + kappa * (z_te @ mu_v)).sum()
    # Jacobian of stereographic from R^2 onto S^2: roughly 4/(1+|p|^2)^2
    # include it so log-likelihoods are comparable (density on R^2)
    sc_te = test / (np.abs(test).max() + 1e-9)
    n2_te = (sc_te ** 2).sum(1)
    log_jac = (np.log(4) - 2 * np.log(1 + n2_te)).sum()
    return float(ll + log_jac), 3  # mu (2 dof on sphere) + kappa


def fit_score_cylinder(train, test):
    """S^1 (angle for one feature) × R (real for the other).
    Independent von Mises on theta_x = atan2(y, x) and Gaussian on a
    secondary 'radius' coordinate built from the second principal direction.

    Actually: pick the dominant variance axis as the 'angular' lift via
    angle = 2*pi*F(x_1) (rank-based wrap), then Gaussian on x_2.
    """
    # use auto-orient: angular direction = first PC of train
    mu_tr = train.mean(0)
    Tc_tr = train - mu_tr
    U, S, Vt = np.linalg.svd(Tc_tr, full_matrices=False)
    e1 = Vt[0]  # angular dir
    e2 = Vt[1]  # linear dir
    proj_tr_1 = Tc_tr @ e1
    proj_tr_2 = Tc_tr @ e2
    # wrap proj_tr_1 to [-pi, pi] by rank-quantile
    ranks_tr = proj_tr_1.argsort().argsort()
    th_tr = 2 * np.pi * (ranks_tr / len(ranks_tr)) - np.pi
    # von Mises MLE on th_tr (will be ~uniform by construction; use kappa=0)
    # Better: model theta as uniform on S^1 (entropy = log(2 pi)), no params.
    # then score linear part as Gaussian
    lin_mean = proj_tr_2.mean()
    lin_var = proj_tr_2.var() + 1e-6
    Tc_te = test - mu_tr
    proj_te_2 = Tc_te @ e2
    proj_te_1 = Tc_te @ e1
    # Embed test into the same angular grid by quantile of train CDF
    sorted_tr = np.sort(proj_tr_1)
    qte = np.searchsorted(sorted_tr, proj_te_1) / max(len(sorted_tr), 1)
    qte = np.clip(qte, 1e-6, 1 - 1e-6)
    # density on circle: 1/(2 pi); ll uniform = -log(2 pi) per pt
    ll_th = -np.log(2 * np.pi) * len(test)
    ll_lin = (-0.5 * np.log(2 * np.pi * lin_var)
              - (proj_te_2 - lin_mean) ** 2 / (2 * lin_var)).sum()
    # account for the change-of-variable Jacobian from R -> S^1
    # angular density on R is uniform(min..max), const = 1/(max-min)
    tr_range = sorted_tr[-1] - sorted_tr[0] + 1e-9
    # We replace 1/(2pi) with 1/range to keep densities on R comparable
    ll_th_R = -np.log(tr_range) * len(test)
    return float(ll_th_R + ll_lin), 4  # mu(2) + e1(1 angle) + lin_var


SCORERS = {
    "Euclidean": fit_score_euclidean,
    "Circle":    fit_score_circle,
    "Sphere":    fit_score_sphere,
    "Cylinder":  fit_score_cylinder,
}


def bic(ll, k, n):
    return -2 * ll + k * np.log(max(n, 2))


def tk_score(ll, k, n, eff_dim=2):
    # TK-style normalization (per gamfit auto_76 convention):
    # log_marginal / sqrt(n) * eff_dim. Larger = better.
    # Use the (ll - 0.5 k log n) Laplace marginal approx.
    log_marg = ll - 0.5 * k * np.log(max(n, 2))
    return log_marg / np.sqrt(max(n, 1)) * eff_dim


# ---------- main ----------------------------------------------------------
def main():
    t0 = time.time()
    print("[auto_exp_41] free-block topology sweep")
    print(f"[gamfit] checking for TopologyAutoSelector...")
    try:
        import gamfit
        has_topo = hasattr(gamfit, "TopologyAutoSelector")
        print(f"[gamfit] version={gamfit.__version__} "
              f"TopologyAutoSelector={has_topo}")
    except Exception as e:
        has_topo = False
        print(f"[gamfit] unavailable: {e!r}")
    path = "production" if has_topo else "fallback (scipy KDE/vonMises/vMF)"
    print(f"[path] {path}")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)

    # ---- HSV supervised
    sup = fit_aux_supervised_hsv(T0, hsv)
    print(f"[hsv] R2={np.round(sup['r2_hsv'], 3)}")

    # ---- HSV-orthogonal residual
    Tc = T0 - T0.mean(0, keepdims=True)
    Q, _ = np.linalg.qr(sup["W_sup"])
    T_perp = Tc @ (np.eye(sup["W_sup"].shape[0]) - Q @ Q.T)

    # ---- PPCA-ARD on free block (auto_exp_40, dominant w_ard=0.01)
    fit = fit_free_ard_ppca(T_perp, D_FREE, W_ARD_PICK)
    T_free = fit["T_free"]
    # Take the 2 axes with largest variance (auto_exp_40 reported axis-5 was noise)
    var = T_free.var(0)
    order = np.argsort(var)[::-1]
    keep = order[:2]
    T2 = T_free[:, keep].astype(np.float64)
    # standardize for numerical sanity (topology choice is rotation-equivariant
    # for the metrics that are direction-invariant; cylinder picks dir via SVD)
    T2 = T2 - T2.mean(0, keepdims=True)
    T2 = T2 / (T2.std(0, keepdims=True) + 1e-9)
    print(f"[free2d] shape={T2.shape}; var_kept={var[keep]}; "
          f"axis_dropped_var={var[order[2]]:.4g}")

    # ---- In-sample log-lik per topology
    print()
    print("=== IN-SAMPLE FIT ===")
    in_sample = {}
    n = T2.shape[0]
    for name, fn in SCORERS.items():
        ll, k = fn(T2, T2)
        in_sample[name] = (ll, k)
        print(f"  {name:>10}: ll={ll:>10.2f} k={k} BIC={bic(ll,k,n):>10.2f} "
              f"TK={tk_score(ll,k,n):.4f}")

    # ---- 5-fold CV by color
    print()
    print(f"=== {N_FOLDS}-FOLD CV BY COLOR ===")
    idx = np.arange(n)
    RNG.shuffle(idx)
    folds = np.array_split(idx, N_FOLDS)
    cv_ll = {name: [] for name in SCORERS}
    for fi, te_idx in enumerate(folds):
        tr_idx = np.setdiff1d(idx, te_idx)
        tr, te = T2[tr_idx], T2[te_idx]
        for name, fn in SCORERS.items():
            ll, _ = fn(tr, te)
            cv_ll[name].append(ll / len(te))  # per-point
    print(f"  {'topology':>10} {'mean_ll/pt':>12} {'sd':>8}")
    for name in SCORERS:
        m = float(np.mean(cv_ll[name]))
        s = float(np.std(cv_ll[name]))
        print(f"  {name:>10} {m:>12.4f} {s:>8.4f}")

    # ---- Final ranking
    print()
    print("=" * 78)
    print(" RANKING (by BIC, lower = better)")
    print("=" * 78)
    print(f"  {'topology':>10} {'log_lik':>10} {'k':>3} "
          f"{'BIC':>10} {'TK':>9}  rank")
    rows = []
    for name in SCORERS:
        ll, k = in_sample[name]
        rows.append((name, ll, k, bic(ll, k, n), tk_score(ll, k, n)))
    rows.sort(key=lambda r: r[3])  # by BIC ascending
    for rank, (name, ll, k, b, tk) in enumerate(rows, 1):
        print(f"  {name:>10} {ll:>10.2f} {k:>3} {b:>10.2f} {tk:>9.4f}  #{rank}")
    print("=" * 78)
    winner = rows[0][0]
    margin = rows[1][3] - rows[0][3]
    print(f"[WINNER] {winner}  (BIC margin over #2 = {margin:.2f})")

    # ---- save
    np.savez(
        OUT_NPZ,
        T2=T2,
        free_var=var,
        kept_axes=keep,
        in_sample_ll=np.array([in_sample[nm][0] for nm in SCORERS]),
        in_sample_k=np.array([in_sample[nm][1] for nm in SCORERS]),
        bic=np.array([bic(in_sample[nm][0], in_sample[nm][1], n)
                      for nm in SCORERS]),
        tk=np.array([tk_score(in_sample[nm][0], in_sample[nm][1], n)
                     for nm in SCORERS]),
        topology_names=np.array(list(SCORERS.keys())),
        cv_ll_mean=np.array([np.mean(cv_ll[nm]) for nm in SCORERS]),
        cv_ll_sd=np.array([np.std(cv_ll[nm]) for nm in SCORERS]),
        winner=winner,
        bic_margin=margin,
        path_taken=path,
    )
    print(f"[npz] {OUT_NPZ}")

    # ---- append memo
    try:
        snippet = (
            "\n## auto_exp_41: free-block topology sweep (2026-05-23)\n"
            f"Reused HSV-supervised + PPCA-ARD (w_ard=0.01) from auto_exp_40; "
            f"kept axes by descending free_var. gamfit path: {path}.\n\n"
            f"| topology | log_lik | k | BIC | TK | rank |\n"
            f"|---|---|---|---|---|---|\n"
        )
        for rank, (name, ll, k, b, tk) in enumerate(rows, 1):
            snippet += f"| {name} | {ll:.2f} | {k} | {b:.2f} | {tk:.4f} | #{rank} |\n"
        snippet += (
            f"\n**Winner: {winner}** (BIC margin over #2 = {margin:.2f}).\n"
            f"5-fold CV by color (mean log-lik per point):\n"
        )
        for name in SCORERS:
            snippet += (f"- {name}: {np.mean(cv_ll[name]):.4f} "
                        f"(sd {np.std(cv_ll[name]):.4f})\n")
        snippet += f"\nArchive: `runs/auto_exp_41_results.npz`.\n"
        with open(MEMO, "a") as f:
            f.write(snippet)
        print(f"[memo] appended to {MEMO}")
    except Exception as e:
        print(f"[memo] FAILED to append: {e!r}")

    print(f"[runtime] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
