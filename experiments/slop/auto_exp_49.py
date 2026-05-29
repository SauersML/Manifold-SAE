"""auto_exp_49: compose ARD-over-PCs with Circle latent for hue recovery.

Tests auto_exp_47's takeaway directly: Circle alone fails (|rho|=0.041) because
unsupervised reconstruction lands on top-2 PCs while hue lives in PC2+PC4
(oracle |rho|=0.72). ARD over PCs should DISCOVER the right axes so Circle
can exploit the cyclic structure on the chosen plane.

Strategies (all unsupervised; true hue used ONLY for scoring):
  1. Euclidean baseline (auto_exp_47 setup, top-1 PC).
  2. Oracle Circle on PC2+PC4 (auto_exp_47 ceiling).
  3. ARD + Circle composed: per-PC weight alpha_k learned alternately with
     a Circle latent fit on alpha-weighted PCs.
  4. ARD + Euclidean composed: same alpha schedule, Euclidean 1D latent.

gamfit path: auto_exp_47 already confirmed gamfit 0.1.112 has no Circle
wrapper, so we use the same Python emulator.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
)
from auto_exp_47 import (  # type: ignore
    fit_euclidean_1d, fit_circle_1d, _fit_circle_1d_single,
    circular_spearman, circular_mse, best_align_theta_to_hue,
    best_align_linear_to_hue,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_NPZ = ROOT / "runs" / "auto_exp_49_results.npz"

K_PCS = 16
N_OUTER = 25         # ARD outer iterations
N_INNER = 100        # Circle alt-fit inner iterations
RIDGE = 1e-3
ALPHA_EPS = 1e-6


def _circle_recon_loss(Tw, W, theta):
    Phi = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return float(((Tw - Phi @ W) ** 2).sum())


def fit_ard_circle(T0, K_pcs=K_PCS, n_outer=N_OUTER, n_inner=N_INNER,
                   ridge=RIDGE, seed=49, n_restarts=8):
    """ARD over PC indices composed with a Circle latent.

    Model:  alpha_k * T0[:,k]  ~  [cos theta_n, sin theta_n] @ W[:,k] + noise
    Equivalently, treat the latent as a Circle fit on alpha-weighted PCs.
    ARD update (mean-field VB approx for a Gaussian-Gamma prior):
        alpha_k <- d_k / (||W[:, k]||^2 + tr_Sigma_k)
    With Phi^T Phi approx (n/2) I in steady state, posterior var per K-loading
    is ~ sigma2 / (n/2). We use a simple proxy:
        alpha_k <- 1 / (||W[:, k]||^2 / d_k + eps)
    and renormalize sum(alpha)=K so the latent scale stays stable.
    d_k=2 (cos/sin contribute equally to PC k).

    Returns (theta, W, alpha, losses).
    """
    Tc = T0 - T0.mean(0, keepdims=True)
    n, K = Tc.shape
    alpha = np.ones(K)
    rng = np.random.default_rng(seed)
    # init theta from the multi-restart Circle fit on unweighted PCs
    theta, W, _ = fit_circle_1d(T0, n_iter=n_inner, ridge=ridge, seed=seed,
                                n_restarts=n_restarts)
    losses = []
    for it in range(n_outer):
        # weighted features
        Tw = Tc * alpha[None, :]                # (n, K)
        # inner: refine theta + W on weighted features (single warm-start run)
        theta, W, _ = _fit_circle_1d_single(Tw, n_inner, ridge, theta)
        loss = _circle_recon_loss(Tw, W, theta)
        losses.append(loss)
        # ARD update: per-PC loading strength
        d_k = 2.0
        wnorm2 = (W ** 2).sum(axis=0)           # (K,)  ||W[:, k]||^2
        alpha_new = d_k / (wnorm2 + ALPHA_EPS)
        # normalize so sum(alpha) == K (keeps latent scale comparable)
        alpha_new = alpha_new * (K / alpha_new.sum())
        # damping
        alpha = 0.5 * alpha + 0.5 * alpha_new
    return theta, W, alpha, np.asarray(losses)


def fit_ard_euclidean(T0, K_pcs=K_PCS, n_outer=N_OUTER, ridge=RIDGE, seed=49):
    """ARD over PCs composed with a Euclidean 1-D latent (top-1 PC of weighted)."""
    Tc = T0 - T0.mean(0, keepdims=True)
    n, K = Tc.shape
    alpha = np.ones(K)
    losses = []
    t = None
    w = None
    for it in range(n_outer):
        Tw = Tc * alpha[None, :]
        U, S, Vt = np.linalg.svd(Tw, full_matrices=False)
        w = Vt[0]
        t = Tw @ w
        # recon loss (rank-1)
        recon = np.outer(t, w)
        loss = float(((Tw - recon) ** 2).sum())
        losses.append(loss)
        # ARD update: per-PC loading strength (d_k=1 here)
        w2 = w ** 2 + ALPHA_EPS
        alpha_new = 1.0 / w2
        alpha_new = alpha_new * (K / alpha_new.sum())
        alpha = 0.5 * alpha + 0.5 * alpha_new
    return t, w, alpha, np.asarray(losses)


def oracle_circle(T0, hue, top_n=8):
    Tc = T0 - T0.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    best = None
    for i in range(top_n):
        for j in range(i + 1, top_n):
            plane = Tc @ Vt[[i, j]].T
            th = np.arctan2(plane[:, 1], plane[:, 0])
            cc = circular_spearman(th, hue)
            if best is None or abs(cc) > abs(best[0]):
                best = (cc, i, j, th)
    return best


def kfold_circle_with_alpha(T0, hue, alpha, k=5, seed=49):
    rng = np.random.default_rng(seed)
    n = T0.shape[0]
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int); fold_ids[perm] = np.arange(n) % k
    mses = []
    for i in range(k):
        te = np.where(fold_ids == i)[0]
        tr = np.where(fold_ids != i)[0]
        mu_tr = T0[tr].mean(0, keepdims=True)
        Tc_tr = (T0[tr] - mu_tr) * alpha[None, :]
        theta_tr, W, _ = fit_circle_1d(Tc_tr + Tc_tr.mean(0, keepdims=True),
                                       n_iter=N_INNER, ridge=RIDGE,
                                       seed=seed, n_restarts=4)
        # refit on weighted-centered:
        theta_tr, W, _ = _fit_circle_1d_single(Tc_tr, N_INNER, RIDGE, theta_tr)
        Tc_te = (T0[te] - mu_tr) * alpha[None, :]
        proj = Tc_te @ W.T
        theta_te = np.arctan2(proj[:, 1], proj[:, 0])
        _, s, phi, _ = best_align_theta_to_hue(theta_tr, hue[tr])
        pred_te_01 = ((s * theta_te + phi) % (2 * np.pi)) / (2 * np.pi)
        mses.append(circular_mse(hue[te], pred_te_01))
    return float(np.mean(mses)), float(np.std(mses))


def kfold_euclidean_with_alpha(T0, hue, alpha, k=5, seed=49):
    rng = np.random.default_rng(seed)
    n = T0.shape[0]
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int); fold_ids[perm] = np.arange(n) % k
    mses = []
    for i in range(k):
        te = np.where(fold_ids == i)[0]
        tr = np.where(fold_ids != i)[0]
        mu_tr = T0[tr].mean(0, keepdims=True)
        Tc_tr = (T0[tr] - mu_tr) * alpha[None, :]
        U, S, Vt = np.linalg.svd(Tc_tr, full_matrices=False)
        w = Vt[0]
        t_tr = Tc_tr @ w
        _, s, _, _ = best_align_linear_to_hue(t_tr, hue[tr])
        t_te = (T0[te] - mu_tr) * alpha[None, :] @ w
        st_tr = s * t_tr; st_te = s * t_te
        sort_tr = np.sort(st_tr)
        ranks_te = np.searchsorted(sort_tr, st_te) / max(len(sort_tr) - 1, 1)
        ranks_tr = np.argsort(np.argsort(st_tr)) / max(len(st_tr) - 1, 1)
        phi = np.angle(np.mean(np.exp(1j * 2 * np.pi * (hue[tr] - ranks_tr))))
        pred = (ranks_te + phi / (2 * np.pi)) % 1.0
        mses.append(circular_mse(hue[te], pred))
    return float(np.mean(mses)), float(np.std(mses))


def check_gamfit():
    try:
        import gamfit
        return getattr(gamfit, "__version__", "unknown")
    except Exception as exc:
        return f"unavailable:{exc!r}"


def main():
    t0 = time.time()
    print("[auto_exp_49] ARD-over-PCs composed with Circle latent")
    print(f"[gamfit] version={check_gamfit()}  path=fallback_python_emulator")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")
    names, rgb = load_xkcd_rgb(n)
    hue = hsv_from_rgb(rgb)[:, 0]

    results = {}

    # --- Setup 1: Euclidean baseline ---
    t_eu, w_eu = fit_euclidean_1d(T0)
    rho_eu, s_eu, mse_eu_full, pred_eu = best_align_linear_to_hue(t_eu, hue)
    rho_eu_circ = circular_spearman(2 * np.pi * pred_eu, hue)
    print(f"[1 Eucl baseline]    |rho|={abs(rho_eu):.3f}  circ-rho={rho_eu_circ:+.3f}  "
          f"full-MSE={mse_eu_full:.4f}")
    results["eu"] = dict(t=t_eu, alpha=None, rho=abs(rho_eu),
                         circ_rho=rho_eu_circ, mse_full=mse_eu_full)

    # --- Setup 2: Oracle Circle on PC2+PC4 (auto-discovered best pair) ---
    cc_o, oi, oj, theta_o = oracle_circle(T0, hue, top_n=8)
    mse_o_full, s_o, phi_o, pred_o = best_align_theta_to_hue(theta_o, hue)
    rho_o, _ = spearmanr(pred_o, hue)
    print(f"[2 Oracle PC{oi},{oj}]   |rho|={abs(rho_o):.3f}  circ-rho={cc_o:+.3f}  "
          f"full-MSE={mse_o_full:.4f}")
    results["oracle"] = dict(theta=theta_o, pc_i=oi, pc_j=oj,
                             rho=abs(rho_o), circ_rho=cc_o,
                             mse_full=mse_o_full)

    # --- Setup 3: ARD + Circle composed ---
    print("[3 ARD+Circle]  fitting...")
    theta_ac, W_ac, alpha_ac, losses_ac = fit_ard_circle(T0)
    mse_ac_full, s_ac, phi_ac, pred_ac = best_align_theta_to_hue(theta_ac, hue)
    rho_ac, _ = spearmanr(pred_ac, hue)
    rho_ac_circ = circular_spearman(s_ac * theta_ac + phi_ac, hue)
    top3_ac = np.argsort(-alpha_ac)[:3]
    print(f"[3 ARD+Circle]   |rho|={abs(rho_ac):.3f}  circ-rho={rho_ac_circ:+.3f}  "
          f"full-MSE={mse_ac_full:.4f}  top3 alpha={top3_ac.tolist()} "
          f"({alpha_ac[top3_ac].round(2).tolist()})")
    mse_ac_cv, sd_ac_cv = kfold_circle_with_alpha(T0, hue, alpha_ac)
    print(f"                5-fold circ-MSE={mse_ac_cv:.4f} +/- {sd_ac_cv:.4f}")
    results["ard_circle"] = dict(theta=theta_ac, alpha=alpha_ac,
                                 rho=abs(rho_ac), circ_rho=rho_ac_circ,
                                 mse_full=mse_ac_full,
                                 mse_cv=mse_ac_cv, top3=top3_ac)

    # --- Setup 4: ARD + Euclidean composed ---
    print("[4 ARD+Eucl]    fitting...")
    t_ae, w_ae, alpha_ae, losses_ae = fit_ard_euclidean(T0)
    rho_ae, s_ae, mse_ae_full, pred_ae = best_align_linear_to_hue(t_ae, hue)
    rho_ae_circ = circular_spearman(2 * np.pi * pred_ae, hue)
    top3_ae = np.argsort(-alpha_ae)[:3]
    print(f"[4 ARD+Eucl]     |rho|={abs(rho_ae):.3f}  circ-rho={rho_ae_circ:+.3f}  "
          f"full-MSE={mse_ae_full:.4f}  top3 alpha={top3_ae.tolist()} "
          f"({alpha_ae[top3_ae].round(2).tolist()})")
    mse_ae_cv, sd_ae_cv = kfold_euclidean_with_alpha(T0, hue, alpha_ae)
    print(f"                5-fold circ-MSE={mse_ae_cv:.4f} +/- {sd_ae_cv:.4f}")
    results["ard_eu"] = dict(t=t_ae, alpha=alpha_ae,
                             rho=abs(rho_ae), circ_rho=rho_ae_circ,
                             mse_full=mse_ae_full,
                             mse_cv=mse_ae_cv, top3=top3_ae)

    # --- Table ---
    print()
    print("setup                       | |rho|  | circ-rho | full-MSE | 5-fold MSE      | top-3 alpha PCs")
    print("----------------------------+--------+----------+----------+-----------------+----------------")
    print(f"1 Euclidean baseline        | {abs(rho_eu):.3f}  | {rho_eu_circ:+.3f}   | {mse_eu_full:.4f}   | (n/a)           | (n/a)")
    print(f"2 Oracle Circle PC{oi},{oj}      | {abs(rho_o):.3f}  | {cc_o:+.3f}   | {mse_o_full:.4f}   | (n/a)           | (n/a)")
    print(f"3 ARD+Circle composed       | {abs(rho_ac):.3f}  | {rho_ac_circ:+.3f}   | {mse_ac_full:.4f}   | {mse_ac_cv:.4f} +/- {sd_ac_cv:.4f} | {top3_ac.tolist()}")
    print(f"4 ARD+Euclidean composed    | {abs(rho_ae):.3f}  | {rho_ae_circ:+.3f}   | {mse_ae_full:.4f}   | {mse_ae_cv:.4f} +/- {sd_ae_cv:.4f} | {top3_ae.tolist()}")

    # --- Verdict ---
    primary = abs(rho_ac) > 0.3
    pc24 = set(top3_ac.tolist()) >= {1, 3}  # PC2, PC4 -> indices 1, 3
    print()
    print(f"[verdict] PRIMARY  ARD+Circle |rho|>0.3 ?  {primary}  "
          f"(|rho|={abs(rho_ac):.3f}, baseline unsup Circle was 0.041)")
    print(f"[verdict] SECONDARY top-3 alpha contains PC2 and PC4 ?  {pc24}  "
          f"(top3={top3_ac.tolist()})")
    if primary and pc24:
        verdict = "BOTH_YES: composition works -- ARD discovers, Circle exploits"
    elif primary:
        verdict = "PRIMARY_ONLY: Circle topology helps without explicit PC selection"
    elif pc24:
        verdict = "SECONDARY_ONLY: ARD finds plane but Circle latent fails on it"
    else:
        verdict = "BOTH_NO: unsupervised ARD-over-PCs cannot find sub-dominant cyclic plane"
    print(f"[verdict] {verdict}")

    # --- Save ---
    np.savez(
        OUT_NPZ,
        hue_true=hue,
        t_euclidean=t_eu,
        theta_oracle=theta_o, oracle_pc_i=oi, oracle_pc_j=oj,
        theta_ard_circle=theta_ac, alpha_ard_circle=alpha_ac,
        W_ard_circle=W_ac, losses_ard_circle=losses_ac,
        t_ard_euclidean=t_ae, alpha_ard_euclidean=alpha_ae,
        losses_ard_euclidean=losses_ae,
        rho_eu=abs(rho_eu), rho_oracle=abs(rho_o),
        rho_ard_circle=abs(rho_ac), rho_ard_eu=abs(rho_ae),
        mse_ac_cv=mse_ac_cv, mse_ae_cv=mse_ae_cv,
        verdict=verdict,
    )
    print(f"[npz] saved {OUT_NPZ}")
    print(f"[runtime] {time.time() - t0:.1f}s")
    return verdict


if __name__ == "__main__":
    main()
