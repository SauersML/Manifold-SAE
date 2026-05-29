"""auto_exp_47: Riemannian Circle (S^1) latent vs Euclidean for hue recovery.

Builds on auto_79 (which found the unsupervised U_3d t3 axis has Spearman
rho=0.41 with true hue in a flat-Euclidean fit). Hypothesis: putting the
hue latent on a circle (S^1) accommodates wrap and substantially improves
recovery. If Circle <= Euclidean, the L40 hue-shaped feature lives on a
line, not a ring -- a generalizable lesson for cyclic concept manifolds.

gamfit path:
  - gamfit 0.1.112 has NO LatentCoord(manifold="circle") wrapper exposed
    (checked at runtime: dir(gamfit) contains no Circle/Manifold/Retract
    entries). So Setup B uses a Python fallback: alternating
    von-Mises-MAP angle estimation per row + ridge regression of
    sin/cos(theta) onto PCA features (a 1-D S^1 GP-LVM by hand).

Setup A (Euclidean): top-1 PC of T0_K16 (top PC has largest data variance,
identical to a 1-D LatentCoord(manifold="euclidean") init).

Setup B (Circle): jointly fit (theta_n in S^1, W in R^{K x 2}) such that
T0_n ~ W @ [cos theta_n; sin theta_n] + noise, by alternating:
  - given W: theta_n = atan2(W2 . t_n, W1 . t_n)   (von-Mises MAP, k -> inf)
  - given theta: W = ridge-LS of T0 on [cos theta, sin theta]
Init theta from the top-PC angle of (T0 projected to its top-2 PC plane).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # type: ignore
    N_TEMPLATES, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_NPZ = ROOT / "runs" / "auto_exp_47_circle_vs_euclidean.npz"
OUT_PNG = ROOT / "runs" / "auto_exp_47_circle_vs_euclidean.png"

K_PCS = 16
N_ITER = 200
RIDGE = 1e-3


def check_gamfit_circle():
    try:
        import gamfit
        ver = getattr(gamfit, "__version__", "unknown")
        has_circle = any(
            "circle" in n.lower() or "manifold" in n.lower()
            or "retract" in n.lower()
            for n in dir(gamfit)
        )
        if has_circle:
            return ver, "gamfit_circle_wrapper"
        return ver, "fallback_python_vonmises_alt"
    except Exception as exc:
        return f"unavailable:{exc!r}", "fallback_no_gamfit"


def fit_euclidean_1d(T0):
    """Top-1 PC of T0. Returns t (n,) and recon weight w (K,)."""
    Tc = T0 - T0.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    w = Vt[0]                  # (K,)
    t = Tc @ w                 # (n,)
    return t, w


def _fit_circle_1d_single(Tc, n_iter, ridge, theta_init):
    """One alternating run from a given theta init. Returns (theta, W, losses)."""
    theta = theta_init.copy()
    losses = []
    W = None
    for it in range(n_iter):
        Phi = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        A = Phi.T @ Phi + ridge * np.eye(2)
        W = np.linalg.solve(A, Phi.T @ Tc)  # (2, K)
        proj = Tc @ W.T                      # (n, 2)
        theta_new = np.arctan2(proj[:, 1], proj[:, 0])
        resid = Tc - Phi @ W
        losses.append(float((resid ** 2).sum()))
        if np.max(np.abs(np.angle(np.exp(1j * (theta_new - theta))))) < 1e-6:
            theta = theta_new
            break
        theta = theta_new
    return theta, W, np.asarray(losses)


def fit_circle_1d(T0, n_iter=N_ITER, ridge=RIDGE, seed=47, n_restarts=12):
    """Multi-restart alternating fit of theta in S^1 and W in R^{K x 2}.

    Model: T0_centered_n approx [cos theta_n, sin theta_n] @ W.T.
    Restarts: (a) top-2 PC plane angle, (b) every (PC_i, PC_j) pair for
    i<j in top-6, (c) random theta restarts. Pick lowest recon loss.
    """
    rng = np.random.default_rng(seed)
    Tc = T0 - T0.mean(0, keepdims=True)
    n, K = Tc.shape
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    inits = []
    # PC pair inits in top-6 (15 pairs)
    for i in range(6):
        for j in range(i + 1, 6):
            plane = Tc @ Vt[[i, j]].T
            inits.append(np.arctan2(plane[:, 1], plane[:, 0]))
    # random restarts
    for _ in range(n_restarts):
        inits.append(rng.uniform(-np.pi, np.pi, size=n))
    best = None
    for theta0 in inits:
        theta, W, losses = _fit_circle_1d_single(Tc, n_iter, ridge, theta0)
        final_loss = losses[-1]
        if best is None or final_loss < best[2][-1]:
            best = (theta, W, losses)
    return best


def circular_distance(a, b):
    """Smallest signed angular distance in radians. a,b in [0,1) (hue units)."""
    d = (a - b) % 1.0
    d = np.where(d > 0.5, d - 1.0, d)
    return d  # in [-0.5, 0.5)


def circular_mse(true_hue_01, pred_hue_01):
    """Mean squared circular distance, units = hue^2 in [0,1) space."""
    d = circular_distance(true_hue_01, pred_hue_01)
    return float((d ** 2).mean())


def circular_spearman(theta_pred_rad, hue_true_01):
    """Circular-vs-linear association: max over rotations of standard Spearman.

    Treat hue_true as angle (0..2pi). Compute Spearman between
    cos(theta - phi) and cos(hue - phi) -- but the cleanest scalar is
    Fisher's circular correlation r_c between two circular variables.
    Implement the Jammalamadaka-Sarma rho_c (symmetric, in [-1, 1]).
    """
    a = theta_pred_rad
    b = 2 * np.pi * hue_true_01
    a_bar = np.angle(np.mean(np.exp(1j * a)))
    b_bar = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - a_bar) * np.sin(b - b_bar))
    den = np.sqrt(np.sum(np.sin(a - a_bar) ** 2) * np.sum(np.sin(b - b_bar) ** 2))
    return float(num / den) if den > 0 else float("nan")


def best_align_theta_to_hue(theta_rad, hue_01):
    """Find (sign, phase) that align theta to true hue, return predicted hue in [0,1)."""
    hue_rad = 2 * np.pi * hue_01
    best = None
    for s in (+1, -1):
        a = s * theta_rad
        # MLE phase shift: phi = circ_mean(hue - a)
        phi = np.angle(np.mean(np.exp(1j * (hue_rad - a))))
        pred_rad = (a + phi) % (2 * np.pi)
        pred_01 = pred_rad / (2 * np.pi)
        mse = circular_mse(hue_01, pred_01)
        if best is None or mse < best[0]:
            best = (mse, s, phi, pred_01)
    return best  # (mse, sign, phase, pred_hue_01)


def best_align_linear_to_hue(t, hue_01):
    """For Euclidean t: find best 1-D mapping t -> hue_01 by rank correlation.

    Use Spearman with sign flip; predicted hue is the empirical rank of (sign*t).
    """
    rho_pos, _ = spearmanr(t, hue_01)
    rho_neg, _ = spearmanr(-t, hue_01)
    if abs(rho_pos) >= abs(rho_neg):
        s = +1
        rho = rho_pos
    else:
        s = -1
        rho = rho_neg
    st = s * t
    # convert to [0,1) via rank
    ranks = np.argsort(np.argsort(st)) / max(len(st) - 1, 1)
    # circular MSE makes no sense for a linear axis: also compute it
    # against the best phase by treating ranks as hue.
    phi = np.angle(np.mean(np.exp(1j * 2 * np.pi * (hue_01 - ranks))))
    pred_01 = (ranks + phi / (2 * np.pi)) % 1.0
    mse = circular_mse(hue_01, pred_01)
    return rho, s, mse, pred_01


def kfold_circular_mse_circle(T0, hue_01, k=5, seed=47):
    """5-fold CV: fit Circle on train, predict theta on test, score circular MSE."""
    rng = np.random.default_rng(seed)
    n = T0.shape[0]
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    fold_ids[perm] = np.arange(n) % k
    mses = []
    for i in range(k):
        te = np.where(fold_ids == i)[0]
        tr = np.where(fold_ids != i)[0]
        mu_tr = T0[tr].mean(0, keepdims=True)
        Tc_tr = T0[tr] - mu_tr
        theta_tr, W, _ = fit_circle_1d(T0[tr])
        # predict theta on held-out via the SAME W
        Tc_te = T0[te] - mu_tr
        proj = Tc_te @ W.T
        theta_te = np.arctan2(proj[:, 1], proj[:, 0])
        # align (sign+phase) on TRAIN only
        _, s, phi, _ = best_align_theta_to_hue(theta_tr, hue_01[tr])
        pred_te_rad = (s * theta_te + phi) % (2 * np.pi)
        pred_te_01 = pred_te_rad / (2 * np.pi)
        mses.append(circular_mse(hue_01[te], pred_te_01))
    return float(np.mean(mses)), float(np.std(mses))


def kfold_circular_mse_euclidean(T0, hue_01, k=5, seed=47):
    """5-fold CV: top-1 PC on train, project test on same direction."""
    rng = np.random.default_rng(seed)
    n = T0.shape[0]
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    fold_ids[perm] = np.arange(n) % k
    mses = []
    for i in range(k):
        te = np.where(fold_ids == i)[0]
        tr = np.where(fold_ids != i)[0]
        mu_tr = T0[tr].mean(0, keepdims=True)
        Tc_tr = T0[tr] - mu_tr
        U, S, Vt = np.linalg.svd(Tc_tr, full_matrices=False)
        w = Vt[0]
        t_tr = Tc_tr @ w
        # align on TRAIN
        _, s, _, _ = best_align_linear_to_hue(t_tr, hue_01[tr])
        t_te = (T0[te] - mu_tr) @ w
        st_te = s * t_te
        # to hue: empirical rank within train+test combined for monotone mapping
        st_tr = s * t_tr
        # quantile-map test through train ranks
        sort_tr = np.sort(st_tr)
        ranks_te = np.searchsorted(sort_tr, st_te) / max(len(sort_tr) - 1, 1)
        # apply best phase from train
        ranks_tr = np.argsort(np.argsort(st_tr)) / max(len(st_tr) - 1, 1)
        phi = np.angle(np.mean(np.exp(1j * 2 * np.pi * (hue_01[tr] - ranks_tr))))
        pred_te_01 = (ranks_te + phi / (2 * np.pi)) % 1.0
        mses.append(circular_mse(hue_01[te], pred_te_01))
    return float(np.mean(mses)), float(np.std(mses))


def main():
    t_start = time.time()
    print("[auto_exp_47] Riemannian Circle vs Euclidean for hue recovery")

    ver, path = check_gamfit_circle()
    print(f"[gamfit] version={ver} path={path}")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n)
    hsv = hsv_from_rgb(rgb)
    hue = hsv[:, 0]   # in [0, 1)
    print(f"[hue] range=[{hue.min():.3f},{hue.max():.3f}]  n={n}")

    # ----- Setup A: Euclidean baseline -----
    t_eu, w_eu = fit_euclidean_1d(T0)
    rho_eu_linear, sign_eu, mse_eu_full, pred_eu_full = best_align_linear_to_hue(t_eu, hue)
    rho_eu_circ = circular_spearman(2 * np.pi * pred_eu_full, hue)
    mse_eu_cv, sd_eu_cv = kfold_circular_mse_euclidean(T0, hue, k=5)
    print(f"[Eucl] |Spearman rho|={abs(rho_eu_linear):.3f}  "
          f"circ-rho={rho_eu_circ:+.3f}  "
          f"full-data circ-MSE={mse_eu_full:.4f}  "
          f"5-fold circ-MSE={mse_eu_cv:.4f} +/- {sd_eu_cv:.4f}")

    # ----- Setup B: Riemannian Circle -----
    theta_ci, W_ci, losses_ci = fit_circle_1d(T0)
    print(f"[Circle] iters={len(losses_ci)} final-recon={losses_ci[-1]:.2f}")
    mse_ci_full, sign_ci, phi_ci, pred_ci_full = best_align_theta_to_hue(theta_ci, hue)
    rho_ci_circ = circular_spearman(sign_ci * theta_ci + phi_ci, hue)
    # linear Spearman on the aligned predicted hue
    rho_ci_linear, _ = spearmanr(pred_ci_full, hue)
    mse_ci_cv, sd_ci_cv = kfold_circular_mse_circle(T0, hue, k=5)
    print(f"[Circle] |Spearman rho|={abs(rho_ci_linear):.3f}  "
          f"circ-rho={rho_ci_circ:+.3f}  "
          f"full-data circ-MSE={mse_ci_full:.4f}  "
          f"5-fold circ-MSE={mse_ci_cv:.4f} +/- {sd_ci_cv:.4f}")

    # ----- Setup C: SUPERVISED-plane Circle (oracle PC pair) -----
    # Sanity-check whether a hue-shaped circle exists ANYWHERE in PCA(K=16).
    # Sweep all PC pairs (i,j) in top-8, pick the angle with highest absolute
    # circular correlation to true hue, then evaluate that fixed 2-plane.
    Tc_full = T0 - T0.mean(0, keepdims=True)
    U, S, Vt_full = np.linalg.svd(Tc_full, full_matrices=False)
    best_pair = None  # (|circ_corr|, i, j, theta)
    for i in range(8):
        for j in range(i + 1, 8):
            plane = Tc_full @ Vt_full[[i, j]].T
            th = np.arctan2(plane[:, 1], plane[:, 0])
            cc = circular_spearman(th, hue)
            if best_pair is None or abs(cc) > abs(best_pair[0]):
                best_pair = (cc, i, j, th)
    cc_oracle, oi, oj, theta_oracle = best_pair
    mse_oracle_full, s_o, phi_o, pred_oracle = best_align_theta_to_hue(theta_oracle, hue)
    rho_oracle_linear, _ = spearmanr(pred_oracle, hue)
    print(f"[Circle-oracle] best PC pair = ({oi},{oj})  circ-corr={cc_oracle:+.3f}  "
          f"|Spearman rho|={abs(rho_oracle_linear):.3f}  full-data circ-MSE={mse_oracle_full:.4f}")

    # ----- Table -----
    print()
    print("setup                | Spearman |rho|  | circ-rho   | 5-fold circ-MSE | full circ-MSE")
    print("---------------------+------------------+------------+-----------------+---------------")
    print(f"Euclidean (top-1 PC) |   {abs(rho_eu_linear):.3f}          |  {rho_eu_circ:+.3f}    |  {mse_eu_cv:.4f} +/- {sd_eu_cv:.4f} |  {mse_eu_full:.4f}")
    print(f"Circle  (unsup recon)|   {abs(rho_ci_linear):.3f}          |  {rho_ci_circ:+.3f}    |  {mse_ci_cv:.4f} +/- {sd_ci_cv:.4f} |  {mse_ci_full:.4f}")
    print(f"Circle  (oracle PC{oi},{oj})|   {abs(rho_oracle_linear):.3f}          |  {cc_oracle:+.3f}    |  (no CV)         |  {mse_oracle_full:.4f}")

    # ----- Hypothesis verdict -----
    hyp_circle_wins = (abs(rho_ci_linear) >= 0.65) and (mse_ci_cv < mse_eu_cv)
    print(f"\n[hypothesis] Circle |rho|>=0.65 AND CV-MSE < Euclidean: {hyp_circle_wins}")

    # ----- Plot -----
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    rgb_clip = np.clip(rgb, 0, 1)
    ax = axs[0]
    ax.scatter(hue, pred_eu_full, c=rgb_clip, s=15, edgecolors="k", linewidths=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5)
    ax.set_xlabel("true hue (HSV[0])")
    ax.set_ylabel("recovered (Euclidean, aligned)")
    ax.set_title(f"Euclidean baseline: |rho|={abs(rho_eu_linear):.3f}  "
                 f"5-fold circ-MSE={mse_eu_cv:.4f}")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axs[1]
    ax.scatter(hue, pred_ci_full, c=rgb_clip, s=15, edgecolors="k", linewidths=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5)
    # show the wrap by drawing the two off-diagonal y=x+/-1 lines
    ax.plot([0, 1], [1, 2], "k:", lw=0.4, alpha=0.4)
    ax.plot([0, 1], [-1, 0], "k:", lw=0.4, alpha=0.4)
    ax.set_xlabel("true hue (HSV[0])")
    ax.set_ylabel("recovered (Circle, sign+phase aligned)")
    ax.set_title(f"Riemannian Circle (S^1): |rho|={abs(rho_ci_linear):.3f}  "
                 f"5-fold circ-MSE={mse_ci_cv:.4f}")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle(
        f"auto_exp_47: Riemannian Circle vs Euclidean hue recovery from cogito-L40 PCA(K=16)\n"
        f"gamfit path = {path} | Circle wins hypothesis = {hyp_circle_wins}",
        fontsize=11)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    np.savez(
        OUT_NPZ,
        hue_true=hue,
        t_euclidean=t_eu,
        pred_hue_euclidean=pred_eu_full,
        theta_circle=theta_ci,
        pred_hue_circle=pred_ci_full,
        theta_circle_oracle=theta_oracle,
        pred_hue_circle_oracle=pred_oracle,
        oracle_pc_i=oi, oracle_pc_j=oj,
        rho_oracle_linear=rho_oracle_linear,
        cc_oracle=cc_oracle,
        mse_oracle_full=mse_oracle_full,
        circle_W=W_ci,
        circle_losses=losses_ci,
        rho_eu_linear=rho_eu_linear,
        rho_eu_circ=rho_eu_circ,
        rho_ci_linear=rho_ci_linear,
        rho_ci_circ=rho_ci_circ,
        mse_eu_full=mse_eu_full,
        mse_ci_full=mse_ci_full,
        mse_eu_cv=mse_eu_cv,
        mse_ci_cv=mse_ci_cv,
        sd_eu_cv=sd_eu_cv,
        sd_ci_cv=sd_ci_cv,
        gamfit_version=ver,
        path_taken=path,
        circle_wins=hyp_circle_wins,
    )
    print(f"[npz] saved {OUT_NPZ}")
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
