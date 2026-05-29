"""auto_exp_53: latent-dimension sweep d ∈ {2,3,4,5,6} for HSV supervision on cogito-L40.

Question: was auto_exp_38's d=3 the right choice, or just a "matches HSV channel count"
default? Sweep d, fit HSV-supervised projection W ∈ R^{K×d}, score:
  - in-sample R²(h), R²(s), R²(v), mean
  - 5-fold CV R² (grouped by color — no leakage since centroids are per-color)
  - BIC = -2·log_lik + n_params·log(n_obs)
  - sum of explained-variance fractions of W's components > 5% threshold (active axes)

Pipeline: identical PCA(K=16) centroids + per-feature standardization as auto_exp_38.
Fit reuses fit_aux_supervised_hsv (weighted-LS + ARD coordinate-descent), but with
variable d_aux. For d > 3, HSV is the 3-D target zero-padded to d → that's wrong; the
right thing is to fit a (K, d)-dim subspace whose first 3 cols predict HSV best — i.e.
extract HSV via Ridge into d cols simultaneously is ill-posed (rank-3 target).

Cleaner formulation: fit W ∈ R^{K×d} as the d-dim subspace that minimizes the
HSV-prediction loss with d free latent axes mapped through a learned head A ∈ R^{d×3}:
    HSV ≈ Tc @ W @ A
This is the canonical "low-rank ridge" fit; equivalently, top-d reduced-rank-regression
(RRR) of HSV on Tc. For d ≥ 3 it saturates to full-rank ridge of HSV-on-Tc; for d < 3
it's the rank-d best approximation. We use this for ALL d so the comparison is honest.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (
    X_PATH, N_TEMPLATES, K_PCS, AUX_LABELS_HSV,
    per_color_stats_mmap, load_xkcd_rgb, hsv_from_rgb,
)

ROOT = Path("/Users/user/Manifold-SAE")
OUT_NPZ = ROOT / "runs" / "auto_exp_53_d_sweep.npz"
OUT_PNG = ROOT / "runs" / "auto_exp_53_d_sweep.png"

D_LIST = [2, 3, 4, 5, 6]
RIDGE_LAM = 1.0      # match auto_exp_38 effective regularization scale
N_FOLDS = 5
ACTIVE_FRAC = 0.05   # 5%-variance threshold for "meaningful axis"
SEED = 53


def fit_rrr(Tc: np.ndarray, Y: np.ndarray, d: int, lam: float = RIDGE_LAM):
    """Reduced-rank ridge regression: Y ≈ Tc @ W @ A, W ∈ R^{K×d}, A ∈ R^{d×3}.

    Closed form: full-rank ridge B = (Tc^T Tc + lam I)^{-1} Tc^T Y ∈ R^{K×3}.
    Then SVD of Y_hat_full = Tc @ B; top-d SVD gives the rank-d projector.
    Equivalently: B = U_d S_d V_d^T → W = (Tc^T Tc + lam I)^{-1} Tc^T (U_d S_d), A = V_d^T
    A simpler practical recipe: do full ridge, then SVD the fitted matrix B itself with
    weighting by Tc covariance; for stability use SVD of fitted values Y_hat = Tc B.
    Returns dict with W (K,d), A (d,3), pred_in (n,3).
    """
    n, K = Tc.shape
    A_full = Tc.T @ Tc + lam * np.eye(K)
    B = np.linalg.solve(A_full, Tc.T @ Y)         # (K, 3)
    Y_hat_full = Tc @ B                            # (n, 3)
    # Rank-d SVD of fitted values
    U, S, Vt = np.linalg.svd(Y_hat_full, full_matrices=False)
    d_use = min(d, S.shape[0], 3)
    # Project Y_hat_full onto top-d subspace of its column space
    # latent T = U_d * S_d  ∈ (n, d_use); head A = Vt[:d_use]  ∈ (d_use, 3)
    T_d = U[:, :d_use] * S[:d_use]
    A_head = Vt[:d_use]
    # W such that Tc @ W ≈ T_d : W = (Tc^T Tc + lam I)^{-1} Tc^T T_d
    W = np.linalg.solve(A_full, Tc.T @ T_d)        # (K, d_use)
    # If d > rank(Y), pad with zeros so W is (K, d)
    if d_use < d:
        W = np.concatenate([W, np.zeros((K, d - d_use))], axis=1)
        A_head = np.concatenate([A_head, np.zeros((d - d_use, 3))], axis=0)
        T_d = np.concatenate([T_d, np.zeros((n, d - d_use))], axis=1)
    pred = T_d @ A_head                            # (n, 3) in centered Y space
    return {"W": W, "A": A_head, "T": T_d, "pred": pred, "B": B, "sing": S}


def r2_per_channel(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-channel R² assuming Y, pred are already centered (or both raw — uses sum of
    squares relative to Y mean)."""
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0).clip(min=1e-12)
    return 1.0 - ss_res / ss_tot


def cv_r2(Tc: np.ndarray, Y: np.ndarray, d: int, n_folds: int = N_FOLDS,
          seed: int = SEED) -> np.ndarray:
    """5-fold CV per channel. Group is just per-row (centroids already aggregated)."""
    rng = np.random.default_rng(seed)
    n = Tc.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    preds = np.zeros_like(Y)
    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[k] for k in range(n_folds) if k != f])
        # Center on train only
        mu_T = Tc[tr].mean(0, keepdims=True)
        mu_Y = Y[tr].mean(0, keepdims=True)
        fit = fit_rrr(Tc[tr] - mu_T, Y[tr] - mu_Y, d)
        # Apply to test: W maps Tc → T_d, A_head maps T_d → Y_centered
        T_te = (Tc[te] - mu_T) @ fit["W"]
        pred_te = T_te @ fit["A"] + mu_Y
        preds[te] = pred_te
    return r2_per_channel(Y, preds)


def gaussian_loglik(Y: np.ndarray, pred: np.ndarray) -> float:
    """Per-channel Gaussian log-likelihood at MLE sigma (sum across channels and rows)."""
    n = Y.shape[0]
    resid = Y - pred
    ll = 0.0
    for j in range(Y.shape[1]):
        s2 = max(float((resid[:, j] ** 2).mean()), 1e-12)
        ll += -0.5 * n * (np.log(2 * np.pi * s2) + 1.0)
    return ll


def active_axes(W: np.ndarray, frac: float = ACTIVE_FRAC) -> tuple[int, np.ndarray]:
    """Count columns of W whose column-norm² contributes >= frac of the total."""
    col_var = (W ** 2).sum(axis=0)
    tot = col_var.sum()
    if tot <= 0:
        return 0, col_var
    shares = col_var / tot
    return int((shares >= frac).sum()), shares


def main():
    t_start = time.time()
    print("[auto_exp_53] latent-d sweep for HSV supervision on cogito-L40")
    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c, K = T0.shape
    print(f"[centroids] T0={T0.shape}")
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    print(f"[aux] hsv={hsv.shape}")

    Tc = T0 - T0.mean(0, keepdims=True)
    Yc = hsv - hsv.mean(0, keepdims=True)

    rows = []
    archive = {}
    for d in D_LIST:
        fit = fit_rrr(Tc, Yc, d)
        r2_in = r2_per_channel(hsv, fit["pred"] + hsv.mean(0, keepdims=True))
        r2_cv = cv_r2(Tc, hsv, d)
        ll = gaussian_loglik(hsv, fit["pred"] + hsv.mean(0, keepdims=True))
        # n_params = K*d (W) + d*3 (A_head) ; baseline mu accounted for via centering
        n_params = K * d + d * 3
        n_obs = n_c * 3  # treating each (color, channel) as an obs
        bic = -2.0 * ll + n_params * np.log(n_obs)
        n_active, shares = active_axes(fit["W"])
        rows.append({
            "d": d,
            "r2_in": r2_in,
            "r2_cv": r2_cv,
            "r2_in_mean": float(r2_in.mean()),
            "r2_cv_mean": float(r2_cv.mean()),
            "log_lik": float(ll),
            "n_params": int(n_params),
            "bic": float(bic),
            "n_active": int(n_active),
            "var_shares": shares,
            "sing": fit["sing"],
        })
        archive[f"d{d}_W"] = fit["W"]
        archive[f"d{d}_A"] = fit["A"]
        archive[f"d{d}_T"] = fit["T"]
        archive[f"d{d}_r2_in"] = r2_in
        archive[f"d{d}_r2_cv"] = r2_cv
        archive[f"d{d}_var_shares"] = shares
        print(f"[d={d}] r2_in h={r2_in[0]:.3f} s={r2_in[1]:.3f} v={r2_in[2]:.3f} "
              f"| r2_cv h={r2_cv[0]:.3f} s={r2_cv[1]:.3f} v={r2_cv[2]:.3f} "
              f"| LL={ll:.1f} BIC={bic:.1f} active={n_active} shares={np.round(shares,3)}")

    # --- Stdout table
    print()
    print("=" * 92)
    print(" auto_exp_53: latent-d sweep summary")
    print("=" * 92)
    hdr = f"{'d':>3} {'R2_h_in':>9} {'R2_s_in':>9} {'R2_v_in':>9} {'R2_h_CV':>9} {'mean_CV':>9} {'BIC':>11} {'active':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['d']:>3} {r['r2_in'][0]:>9.3f} {r['r2_in'][1]:>9.3f} {r['r2_in'][2]:>9.3f} "
              f"{r['r2_cv'][0]:>9.3f} {r['r2_cv_mean']:>9.3f} {r['bic']:>11.1f} {r['n_active']:>7}")
    print("=" * 92)

    # --- Verdict
    cv_h = np.array([r['r2_cv'][0] for r in rows])
    cv_mean = np.array([r['r2_cv_mean'] for r in rows])
    bic = np.array([r['bic'] for r in rows])
    d_arr = np.array(D_LIST)
    cv_h_at_d3 = cv_h[D_LIST.index(3)]
    delta_max = float(cv_h.max() - cv_h_at_d3)
    best_d_cv = int(d_arr[int(cv_mean.argmax())])
    best_d_bic = int(d_arr[int(bic.argmin())])
    primary = "d=3 SUBOPTIMAL" if delta_max > 0.03 else "d=3 OPTIMAL (Δ_CV_h ≤ 0.03)"
    print(f"[verdict.primary] {primary} (Δ_CV_h_max-vs-d3 = {delta_max:+.4f})")
    print(f"[verdict.secondary] BIC-optimal d = {best_d_bic}")
    print(f"[verdict.tertiary] active-axes per d: {[(r['d'], r['n_active']) for r in rows]}")
    print(f"[verdict] mean-CV-optimal d = {best_d_cv}")

    # --- Save
    np.savez(
        OUT_NPZ,
        d_list=np.array(D_LIST),
        r2_in=np.stack([r['r2_in'] for r in rows]),
        r2_cv=np.stack([r['r2_cv'] for r in rows]),
        r2_in_mean=np.array([r['r2_in_mean'] for r in rows]),
        r2_cv_mean=np.array([r['r2_cv_mean'] for r in rows]),
        log_lik=np.array([r['log_lik'] for r in rows]),
        bic=bic,
        n_params=np.array([r['n_params'] for r in rows]),
        n_active=np.array([r['n_active'] for r in rows]),
        delta_max_cv_h_vs_d3=delta_max,
        best_d_cv=best_d_cv,
        best_d_bic=best_d_bic,
        **archive,
    )
    print(f"[npz] saved {OUT_NPZ}")

    # --- Plot
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    # P1: R² in-sample per channel vs d
    ax = axs[0]
    for j, lab in enumerate(AUX_LABELS_HSV):
        ax.plot(D_LIST, [r['r2_in'][j] for r in rows], 'o-', label=f"in-sample {lab}")
        ax.plot(D_LIST, [r['r2_cv'][j] for r in rows], 's--', label=f"5-fold CV {lab}", alpha=0.7)
    ax.set_xlabel("latent dim d")
    ax.set_ylabel("R²")
    ax.set_title("HSV recovery R² vs d (in-sample + CV)")
    ax.axvline(3, color='k', ls=':', alpha=0.5, label='d=3 (auto_exp_38)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # P2: BIC + mean CV R² (twin y)
    ax = axs[1]
    color1 = "#1f77b4"
    color2 = "#d62728"
    ax.plot(D_LIST, bic, 'o-', color=color1, label='BIC')
    ax.set_xlabel("latent dim d")
    ax.set_ylabel("BIC", color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax2 = ax.twinx()
    ax2.plot(D_LIST, cv_mean, 's-', color=color2, label='mean CV R²')
    ax2.set_ylabel("mean CV R²", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax.set_title(f"BIC (blue) vs mean CV R² (red) | BIC-best d={best_d_bic}, CV-best d={best_d_cv}")
    ax.axvline(3, color='k', ls=':', alpha=0.5)
    ax.grid(alpha=0.3)

    # P3: variance shares per axis stacked
    ax = axs[2]
    bottom = np.zeros(len(D_LIST))
    max_axes = max(r['n_active'] for r in rows)
    for k in range(max(D_LIST)):
        heights = []
        for r in rows:
            shares = r['var_shares']
            h = shares[k] if k < len(shares) else 0.0
            heights.append(h)
        heights = np.array(heights)
        ax.bar(D_LIST, heights, bottom=bottom, label=f"axis {k}",
               edgecolor='white', linewidth=0.5)
        bottom += heights
    ax.set_xlabel("latent dim d")
    ax.set_ylabel("var share of W columns")
    ax.axhline(ACTIVE_FRAC, color='k', ls=':', alpha=0.5, label=f"{ACTIVE_FRAC:.0%} threshold")
    ax.set_title("W column-norm² shares (stacked)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f"auto_exp_53: HSV-supervised d sweep on cogito-L40 (n={n_c}, K={K})\n"
        f"primary verdict: {primary} | BIC-best d={best_d_bic} | CV-best d={best_d_cv}",
        fontsize=11, y=1.04,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    print(f"[runtime] {time.time() - t_start:.1f}s")

    summary = {
        "rows": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in r.items()} for r in rows],
        "delta_max_cv_h_vs_d3": delta_max,
        "best_d_cv": best_d_cv,
        "best_d_bic": best_d_bic,
        "primary_verdict": primary,
    }
    (ROOT / "runs" / "auto_exp_53_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
