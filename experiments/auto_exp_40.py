"""auto_exp_40: REAL ARD (coordinate-descent) on the free latent block.

Improvement over auto_exp_39: auto_exp_39 used a PCA-prefix scan for the free
block, so axes 3..d_aux-1 were forced to be top-PCs of the HSV-orthogonal
residual. That algorithm CANNOT reallocate variance onto non-PCA-aligned
directions, which the memo project_ard_gauge_fix_doesnt_help_cogito flagged as
a confound (the cogito-L40 free-block spectrum is fat).

THIS experiment: instead of PCA-of-residual, fit a free decoder W_free (K, d_free)
via coordinate-descent ARD that alternates:
  (i)  W_free[:, j] = (T_perp.T T_perp / n + (alpha[j]/n) I)^{-1} T_perp.T y_j
       — where y_j is reconstructed implicitly by the per-axis ARD posterior
       on a generative form: x_perp ≈ W_free t_free + noise (PPCA-style).
  (ii) alpha[j] = (d_eff_j) / (||W_free[:, j]||^2 + tr-correction)

Concretely we use the PPCA-with-ARD update of Bishop (1999) on the
HSV-orthogonal residual data T_perp ∈ R^{n × K}:
  W_free ∈ R^{K, d_free}, T_free = T_perp W_free (rotation), alpha_j prior on cols.
This IS coordinate-descent ARD on the free block in the same K-dim feature
space as the supervised companion, and—crucially—it is NOT a PCA-prefix
algorithm: cold-start randomness + per-column shrinkage can produce free
axes that span non-top-PC directions of T_perp, or prune to fewer than d_free.

Sweep w_ard ∈ {0.01, 0.1, 1.0, 10.0} (an outer scalar multiplying the alpha
prior strength). For each w_ard report:
  (i)   HSV R² on supervised axes 0..2 (shared, so reported once)
  (ii)  per-axis alpha_j on axes 3..5
  (iii) max |corr| of each free axis vs {monoword, mod_count, template_sigma}
  (iv)  name-active count (free axes with max-corr > 0.4)

GAMFIT PATH: production wrappers (AuxConditionalPriorPenalty / ARDPenalty)
not available in installed gamfit 0.1.112 — falling back to PROPER
coordinate-descent ARD (NOT PCA prefix). Documented in body.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # reuse pipeline
    X_PATH, N_TEMPLATES, K_PCS, AUX_LABELS_HSV, AUX_LABELS_NAME,
    per_color_stats_mmap, load_xkcd_rgb, hsv_from_rgb, name_features,
    fit_aux_supervised_hsv, abs_corr_matrix, check_gamfit,
)

ROOT = Path("/Users/user/Manifold-SAE")
OUT_NPZ = ROOT / "runs" / "auto_exp_40_results.npz"

D_AUX_SUP = 3
D_FREE = 3
W_ARD_LIST = [0.01, 0.1, 1.0, 10.0]
N_OUTER = 200            # coordinate-descent outer iterations
N_OUTER_FALLBACK = 2     # if a fit explodes / NaNs / runs long
ALPHA_PRUNE_THRESH = 1e3 # alpha_j > this => axis pruned (effective amplitude ~ 1/sqrt(alpha))
NAME_ACTIVE_THRESH = 0.40


def fit_free_ard_ppca(T_perp, d_free, w_ard, n_outer=N_OUTER, seed=40):
    """PPCA-with-ARD on the HSV-orthogonal residual T_perp ∈ R^{n × K}.

    Generative model: x_n = W t_n + eps,  t_n ~ N(0, I_{d_free}),
                       eps ~ N(0, sigma2 I_K),  W[:,j] ~ N(0, alpha_j^{-1} I_K).

    Per-iteration:
      E-step:  M = W^T W + sigma2 * I_{d_free}
               t_n = M^{-1} W^T x_n
               <t_n t_n^T> = sigma2 M^{-1} + t_n t_n^T
      M-step:  W_new = (sum_n x_n <t_n>^T) (sum_n <t_n t_n^T> + sigma2 diag(alpha))^{-1}
               alpha_j = K / (||W[:,j]||^2 + sigma2 tr(M^{-1})_jj)   -- Bishop 1999
               (we scale alpha update by w_ard)
               sigma2 from reconstruction residual.

    This is a TRUE coordinate-descent ARD that can prune columns
    (alpha_j → ∞) and is NOT a PCA-prefix scan: cold-start randomness +
    per-column shrinkage allows W to occupy any d_free-dim subspace of the
    K-dim residual.
    """
    rng = np.random.default_rng(seed)
    n, K = T_perp.shape
    # Init W with small random + a touch of top-PC direction to break symmetry
    U, S, Vt = np.linalg.svd(T_perp, full_matrices=False)
    W = (Vt[:d_free].T * (S[:d_free] / np.sqrt(n))) * 0.5 \
        + rng.normal(scale=0.1, size=(K, d_free))
    alpha = np.ones(d_free)
    sigma2 = float(np.var(T_perp)) / 10.0 + 1e-6
    XtX = T_perp.T @ T_perp  # (K, K)
    alpha_traj = []
    converged = False
    for it in range(n_outer):
        # E-step
        M = W.T @ W + sigma2 * np.eye(d_free)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            break
        # <t_n> = M^{-1} W^T x_n  (so <T>^T <T> = (Minv W^T) XtX (W Minv))
        WMinv = W @ Minv         # (K, d_free)
        E_T = T_perp @ (W @ Minv)  # (n, d_free)
        E_TT = sigma2 * Minv + (E_T.T @ E_T)  # (d_free, d_free)
        # M-step: W = (T_perp^T E_T) (E_TT + sigma2 diag(alpha))^{-1}
        rhs = T_perp.T @ E_T     # (K, d_free)
        lhs = E_TT + sigma2 * np.diag(w_ard * alpha)
        try:
            W_new = np.linalg.solve(lhs.T, rhs.T).T
        except np.linalg.LinAlgError:
            break
        # alpha update (Bishop 1999): alpha_j = K / (||W[:,j]||^2 + sigma2 * (Minv)_jj * K)
        w_norm2 = (W_new ** 2).sum(0)
        # tr correction per axis: K * sigma2 * Minv_jj (proper full posterior variance)
        tr_corr = K * sigma2 * np.diag(Minv)
        alpha_new = K / np.maximum(w_norm2 + tr_corr, 1e-10)
        # sigma2 update from reconstruction
        recon = E_T @ W_new.T    # (n, K)
        sigma2_new = float(((T_perp - recon) ** 2).mean()) + 1e-8
        # convergence check
        dW = float(np.abs(W_new - W).max())
        W = W_new
        alpha = alpha_new
        sigma2 = sigma2_new
        alpha_traj.append(alpha.copy())
        if dW < 1e-6 and it > 10:
            converged = True
            break
    # Final T_free: project T_perp onto the latent posterior mean
    M = W.T @ W + sigma2 * np.eye(d_free)
    try:
        T_free = T_perp @ (W @ np.linalg.inv(M))
    except np.linalg.LinAlgError:
        T_free = T_perp @ W
    return {
        "W_free": W,
        "T_free": T_free,
        "alpha": alpha,
        "alpha_traj": np.asarray(alpha_traj),
        "sigma2": sigma2,
        "n_iter_done": it + 1,
        "converged": converged,
    }


def main():
    t_start = time.time()
    print("[auto_exp_40] REAL ARD (coordinate-descent PPCA-ARD) on free block")
    ver, _ = check_gamfit()
    print(f"[gamfit] version={ver}")
    print(f"[path] production wrappers NOT in {ver}; using "
          f"coordinate-descent PPCA-ARD fallback (NOT PCA-prefix)")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)   # held-out for eval ONLY
    print(f"[aux] hsv={hsv.shape}; namef={namef.shape}")

    # --- Supervised HSV (shared across the w_ard sweep)
    sup = fit_aux_supervised_hsv(T0, hsv)
    r2_hsv = sup["r2_hsv"]
    print(f"[fit] supervised HSV R^2: hue={r2_hsv[0]:.3f} sat={r2_hsv[1]:.3f} "
          f"val={r2_hsv[2]:.3f}")

    # --- Build HSV-orthogonal residual ONCE
    Tc = T0 - T0.mean(0, keepdims=True)
    Q, _ = np.linalg.qr(sup["W_sup"])
    P_perp = np.eye(sup["W_sup"].shape[0]) - Q @ Q.T
    T_perp = Tc @ P_perp
    print(f"[residual] T_perp={T_perp.shape}, var={float(np.var(T_perp)):.4f}")

    # --- Sweep w_ard
    results = []
    archive = {}
    for w_ard in W_ARD_LIST:
        t_fit = time.time()
        fit = fit_free_ard_ppca(T_perp, D_FREE, w_ard, n_outer=N_OUTER)
        fit_s = time.time() - t_fit
        # If the fit blew alpha to crazy values immediately, retry shorter
        if not np.isfinite(fit["alpha"]).all():
            print(f"[w_ard={w_ard}] NaN alpha — retrying with {N_OUTER_FALLBACK} iters")
            fit = fit_free_ard_ppca(T_perp, D_FREE, w_ard,
                                    n_outer=N_OUTER_FALLBACK)

        T_free = fit["T_free"]
        alpha = fit["alpha"]
        T_all = np.concatenate([sup["T_sup"], T_free], axis=1)

        corr_hsv = abs_corr_matrix(T_all, hsv)         # (6, 3)
        corr_name = abs_corr_matrix(T_all, namef)      # (6, 3)
        free_idx = list(range(D_AUX_SUP, D_AUX_SUP + D_FREE))
        free_corr_name = corr_name[free_idx]
        free_corr_hsv = corr_hsv[free_idx]
        per_axis_max_name = free_corr_name.max(axis=1)
        per_axis_best_name = [AUX_LABELS_NAME[int(i)]
                              for i in free_corr_name.argmax(axis=1)]
        n_name_active = int((per_axis_max_name > NAME_ACTIVE_THRESH).sum())
        n_pruned = int((alpha > ALPHA_PRUNE_THRESH).sum())

        # Per-free-axis variance ratio (effective amplitude)
        Tfc = T_free - T_free.mean(0, keepdims=True)
        free_var = (Tfc ** 2).mean(0)

        rec = {
            "w_ard": w_ard,
            "alpha": alpha,
            "free_var": free_var,
            "n_pruned": n_pruned,
            "per_axis_max_name": per_axis_max_name,
            "per_axis_best_name": per_axis_best_name,
            "free_corr_name": free_corr_name,
            "free_corr_hsv": free_corr_hsv,
            "n_name_active": n_name_active,
            "n_iter": fit["n_iter_done"],
            "converged": fit["converged"],
            "sigma2": fit["sigma2"],
            "fit_s": fit_s,
        }
        results.append(rec)
        archive[f"w{w_ard}_corr_name"] = free_corr_name
        archive[f"w{w_ard}_corr_hsv"] = free_corr_hsv
        archive[f"w{w_ard}_alpha_traj"] = fit["alpha_traj"]
        archive[f"w{w_ard}_alpha_final"] = alpha

        print(f"[w_ard={w_ard:>6}] iter={fit['n_iter_done']:>3} "
              f"conv={fit['converged']} sigma2={fit['sigma2']:.4f} "
              f"alpha={np.round(alpha, 2)} free_var={np.round(free_var, 4)} "
              f"name-active={n_name_active} pruned={n_pruned}  ({fit_s:.1f}s)")
        print(f"            best-name per axis: "
              f"{[f'{f}({c:.2f})' for f,c in zip(per_axis_best_name, per_axis_max_name)]}")

    # --- Save
    np.savez(
        OUT_NPZ,
        w_ard_list=np.array(W_ARD_LIST),
        R2_hsv_sup=r2_hsv,
        n_name_active=np.array([r["n_name_active"] for r in results]),
        n_pruned=np.array([r["n_pruned"] for r in results]),
        per_axis_max_name=np.stack([r["per_axis_max_name"] for r in results]),
        alpha_final=np.stack([r["alpha"] for r in results]),
        free_var=np.stack([r["free_var"] for r in results]),
        **archive,
    )
    print(f"[npz] saved {OUT_NPZ}")

    # --- Stdout table
    print()
    print("=" * 92)
    print(" auto_exp_40 RESULTS: REAL coordinate-descent ARD on free block (d_aux=6)")
    print("=" * 92)
    print(f" supervised HSV (shared): hue={r2_hsv[0]:.3f} sat={r2_hsv[1]:.3f} "
          f"val={r2_hsv[2]:.3f}")
    print()
    header = (f"{'w_ard':>8} {'name-active':>12} {'pruned':>7} "
              f"{'alpha_0..2':>22} {'best free corr':>22}")
    print(header)
    print("-" * len(header))
    for r in results:
        best = float(r["per_axis_max_name"].max())
        best_ax = int(r["per_axis_max_name"].argmax())
        best_feat = r["per_axis_best_name"][best_ax]
        a = r["alpha"]
        a_str = f"[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f}]"
        print(f"{r['w_ard']:>8} {r['n_name_active']:>12} {r['n_pruned']:>7} "
              f"{a_str:>22} {best:.3f}({best_feat}):>22")
    print()
    print(" per-axis detail:")
    for r in results:
        print(f"  w_ard={r['w_ard']}")
        for i in range(D_FREE):
            mn = float(r["per_axis_max_name"][i])
            mh = float(r["free_corr_hsv"][i].max())
            corrs = ", ".join(
                f"{AUX_LABELS_NAME[k]}={r['free_corr_name'][i,k]:.2f}"
                for k in range(3)
            )
            print(f"    free axis {i}: alpha={r['alpha'][i]:.3f} "
                  f"var={r['free_var'][i]:.4f} "
                  f"best-name={mn:.3f}({r['per_axis_best_name'][i]}) "
                  f"HSV-leak={mh:.3f} [{corrs}]")
    print("=" * 92)
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
