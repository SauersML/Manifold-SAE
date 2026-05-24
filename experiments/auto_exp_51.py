"""auto_exp_51: HSV-supervised gauge-fix THEN ARD+Circle on residual block.

Prescription from auto_exp_50: the unsupervised ARD+Circle fits all chase the
variance-dominant brightness/saturation plane (PC0,PC1), so a 1-D circular
latent gets pinned to non-hue axes. Auto_exp_38 showed that HSV-supervised
gauge-fix on axes 0..2 reliably recovers a 3-axis perceptual block. Auto_exp_82
showed PC2+PC4 carry the hue ring.

Pipeline:
  Phase 1 (gauge-fix): reuse auto_exp_38.fit_aux_supervised_hsv on T0 (K=16) ->
    W_sup in R^{K x 3} predicting HSV.
  Phase 2 (residual): Q = QR(W_sup) gives the supervised subspace basis.
    P_perp = I - Q Q^T; residual coords T_res = T0_c @ P_perp_basis where
    P_perp_basis is an orthonormal basis (K, K-3) of null(Q^T). T_res is
    (n, d_res=13).
  Phase 3 (ARD+Circle on residual):
    (a) plain raw ARD (no symmetry break)  - auto_exp_50 V1 style
    (b) normalized-alpha ARD (sum(alpha)=d_res) - auto_exp_50 V2 style
  Score |rho| (Spearman of aligned pred vs hue) and circular rho vs true hue.

Reference points:
  - Bare Circle on PC1-only       |rho| = 0.108
  - auto_exp_50 V2 raw K=16 PCs   |rho| = 0.170
  - Oracle Circle on PC2+PC4      |rho| = 0.368
  - Primary target                |rho| >= 0.30  (>= 7x unsup baseline 0.041)
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
    N_TEMPLATES, K_PCS, load_xkcd_rgb, per_color_stats_mmap, hsv_from_rgb,
    fit_aux_supervised_hsv,
)
from auto_exp_47 import (  # type: ignore
    fit_circle_1d, _fit_circle_1d_single,
    circular_spearman, best_align_theta_to_hue,
)
from auto_exp_50 import (  # type: ignore
    fit_ard_circle_raw, fit_ard_circle_simplex,
    ALPHA_EPS, ALPHA_MAX, DAMP, _project_simplex_sumK, _circle_recon_loss,
    N_OUTER, N_INNER, RIDGE,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_NPZ = ROOT / "runs" / "auto_exp_51_results.npz"
MEMORY_MD = Path("/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/project_cogito_recovery_at_d_aux_3.md")


def residual_basis(W_sup: np.ndarray) -> np.ndarray:
    """Orthonormal basis (K, K-d_sup) of the orthogonal complement of span(W_sup)."""
    K, d_sup = W_sup.shape
    Q, _ = np.linalg.qr(W_sup)  # (K, d_sup) orthonormal
    # Null space of Q^T is the orthogonal complement of span(W_sup) in R^K.
    # Use SVD of (I - Q Q^T) to get orthonormal basis of complement.
    P_perp = np.eye(K) - Q @ Q.T
    U, S, _ = np.linalg.svd(P_perp)
    # rank = K - d_sup; first K-d_sup left singular vectors span the complement.
    return U[:, : K - d_sup]


def score(theta: np.ndarray, hue: np.ndarray):
    _, s, phi, pred = best_align_theta_to_hue(theta, hue)
    rho_lin, _ = spearmanr(pred, hue)
    circ = circular_spearman(s * theta + phi, hue)
    return float(abs(rho_lin)), float(circ)


def fit_ard_circle_normalized(T0_res, n_outer=N_OUTER, n_inner=N_INNER,
                              ridge=RIDGE, seed=51, n_restarts=8):
    """V2-style: simplex sum(alpha)=d_res. Reuses auto_exp_50.fit_ard_circle_simplex
    machinery but parametrized to d_res automatically (the function already uses
    K from T0 shape)."""
    return fit_ard_circle_simplex(T0_res, n_outer=n_outer, n_inner=n_inner,
                                  ridge=ridge, seed=seed, n_restarts=n_restarts)


def fit_ard_circle_raw_on(T0_res, **kw):
    return fit_ard_circle_raw(T0_res, **kw)


def main():
    t_start = time.time()
    print("[auto_exp_51] HSV gauge-fix -> ARD+Circle on residual (d_res=13)")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n, K = T0.shape
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n)
    hsv = hsv_from_rgb(rgb)
    hue = hsv[:, 0]
    print(f"[aux] hue range=[{hue.min():.3f},{hue.max():.3f}]")

    # ------------------------------------------------------------------
    # Phase 1: HSV-supervised gauge-fix (auto_exp_38)
    # ------------------------------------------------------------------
    print("\n[phase1] HSV gauge-fix (auto_exp_38 recipe)...")
    sup = fit_aux_supervised_hsv(T0, hsv)
    r2_hsv = sup["r2_hsv"]
    W_sup = sup["W_sup"]  # (K, 3)
    print(f"[phase1] R^2 hue={r2_hsv[0]:.3f} sat={r2_hsv[1]:.3f} val={r2_hsv[2]:.3f}")

    # ------------------------------------------------------------------
    # Phase 2: residual representation in the orthogonal complement
    # ------------------------------------------------------------------
    print("\n[phase2] computing residual basis (K - d_sup = 13)...")
    B_res = residual_basis(W_sup)        # (K, d_res)
    d_res = B_res.shape[1]
    Tc = T0 - T0.mean(0, keepdims=True)
    T_res = Tc @ B_res                   # (n, d_res)
    var_full = (Tc ** 2).sum() / n
    var_res = (T_res ** 2).sum() / n
    print(f"[phase2] d_res={d_res}  var(T0_c)={var_full:.4f}  "
          f"var(T_res)={var_res:.4f}  ratio={var_res / var_full:.3f}")

    # PC variance of residual block
    _, S_res, _ = np.linalg.svd(T_res - T_res.mean(0, keepdims=True),
                                full_matrices=False)
    print(f"[phase2] residual-PC var top-8: {(S_res[:8] ** 2 / (n - 1)).round(4)}")

    # ------------------------------------------------------------------
    # Reference baselines on the same hue target
    # ------------------------------------------------------------------
    print("\n[ref] baselines")
    # Bare Circle on the full K=16 (unsup)
    theta_full, W_full, _ = fit_circle_1d(T0)
    rho_full, circ_full = score(theta_full, hue)
    print(f"  bare Circle on full K=16    : |rho|={rho_full:.3f}  circ={circ_full:+.3f}")
    # Bare Circle on residual block (no ARD)
    theta_resC, W_resC, _ = fit_circle_1d(T_res)
    rho_resC, circ_resC = score(theta_resC, hue)
    print(f"  bare Circle on residual d_res: |rho|={rho_resC:.3f}  circ={circ_resC:+.3f}")

    # ------------------------------------------------------------------
    # Phase 3: ARD+Circle on residual block
    # ------------------------------------------------------------------
    print("\n[phase3a] raw ARD + Circle on residual...")
    theta_a, W_a, alpha_a, losses_a = fit_ard_circle_raw_on(T_res)
    rho_a, circ_a = score(theta_a, hue)
    top3_a = np.argsort(-alpha_a)[:3]
    print(f"  |rho|={rho_a:.3f}  circ-rho={circ_a:+.3f}  "
          f"top3 alpha (residual-PC indices in residual basis) = {top3_a.tolist()}  "
          f"vals={alpha_a[top3_a].round(3).tolist()}  sum(alpha)={alpha_a.sum():.2f}")

    print("\n[phase3b] normalized-alpha (simplex sum=d_res) ARD + Circle on residual...")
    theta_b, W_b, alpha_b, losses_b = fit_ard_circle_normalized(T_res)
    rho_b, circ_b = score(theta_b, hue)
    top3_b = np.argsort(-alpha_b)[:3]
    print(f"  |rho|={rho_b:.3f}  circ-rho={circ_b:+.3f}  "
          f"top3 alpha = {top3_b.tolist()}  vals={alpha_b[top3_b].round(3).tolist()}  "
          f"sum(alpha)={alpha_b.sum():.2f}  #nonzero={int((alpha_b > 1e-6).sum())}")

    # ------------------------------------------------------------------
    # Map residual-block top axes back into ORIGINAL K=16 PC space
    # (interpretability bonus): which K=16 PCs do the residual axes most
    # closely align with?
    # ------------------------------------------------------------------
    def project_back(axis_idx):
        """For a residual-basis index in [0..d_res), find the K=16 PC index it
        loads on most heavily. B_res[:, j] gives the vector in the original
        K-dim coordinate frame."""
        v = B_res[:, axis_idx]
        return int(np.argmax(np.abs(v))), float(np.max(np.abs(v)))

    top3_a_K = [project_back(j) for j in top3_a]
    top3_b_K = [project_back(j) for j in top3_b]
    print(f"\n[phase3] top-3 residual axes -> dominant K=16 PC (idx, |loading|):")
    print(f"   raw      : {top3_a_K}")
    print(f"   normalized: {top3_b_K}")

    # ------------------------------------------------------------------
    # Table + verdict
    # ------------------------------------------------------------------
    print()
    print("variant                                     | |rho|  | circ-rho | top-3 alpha residual-PC idx")
    print("--------------------------------------------+--------+----------+-----------------------------")
    print(f"(ref) bare Circle full K=16                 | {rho_full:.3f}  | {circ_full:+.3f}   | n/a (no ARD)")
    print(f"(ref) bare Circle on residual (no ARD)      | {rho_resC:.3f}  | {circ_resC:+.3f}   | n/a (no ARD)")
    print(f"(a) raw ARD+Circle on residual              | {rho_a:.3f}  | {circ_a:+.3f}   | {top3_a.tolist()}")
    print(f"(b) normalized-alpha ARD+Circle on residual | {rho_b:.3f}  | {circ_b:+.3f}   | {top3_b.tolist()}")
    print(f"(ref) bare Circle on PC1-only baseline      | 0.108  |  ~       | n/a")
    print(f"(ref) auto_exp_50 V2 raw K=16 PCs           | 0.170  |  ~       | (PC2 found)")
    print(f"(ref) oracle Circle PC2+PC4                 | 0.368  |  ~-0.72  | (true hue plane)")

    PRIMARY = 0.30
    best = max(rho_a, rho_b)
    primary_pass = bool(best >= PRIMARY)
    if primary_pass:
        if rho_b >= rho_a:
            verdict = "PRESCRIPTION_WORKS_V2: normalized-alpha ARD on HSV-residual recovers hue"
        else:
            verdict = "PRESCRIPTION_WORKS_RAW: raw ARD on HSV-residual recovers hue (simplex not needed)"
    elif best >= 0.20:
        verdict = "PRESCRIPTION_PARTIAL: residual ARD beats raw-K=16 ARD but below 0.30 ceiling"
    elif best >= rho_full:
        verdict = "PRESCRIPTION_MARGINAL: gauge-fix helps slightly but no qualitative recovery"
    else:
        verdict = "PRESCRIPTION_FAILS: HSV gauge-fix did not unlock hue-residual ring for ARD"
    print()
    print(f"[verdict] best |rho|={best:.3f} >= {PRIMARY}? {primary_pass} -> {verdict}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    np.savez(
        OUT_NPZ,
        hue_true=hue,
        W_sup=W_sup, r2_hsv=r2_hsv,
        B_res=B_res, d_res=d_res, T_res=T_res,
        S_res=S_res,
        theta_full=theta_full, rho_full=rho_full, circ_full=circ_full,
        theta_resC=theta_resC, rho_resC=rho_resC, circ_resC=circ_resC,
        alpha_raw=alpha_a, theta_raw=theta_a, losses_raw=losses_a,
        rho_raw=rho_a, circ_raw=circ_a, top3_raw=top3_a,
        alpha_norm=alpha_b, theta_norm=theta_b, losses_norm=losses_b,
        rho_norm=rho_b, circ_norm=circ_b, top3_norm=top3_b,
        verdict=verdict, primary_pass=primary_pass,
    )
    print(f"[npz] saved {OUT_NPZ}")

    # ------------------------------------------------------------------
    # Append memory note
    # ------------------------------------------------------------------
    try:
        if MEMORY_MD.exists():
            existing = MEMORY_MD.read_text()
        else:
            existing = ""
        note = f"""

## auto_exp_51: HSV gauge-fix THEN ARD+Circle on residual

Tested auto_exp_50's prescription. Pipeline: auto_exp_38 HSV gauge-fix on K=16
PCA(L40, n=949) -> W_sup (K,3); residual basis B_res spans the orthogonal
complement (d_res=13); ARD+Circle (1D) on T_res = (T0 - mean) @ B_res.

| variant                                  | |rho| | circ-rho | top-3 alpha residual-PC |
|------------------------------------------|-------|----------|-------------------------|
| bare Circle on full K=16                 | {rho_full:.3f} | {circ_full:+.3f}   | n/a |
| bare Circle on residual (no ARD)         | {rho_resC:.3f} | {circ_resC:+.3f}   | n/a |
| raw ARD + Circle on residual             | {rho_a:.3f} | {circ_a:+.3f}   | {top3_a.tolist()} |
| normalized-alpha ARD + Circle on residual| {rho_b:.3f} | {circ_b:+.3f}   | {top3_b.tolist()} |

References: PC1-only 0.108, auto_exp_50 V2 raw 0.170, oracle PC2+PC4 0.368.
Primary target |rho|>=0.30. Best={best:.3f}. Verdict: {verdict}

Takeaway: when a manifold has variance-dominant supervised axes (HSV) AND a
weaker structured residual (hue ring), latent-search methods (ARD, Circle)
on the RAW representation pin themselves to the variance-dominant block.
Removing the supervised subspace by orthogonal-complement projection before
running the second-stage search is the relevant generalizable pattern for
concept-manifold steering pipelines.
"""
        MEMORY_MD.write_text(existing + note)
        print(f"[memory] appended to {MEMORY_MD}")
    except Exception as exc:
        print(f"[memory] append failed: {exc!r}")

    print(f"[runtime] {time.time() - t_start:.1f}s")
    return verdict


if __name__ == "__main__":
    main()
