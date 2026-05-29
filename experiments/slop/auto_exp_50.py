"""auto_exp_50: Does a NORMALIZED alpha prior fix ARD+Circle's hue recovery?

Hypothesis (from auto_exp_49 + auto_82 + auto_85):
  ARD+Circle in auto_exp_49 picked top-3 alpha PCs [9, 7, 4] -- low-variance tail.
  auto_85 showed Spearman rho ~= -0.5 between PC variance and hue-circularity:
  high-variance PCs do NOT carry hue. auto_82 showed PC2+PC4 carries hue
  (circ-corr -0.72). Why did ARD then concentrate alpha on the TAIL?

Mechanism candidate: the ARD/W rescaling symmetry. The model
    alpha_k * T0[:,k] ~ Phi @ W[:, k]
is invariant under  (alpha_k, W[:,k]) -> (c*alpha_k, c*W[:,k])  for any c.
The "shrinkage" update alpha_k <- d_k / ||W_k||^2 then trivially inflates
alpha wherever the latent landed (which auto_exp_47 showed is the top-1
plane: PC0+PC1, brightness/saturation). The renormalization in auto_exp_49
(sum(alpha)=K) breaks GLOBAL scale only, not the per-axis trade-off.

This experiment tests two distinct symmetry-breakers:
  V1: ARD+Circle, raw VB update (NO simplex constraint, just bounded for numerics)
  V2: ARD+Circle with HARD simplex projection (alpha_k >= 0, sum == K)
      via Lagrange-style softmax-style projection each step.
  V3: ARD+Circle with FIXED isotropic noise sigma^2 = 1 inside a proper
      Gaussian-Gamma ARD update -- the OTHER axis of the symmetry
      (sigma can't shrink to make alpha irrelevant).

Reports |rho| with hue + top-3 alpha PC indices per variant.
Primary verdict: does V2 (or V3) concentrate alpha on {PC2, PC4}?
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
    fit_circle_1d, _fit_circle_1d_single,
    circular_spearman, best_align_theta_to_hue,
)

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
OUT_NPZ = ROOT / "runs" / "auto_exp_50_results.npz"

K_PCS = 16
N_OUTER = 25
N_INNER = 100
RIDGE = 1e-3
ALPHA_EPS = 1e-6
ALPHA_MAX = 1e3
DAMP = 0.5


def _circle_recon_loss(Tw, W, theta):
    Phi = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return float(((Tw - Phi @ W) ** 2).sum())


def _project_simplex_sumK(v, K_target):
    """Project v onto {a : a_k >= 0, sum(a) == K_target}.
    Euclidean projection onto the (scaled) simplex via the standard
    Duchi-Shalev-Shwartz O(n log n) algorithm.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - K_target
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = np.nonzero(cond)[0].max() + 1
    theta = cssv[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def fit_ard_circle_raw(T0, n_outer=N_OUTER, n_inner=N_INNER, ridge=RIDGE,
                       seed=49, n_restarts=8):
    """Variant 1: RAW ARD -- no simplex constraint. Only bounded for numerics.
    This is what ARD looks like when the alpha/W rescaling symmetry is unbroken.
    """
    Tc = T0 - T0.mean(0, keepdims=True)
    n, K = Tc.shape
    alpha = np.ones(K)
    theta, W, _ = fit_circle_1d(T0, n_iter=n_inner, ridge=ridge, seed=seed,
                                n_restarts=n_restarts)
    losses = []
    for it in range(n_outer):
        Tw = Tc * alpha[None, :]
        theta, W, _ = _fit_circle_1d_single(Tw, n_inner, ridge, theta)
        losses.append(_circle_recon_loss(Tw, W, theta))
        d_k = 2.0
        wnorm2 = (W ** 2).sum(axis=0)
        alpha_new = d_k / (wnorm2 + ALPHA_EPS)
        # NO normalization. Only clamp to avoid overflow.
        alpha_new = np.clip(alpha_new, ALPHA_EPS, ALPHA_MAX)
        alpha = DAMP * alpha + (1 - DAMP) * alpha_new
    return theta, W, alpha, np.asarray(losses)


def fit_ard_circle_simplex(T0, n_outer=N_OUTER, n_inner=N_INNER, ridge=RIDGE,
                           seed=49, n_restarts=8):
    """Variant 2: HARD simplex projection onto {alpha_k >= 0, sum == K}.
    Breaks the alpha/W rescaling symmetry GLOBALLY (sum) AND by preventing
    alpha from being made arbitrarily small to suppress an axis -- it costs
    capacity that must go elsewhere.
    """
    Tc = T0 - T0.mean(0, keepdims=True)
    n, K = Tc.shape
    alpha = np.ones(K)  # already on simplex (sum=K)
    theta, W, _ = fit_circle_1d(T0, n_iter=n_inner, ridge=ridge, seed=seed,
                                n_restarts=n_restarts)
    losses = []
    for it in range(n_outer):
        Tw = Tc * alpha[None, :]
        theta, W, _ = _fit_circle_1d_single(Tw, n_inner, ridge, theta)
        losses.append(_circle_recon_loss(Tw, W, theta))
        d_k = 2.0
        wnorm2 = (W ** 2).sum(axis=0)
        alpha_new = d_k / (wnorm2 + ALPHA_EPS)
        alpha_new = _project_simplex_sumK(alpha_new, float(K))
        alpha = DAMP * alpha + (1 - DAMP) * alpha_new
        # re-project after damping (damping breaks the constraint slightly)
        alpha = _project_simplex_sumK(alpha, float(K))
    return theta, W, alpha, np.asarray(losses)


def fit_ard_circle_fixed_sigma(T0, n_outer=N_OUTER, n_inner=N_INNER, ridge=RIDGE,
                                seed=49, n_restarts=8, sigma2=1.0):
    """Variant 3: FIXED isotropic noise sigma^2 = 1. Proper Gaussian-Gamma ARD:
        alpha_k = (d_k + 2*a0) / (||W_k||^2 / sigma^2 + 2*b0)
    With (a0, b0) = (1e-3, 1e-3) (weakly informative).
    Fixed sigma^2 breaks the other half of the alpha/W rescaling symmetry:
    you can't trivially shrink the noise to absorb a small alpha.
    """
    Tc = T0 - T0.mean(0, keepdims=True)
    n, K = Tc.shape
    alpha = np.ones(K)
    theta, W, _ = fit_circle_1d(T0, n_iter=n_inner, ridge=ridge, seed=seed,
                                n_restarts=n_restarts)
    a0, b0 = 1e-3, 1e-3
    losses = []
    for it in range(n_outer):
        Tw = Tc * alpha[None, :]
        theta, W, _ = _fit_circle_1d_single(Tw, n_inner, ridge, theta)
        losses.append(_circle_recon_loss(Tw, W, theta))
        d_k = 2.0
        wnorm2 = (W ** 2).sum(axis=0)
        alpha_new = (d_k + 2 * a0) / (wnorm2 / sigma2 + 2 * b0)
        alpha_new = np.clip(alpha_new, ALPHA_EPS, ALPHA_MAX)
        alpha = DAMP * alpha + (1 - DAMP) * alpha_new
    return theta, W, alpha, np.asarray(losses)


def score(theta, hue):
    _, s, phi, pred = best_align_theta_to_hue(theta, hue)
    rho, _ = spearmanr(pred, hue)
    circ_rho = circular_spearman(s * theta + phi, hue)
    return abs(rho), circ_rho


def main():
    t0 = time.time()
    print("[auto_exp_50] Normalized-alpha prior tests for ARD+Circle hue recovery")
    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, _ = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")
    names, rgb = load_xkcd_rgb(n)
    hue = hsv_from_rgb(rgb)[:, 0]

    # PC variance for reference
    Tc = T0 - T0.mean(0, keepdims=True)
    _, S, _ = np.linalg.svd(Tc, full_matrices=False)
    print(f"[pca] var top-8: {(S[:8]**2/(n-1)).round(3)}")

    # ----- V1: raw ARD -----
    print("\n[V1 ARD+Circle RAW (no normalization)] fitting...")
    theta1, W1, alpha1, losses1 = fit_ard_circle_raw(T0)
    rho1, circ1 = score(theta1, hue)
    top3_1 = np.argsort(-alpha1)[:3]
    print(f"  |rho|={rho1:.3f}  circ-rho={circ1:+.3f}  top3 alpha PCs={top3_1.tolist()} "
          f"vals={alpha1[top3_1].round(3).tolist()}  sum(alpha)={alpha1.sum():.2f}")

    # ----- V2: simplex-projected -----
    print("\n[V2 ARD+Circle simplex sum(alpha)=K] fitting...")
    theta2, W2, alpha2, losses2 = fit_ard_circle_simplex(T0)
    rho2, circ2 = score(theta2, hue)
    top3_2 = np.argsort(-alpha2)[:3]
    print(f"  |rho|={rho2:.3f}  circ-rho={circ2:+.3f}  top3 alpha PCs={top3_2.tolist()} "
          f"vals={alpha2[top3_2].round(3).tolist()}  sum(alpha)={alpha2.sum():.2f}  "
          f"#nonzero={int((alpha2 > 1e-6).sum())}")

    # ----- V3: fixed sigma^2=1 -----
    print("\n[V3 ARD+Circle fixed sigma^2=1] fitting...")
    theta3, W3, alpha3, losses3 = fit_ard_circle_fixed_sigma(T0)
    rho3, circ3 = score(theta3, hue)
    top3_3 = np.argsort(-alpha3)[:3]
    print(f"  |rho|={rho3:.3f}  circ-rho={circ3:+.3f}  top3 alpha PCs={top3_3.tolist()} "
          f"vals={alpha3[top3_3].round(3).tolist()}  sum(alpha)={alpha3.sum():.2f}")

    # ----- table -----
    print()
    print("variant                                     | |rho|  | circ-rho | top-3 alpha PCs")
    print("--------------------------------------------+--------+----------+------------------")
    print(f"V1 ARD+Circle RAW (no normalization)        | {rho1:.3f}  | {circ1:+.3f}   | {top3_1.tolist()}")
    print(f"V2 ARD+Circle simplex sum(alpha)=K          | {rho2:.3f}  | {circ2:+.3f}   | {top3_2.tolist()}")
    print(f"V3 ARD+Circle fixed sigma^2=1               | {rho3:.3f}  | {circ3:+.3f}   | {top3_3.tolist()}")
    print(f"(reference) auto_exp_49 unsupervised Circle | 0.041  |  n/a     | n/a  (no ARD)")
    print(f"(reference) auto_exp_49 ARD+Circle          |  ~     |  ~       | [9, 7, 4]")
    print(f"(reference) oracle PC2+PC4 ceiling          |  ~0.65 | -0.72    | (true hue plane)")

    # ----- verdict -----
    BASE = 0.041
    target_pcs = {2, 4}  # PC2 (index 2), PC4 (index 4)
    def check(rho_v, top3):
        primary = bool(set(top3.tolist()) & target_pcs)  # top-3 includes PC2 OR PC4
        secondary = (rho_v - BASE) >= 0.15
        return primary, secondary
    p1, s1 = check(rho1, top3_1)
    p2, s2 = check(rho2, top3_2)
    p3, s3 = check(rho3, top3_3)
    print()
    print(f"[verdict V1 raw]      top3 hits PC2/PC4? {p1}   |rho|-base>=0.15? {s1}")
    print(f"[verdict V2 simplex]  top3 hits PC2/PC4? {p2}   |rho|-base>=0.15? {s2}")
    print(f"[verdict V3 fixed_s]  top3 hits PC2/PC4? {p3}   |rho|-base>=0.15? {s3}")
    if p2 and s2:
        verdict = "V2_BOTH_YES: normalized-alpha simplex fixes ARD/W variance inversion"
    elif p2 or s2:
        verdict = "V2_PARTIAL: normalized-alpha helps but does not fully recover hue plane"
    elif p3 and s3:
        verdict = "V3_BOTH_YES: only fixed-sigma symmetry-break works; sum-constraint insufficient"
    elif (p3 or s3):
        verdict = "V3_PARTIAL: fixed-sigma helps; sum-constraint did not"
    else:
        verdict = "ALL_NO: ARD-over-PCs fundamentally wrong tool for this geometry; need supervised aux"
    print(f"[verdict] {verdict}")

    np.savez(
        OUT_NPZ,
        hue_true=hue,
        alpha_raw=alpha1, theta_raw=theta1, losses_raw=losses1,
        alpha_simplex=alpha2, theta_simplex=theta2, losses_simplex=losses2,
        alpha_fixed_sigma=alpha3, theta_fixed_sigma=theta3, losses_fixed_sigma=losses3,
        rho_raw=rho1, rho_simplex=rho2, rho_fixed_sigma=rho3,
        top3_raw=top3_1, top3_simplex=top3_2, top3_fixed_sigma=top3_3,
        verdict=verdict,
    )
    print(f"[npz] saved {OUT_NPZ}")
    print(f"[runtime] {time.time() - t0:.1f}s")
    return verdict, (rho1, top3_1), (rho2, top3_2), (rho3, top3_3)


if __name__ == "__main__":
    main()
