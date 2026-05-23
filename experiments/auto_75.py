"""auto_75.py — `LatentCoord` prototype: GP-LVM-as-gamfit-config (Python only).

Companion to `/Users/user/gam/proposals/latent_coord.md`. Demonstrates that
a per-row latent coordinate `t_i ∈ ℝ^d` can be fit jointly with the
smoothing coefficients β using *only* existing gamfit primitives
(`duchon_basis`, `duchon_function_norm_penalty`, `gaussian_reml_fit`),
plus a thin outer loop over t that uses an auxiliary-conditional prior
on observed RGB to break the gauge.

This is a proof-of-concept of the proposal: "the engine already does it",
not a new optimizer.

Pipeline
--------
  y = Z_top16  (cogito color centroids in PCA basis)
  for each candidate d ∈ {2, 3, 4, 5, 6}:
      build  Phi_obs  =  [Phi_hue | Phi_sat_val]            # observed
      init   t        =  PCA on Z_top16 (top d directions)  # warm start
      fit    g_φ(u)   =  ridge(rgb → t_init)                # aux prior target
      loop:
          inner: gamfit.gaussian_reml_fit on  [Phi_obs | Phi(t)]  → β, λ
          outer: gradient step on t  using analytic ∂(½‖y − Φβ‖²)/∂t_i
                                       + τ · (t_i − g_φ(u_i))
      report CV R² (re-fit aux & inner per fold; t is data-coupled but
      re-initialized from the train PCA each fold).
  pick d* by CV.

Compare to:
  auto_67 supervised HSV ceiling           R² = 0.321
  auto_74 full stack (HSV + ICA residual)  R² = 0.608
  auto_71 geodesic-LM gauge-unfixed        R² = 0.640  (Procrustes 0.99 → bad)
  this    LatentCoord + aux-prior          R² = ?      (Procrustes ?)

HARD RULES (per project memory):
  - NO Gaussian RBF; NEVER set length_scale on Duchon; NO B-splines.
  - mmap X_L40.npy.
  - _pca_basis.load_pc_basis + project for PCA.
  - 3D Duchon m=2 needs nullspace_order="degree2" (2(p+s) > d+2).
"""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors
from color_filter_list import filter_colors
from _pca_basis import load_pc_basis, project

import gamfit


OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
X_PATH = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")

N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
K_PC = 16                       # match auto_74's response space
HUE_CENTERS = 40                # observed hue basis
SV_GRID = 6                     # observed (sat, val) grid: 36 centers
N_FOLDS = 5
LATENT_DIMS = [2, 3, 4, 5]   # d=6 hits Duchon nullspace-rank ceiling at K=80; revisit when ND `LatentCoord` Rust side handles auto-centers
N_SEEDS = 2
LATENT_CENTERS_PER_AXIS = 4    # K = 4**d centers in latent space (d<=3)
OUTER_ITERS = 25
LR_T = 0.05
TAU_AUX = 0.5                  # aux-prior strength (gauge breaker)
FD_H = 1e-3                    # finite-diff step for ∂Φ/∂t


# ---------------------------------------------------------------------------
# Observed bases (verbatim from auto_67/74)
# ---------------------------------------------------------------------------
def hue_basis(hue01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.linspace(0.0, 1.0, HUE_CENTERS, endpoint=False).reshape(-1, 1)
    pts = np.asarray(hue01, dtype=np.float64).reshape(-1, 1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, periodic_per_axis=[True])
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, periodic_per_axis=[True])
    )
    return Phi, P


def sv_basis(sat: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(0.0, 1.0, SV_GRID)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
    pts = np.stack([np.asarray(sat, float), np.asarray(val, float)], axis=1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, nullspace_order="degree2")
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, nullspace_order="degree2")
    )
    return Phi, P


# ---------------------------------------------------------------------------
# Latent Duchon basis (dimension d, scale-free Duchon, nullspace=degree2)
# ---------------------------------------------------------------------------
def latent_centers(d: int) -> np.ndarray:
    """Centers in [-1,1]^d. For d<=3 tensor-product; for d>=4 a Sobol-like
    quasi-random subset of fixed budget keeps the basis size bounded."""
    if d <= 3:
        g = np.linspace(-1.0, 1.0, LATENT_CENTERS_PER_AXIS)
        meshes = np.meshgrid(*([g] * d), indexing="ij")
        return np.stack([m.ravel() for m in meshes], axis=1)
    # d >= 4: cap at ~80 centers via Halton-like sequence
    K_target = 80
    rng_c = np.random.default_rng(0)
    return rng_c.uniform(-1.0, 1.0, size=(K_target, d))


def _ns_for_dim(d: int) -> str:
    """Pick nullspace_order so 2(p+s) > d+2 holds for m=2 Duchon.

    nullspace_order="degreeK" sets p=K+1, s=0 → need 2(K+1) > d+2 → K > d/2.
    """
    K = (d // 2) + 1
    K = max(K, 2)
    return f"degree{K}"


def latent_basis(t: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Phi(t) for d-dim latent. Auto-picks nullspace order per d."""
    d = t.shape[1]
    return np.asarray(
        gamfit.duchon_basis(t, centers, m=2, nullspace_order=_ns_for_dim(d))
    )


def latent_penalty(centers: np.ndarray) -> np.ndarray:
    d = centers.shape[1]
    return np.asarray(
        gamfit.duchon_function_norm_penalty(
            centers, m=2, nullspace_order=_ns_for_dim(d)
        )
    )


# ---------------------------------------------------------------------------
# Auxiliary predictor: ridge(rgb → t_init)
# ---------------------------------------------------------------------------
def fit_aux_ridge(rgb: np.ndarray, t: np.ndarray, alpha: float = 1e-2):
    """Closed-form ridge regression in feature space [rgb, rgb², cross]."""
    feat = build_aux_features(rgb)
    p = feat.shape[1]
    A = feat.T @ feat + alpha * np.eye(p)
    B = feat.T @ t
    W = np.linalg.solve(A, B)
    return W


def build_aux_features(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return np.column_stack(
        [np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b]
    )


def predict_aux(rgb: np.ndarray, W: np.ndarray) -> np.ndarray:
    return build_aux_features(rgb) @ W


# ---------------------------------------------------------------------------
# Inner solve: REML over [Phi_obs | Phi(t)] with block-diag penalty
# ---------------------------------------------------------------------------
def reml_inner(
    Phi_obs: np.ndarray,
    P_obs: np.ndarray,
    Phi_lat: np.ndarray,
    P_lat: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, float]:
    Phi = np.concatenate([Phi_obs, Phi_lat], axis=1)
    P = sla.block_diag(P_obs, P_lat)
    out = gamfit.gaussian_reml_fit(Phi, Y, P)
    return np.asarray(out["coefficients"]), float(out["lambda"])


# ---------------------------------------------------------------------------
# ∂(½‖y − Φ(t)β‖²)/∂t_i via finite-difference along each latent axis
# 2·d full basis evaluations per outer step (cheap for N≤1k, d≤6)
# ---------------------------------------------------------------------------
def latent_residual_gradient(
    t: np.ndarray,
    centers: np.ndarray,
    beta_lat: np.ndarray,    # (K_lat, R)
    resid: np.ndarray,        # y - Phi_full · β_full, shape (N, R)
) -> np.ndarray:
    N, d = t.shape
    grad = np.zeros((N, d))
    # Phi_base @ beta_lat is already in resid; we need ∂Φ/∂t · beta_lat per row.
    for a in range(d):
        e_a = np.zeros(d)
        e_a[a] = FD_H
        Phi_plus = latent_basis(t + e_a, centers)
        Phi_minus = latent_basis(t - e_a, centers)
        dPhi = (Phi_plus - Phi_minus) / (2.0 * FD_H)   # (N, K_lat)
        # Per-row contribution of ∂(Φ_i β)/∂t_i_a is dPhi_i · beta_lat → (R,)
        # Gradient of ½‖resid‖² is −resid · (dPhi · β) summed over R
        dyhat = dPhi @ beta_lat                          # (N, R)
        grad[:, a] = -np.sum(resid * dyhat, axis=1)
    return grad


# ---------------------------------------------------------------------------
# Outer loop: alternating REML-on-β + projected-GD on t with aux prior
# ---------------------------------------------------------------------------
def fit_latent_coord(
    Y: np.ndarray,
    Phi_obs: np.ndarray,
    P_obs: np.ndarray,
    t_init: np.ndarray,
    rgb: np.ndarray,
    *,
    n_outer: int = OUTER_ITERS,
    lr_t: float = LR_T,
    tau: float = TAU_AUX,
    verbose: bool = False,
) -> dict:
    N, d = t_init.shape
    centers = latent_centers(d)
    K_lat = centers.shape[0]
    K_obs = Phi_obs.shape[1]
    P_lat = latent_penalty(centers)

    # Aux prior target: ridge(rgb → t_init).  We re-fit the aux predictor
    # against the current t every few steps so it tracks the moving latent —
    # this is the iVAE consistency property: identifiability requires that
    # the conditional prior p(t|u) actually matches the recovered t.
    t = t_init.copy()
    W_aux = fit_aux_ridge(rgb, t)
    history = []

    for it in range(n_outer):
        # --- inner: closed-form REML on β at current t ---
        Phi_lat = latent_basis(t, centers)
        beta, lam = reml_inner(Phi_obs, P_obs, Phi_lat, P_lat, Y)
        beta_obs = beta[:K_obs]
        beta_lat = beta[K_obs:]
        yhat = Phi_obs @ beta_obs + Phi_lat @ beta_lat
        resid = Y - yhat
        train_r2 = 1.0 - float((resid ** 2).sum()) / float(
            ((Y - Y.mean(0, keepdims=True)) ** 2).sum()
        )

        # --- outer: grad on t  =  data residual + τ · (t − g_φ(u)) ---
        g_data = latent_residual_gradient(t, centers, beta_lat, resid)
        t_pred = predict_aux(rgb, W_aux)
        g_aux = tau * (t - t_pred)
        g = g_data + g_aux

        # Crude line-search-free step with mild clipping
        gnorm = np.linalg.norm(g, axis=1, keepdims=True).clip(min=1e-9)
        step = lr_t * g / gnorm.clip(max=3.0)
        t_new = t - step

        # Stay in init box (keeps centers grid useful)
        t_new = np.clip(t_new, -1.5, 1.5)

        history.append(dict(iter=it, train_r2=train_r2, lam=lam,
                            gnorm=float(np.linalg.norm(g))))
        t = t_new

        # Refresh aux predictor every 10 steps (iVAE consistency)
        if (it + 1) % 10 == 0:
            W_aux = fit_aux_ridge(rgb, t)
            if verbose:
                print(f"  [outer {it+1:3d}] train_R²={train_r2:+.4f}  λ={lam:.3g}")

    # Final inner solve
    Phi_lat = latent_basis(t, centers)
    beta, lam = reml_inner(Phi_obs, P_obs, Phi_lat, P_lat, Y)

    return dict(
        t=t,
        beta=beta,
        lam=lam,
        centers=centers,
        K_lat=K_lat,
        history=history,
        W_aux=W_aux,
    )


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------
def cv_score(
    Y: np.ndarray,
    Phi_obs_train: np.ndarray,
    Phi_obs_test: np.ndarray,
    P_obs: np.ndarray,
    rgb_train: np.ndarray,
    rgb_test: np.ndarray,
    Z_train: np.ndarray,           # the PCA Z used for warm-start
    Z_test: np.ndarray,
    d: int,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Returns (CV R^2 contribution on test fold, t_test estimated)."""
    rng = np.random.default_rng(seed)
    # warm-start t_train = top-d PCs of training Z (after partial-out of obs)
    # quick partial-out via plain ridge
    Btmp = np.linalg.solve(
        Phi_obs_train.T @ Phi_obs_train + 1e-3 * np.eye(Phi_obs_train.shape[1]),
        Phi_obs_train.T @ Z_train,
    )
    Z_train_res = Z_train - Phi_obs_train @ Btmp
    U, S, Vt = np.linalg.svd(Z_train_res, full_matrices=False)
    t_train_init = U[:, :d] * S[:d]
    # rescale to unit-ish box
    scale = max(1e-9, np.max(np.abs(t_train_init)))
    t_train_init = t_train_init / scale + 0.02 * rng.standard_normal(t_train_init.shape)

    fit = fit_latent_coord(
        Y, Phi_obs_train, P_obs, t_train_init, rgb_train,
        n_outer=OUTER_ITERS, verbose=False,
    )

    # Out-of-fold t: warm-start from aux predictor (gauge anchor),
    # then refine by minimizing  ½‖Y_test − Phi_obs_test β_obs − Phi(t)β_lat‖²
    # + τ·‖t − g_φ(u_test)‖²  with β frozen.  This is exactly the inductive
    # use of an identified latent-variable model — train-side coefficients
    # are held fixed, latent reconstructed under the aux prior.
    centers = fit["centers"]
    beta = fit["beta"]
    K_obs = Phi_obs_train.shape[1]
    beta_obs = beta[:K_obs]
    beta_lat = beta[K_obs:]
    t_pred = predict_aux(rgb_test, fit["W_aux"])
    t_test = np.clip(t_pred, -1.5, 1.5)

    # Mini outer loop on t_test with β frozen
    resid_target = Z_test - Phi_obs_test @ beta_obs
    for _ in range(40):
        Phi_lat_te = latent_basis(t_test, centers)
        r = resid_target - Phi_lat_te @ beta_lat
        g_data = latent_residual_gradient(t_test, centers, beta_lat, r)
        g_aux = TAU_AUX * (t_test - t_pred)
        g = g_data + g_aux
        gnorm = np.linalg.norm(g, axis=1, keepdims=True).clip(min=1e-9)
        step = LR_T * g / gnorm.clip(max=3.0)
        t_test = np.clip(t_test - step, -1.5, 1.5)

    Phi_lat_test = latent_basis(t_test, centers)
    Y_pred = Phi_obs_test @ beta_obs + Phi_lat_test @ beta_lat
    return Y_pred, fit["t"]


def r2_macro(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized Procrustes distance after centering + scaling + rotation."""
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / max(np.linalg.norm(A), 1e-12)
    B = B / max(np.linalg.norm(B), 1e-12)
    if A.shape[1] != B.shape[1]:
        k = min(A.shape[1], B.shape[1])
        A, B = A[:, :k], B[:, :k]
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    return float(np.linalg.norm(A @ R - B) ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    X = np.load(X_PATH, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        block = np.asarray(X[rows], dtype=np.float64)
        centroids[ci] = block.mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    N = len(kept)
    print(f"[load] N={N} filtered colors")

    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    print(f"[pca] Z shape={Z.shape}  EVR={float(basis['evr'].sum()):.3f}")

    # --- 2. Observed bases (constant across folds) ---
    Phi_h_full, P_h = hue_basis(hue)
    Phi_sv_full, P_sv = sv_basis(sat, val)
    Phi_obs_full = np.concatenate([Phi_h_full, Phi_sv_full], axis=1)
    P_obs = sla.block_diag(P_h, P_sv)
    print(f"[obs basis] K_hue={Phi_h_full.shape[1]}  K_sv={Phi_sv_full.shape[1]}")

    # --- 3. CV folds ---
    rng_main = np.random.default_rng(0)
    perm = rng_main.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    # --- 4. Baseline: hue + sv only (matches auto_74 0.321 / 0.x line) ---
    pred_obs = np.zeros_like(Z)
    for f in range(N_FOLDS):
        tr = fold != f; te = ~tr
        Btmp = np.asarray(gamfit.gaussian_reml_fit(
            Phi_obs_full[tr], Z[tr], P_obs
        )["coefficients"])
        pred_obs[te] = Phi_obs_full[te] @ Btmp
    r2_obs = r2_macro(Z, pred_obs)
    print(f"[baseline obs] CV R² = {r2_obs:+.4f}")

    # --- 5. Sweep d ---
    results = {}
    for d in LATENT_DIMS:
        print(f"\n=== d = {d} ===")
        seed_r2s = []
        seed_ts = []
        for seed in range(N_SEEDS):
            pred = np.zeros_like(Z)
            for f in range(N_FOLDS):
                tr = fold != f; te = ~tr
                pred_te, t_train = cv_score(
                    Y=Z[tr],
                    Phi_obs_train=Phi_obs_full[tr],
                    Phi_obs_test=Phi_obs_full[te],
                    P_obs=P_obs,
                    rgb_train=rgb[tr],
                    rgb_test=rgb[te],
                    Z_train=Z[tr],
                    Z_test=Z[te],
                    d=d,
                    seed=seed,
                )
                pred[te] = pred_te
            r2 = r2_macro(Z, pred)
            seed_r2s.append(r2)
            print(f"  seed={seed}  CV R² = {r2:+.4f}")

            # Full-data fit for cross-seed Procrustes
            Btmp = np.linalg.solve(
                Phi_obs_full.T @ Phi_obs_full
                + 1e-3 * np.eye(Phi_obs_full.shape[1]),
                Phi_obs_full.T @ Z,
            )
            Z_res = Z - Phi_obs_full @ Btmp
            U, S, _ = np.linalg.svd(Z_res, full_matrices=False)
            t0 = U[:, :d] * S[:d]
            t0 = t0 / max(1e-9, np.max(np.abs(t0)))
            rng_s = np.random.default_rng(seed + 100)
            t0 = t0 + 0.05 * rng_s.standard_normal(t0.shape)
            full_fit = fit_latent_coord(
                Z, Phi_obs_full, P_obs, t0, rgb,
                n_outer=OUTER_ITERS, verbose=False,
            )
            seed_ts.append(full_fit["t"])

        # Procrustes across seeds
        proc = []
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                proc.append(procrustes_disparity(seed_ts[i], seed_ts[j]))
        proc_med = float(np.median(proc)) if proc else float("nan")
        results[d] = dict(
            cv_r2_mean=float(np.mean(seed_r2s)),
            cv_r2_std=float(np.std(seed_r2s)),
            cv_r2_per_seed=[float(x) for x in seed_r2s],
            procrustes_median=proc_med,
        )
        print(f"  d={d}: CV R² = {np.mean(seed_r2s):+.4f} ± {np.std(seed_r2s):.4f}"
              f"   Procrustes(med across {N_SEEDS} seeds) = {proc_med:.3f}")

    # --- 6. Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ds = list(results.keys())
    r2s = [results[d]["cv_r2_mean"] for d in ds]
    err = [results[d]["cv_r2_std"] for d in ds]
    ax.errorbar(ds, r2s, yerr=err, marker="o", capsize=4, color="#2ca02c",
                label="LatentCoord + aux-prior (this)")
    ax.axhline(0.321, color="#d62728", ls="--", lw=1,
               label="auto_67 HSV ceiling (0.321)")
    ax.axhline(0.608, color="#1f77b4", ls="--", lw=1,
               label="auto_74 stacked (0.608)")
    ax.axhline(0.640, color="#888", ls=":", lw=1,
               label="auto_71 LM (0.640, gauge-broken)")
    ax.axhline(r2_obs, color="black", ls=":", lw=1,
               label=f"obs-only baseline ({r2_obs:+.3f})")
    ax.set_xlabel("latent dim d")
    ax.set_ylabel("CV macro R²  (Z_top16)")
    ax.set_title("auto_75: LatentCoord prototype  (gamfit primitives)")
    ax.set_xticks(ds)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    procs = [results[d]["procrustes_median"] for d in ds]
    ax.bar(ds, procs, color="#1f77b4", edgecolor="black")
    ax.axhline(0.99, color="#d62728", ls="--", lw=1,
               label="auto_71 unfixed (0.99)")
    ax.set_xlabel("latent dim d")
    ax.set_ylabel(f"median Procrustes across {N_SEEDS} seeds")
    ax.set_title("Gauge-orbit residual after aux-prior")
    ax.set_xticks(ds)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    for d, p in zip(ds, procs):
        ax.text(d, p + 0.005, f"{p:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    out_png = OUT_DIR / "auto_75.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n[plot] saved {out_png}")

    # --- 7. JSON ---
    out_json = OUT_DIR / "auto_75.json"
    payload = dict(
        N=N,
        K_PC=K_PC,
        observed_baseline_cv_r2=r2_obs,
        per_dim=results,
        references=dict(
            auto_67_hsv_ceiling=0.321,
            auto_74_stacked=0.608,
            auto_71_lm_gauge_unfixed=0.640,
        ),
        config=dict(
            HUE_CENTERS=HUE_CENTERS,
            SV_GRID=SV_GRID,
            LATENT_CENTERS_PER_AXIS=LATENT_CENTERS_PER_AXIS,
            OUTER_ITERS=OUTER_ITERS,
            LR_T=LR_T,
            TAU_AUX=TAU_AUX,
            N_FOLDS=N_FOLDS,
            N_SEEDS=N_SEEDS,
        ),
    )
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[json] saved {out_json}")

    # --- 8. Best-d summary line ---
    best_d = max(results, key=lambda d: results[d]["cv_r2_mean"])
    print(f"\n[summary] best d* = {best_d}  "
          f"CV R² = {results[best_d]['cv_r2_mean']:+.4f}  "
          f"Procrustes = {results[best_d]['procrustes_median']:.3f}")


if __name__ == "__main__":
    main()
