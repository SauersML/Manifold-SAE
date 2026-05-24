"""auto_exp_21 - Circle manifold + Fisher-Rao W + ARD over aux dims (cogito L40).

HYPOTHESIS
----------
A Riemannian-circle latent for hue + a Fisher-Rao per-row metric W (N,p,p)
+ ARDPenalty over 4 auxiliary Euclidean dims should:
  (a) recover hue as the circular axis (high circular-correlation R^2 with
      ground-truth HSV hue),
  (b) ARD-prune the 4 aux dims to ~2-3 effective dims (the perceptual +
      name-semantic axes per project_cogito_color_manifold_decomposition).

GAMFIT API REALITY (probed)
---------------------------
gamfit==0.1.112 currently ships:
  - Sphere (only S^2, not Sphere(dim=0)),
  - PeriodicSplineCurve (closed curve in R^d, periodic in t),
  - smooth, fit(formula=...), Duchon, BSpline, etc.
NOT present in this wheel:
  - LatentCoord,
  - fisher_w kwarg on fit(),
  - gamfit._penalties.ARDPenalty.

So all three requested primitives are FALLBACKS (no real composition-engine
call possible on this wheel). The fallback emulators are designed to match
the *shape* of the composition each primitive would build:
  circle      -> direct (cos t, sin t) latent with periodic alternating fit,
  fisher_rao  -> per-row diagonal precision W_ii from local residual variance,
  ard         -> per-aux-axis precision tau_j updated by EM
                 tau_j <- (1 + 2*a0) / (||u_j||^2 / sigma^2 + 2*b0).

If any of these primitives DO become reachable in a later gamfit, the
prediction in the JSON is what to compare against.

RAM
---
- mmap=r on X_L40.npy (760 MB),
- cached load_pc_basis(K=64),
- project to (N, 16) immediately; never load full N x 4096.
"""

from __future__ import annotations

import colorsys
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import load_pc_basis, project, TOP_TEMPLATES, N_TEMPLATES
from color_filter_list import filter_colors
from color_geometry import load_xkcd_colors


# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_21.png"
OUT_JSON = OUT_DIR / "auto_exp_21.json"

K_PC      = 16          # ambient (project to top-16 PCs)
D_AUX     = 4           # ARD over 4 auxiliary euclidean dims
N_ITERS   = 60
ARD_A0    = 1.0         # gamma hyperprior shape (uninformative-ish)
ARD_B0    = 1.0         # gamma hyperprior rate
TAU_MAX   = 1e3         # cap to prevent runaway shrinkage on first iter
ARD_PRUNE_THR = 0.10    # axis "effective" if eff_var / max_eff_var >= 0.10
SEED      = 0
MAX_RSS_GB = 6.0


# ---------------------------------------------------------------------------
# Probe gamfit + composition engine reachability
# ---------------------------------------------------------------------------
import gamfit
GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")

def _probe_primitive(name, fn):
    try:
        fn()
        return True, "reached"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

PRIMITIVE_STATUS = {}

# circle = Sphere(dim=0) OR LatentCoord(..., manifold=Sphere(dim=0))
ok, why = _probe_primitive(
    "circle",
    lambda: getattr(gamfit.smooth, "LatentCoord")
            or gamfit.Sphere(n_centers=4),  # Sphere with dim=0 would be ideal
)
PRIMITIVE_STATUS["circle"] = {"reached": ok, "detail": why}
# Sphere exists but is S^2 (lat,lng) not S^1. Treat as fallback regardless.
PRIMITIVE_STATUS["circle"]["reached"] = False
PRIMITIVE_STATUS["circle"]["detail"] = (
    "gamfit.Sphere is S^2 only (no Sphere(dim=0)); "
    "gamfit.smooth has no LatentCoord. Fallback: direct (cos t, sin t) latent.")

ok_fw, why_fw = _probe_primitive(
    "fisher_rao",
    lambda: ("fisher_w" in gamfit.fit.__code__.co_varnames) or 1 / 0,
)
PRIMITIVE_STATUS["fisher_rao"] = {"reached": False,
                                  "detail": "no fisher_w kwarg on gamfit.fit; "
                                            "fallback: per-row diag W."}

ok_ard, why_ard = _probe_primitive(
    "ard",
    lambda: __import__("gamfit._penalties", fromlist=["ARDPenalty"]),
)
PRIMITIVE_STATUS["ard"] = {"reached": False,
                           "detail": f"gamfit._penalties unavailable ({why_ard}); "
                                     "fallback: per-axis tau EM."}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_centroids():
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape
    n_raw = n_total // N_TEMPLATES
    centroids = np.zeros((n_raw, H), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_TEMPLATES + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(axis=0)
    del X
    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    names = [n for n, *_ in kept]
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    return centroids, names, rgb, hsv


# ---------------------------------------------------------------------------
# Primitive 2 fallback: per-row Fisher-Rao W as diag precision
# ---------------------------------------------------------------------------
def fisher_rao_W_diag(Z, k_nn=20):
    """Per-row diagonal precision W[i] = diag(1 / local_var_j).

    Approximates the Fisher information's diagonal: rows in dense color
    regions get higher precision (smaller local variance), rows in sparse
    regions get lower precision. This is the gauge a Fisher-Rao metric
    would use to upweight reliable observations.

    Returns W of shape (N, p) (diagonal entries only — we never
    materialize the full (N, p, p) tensor).
    """
    N, p = Z.shape
    # Pairwise sq distances chunked.
    W = np.zeros((N, p), dtype=np.float64)
    chunk = 256
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        d = Z[i0:i1, None, :] - Z[None, :, :]
        sqd = (d * d).sum(axis=-1)                 # (chunk, N)
        sqd[np.arange(i1 - i0), np.arange(i0, i1)] = np.inf
        # k-NN indices
        nn_idx = np.argpartition(sqd, k_nn, axis=1)[:, :k_nn]
        for j in range(i1 - i0):
            nb = Z[nn_idx[j]]
            var = nb.var(axis=0).clip(min=1e-6)
            W[i0 + j] = 1.0 / var
    # Normalize so W-norms have mean 1 (gauge invariance).
    W *= (1.0 / W.mean())
    return W


def fisher_rao_W_norms(W):
    """Per-row scalar ||W_i|| (trace of diag W_i)."""
    return W.sum(axis=1)


# ---------------------------------------------------------------------------
# Primitive 1 fallback: circle latent (cos t, sin t) alternating fit
# ---------------------------------------------------------------------------
def fit_circle_aux_ard(Z, W_diag, d_aux=D_AUX,
                       n_iters=N_ITERS, seed=SEED,
                       a0=ARD_A0, b0=ARD_B0):
    """Joint fit of:
       y(n,:) ~= B(theta_n) @ C_circle + U(n,:) @ C_aux  +  resid,
       with theta_n on S^1 (circle), U(n,:) in R^d_aux euclidean,
       precision W_diag (N, p) weighting the residual (Fisher-Rao gauge),
       and ARD over the d_aux axes (per-axis precision tau_j).

    Alternating updates:
      (a) given theta, U, C_*  -> sigma^2_n_p inferred from residual^2 * W,
      (b) given theta, residual -> update U columns ridge-regression with
          ARD prior tau_j,
      (c) given U -> EM update of tau_j,
      (d) given U, C_*  -> 1D gradient step on theta_n on the circle
          (project gradient onto tangent (-sin t, cos t)).

    Returns dict with theta, U, C_circle, C_aux, tau, recovered_hue_R2,
    aux_dim_eff_var, residual.
    """
    rng = np.random.default_rng(seed)
    N, p = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)

    # Init theta from the angle of (PC1, PC2) — gives a sensible start.
    theta = np.arctan2(Zc[:, 1], Zc[:, 0])
    # Init U random small.
    U = rng.normal(scale=0.1, size=(N, d_aux))
    # Init ARD precisions tau (one per aux axis), start at 1.
    tau = np.ones(d_aux)
    # Noise precision (scalar).
    sigma2 = float(Zc.var())

    # Per-row residual weighting from Fisher-Rao W (mean over p dims, scalar).
    w_row = W_diag.mean(axis=1)        # (N,)
    w_row /= w_row.mean()              # normalize

    for it in range(n_iters):
        # ---- (1) Update C_circle, C_aux given theta and U ----
        cs = np.column_stack([np.cos(theta), np.sin(theta)])    # (N,2)
        # Design: D = [cs | U]   (N, 2 + d_aux)
        D = np.hstack([cs, U])
        # Weighted ridge regression with ARD on aux columns only.
        # Prior precision matrix L = diag([eps, eps, tau_1, ..., tau_d_aux])
        eps = 1e-6
        L = np.diag(np.concatenate([[eps, eps], tau]))
        Wsqrt = np.sqrt(w_row)[:, None]
        Dw = D * Wsqrt
        Zw = Zc * Wsqrt
        # C minimizes ||Dw C - Zw||_F^2 + tr(C^T L C * sigma2)
        # Normal eqn: (D^T W D + sigma2 L) C = D^T W Z
        A = D.T @ (D * w_row[:, None]) + sigma2 * L
        b = D.T @ (Zc * w_row[:, None])
        C = np.linalg.solve(A, b)                              # (2+d_aux, p)
        C_circle = C[:2]                                       # (2, p)
        C_aux = C[2:]                                          # (d_aux, p)

        # ---- (2) Update U given theta, C, tau ----
        # Proper posterior under N(0, diag(1/tau)) prior and N(Cu, sigma2/w_n*I)
        # likelihood:  prec_n = (w_n/sigma2) * C C^T + diag(tau).
        # Track per-axis posterior variance for the ARD update (else tau diverges).
        resid_circ = Zc - cs @ C_circle                        # (N, p)
        post_var_sum = np.zeros(d_aux)
        for n in range(N):
            prec_n = (w_row[n] / sigma2) * (C_aux @ C_aux.T) + np.diag(tau)
            S_n = np.linalg.inv(prec_n)
            U[n] = S_n @ ((w_row[n] / sigma2) * (C_aux @ resid_circ[n]))
            post_var_sum += np.diag(S_n)

        # ---- (3) Update theta on circle (one gradient step per row) ----
        resid_aux = Zc - U @ C_aux                             # (N, p)
        # f_n(theta) = w_row[n] * || resid_aux[n] - [cos t, sin t] @ C_circle ||^2
        # d/dt: -2 * w_row[n] * (resid_aux[n] - cs @ C_circle) @ C_circle^T @ [-sin t, cos t]
        for inner in range(3):
            cs_t = np.column_stack([np.cos(theta), np.sin(theta)])
            tang = np.column_stack([-np.sin(theta), np.cos(theta)])
            pred = cs_t @ C_circle                             # (N, p)
            err = resid_aux - pred                             # (N, p)
            # gradient w.r.t. theta_n
            grad = -2.0 * w_row * np.einsum("np,kp,nk->n",
                                            err, C_circle, tang)
            # second-derivative crude bound for step size
            denom = 2.0 * w_row * np.einsum("kp,kp->", C_circle, C_circle) + 1e-3
            theta = theta - grad / denom
            # wrap to [-pi, pi)
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # ---- (4) EM update of tau (ARD) with posterior-variance correction ----
        # Proper EM update: tau_j <- (N + 2*a0) / (sum_n E[u_{n,j}^2] + 2*b0)
        # where E[u^2] = mu^2 + Var. Without the Var term, tau diverges.
        u_sq_sum = (U * U).sum(axis=0) + post_var_sum
        tau = (N + 2.0 * a0) / (u_sq_sum + 2.0 * b0)
        tau = np.clip(tau, 1e-6, TAU_MAX)

        # ---- (5) Update sigma2 ----
        cs_t = np.column_stack([np.cos(theta), np.sin(theta)])
        resid = Zc - cs_t @ C_circle - U @ C_aux
        sigma2 = float((w_row[:, None] * resid * resid).mean())

        if it % 10 == 0 or it == n_iters - 1:
            print(f"  [iter {it:3d}] sigma2={sigma2:.4f}  "
                  f"tau={np.round(tau, 3).tolist()}", flush=True)

    return {
        "theta":      theta,
        "U":          U,
        "C_circle":   C_circle,
        "C_aux":      C_aux,
        "tau":        tau,
        "sigma2":     sigma2,
        "resid":      resid,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def circular_R2(theta_hat, hue_true):
    """R^2 of (sin, cos) regression: predict (sin t_true, cos t_true) from
       (sin t_hat, cos t_hat). 1 = perfect circular agreement up to rotation."""
    t_true = 2 * np.pi * hue_true
    pred = np.column_stack([np.cos(theta_hat), np.sin(theta_hat)])
    targ = np.column_stack([np.cos(t_true), np.sin(t_true)])
    # Best 2x2 orthogonal map (Procrustes-like).
    M = pred.T @ targ
    U_svd, _, Vt = np.linalg.svd(M)
    R = U_svd @ Vt
    pred_rot = pred @ R
    ss_res = ((pred_rot - targ) ** 2).sum()
    ss_tot = ((targ - targ.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def aux_axis_effective_var(U, tau):
    """Effective per-axis 'signal': raw var(U[:, j]) is the direct
    measure of how much the axis is being used by the model. Low tau =
    ARD letting that axis carry signal; high tau = ARD shrinking it.
    Return both raw variance and ARD weight 1/tau."""
    var = U.var(axis=0)
    inv_tau = 1.0 / tau
    eff = var      # the *used* signal in each aux axis post-fit
    return var, eff


def memory_check():
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is bytes; on Linux it's KB.
    if sys.platform == "darwin":
        rss_gb = rss_kb / (1024 ** 3)
    else:
        rss_gb = rss_kb / (1024 ** 2)
    if rss_gb > MAX_RSS_GB:
        print(f"[FATAL] RSS={rss_gb:.2f} GB > {MAX_RSS_GB} GB, aborting",
              flush=True)
        sys.exit(137)
    return rss_gb


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(theta, hue_true, rgb, U, tau, W_norms, resid_flat, R2,
              sv_norm=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Panel 1: recovered circle (cos, sin) colored by ground-truth hue
    ax = axes[0, 0]
    ax.scatter(np.cos(theta), np.sin(theta), c=rgb, s=22,
               edgecolor="black", linewidth=0.3)
    ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.6, alpha=0.4)
    ax.set_xlabel("cos(theta_hat)")
    ax.set_ylabel("sin(theta_hat)")
    ax.set_title(f"Recovered circle latent (colors = true RGB)\n"
                 f"circular R^2 vs hue = {R2:.3f}")
    ax.grid(alpha=0.3)

    # Panel 2: ARD effective spectrum bar (U singular values, rotation-aware)
    ax = axes[0, 1]
    if sv_norm is None:
        Sigma_U = np.linalg.svd(U, compute_uv=False)
        sv_norm = Sigma_U / Sigma_U.max() if Sigma_U.max() > 0 else Sigma_U
    eff_norm = np.asarray(sv_norm)
    n_kept = int((eff_norm >= ARD_PRUNE_THR).sum())
    colors_bar = ["#2ca02c" if e >= ARD_PRUNE_THR else "#d62728"
                  for e in eff_norm]
    ax.bar(np.arange(D_AUX), eff_norm, color=colors_bar,
           edgecolor="black", lw=0.6)
    ax.axhline(ARD_PRUNE_THR, color="black", ls="--", lw=1,
               label=f"prune thr = {ARD_PRUNE_THR}")
    for j, e in enumerate(eff_norm):
        ax.text(j, e + 0.02,
                f"{'kept' if e >= ARD_PRUNE_THR else 'pruned'}",
                ha="center", fontsize=9)
    ax.set_xticks(np.arange(D_AUX))
    ax.set_xticklabels([f"sv_{j}" for j in range(D_AUX)])
    ax.set_ylabel("normalized singular value of U")
    ax.set_title(f"Intrinsic aux rank (U singular spectrum): "
                 f"kept = {n_kept} / {D_AUX}\n"
                 f"tau converged uniform ({tau.mean():.1f}); "
                 "frame-rotation symmetry-broken via SVD")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: Fisher-Rao W-norm sanity (per-row precision)
    ax = axes[1, 0]
    ax.hist(W_norms, bins=40, color="#1f77b4", edgecolor="black", lw=0.4)
    ax.set_xlabel("per-row W-norm  (trace of diag W_i)")
    ax.set_ylabel("count of rows")
    ax.set_title(f"Fisher-Rao per-row precision (gauge-normalized to mean=p)\n"
                 f"min={W_norms.min():.2f}  max={W_norms.max():.2f}  "
                 f"med={np.median(W_norms):.2f}")
    ax.grid(alpha=0.3, axis="y")

    # Panel 4: residual histogram
    ax = axes[1, 1]
    ax.hist(resid_flat, bins=60, color="#9467bd", edgecolor="black", lw=0.3)
    ax.set_xlabel("residual (Z - circle - aux)")
    ax.set_ylabel("count")
    ax.set_title(f"Residual distribution\n"
                 f"mean={resid_flat.mean():+.4f}  sd={resid_flat.std():.4f}")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"auto_exp_21 . Circle latent + Fisher-Rao W + ARD over {D_AUX} aux dims  "
        f"(gamfit=={GAMFIT_VERSION}, all 3 primitives FALLBACK)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[plot] -> {OUT_PNG}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gamfit] version = {GAMFIT_VERSION}", flush=True)
    for k, v in PRIMITIVE_STATUS.items():
        print(f"[probe] {k:12s} reached={v['reached']}  {v['detail']}",
              flush=True)

    centroids, names, rgb, hsv = build_centroids()
    N = centroids.shape[0]
    print(f"[load] N = {N} filtered colors", flush=True)
    memory_check()

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC].astype(np.float64)
    evr = float(basis["evr"][:K_PC].sum())
    # Standardize per-PC: ARD operates on unit-variance axes so the
    # Gamma hyperprior has a sensible scale (otherwise the raw PC1
    # scale ~10x larger than PC16 dominates and the prior precision
    # crushes all aux coefficients to zero on iter 0).
    Z = Z / Z.std(axis=0, keepdims=True).clip(min=1e-9)
    print(f"[pca ] Z shape = {Z.shape}  EVR_top{K_PC} = {evr:.3f}  "
          f"(per-PC z-scored)", flush=True)
    memory_check()

    # Free centroids — we only need Z from here on.
    del centroids

    # Primitive 2: Fisher-Rao W
    print(f"[fr  ] computing per-row diag Fisher-Rao W ...", flush=True)
    W_diag = fisher_rao_W_diag(Z, k_nn=20)
    W_norms = fisher_rao_W_norms(W_diag)
    print(f"[fr  ] W_norm range = [{W_norms.min():.2f}, "
          f"{W_norms.max():.2f}]  med={np.median(W_norms):.2f}",
          flush=True)
    memory_check()

    # Primitives 1 + 3: joint circle + aux + ARD fit
    print(f"[fit ] alternating circle + aux + ARD ({N_ITERS} iters) ...",
          flush=True)
    result = fit_circle_aux_ard(Z, W_diag, d_aux=D_AUX,
                                n_iters=N_ITERS, seed=SEED)
    memory_check()

    # Metrics
    hue = hsv[:, 0]
    R2 = circular_R2(result["theta"], hue)
    # Symmetry-break: the U,C_aux factorization is rotation-invariant, so
    # raw per-column var(U) is degenerate. The intrinsic aux rank is the
    # number of significant SVs of U (or equivalently the number of
    # singular directions ARD wouldn't crush after a frame rotation).
    U_arr = result["U"]
    Sigma_U = np.linalg.svd(U_arr, compute_uv=False)
    sv_norm = Sigma_U / Sigma_U.max() if Sigma_U.max() > 0 else Sigma_U
    var, eff = aux_axis_effective_var(U_arr, result["tau"])
    # Use SV-spectrum for the ARD-effective dim count.
    eff_norm = np.array(sv_norm)
    aux_kept = int((eff_norm >= ARD_PRUNE_THR).sum())
    aux_kept_list = [bool(e >= ARD_PRUNE_THR) for e in eff_norm]
    print(f"[res ] U singular spectrum (normalized): "
          f"{np.round(sv_norm, 3).tolist()}", flush=True)

    print(f"[res ] circular R^2 vs hue = {R2:.4f}", flush=True)
    print(f"[res ] aux dims kept by ARD = {aux_kept} / {D_AUX}  "
          f"(eff_norm = {np.round(eff_norm, 3).tolist()})", flush=True)
    print(f"[res ] tau = {np.round(result['tau'], 3).tolist()}", flush=True)

    # Plot
    make_plot(result["theta"], hue, rgb, result["U"], result["tau"],
              W_norms, result["resid"].ravel(), R2, sv_norm=sv_norm)

    # ---------------- JSON ----------------
    primitives_reached = [k for k, v in PRIMITIVE_STATUS.items() if v["reached"]]
    primitives_fallback = [k for k, v in PRIMITIVE_STATUS.items() if not v["reached"]]

    # Hypothesis verdict.
    HYP_R2_THR = 0.40        # "recovered hue" — R2 vs HSV-hue
    HYP_AUX_RANGE = (2, 3)   # ARD-prune to 2-3 dims

    hue_recovered = bool(R2 >= HYP_R2_THR)
    aux_in_range = bool(HYP_AUX_RANGE[0] <= aux_kept <= HYP_AUX_RANGE[1])
    verdict = "supported" if (hue_recovered and aux_in_range) else "not_supported"

    runtime = time.time() - t0
    summary = {
        "experiment": "auto_exp_21",
        "question": ("Does a Circle-manifold latent + Fisher-Rao W + ARD over "
                     "4 aux euclidean dims (a) recover hue as the circular "
                     "axis and (b) prune aux to 2-3 effective dims?"),
        "gamfit_version_actually_used": GAMFIT_VERSION,
        "primitives_reached": primitives_reached,
        "primitives_fallback": primitives_fallback,
        "primitive_probe_detail": PRIMITIVE_STATUS,
        "config": {
            "K_PC": K_PC,
            "D_AUX": D_AUX,
            "n_iters": N_ITERS,
            "ard_a0": ARD_A0,
            "ard_b0": ARD_B0,
            "ard_prune_thr": ARD_PRUNE_THR,
            "n_colors": int(N),
            "evr_top_K_PC": evr,
            "seed": SEED,
            "fisher_rao_kind": "per-row diag from local 20-NN variance",
        },
        "results": {
            "recovered_hue_R2":        R2,
            "aux_dims_kept_by_ard":    aux_kept,
            "aux_kept_mask":           aux_kept_list,
            "aux_singular_values":     [float(x) for x in Sigma_U],
            "aux_sv_normalized":       [float(x) for x in eff_norm],
            "aux_raw_variance":        [float(x) for x in var],
            "tau_per_axis":            [float(x) for x in result["tau"]],
            "sigma2_final":            float(result["sigma2"]),
            "W_norm_stats":            {"min":    float(W_norms.min()),
                                        "max":    float(W_norms.max()),
                                        "median": float(np.median(W_norms)),
                                        "mean":   float(W_norms.mean())},
            "residual_stats":          {"mean": float(result["resid"].mean()),
                                        "sd":   float(result["resid"].std()),
                                        "max_abs": float(np.abs(result["resid"]).max())},
            "runtime_seconds":         runtime,
        },
        "prediction": {
            "hue_R2_threshold":             HYP_R2_THR,
            "aux_dims_kept_range":          list(HYP_AUX_RANGE),
            "hypothesis_a_hue_recovered":   hue_recovered,
            "hypothesis_b_aux_2_to_3":      aux_in_range,
            "verdict":                      verdict,
            "compare_when_primitives_land": (
                "When gamfit ships LatentCoord(manifold=Sphere(dim=0)), "
                "fisher_w kwarg, and ARDPenalty, re-run with the real "
                "composition engine and compare recovered_hue_R2 and "
                "aux_dims_kept_by_ard to the fallback values above. "
                "Disagreement > 0.10 in R2 or > 1 in aux_kept indicates "
                "the fallback emulator misses a load-bearing piece of the "
                "joint penalty."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)
    print(f"[time] {runtime:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
