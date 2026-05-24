"""auto_exp_30 — SOFT-emulator retest of auto_exp_23 (Ortho × ARD on cogito L40).

Correction motivation
---------------------
auto_exp_23 paired OrthogonalityPenalty + ARDPenalty on the cogito L40 color
manifold, but fell back to a per-iter QR-orthonormalize step on the aux block U.
QR is the `w → ∞` limit of the real penalty `(w/2)·‖T^T T − I‖²_F`: it forces
both `T^T T = I` and equal radii `‖t_j‖² = 1`. The ARD EM update
`τ_j ∝ N / ‖u_j‖²` is then necessarily uniform — the pairing test is null by
construction.

Fix
---
This experiment replaces the QR projection with a SOFT gradient-step emulator
on the real Frobenius objective, sweeps the finite weight `w_ortho`, picks the
best on held-out 5-fold-by-color CV, and asks whether ARD now prunes
4 → {2, 3} aux dims (composition_engine.md §4(c)).

Hypotheses (preregistered, strict TRUE/FALSE)
---------------------------------------------
(a) aux_dims_kept at best w_ortho ∈ {2, 3}
(b) recovered_hue_R2 at best w_ortho ≥ 0.45
(c) ‖T^T T − I‖_F at convergence is small (< 0.5) but NON-ZERO (> 1e-3)
    — evidence the emulator behaves softly, not as a QR projection.

Outputs
-------
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_30.png    (4-panel)
- runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_30.json   (full results)
"""

from __future__ import annotations

import colorsys
import json
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
# Config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_30.png"
OUT_JSON = OUT_DIR / "auto_exp_30.json"

K_PC          = 16
D_AUX         = 4                     # match auto_exp_23 exactly
N_ITERS       = 60
ARD_A0        = 1.0
ARD_B0        = 1.0
TAU_MAX       = 1e3
ARD_PRUNE_THR = 0.10
SEED          = 0
MAX_RSS_GB    = 6.0

# SOFT Ortho gradient-step config (this is the auto_exp_23 fix).
ORTH_W_SWEEP  = [0.0, 0.1, 1.0, 10.0]
ORTH_LR       = 0.05                  # gradient step on T from joint loss
ORTH_RAMP_END = 20                    # linear ramp 0 → w_ortho across iters
N_CV_FOLDS    = 5


# ---------------------------------------------------------------------------
# gamfit probe (will fall back; we record what version we ran against)
# ---------------------------------------------------------------------------
import gamfit
GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")

def _probe():
    status = {}
    try:
        from gamfit import OrthogonalityPenalty  # noqa
        status["OrthogonalityPenalty"] = {"reached": True,
                                          "detail": "imported from gamfit"}
    except Exception as e:
        status["OrthogonalityPenalty"] = {
            "reached": False,
            "detail": f"{type(e).__name__}: {e}; "
                      "fallback: SOFT gradient-step on (w/2)*||T^T T - I||^2_F.",
        }
    try:
        from gamfit._penalties import ARDPenalty  # noqa
        status["ARDPenalty"] = {"reached": True,
                                "detail": "imported from gamfit._penalties"}
    except Exception as e:
        status["ARDPenalty"] = {
            "reached": False,
            "detail": f"{type(e).__name__}: {e}; fallback: per-axis tau EM.",
        }
    return status

PRIMITIVE_STATUS = _probe()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_centroids():
    print(f"[load] mmap {HARVEST}", flush=True)
    X = np.load(HARVEST, mmap_mode="r")
    n_total, H = X.shape
    print(f"[load] X shape = ({n_total}, {H})", flush=True)
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
# Fisher-Rao per-row diagonal precision
# ---------------------------------------------------------------------------
def fisher_rao_W_diag(Z, k_nn=20):
    N, p = Z.shape
    W = np.zeros((N, p), dtype=np.float64)
    chunk = 256
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        d = Z[i0:i1, None, :] - Z[None, :, :]
        sqd = (d * d).sum(axis=-1)
        sqd[np.arange(i1 - i0), np.arange(i0, i1)] = np.inf
        nn_idx = np.argpartition(sqd, k_nn, axis=1)[:, :k_nn]
        for j in range(i1 - i0):
            nb = Z[nn_idx[j]]
            var = nb.var(axis=0).clip(min=1e-6)
            W[i0 + j] = 1.0 / var
    W *= (1.0 / W.mean())
    return W


# ---------------------------------------------------------------------------
# SOFT-emulator joint fit
# ---------------------------------------------------------------------------
def fit_circle_aux_soft(Z, W_diag, *,
                        w_ortho, d_aux=D_AUX, n_iters=N_ITERS,
                        seed=SEED, a0=ARD_A0, b0=ARD_B0,
                        lr=ORTH_LR, train_mask=None):
    """Joint fit with SOFT (w/2)*||T^T T - I||^2_F + ARD per-axis penalty.

    Joint per-iter objective on T (= U here, since we factor out C_circle):
       L(U) = (1/2) ||W^{1/2} (Z_c - cs C_circle - U C_aux)||^2
            + (w/2)*||U^T U - I||^2_F
            + (1/2) sum_k tau_k ||U[:, k]||^2

    Gradient (for the U block):
       dL/dU = -W * R @ C_aux^T + 2*w*U(U^T U - I) + U @ diag(tau)
    where R = Z_c - cs C_circle - U C_aux.

    train_mask : boolean array length N. If given, only rows where True
    contribute to the gradient updates; the held-out rows still receive a U
    update each iter so we can score them later for CV.
    """
    rng = np.random.default_rng(seed)
    N, p = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)

    if train_mask is None:
        train_mask = np.ones(N, dtype=bool)
    tr = train_mask
    w_row = W_diag.mean(axis=1)
    w_row /= w_row.mean()
    w_row_tr = w_row.copy()
    w_row_tr[~tr] = 0.0  # mask out CV-held-out rows for the linear solve

    theta = np.arctan2(Zc[:, 1], Zc[:, 0])
    U = rng.normal(scale=0.1, size=(N, d_aux))
    tau = np.ones(d_aux)
    sigma2 = float(Zc.var())

    history = {
        "tau":   [],
        "u_var": [],
        "sigma2":[],
        "frob":  [],
        "w_eff": [],
    }

    for it in range(n_iters):
        alpha = min(1.0, (it + 1) / float(ORTH_RAMP_END))
        w_eff = float(w_ortho) * alpha

        # (1) Linear solve for C_circle and C_aux on TRAIN rows only.
        cs = np.column_stack([np.cos(theta), np.sin(theta)])
        D = np.hstack([cs, U])
        eps = 1e-6
        L_diag = np.concatenate([[eps, eps], tau])
        A = D.T @ (D * w_row_tr[:, None]) + sigma2 * np.diag(L_diag)
        b = D.T @ (Zc * w_row_tr[:, None])
        C = np.linalg.solve(A, b)
        C_circle = C[:2]
        C_aux = C[2:]

        # (2a) Per-row closed-form U posterior from (data + ARD).
        # prec_n = (w_row_n / sigma2) * C_aux @ C_aux^T + diag(tau)
        # mean_n = prec_n^-1 @ (w_row_n / sigma2) * C_aux @ resid_circ_n
        resid_circ = Zc - cs @ C_circle
        CtC = C_aux @ C_aux.T  # (d_aux, d_aux)
        post_var_sum = np.zeros(d_aux)
        for n in range(N):
            prec_n = (w_row[n] / sigma2) * CtC + np.diag(tau)
            S_n = np.linalg.inv(prec_n)
            U[n] = S_n @ ((w_row[n] / sigma2) * (C_aux @ resid_circ[n]))
            post_var_sum += np.diag(S_n)

        # (2b) SOFT Ortho correction — iterative gradient steps on
        # L_ortho(U) = (w_eff/2)*||U^T U - I||^2_F.
        # Gradient: 2 * w_eff * U @ (U^T U - I).
        # Local quadratic step size: 1 / (2 * w_eff * ||3 U^T U - I||_2 + eps).
        # This is a true SOFT penalty: equilibrium is NOT U^T U = I (ARD pulls
        # against it), and frob_final is finite even at w_ortho -> max sweep.
        if w_eff > 0:
            for _ in range(30):
                UtU = U.T @ U
                M = UtU - np.eye(d_aux)
                grad = 2.0 * w_eff * (U @ M)
                # Lipschitz-ish stepsize: dominant Hessian eigenvalue along U
                H_eig = 2.0 * w_eff * (3.0 * np.abs(UtU).sum()
                                       + d_aux)
                step = 1.0 / (H_eig + 1.0)
                U = U - step * grad

        # (3) Theta gradient steps on the circle (TRAIN rows only).
        resid_aux = Zc - U @ C_aux
        for inner in range(3):
            cs_t = np.column_stack([np.cos(theta), np.sin(theta)])
            tang = np.column_stack([-np.sin(theta), np.cos(theta)])
            pred = cs_t @ C_circle
            err = resid_aux - pred
            grad = -2.0 * w_row_tr * np.einsum(
                "np,kp,nk->n", err, C_circle, tang)
            denom_t = 2.0 * w_row_tr * np.einsum(
                "kp,kp->", C_circle, C_circle) + 1e-3
            theta = theta - grad / denom_t
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # (4) ARD EM update on tau, using TRAIN rows; include posterior
        # variance contribution so high-tau axes don't runaway-collapse.
        u_tr = U[tr]
        u_sq_sum = (u_tr * u_tr).sum(axis=0) + post_var_sum
        N_tr = int(tr.sum())
        tau = (N_tr + 2.0 * a0) / (u_sq_sum + 2.0 * b0)
        tau = np.clip(tau, 1e-6, TAU_MAX)

        # (5) sigma2 update on TRAIN rows.
        cs_t = np.column_stack([np.cos(theta), np.sin(theta)])
        resid_full = Zc - cs_t @ C_circle - U @ C_aux
        sigma2 = float((w_row_tr[:, None] * resid_full * resid_full).sum()
                       / max(1.0, w_row_tr.sum() * p))

        # (6) Track diagnostics.
        frob = float(np.linalg.norm(U.T @ U - np.eye(d_aux), ord="fro"))
        history["tau"].append(tau.tolist())
        history["u_var"].append(U.var(axis=0).tolist())
        history["sigma2"].append(sigma2)
        history["frob"].append(frob)
        history["w_eff"].append(w_eff)

        if it % 15 == 0 or it == n_iters - 1:
            print(f"  [w={w_ortho:>5} iter {it:3d}] "
                  f"sigma2={sigma2:.4f}  "
                  f"frob={frob:.3f}  "
                  f"tau={np.round(tau, 2).tolist()}  "
                  f"var(U)={np.round(U.var(axis=0), 4).tolist()}",
                  flush=True)

    return {
        "theta": theta, "U": U,
        "C_circle": C_circle, "C_aux": C_aux,
        "tau": tau, "sigma2": sigma2, "resid": resid_full,
        "history": history,
        "frob_final": history["frob"][-1],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def circular_R2(theta_hat, hue_true):
    t_true = 2 * np.pi * hue_true
    pred = np.column_stack([np.cos(theta_hat), np.sin(theta_hat)])
    targ = np.column_stack([np.cos(t_true), np.sin(t_true)])
    M = pred.T @ targ
    U_svd, _, Vt = np.linalg.svd(M)
    R = U_svd @ Vt
    pred_rot = pred @ R
    ss_res = ((pred_rot - targ) ** 2).sum()
    ss_tot = ((targ - targ.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def aux_dims_kept(U, thr=ARD_PRUNE_THR):
    var = U.var(axis=0)
    if var.max() <= 0:
        return 0, var, np.zeros_like(var)
    norm = var / var.max()
    return int((norm >= thr).sum()), var, norm


def cv_holdout_loss(res, Z, W_diag, test_mask):
    """MSE on the held-out fold using the fitted (theta, U, C_circle, C_aux)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cs = np.column_stack([np.cos(res["theta"]), np.sin(res["theta"])])
    pred = cs @ res["C_circle"] + res["U"] @ res["C_aux"]
    err = (Zc - pred)[test_mask]
    return float((err * err).mean())


def memory_check():
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / (1024 ** 3) if sys.platform == "darwin" else rss_kb / (1024 ** 2)
    if rss_gb > MAX_RSS_GB:
        print(f"[FATAL] RSS={rss_gb:.2f} GB > {MAX_RSS_GB} GB, aborting",
              flush=True)
        sys.exit(137)
    return rss_gb


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(sweep, best_idx, hue, rgb):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: per-axis variance of U across w_ortho sweep (grouped bars).
    ax = axes[0, 0]
    x = np.arange(D_AUX)
    width = 0.8 / len(sweep)
    cmap = plt.get_cmap("viridis")
    for i, item in enumerate(sweep):
        var = item["res"]["U"].var(axis=0)
        ax.bar(x + (i - (len(sweep) - 1) / 2) * width, var, width,
               label=f"w={item['w']}", color=cmap(i / max(1, len(sweep) - 1)),
               edgecolor="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"aux_{j}" for j in range(D_AUX)])
    ax.set_ylabel("var(U[:, j])")
    ax.set_title("Per-axis aux variance across w_ortho sweep\n"
                 "(SOFT emulator: w_ortho finite, so ARD can differentiate)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Panel 2: recovered hue R^2 bars across sweep.
    ax = axes[0, 1]
    ws = [item["w"] for item in sweep]
    r2s = [item["R2"] for item in sweep]
    keps = [item["kept"] for item in sweep]
    bar_colors = ["#2ca02c" if i == best_idx else "#888888"
                  for i in range(len(sweep))]
    ax.bar([str(w) for w in ws], r2s, color=bar_colors,
           edgecolor="black", lw=0.5)
    for i, (r2, k) in enumerate(zip(r2s, keps)):
        ax.text(i, r2 + 0.01, f"R2={r2:.3f}\nkept={k}",
                ha="center", fontsize=8)
    ax.axhline(0.45, ls="--", color="red", lw=1, label="hyp_b thr=0.45")
    ax.set_xlabel("w_ortho")
    ax.set_ylabel("recovered_hue_R^2")
    ax.set_title("Recovered hue R^2 vs w_ortho (best=green)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Panel 3: Frobenius distance trace per w.
    ax = axes[1, 0]
    for i, item in enumerate(sweep):
        ax.plot(item["res"]["history"]["frob"],
                label=f"w={item['w']}",
                color=cmap(i / max(1, len(sweep) - 1)))
    ax.axhline(0.5, ls="--", color="red", lw=1, label="hyp_c upper=0.5")
    ax.axhline(1e-3, ls="--", color="blue", lw=1, label="hyp_c lower=1e-3")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("||U^T U - I||_F")
    ax.set_title("Frobenius distance trace (SOFT emulator should NOT hit 0)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 4: ARD posterior tau at best w.
    ax = axes[1, 1]
    best = sweep[best_idx]
    tau = best["res"]["tau"]
    ax.bar(np.arange(D_AUX), tau, color="#2ca02c",
           edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(D_AUX))
    ax.set_xticklabels([f"aux_{j}" for j in range(D_AUX)])
    ax.set_ylabel("posterior tau_j (higher = shrunk)")
    ax.set_title(f"ARD tau at best w_ortho={best['w']}\n"
                 f"R^2={best['R2']:.3f}  kept={best['kept']}  "
                 f"frob_final={best['res']['frob_final']:.3g}")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"auto_exp_30 . SOFT Ortho x ARD pairing (fix for auto_exp_23 QR artifact) "
        f"gamfit=={GAMFIT_VERSION} "
        f"orth_reached={PRIMITIVE_STATUS['OrthogonalityPenalty']['reached']} "
        f"ard_reached={PRIMITIVE_STATUS['ARDPenalty']['reached']}",
        fontsize=11)
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
        print(f"[probe] {k:24s} reached={v['reached']}  {v['detail']}",
              flush=True)

    centroids, names, rgb, hsv = build_centroids()
    N = centroids.shape[0]
    print(f"[load] N kept = {N}", flush=True)
    memory_check()

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC].astype(np.float64)
    evr = float(basis["evr"][:K_PC].sum())
    Z = Z / Z.std(axis=0, keepdims=True).clip(min=1e-9)
    print(f"[pca ] Z shape = {Z.shape}  EVR_top{K_PC} = {evr:.3f}",
          flush=True)
    memory_check()
    del centroids

    print(f"[fr  ] Fisher-Rao W_diag (k_nn=20) ...", flush=True)
    W_diag = fisher_rao_W_diag(Z, k_nn=20)
    memory_check()

    hue = hsv[:, 0]

    # 5-fold CV indices "by color" (rows are already per-color centroids).
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    folds = np.array_split(perm, N_CV_FOLDS)

    sweep = []
    for w in ORTH_W_SWEEP:
        print(f"\n[sweep] === w_ortho = {w} ===", flush=True)

        # CV: per-fold holdout loss.
        cv_losses = []
        for fi, fold in enumerate(folds):
            mask = np.ones(N, dtype=bool)
            mask[fold] = False
            res_cv = fit_circle_aux_soft(Z, W_diag, w_ortho=w,
                                         seed=SEED + fi,
                                         train_mask=mask,
                                         n_iters=max(30, N_ITERS // 2))
            test_mask = ~mask
            cv_losses.append(cv_holdout_loss(res_cv, Z, W_diag, test_mask))
            memory_check()
        cv_mean = float(np.mean(cv_losses))
        print(f"  [cv ] w={w} mean_holdout_mse = {cv_mean:.5f}", flush=True)

        # Full fit on all data for final diagnostics.
        res = fit_circle_aux_soft(Z, W_diag, w_ortho=w, seed=SEED)
        R2 = circular_R2(res["theta"], hue)
        k, raw_var, norm_var = aux_dims_kept(res["U"])
        sweep.append({
            "w": float(w),
            "res": res,
            "R2": float(R2),
            "kept": int(k),
            "raw_var": [float(v) for v in raw_var],
            "norm_var": [float(v) for v in norm_var],
            "cv_mean_holdout_mse": cv_mean,
            "cv_losses": [float(x) for x in cv_losses],
        })
        memory_check()

    # Pick best by CV.
    best_idx = int(np.argmin([s["cv_mean_holdout_mse"] for s in sweep]))
    best = sweep[best_idx]
    print(f"\n[best] w_ortho={best['w']}  R2={best['R2']:.4f}  "
          f"kept={best['kept']}  frob_final={best['res']['frob_final']:.4g}",
          flush=True)

    make_plot(sweep, best_idx, hue, rgb)

    # Hypothesis verdicts (strict TRUE/FALSE).
    hyp_a = bool(2 <= best["kept"] <= 3)
    hyp_b = bool(best["R2"] >= 0.45)
    frob_best = float(best["res"]["frob_final"])
    hyp_c = bool((frob_best < 0.5) and (frob_best > 1e-3))

    runtime = time.time() - t0

    out = {
        "experiment": "auto_exp_30",
        "question": ("Does a SOFT (finite-w) Frobenius-Ortho + ARD pairing "
                     "fix the auto_exp_23 QR fallback artifact and prune "
                     "aux 4 -> {2,3} on cogito L40?"),
        "gamfit_version": GAMFIT_VERSION,
        "primitives_reached": [k for k, v in PRIMITIVE_STATUS.items()
                               if v["reached"]],
        "primitives_fallback": [k for k, v in PRIMITIVE_STATUS.items()
                                if not v["reached"]],
        "primitive_probe_detail": PRIMITIVE_STATUS,
        "config": {
            "K_PC": K_PC, "D_AUX": D_AUX, "n_iters": N_ITERS,
            "ard_a0": ARD_A0, "ard_b0": ARD_B0,
            "ard_prune_thr": ARD_PRUNE_THR,
            "orth_ramp_end_iter": ORTH_RAMP_END,
            "orth_lr": ORTH_LR,
            "w_ortho_sweep": ORTH_W_SWEEP,
            "n_cv_folds": N_CV_FOLDS,
            "seed": SEED, "n_colors": int(N),
            "evr_top_K_PC": evr,
        },
        "w_ortho_sweep": [
            {
                "w_ortho": s["w"],
                "recovered_hue_R2": s["R2"],
                "aux_dims_kept": s["kept"],
                "aux_raw_var": s["raw_var"],
                "aux_norm_var": s["norm_var"],
                "cv_mean_holdout_mse": s["cv_mean_holdout_mse"],
                "cv_losses_per_fold": s["cv_losses"],
                "frob_final": float(s["res"]["frob_final"]),
                "tau_final": [float(t) for t in s["res"]["tau"]],
                "sigma2_final": float(s["res"]["sigma2"]),
            }
            for s in sweep
        ],
        "best_w_ortho": best["w"],
        "best_results": {
            "recovered_hue_R2": best["R2"],
            "aux_dims_kept": best["kept"],
            "aux_raw_var": best["raw_var"],
            "aux_norm_var": best["norm_var"],
            "frobenius_distance_at_convergence": frob_best,
            "tau_final": [float(t) for t in best["res"]["tau"]],
            "sigma2_final": float(best["res"]["sigma2"]),
        },
        "hypotheses": {
            "a_aux_dims_kept_in_2_3":        hyp_a,
            "b_recovered_hue_R2_ge_0p45":    hyp_b,
            "c_frob_small_but_nonzero":      hyp_c,
            "frob_window_lo":                1e-3,
            "frob_window_hi":                0.5,
            "verdict_strict_all_true":       bool(hyp_a and hyp_b and hyp_c),
        },
        "runtime_seconds": runtime,
        "prediction_slot_for_v0_1_121_rerun": {
            "note": ("When gamfit >=0.1.121 exposes the real "
                     "OrthogonalityPenalty + ARDPenalty Rust primitives, "
                     "re-run with primitives_reached=['OrthogonalityPenalty',"
                     "'ARDPenalty'] and compare best_w_ortho, aux_dims_kept, "
                     "frobenius_distance_at_convergence against the soft "
                     "emulator slot here."),
            "expected_primitives": ["OrthogonalityPenalty", "ARDPenalty"],
            "expected_aux_dims_kept_range": [2, 3],
            "expected_recovered_hue_R2_ge": 0.45,
            "expected_frob_window": [1e-3, 0.5],
            "emulator_best_w_ortho":       best["w"],
            "emulator_aux_dims_kept":      best["kept"],
            "emulator_recovered_hue_R2":   best["R2"],
            "emulator_frob_final":         frob_best,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)
    print(f"[time] {runtime:.1f} s", flush=True)
    print(f"[hyp ] a(kept in 2..3)={hyp_a}  "
          f"b(R2>=0.45)={hyp_b}  c(frob in (1e-3, 0.5))={hyp_c}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
