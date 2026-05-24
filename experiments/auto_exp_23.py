"""auto_exp_23 - Circle + Fisher-Rao + (Orthogonality x ARD) on cogito L40.

FALSIFIABLE RETEST of auto_exp_21
---------------------------------
auto_exp_21 paired Circle + Fisher-Rao + ARD over 4 aux euclidean dims, and
found:
  recovered_hue_R2          = 0.565  (circle fallback)
  aux_dims_kept_by_ard      = 4 / 4  (ARD failed to prune)

The gamfit composition-engine proposal (sec 4(c) audit caveat) predicts
ARD-alone is rotation-invariant on a free aux frame, so the U-block can
absorb arbitrary in-plane rotations without budget cost. ARD therefore sees
4 roughly equal singular directions and shrinks none of them. The fix is
to PAIR ARD with a gauge-fixing penalty (OrthogonalityPenalty) that locks
the aux frame so per-axis ARD precision tau_j is now meaningful.

THIS EXPERIMENT runs three fits in the SAME inner loop framework:
  fit_a : ARD-only                       (control - reproduce auto_exp_21)
  fit_b : Orthogonality-only             (locks basis, no shrinkage)
  fit_c : Orthogonality + ARD paired     (ortho weight >= ard, ramped)

HYPOTHESES (preregistered):
  (a) recovered_hue_R2_c >= 0.45  (was 0.565 in exp_21)
  (b) aux_dims_kept_c   in {2, 3} (was 4 in exp_21)

GAMFIT VERSION HANDLING
-----------------------
gamfit.OrthogonalityPenalty lands at v0.1.120+ (and gam src
analytic_penalties.rs:2025). On this wheel (probed at runtime) we fall back
to: per-iter QR-orthogonalize-then-renormalize of the latent block U,
paired with the same per-axis ARD shrinkage emulator from auto_exp_21.
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
# Config
# ---------------------------------------------------------------------------
HARVEST  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR  = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG  = OUT_DIR / "auto_exp_23.png"
OUT_JSON = OUT_DIR / "auto_exp_23.json"

K_PC          = 16
D_AUX         = 4
N_ITERS       = 60
ARD_A0        = 1.0
ARD_B0        = 1.0
TAU_MAX       = 1e3
ARD_PRUNE_THR = 0.10        # axis kept if var(U_j)/max(var) >= 0.10
SEED          = 0
MAX_RSS_GB    = 6.0

# Orthogonality ramp (only kicks in for fit_b and fit_c).
# Linear ramp: 0 at iter 0 -> 1 at iter ORTH_RAMP_END.
ORTH_RAMP_END = 20


# ---------------------------------------------------------------------------
# gamfit probe
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
                      "fallback: per-iter QR-orthonormalize U + renormalize.",
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
# Fisher-Rao per-row diagonal precision (same as auto_exp_21)
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
# Joint circle + aux fit, parametrized by penalty config
# ---------------------------------------------------------------------------
def fit_circle_aux(Z, W_diag, *,
                   use_ard, use_orth,
                   d_aux=D_AUX, n_iters=N_ITERS, seed=SEED,
                   a0=ARD_A0, b0=ARD_B0):
    """Generic alternating fit.

    use_ard  : per-axis ARD EM on tau_j (rotation-invariant on aux frame).
    use_orth : gauge-fix the aux block by QR-orthonormalizing U at each
               iteration (then absorbing the upper-triangular factor into
               C_aux so the joint prediction is unchanged). This is the
               fallback for OrthogonalityPenalty: at convergence U has
               orthonormal columns, which is the lambda -> inf limit of
               OrthogonalityPenalty(weight=lambda) on U.

               When ramped in (factor in [0, 1]), we blend
                   U_new = (1 - alpha) * U + alpha * Q
               where Q is the orthonormal factor from QR(U). At alpha=1 the
               aux block has orthonormal columns. C_aux is updated to
               U_new.T @ Z_residual / ||U_new||^2 anyway by the next iter.
    """
    rng = np.random.default_rng(seed)
    N, p = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)

    theta = np.arctan2(Zc[:, 1], Zc[:, 0])
    U = rng.normal(scale=0.1, size=(N, d_aux))
    tau = np.ones(d_aux)
    sigma2 = float(Zc.var())

    w_row = W_diag.mean(axis=1)
    w_row /= w_row.mean()

    history = {"tau": [], "u_var": [], "sigma2": []}

    for it in range(n_iters):
        # (1) Linear solve for C_circle and C_aux.
        cs = np.column_stack([np.cos(theta), np.sin(theta)])
        D = np.hstack([cs, U])
        eps = 1e-6
        ard_diag = tau if use_ard else np.full(d_aux, eps)
        L = np.diag(np.concatenate([[eps, eps], ard_diag]))
        A = D.T @ (D * w_row[:, None]) + sigma2 * L
        b = D.T @ (Zc * w_row[:, None])
        C = np.linalg.solve(A, b)
        C_circle = C[:2]
        C_aux = C[2:]

        # (2) U posterior update.
        resid_circ = Zc - cs @ C_circle
        post_var_sum = np.zeros(d_aux)
        prior_diag = tau if use_ard else np.full(d_aux, eps)
        for n in range(N):
            prec_n = (w_row[n] / sigma2) * (C_aux @ C_aux.T) + np.diag(prior_diag)
            S_n = np.linalg.inv(prec_n)
            U[n] = S_n @ ((w_row[n] / sigma2) * (C_aux @ resid_circ[n]))
            post_var_sum += np.diag(S_n)

        # (2b) ORTHOGONALITY FALLBACK: QR-orthonormalize U with ramp.
        # Fold the upper-triangular R into C_aux so the joint product is
        # preserved before we let theta/C_circle relax against the new aux.
        if use_orth:
            alpha = min(1.0, (it + 1) / float(ORTH_RAMP_END))
            Q, R = np.linalg.qr(U)
            # Match Q's column signs to U's diagonal sign of R, then scale Q
            # columns so that ||Q_j|| matches mean ||U_j|| (preserve overall
            # magnitude, just kill cross-axis correlation).
            col_norms = np.linalg.norm(U, axis=0)
            scale = col_norms.mean()
            Q = Q * scale
            U_new = (1.0 - alpha) * U + alpha * Q
            # Re-fit C_aux against orthogonalized U to keep the prediction
            # consistent (closed form: C_aux = (U^T W U)^-1 U^T W resid_circ).
            UtWU = U_new.T @ (U_new * w_row[:, None])
            UtWr = U_new.T @ (resid_circ * w_row[:, None])
            C_aux = np.linalg.solve(UtWU + 1e-8 * np.eye(d_aux), UtWr)
            U = U_new

        # (3) Theta gradient steps on the circle.
        resid_aux = Zc - U @ C_aux
        for inner in range(3):
            cs_t = np.column_stack([np.cos(theta), np.sin(theta)])
            tang = np.column_stack([-np.sin(theta), np.cos(theta)])
            pred = cs_t @ C_circle
            err = resid_aux - pred
            grad = -2.0 * w_row * np.einsum("np,kp,nk->n", err, C_circle, tang)
            denom = 2.0 * w_row * np.einsum("kp,kp->", C_circle, C_circle) + 1e-3
            theta = theta - grad / denom
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # (4) ARD update.
        if use_ard:
            u_sq_sum = (U * U).sum(axis=0) + post_var_sum
            tau = (N + 2.0 * a0) / (u_sq_sum + 2.0 * b0)
            tau = np.clip(tau, 1e-6, TAU_MAX)

        # (5) sigma2.
        cs_t = np.column_stack([np.cos(theta), np.sin(theta)])
        resid = Zc - cs_t @ C_circle - U @ C_aux
        sigma2 = float((w_row[:, None] * resid * resid).mean())

        history["tau"].append(tau.tolist())
        history["u_var"].append(U.var(axis=0).tolist())
        history["sigma2"].append(sigma2)

        if it % 15 == 0 or it == n_iters - 1:
            print(f"  [iter {it:3d}] sigma2={sigma2:.4f}  "
                  f"tau={np.round(tau, 2).tolist()}  "
                  f"var(U)={np.round(U.var(axis=0), 4).tolist()}",
                  flush=True)

    return {
        "theta": theta, "U": U,
        "C_circle": C_circle, "C_aux": C_aux,
        "tau": tau, "sigma2": sigma2, "resid": resid,
        "history": history,
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
    """Count aux dims with per-axis variance >= thr * max variance.

    NOTE this is the per-axis stat the ARD+Orth pairing is supposed to make
    meaningful. After orthogonalization the per-axis variances are no
    longer rotation-degenerate, so var(U_j) IS the right "is this axis
    being used" measure.
    """
    var = U.var(axis=0)
    if var.max() <= 0:
        return 0, var, np.zeros_like(var)
    norm = var / var.max()
    return int((norm >= thr).sum()), var, norm


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
def make_plot(results, R2s, hue, rgb):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fit_labels = ["fit_a: ARD-only", "fit_b: Ortho-only", "fit_c: Ortho+ARD"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Panel 1: per-axis raw variance of U for each fit (grouped bars).
    ax = axes[0, 0]
    x = np.arange(D_AUX)
    width = 0.27
    for i, (lbl, res, c) in enumerate(zip(fit_labels, results, colors)):
        var = res["U"].var(axis=0)
        ax.bar(x + (i - 1) * width, var, width, label=lbl, color=c,
               edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"aux_{j}" for j in range(D_AUX)])
    ax.set_ylabel("var(U[:, j])")
    ax.set_title("Per-axis variance of aux block under each fit\n"
                 "(orth+ard should leave 2-3 axes >> remainder)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: normalized var (max-norm) + prune threshold for each fit.
    ax = axes[0, 1]
    for i, (lbl, res, c) in enumerate(zip(fit_labels, results, colors)):
        _, _, norm = aux_dims_kept(res["U"])
        ax.bar(x + (i - 1) * width, norm, width, label=lbl, color=c,
               edgecolor="black", lw=0.5)
    ax.axhline(ARD_PRUNE_THR, ls="--", color="black", lw=1,
               label=f"thr={ARD_PRUNE_THR}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"aux_{j}" for j in range(D_AUX)])
    ax.set_ylabel("var / max(var)")
    ax.set_title("Normalized per-axis variance (kept if >= thr)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.15)

    # Panel 3: ARD tau_j bar chart for fit_a and fit_c (fit_b has no tau).
    ax = axes[1, 0]
    width2 = 0.4
    tau_a = results[0]["tau"]
    tau_c = results[2]["tau"]
    ax.bar(x - width2 / 2, tau_a, width2, label="fit_a: ARD-only tau",
           color="#1f77b4", edgecolor="black", lw=0.5)
    ax.bar(x + width2 / 2, tau_c, width2, label="fit_c: Ortho+ARD tau",
           color="#2ca02c", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"aux_{j}" for j in range(D_AUX)])
    ax.set_ylabel("posterior tau_j (higher = shrunk)")
    ax.set_title("ARD posterior precisions: pairing breaks the rotation\n"
                 "symmetry that left fit_a's taus uniform")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel 4: recovered circle, fit_c only.
    ax = axes[1, 1]
    theta_c = results[2]["theta"]
    ax.scatter(np.cos(theta_c), np.sin(theta_c), c=rgb, s=22,
               edgecolor="black", linewidth=0.3)
    ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.6, alpha=0.4)
    ax.set_xlabel("cos(theta_hat) [fit_c]")
    ax.set_ylabel("sin(theta_hat) [fit_c]")
    ax.set_title(f"fit_c recovered hue circle\n"
                 f"R^2(a)={R2s[0]:.3f}  R^2(b)={R2s[1]:.3f}  "
                 f"R^2(c)={R2s[2]:.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"auto_exp_23 . OrthogonalityPenalty x ARDPenalty pairing retest "
        f"(gamfit=={GAMFIT_VERSION}, "
        f"orth_reached={PRIMITIVE_STATUS['OrthogonalityPenalty']['reached']}, "
        f"ard_reached={PRIMITIVE_STATUS['ARDPenalty']['reached']})",
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
    print(f"[load] N = {N}", flush=True)
    memory_check()

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC].astype(np.float64)
    evr = float(basis["evr"][:K_PC].sum())
    Z = Z / Z.std(axis=0, keepdims=True).clip(min=1e-9)
    print(f"[pca ] Z shape = {Z.shape}  EVR_top{K_PC} = {evr:.3f}", flush=True)
    memory_check()
    del centroids

    print(f"[fr  ] Fisher-Rao W_diag (k_nn=20) ...", flush=True)
    W_diag = fisher_rao_W_diag(Z, k_nn=20)
    memory_check()

    hue = hsv[:, 0]

    fits = []
    cfgs = [
        ("fit_a (ARD-only)",   dict(use_ard=True,  use_orth=False)),
        ("fit_b (Ortho-only)", dict(use_ard=False, use_orth=True)),
        ("fit_c (Ortho+ARD)",  dict(use_ard=True,  use_orth=True)),
    ]
    for name, cfg in cfgs:
        print(f"\n[fit ] === {name} ===", flush=True)
        res = fit_circle_aux(Z, W_diag, **cfg)
        fits.append(res)
        memory_check()

    R2s = [circular_R2(r["theta"], hue) for r in fits]
    kept_info = [aux_dims_kept(r["U"]) for r in fits]
    keps  = [ki[0] for ki in kept_info]
    norms = [ki[2] for ki in kept_info]
    vars_ = [ki[1] for ki in kept_info]

    print("\n=== Summary ===", flush=True)
    for (name, _), R2, k, nrm in zip(cfgs, R2s, keps, norms):
        print(f"  {name:22s}: R2={R2:.4f}  kept={k}/{D_AUX}  "
              f"norm_var={np.round(nrm, 3).tolist()}", flush=True)

    make_plot(fits, R2s, hue, rgb)

    HYP_A_R2_THR     = 0.45
    HYP_B_KEPT_RANGE = (2, 3)
    hyp_a = bool(R2s[2] >= HYP_A_R2_THR)
    hyp_b = bool(HYP_B_KEPT_RANGE[0] <= keps[2] <= HYP_B_KEPT_RANGE[1])

    runtime = time.time() - t0
    out = {
        "experiment": "auto_exp_23",
        "question": ("Does pairing the new OrthogonalityPenalty with "
                     "ARDPenalty break the rotation-invariance failure mode "
                     "auto_exp_21 hit and prune aux 4 -> 2-3?"),
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
            "seed": SEED, "n_colors": int(N),
            "evr_top_K_PC": evr,
        },
        "results": {
            "recovered_hue_R2_a": R2s[0],
            "recovered_hue_R2_b": R2s[1],
            "recovered_hue_R2_c": R2s[2],
            "aux_dims_kept_by_ard_a":            keps[0],
            "aux_dims_kept_by_ortho_only_b":     keps[1],
            "aux_dims_kept_by_ortho_plus_ard_c": keps[2],
            "aux_norm_var_a": [float(x) for x in norms[0]],
            "aux_norm_var_b": [float(x) for x in norms[1]],
            "aux_norm_var_c": [float(x) for x in norms[2]],
            "aux_raw_var_a":  [float(x) for x in vars_[0]],
            "aux_raw_var_b":  [float(x) for x in vars_[1]],
            "aux_raw_var_c":  [float(x) for x in vars_[2]],
            "tau_a": [float(x) for x in fits[0]["tau"]],
            "tau_c": [float(x) for x in fits[2]["tau"]],
            "sigma2_a": float(fits[0]["sigma2"]),
            "sigma2_b": float(fits[1]["sigma2"]),
            "sigma2_c": float(fits[2]["sigma2"]),
            "runtime_seconds": runtime,
        },
        "prediction": {
            "hypothesis_a_R2c_ge_0p45":     hyp_a,
            "hypothesis_b_kept_c_in_2_3":   hyp_b,
            "hyp_a_threshold":              HYP_A_R2_THR,
            "hyp_b_range":                  list(HYP_B_KEPT_RANGE),
            "verdict": ("supported" if (hyp_a and hyp_b) else
                        ("partial" if (hyp_a or hyp_b) else "falsified")),
            "auto_exp_21_baseline": {
                "recovered_hue_R2":      0.565,
                "aux_dims_kept_by_ard":  4,
            },
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)
    print(f"[time] {runtime:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
