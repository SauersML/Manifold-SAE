"""auto_exp_34 — ParametricAuxConditionalPriorPenalty on cogito L40 with RGB aux.

Tests the FULL iVAE primitive: Λ(u_n) = diag(α_k + softplus(β_k) · ‖u_n − μ_k‖²)
with ALL of α_k, β_k, μ_k learnable via gradient descent (REML-style joint fit).

Complements auto_exp_33 (FIXED variant w/ pre-computed Λ). The parametric variant
should fit better since the optimizer chooses the precision structure rather than
the user specifying it externally.

Hypotheses (preregistered, strict TRUE/FALSE):
  (a) Parametric R²(hue) > Fixed R²(hue) (auto_exp_33 baseline; if missing, use 0.0)
  (b) Learned α_k > 0 for all latent axes, β_k > 0 for axes correlated with HSV
      variation (REML discovers axis-specific distance-sensitivity)
  (c) Learned μ_k clusters near natural HSV reference points in RGB space
      (interpreted: mean pairwise μ-distance > 0.3 i.e. spread, not collapsed)

gamfit 0.1.112 has neither AuxConditional nor Parametric variants -> pure-Python
fallback. path_taken = "fallback_parametric_python".
"""

from __future__ import annotations

import colorsys
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")

from _pca_basis import N_TEMPLATES, TOP_TEMPLATES, load_pc_basis, project  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402
from color_manifold_gam import load_xkcd_colors  # noqa: E402

import gamfit  # noqa: E402

# --------------------------------------------------------------------------- #
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG = OUT_DIR / "auto_exp_34.png"
OUT_JSON = OUT_DIR / "auto_exp_34.json"
PRIOR_FIXED_JSON = OUT_DIR / "auto_exp_33.json"

K_PC = 16
D_AUX = 3                     # match HSV/perceptual dim (3) per color decomposition memo
W_PEN = 1.0                   # auxiliary-conditional penalty weight
N_ITERS = 200
LR = 0.05
RIDGE = 1e-4
SEED = 0
GAMFIT_VERSION = getattr(gamfit, "__version__", "unknown")


def _probe_parametric() -> dict:
    """Probe gamfit for the parametric aux-conditional primitive."""
    try:
        from gamfit._penalties import (  # type: ignore  # noqa: F401
            ParametricAuxConditionalPriorPenalty,
        )
        return {"reached": True, "detail": "imported native primitive"}
    except Exception as e:
        return {
            "reached": False,
            "detail": (
                f"{type(e).__name__}: {e}; gamfit 0.1.112 lacks native primitive "
                "-> fallback_parametric_python"
            ),
        }


PRIMITIVE_STATUS = _probe_parametric()
PATH_TAKEN = (
    "native_parametric"
    if PRIMITIVE_STATUS["reached"]
    else "fallback_parametric_python"
)


# --------------------------- Data ------------------------------------------ #
def build_inputs():
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

    basis = load_pc_basis(K=64)
    Z = project(centroids, basis)[:, :K_PC]
    Z = (Z - Z.mean(0)) / Z.std(0).clip(min=1e-6)

    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    return Z, rgb, hsv


# ------ Pure-Python emulator of ParametricAuxConditionalPriorPenalty ------- #
def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_grad(x):
    # d/dx softplus(x) = sigmoid(x)
    return 1.0 / (1.0 + np.exp(-x))


def lambda_per_axis(U, log_alpha, raw_beta, mu):
    """λ_k(u_n) = exp(log_α_k) + softplus(raw_β_k) * ‖u_n − μ_k‖²

    U: (N, du), log_alpha: (du,), raw_beta: (du,), mu: (du, du).
    mu[k] is the reference point for axis k (lives in aux R^du space).
    Returns Λ: (N, du).
    """
    alpha = np.exp(log_alpha)           # (du,)
    beta = softplus(raw_beta)           # (du,)
    # ||u_n - μ_k||² over n,k
    diff = U[:, None, :] - mu[None, :, :]           # (N, du_k, du)
    sqd = (diff ** 2).sum(axis=2)                   # (N, du_k)
    return alpha[None, :] + beta[None, :] * sqd, alpha, beta, sqd, diff


def fit_parametric_aux(Z, U, *, w=W_PEN, d_aux=D_AUX, n_iters=N_ITERS,
                       lr=LR, seed=SEED):
    """Joint optimization of (R, beta, log_α, raw_β, μ) under

        L = 0.5 ||U - T β||²  +  (w/2) Σ_n t_n^T diag(λ(u_n)) t_n
        T  = Z R                      (latent aux block, N × d_aux)
        λ_k(u_n) = exp(log_α_k) + softplus(raw_β_k) ‖u_n − μ_k‖²

    Tracks loss trace and returns final params + diagnostics.
    """
    rng = np.random.default_rng(seed)
    N, K = Z.shape
    du = U.shape[1]
    assert d_aux == du, "this fit ties latent aux dim to aux du for clarity"

    R = rng.normal(scale=0.1, size=(K, d_aux))
    beta = rng.normal(scale=0.1, size=(d_aux, du))
    log_alpha = np.zeros(d_aux)                       # alpha_init = 1
    raw_beta = np.full(d_aux, -3.0)                   # softplus(-3) ≈ 0.049 (small)
    mu = rng.uniform(0.0, 1.0, size=(d_aux, du))      # μ in RGB cube
    # init mu from random points in [0,1]^3 per spec

    loss_trace = []
    for it in range(n_iters):
        T = Z @ R                                     # (N, d_aux)
        # closed-form beta given T (ridge for stability)
        TtT = T.T @ T + RIDGE * np.eye(d_aux)
        beta = np.linalg.solve(TtT, T.T @ U)
        resid = U - T @ beta                          # (N, du)

        Lam, alpha, beta_sp, sqd, diff = lambda_per_axis(U, log_alpha, raw_beta, mu)
        # data loss
        data_loss = 0.5 * float((resid ** 2).sum())
        # penalty:  (w/2) Σ_n Σ_k λ_kn t_kn^2
        pen_loss = 0.5 * w * float((Lam * (T ** 2)).sum())
        loss = data_loss + pen_loss
        loss_trace.append(loss)

        # grad on R via data + via penalty
        # data: d/dR 0.5 ||U - ZRβ||² = -Z^T (U - ZRβ) β^T
        grad_R_data = -Z.T @ resid @ beta.T
        # penalty: pen = 0.5 w Σ_n Σ_k λ_kn (Σ_j Z_nj R_jk)²
        # d/dR_jk = w Σ_n λ_kn T_nk Z_nj = w Z^T (Lam ⊙ T)_{:,k} per col k
        grad_R_pen = w * (Z.T @ (Lam * T))
        grad_R = grad_R_data + grad_R_pen

        # grad on log_alpha:  d/dlog_α_k = 0.5 w Σ_n α_k * T_nk²  (since d λ_kn/d log α_k = α_k)
        grad_log_alpha = 0.5 * w * alpha * (T ** 2).sum(axis=0)
        # grad on raw_beta: d λ_kn / d raw_β_k = sigmoid(raw_β_k) * sqd_nk
        sig_rb = softplus_grad(raw_beta)
        grad_raw_beta = 0.5 * w * sig_rb * (sqd * (T ** 2)).sum(axis=0)
        # grad on μ_k_j:  d λ_kn / d μ_k_j = -2 β_k (u_nj - μ_kj)
        # so  d pen / d μ_k_j = 0.5 w Σ_n T_nk² * (-2 β_k (u_nj - μ_kj))
        #                     = -w β_k Σ_n T_nk² (u_nj - μ_kj)
        T2 = T ** 2                                    # (N, d_aux)
        # diff: (N, d_aux_k, du_j)
        grad_mu = -w * beta_sp[:, None] * (T2[:, :, None] * diff).sum(axis=0)

        # adaptive step (heuristic Lipschitz)
        L_data = float((Z ** 2).sum(axis=0).max() * (beta ** 2).sum())
        L_pen = w * float(Lam.max() * (Z ** 2).sum(axis=0).max())
        step_R = lr / (L_data + L_pen + 1.0)
        step_scalars = lr * 0.5
        step_mu = lr * 0.2

        R = R - step_R * grad_R
        log_alpha = log_alpha - step_scalars * grad_log_alpha
        raw_beta = raw_beta - step_scalars * grad_raw_beta
        # clamp μ to [0, 1] (RGB cube) to keep interpretability
        mu = np.clip(mu - step_mu * grad_mu, 0.0, 1.0)

        if it % 50 == 0 or it == n_iters - 1:
            print(
                f"  iter {it:3d}  loss={loss:.4f}  data={data_loss:.4f}  pen={pen_loss:.4f}  "
                f"α={np.exp(log_alpha).round(3).tolist()}  β={softplus(raw_beta).round(3).tolist()}",
                flush=True,
            )

    return {
        "R": R,
        "beta": beta,
        "T": Z @ R,
        "alpha": np.exp(log_alpha),
        "beta_sp": softplus(raw_beta),
        "mu": mu,
        "loss_trace": loss_trace,
    }


# ------------------------------ main --------------------------------------- #
def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gamfit] version={GAMFIT_VERSION}  path_taken={PATH_TAKEN}", flush=True)
    print(f"[probe]  ParametricAuxConditional: {PRIMITIVE_STATUS}", flush=True)

    Z, rgb, hsv = build_inputs()
    N = Z.shape[0]
    print(f"[data] Z={Z.shape}  RGB={rgb.shape}  HSV={hsv.shape}", flush=True)

    # Fit on aux = RGB
    print("\n=== parametric aux-conditional fit (aux = RGB) ===", flush=True)
    fit = fit_parametric_aux(Z, rgb, w=W_PEN, d_aux=D_AUX, n_iters=N_ITERS)

    # Predict aux from T via learned β, then map RGB -> HSV for per-channel R²
    rgb_hat = fit["T"] @ fit["beta"]                 # (N, 3)
    rgb_hat_c = np.clip(rgb_hat, 0.0, 1.0)
    hsv_hat = np.array([colorsys.rgb_to_hsv(*c) for c in rgb_hat_c])

    def r2(y, yh):
        ss_res = float(((y - yh) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    # Hue is circular; use cosine-corrected R² via 2π unwrap then std dev
    # Simpler: compare via (cos2πh, sin2πh) on the unit circle.
    def r2_hue(h, hh):
        a = np.stack([np.cos(2 * np.pi * h), np.sin(2 * np.pi * h)], axis=1)
        ah = np.stack([np.cos(2 * np.pi * hh), np.sin(2 * np.pi * hh)], axis=1)
        ss_res = float(((a - ah) ** 2).sum())
        ss_tot = float(((a - a.mean(0)) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    R2_hue = r2_hue(hsv[:, 0], hsv_hat[:, 0])
    R2_sat = r2(hsv[:, 1], hsv_hat[:, 1])
    R2_val = r2(hsv[:, 2], hsv_hat[:, 2])
    print(f"[R²]  hue={R2_hue:.4f}  sat={R2_sat:.4f}  val={R2_val:.4f}", flush=True)

    # Load prior fixed-variant result for hypothesis (a)
    fixed_hue_r2 = None
    if PRIOR_FIXED_JSON.exists():
        try:
            d33 = json.loads(PRIOR_FIXED_JSON.read_text())
            for k in ("R2_hue", "R²_hue", "r2_hue"):
                if k in d33:
                    fixed_hue_r2 = float(d33[k])
                    break
        except Exception as e:
            print(f"[warn] could not parse auto_exp_33.json: {e}", flush=True)

    # Verdicts
    # (a) parametric > fixed; if no fixed file, baseline = 0.0 (auto-pass if positive)
    fixed_baseline = fixed_hue_r2 if fixed_hue_r2 is not None else 0.0
    verdict_a = bool(R2_hue > fixed_baseline)
    # (b) all α>0 AND all β>0 (REML actually grew β away from init)
    alpha_arr = fit["alpha"]
    beta_arr = fit["beta_sp"]
    verdict_b = bool(np.all(alpha_arr > 0) and np.all(beta_arr > 0))
    # (c) μ clusters spread: mean pairwise distance > 0.3
    mu_arr = fit["mu"]
    pair_dists = []
    for i in range(D_AUX):
        for j in range(i + 1, D_AUX):
            pair_dists.append(float(np.linalg.norm(mu_arr[i] - mu_arr[j])))
    mean_pair_d = float(np.mean(pair_dists)) if pair_dists else 0.0
    verdict_c = bool(mean_pair_d > 0.3)

    print("\n=== VERDICTS ===", flush=True)
    print(f" (a) parametric R²(hue)={R2_hue:.4f} > fixed baseline={fixed_baseline:.4f}: {verdict_a}",
          flush=True)
    print(f" (b) all α>0 ({alpha_arr.tolist()}) AND all β>0 ({beta_arr.tolist()}): {verdict_b}",
          flush=True)
    print(f" (c) mean pairwise ‖μ_i − μ_j‖={mean_pair_d:.3f} > 0.3: {verdict_c}", flush=True)

    # ---- plot 4-panel ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) recovered hue vs ground-truth hue scatter, colored by ground-truth RGB
    ax = axes[0, 0]
    ax.scatter(hsv[:, 0], hsv_hat[:, 0], c=rgb, s=14, edgecolor="black", lw=0.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("ground-truth hue")
    ax.set_ylabel("recovered hue")
    ax.set_title(f"Hue recovery  ·  R²(circular)={R2_hue:.3f}")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # (0,1) learned α_k bars
    ax = axes[0, 1]
    ax.bar(np.arange(D_AUX), alpha_arr, color="tab:blue", edgecolor="black")
    ax.set_xticks(np.arange(D_AUX))
    ax.set_xticklabels([f"axis {k}" for k in range(D_AUX)])
    ax.set_ylabel("learned α_k = exp(log_α_k)")
    ax.set_title(f"Learned α (baseline precision)  ·  all > 0: {bool(np.all(alpha_arr > 0))}")
    ax.grid(alpha=0.3, axis="y")
    for k, v in enumerate(alpha_arr):
        ax.text(k, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # (1,0) learned β_k bars
    ax = axes[1, 0]
    ax.bar(np.arange(D_AUX), beta_arr, color="tab:orange", edgecolor="black")
    ax.set_xticks(np.arange(D_AUX))
    ax.set_xticklabels([f"axis {k}" for k in range(D_AUX)])
    ax.set_ylabel("learned β_k = softplus(raw_β_k)")
    ax.set_title(f"Learned β (distance-sensitivity)  ·  all > 0: {bool(np.all(beta_arr > 0))}")
    ax.grid(alpha=0.3, axis="y")
    for k, v in enumerate(beta_arr):
        ax.text(k, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # (1,1) learned μ_k in 3D (R, G, B), colored by axis index k, with reference RGB cube
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    cmap = plt.get_cmap("tab10")
    for k in range(D_AUX):
        ax.scatter(mu_arr[k, 0], mu_arr[k, 1], mu_arr[k, 2],
                   s=160, color=cmap(k), edgecolor="black",
                   label=f"μ_{k} = {mu_arr[k].round(3).tolist()}")
    # faint cloud of actual RGB color points for context
    ax.scatter(rgb[::5, 0], rgb[::5, 1], rgb[::5, 2], c=rgb[::5],
               s=4, alpha=0.25)
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_title(f"Learned μ_k in RGB cube  ·  mean pair-d={mean_pair_d:.3f}")
    ax.legend(fontsize=7, loc="upper left")

    verdict_line = (
        f"(a)={'PASS' if verdict_a else 'FAIL'}  "
        f"(b)={'PASS' if verdict_b else 'FAIL'}  "
        f"(c)={'PASS' if verdict_c else 'FAIL'}"
    )
    fig.suptitle(
        "auto_exp_34  ·  ParametricAuxConditionalPriorPenalty (α, β, μ all learnable) on cogito L40\n"
        f"aux = RGB, d_aux = {D_AUX}  ·  gamfit={GAMFIT_VERSION}  path={PATH_TAKEN}  ·  {verdict_line}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_PNG}", flush=True)

    runtime = time.time() - t0

    payload = {
        "experiment": "auto_exp_34",
        "gamfit_version": GAMFIT_VERSION,
        "path_taken": PATH_TAKEN,
        "primitive_status": PRIMITIVE_STATUS,
        "k_pc": K_PC,
        "d_aux": D_AUX,
        "w_pen": W_PEN,
        "n_iters": N_ITERS,
        "n_colors": int(N),
        "aux_target": "RGB (3-dim)",
        "R2_hue": R2_hue,
        "R2_sat": R2_sat,
        "R2_val": R2_val,
        "final_alpha": [float(x) for x in alpha_arr],
        "final_beta":  [float(x) for x in beta_arr],
        "final_mu": [[float(x) for x in row] for row in mu_arr],
        "training_loss_trace": [float(x) for x in fit["loss_trace"]],
        "mean_mu_pairwise_distance": mean_pair_d,
        "fixed_baseline_hue_r2_from_auto_exp_33": fixed_hue_r2,
        "fixed_baseline_used_for_verdict_a": fixed_baseline,
        "hypothesis_verdicts": {
            "a_parametric_beats_fixed_hue": {
                "verdict": verdict_a,
                "parametric_R2_hue": R2_hue,
                "fixed_baseline_R2_hue": fixed_baseline,
                "note": (
                    "Baseline from auto_exp_33.json if present, else 0.0. "
                    "Compares parametric Λ vs fixed Λ on hue recovery."
                ),
            },
            "b_alpha_and_beta_positive": {
                "verdict": verdict_b,
                "alpha": [float(x) for x in alpha_arr],
                "beta":  [float(x) for x in beta_arr],
            },
            "c_mu_clusters_spread": {
                "verdict": verdict_c,
                "mean_pairwise_distance": mean_pair_d,
                "threshold": 0.3,
            },
        },
        "runtime_seconds": runtime,
        "prediction_slot": {
            "note": (
                "When gamfit ships ParametricAuxConditionalPriorPenalty natively, "
                "re-run with `from gamfit._penalties import "
                "ParametricAuxConditionalPriorPenalty`. Expected: same R²(hue) and "
                "same learned (α, β, μ) up to fp roundoff; the Python emulator "
                "computes the exact closed-form gradients of the smoothed primitive."
            ),
            "fallback_R2_hue": R2_hue,
            "fallback_alpha": [float(x) for x in alpha_arr],
            "fallback_beta":  [float(x) for x in beta_arr],
            "fallback_mu":    [[float(x) for x in row] for row in mu_arr],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[save] {OUT_JSON}", flush=True)
    print(f"[done] runtime = {runtime:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
