"""auto_exp_33: AuxConditionalPriorPenalty + ARD on cogito L40, with d_aux=3 HSV.

Hypothesis: Prior failed cogito experiments (auto_exp_21/23/30/31/32) used d_aux=4,
mismatched to the U_3d 3+3 decomposition (project_cogito_color_manifold_decomposition).
Matching d_aux=3 to the supervised HSV (R,G,B) aux variable should let AuxConditional's
per-row precision Lambda(u_n) pin the latent basis to HSV-aligned axes, then ARD
either keeps all 3 (correct refusal) or prunes redundant ones.

Hypotheses:
 (a) AuxConditional + ARD recovers HSV-aligned axes with R^2(hue) >= 0.55.
 (b) ARD does NOT prune below 3 (correct refusal).
 (c) Per-axis HSV correlation > 0.30 (axes correlate with R, G, B respectively).

Path: gamfit 0.1.112 lacks AuxConditionalPriorPenalty/ARDPenalty wrappers
(only ships duchon + smoothness penalties). Fallback to soft Python emulator
(per examples/aux_conditional_prior_demo.py), recorded in path_taken.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore

ROOT = Path("/Users/user/Manifold-SAE")
RUN_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_PNG = RUN_DIR / "auto_exp_33.png"
OUT_JSON = RUN_DIR / "auto_exp_33.json"

N_TEMPLATES = 28
K_PCS = 16          # project K=64 basis -> first 16 dims for working latent
D_AUX = 3           # HSV supervised aux (the KEY change vs. 21/23/30/31/32)
N_ITER = 400
AUX_WEIGHT = 8.0
ARD_PRUNE_TAU = 1e-2  # ARD posterior threshold for "kept"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_xkcd_rgb(n_colors: int) -> tuple[list[str], np.ndarray]:
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append((int(hexs[0:2], 16) / 255.0,
                        int(hexs[2:4], 16) / 255.0,
                        int(hexs[4:6], 16) / 255.0))
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float64)


def per_color_centroids_mmap(x_mmap: np.ndarray, n_t: int) -> np.ndarray:
    n_rows, d = x_mmap.shape
    n_c = n_rows // n_t
    out = np.zeros((n_c, d), dtype=np.float64)
    # color-major: row i has color = i // n_t
    block = 4096
    counts = np.zeros(n_c, dtype=np.int64)
    for s in range(0, n_c * n_t, block):
        e = min(s + block, n_c * n_t)
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)
        idx = (np.arange(s, e) // n_t)
        for c in np.unique(idx):
            m = idx == c
            out[c] += chunk[m].sum(axis=0)
            counts[c] += int(m.sum())
    out /= counts[:, None]
    return out


def hsv_from_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


# -----------------------------------------------------------------------------
# Fits.  All four work on the K-dim PCA-projected centroids T0 (n_c, K).
# We learn a (K, d_aux) projection matrix W; latent T = T0 @ W (n_c, d_aux).
# Then map back via OLS to predict HSV channels for the R^2 measures.
# -----------------------------------------------------------------------------
def fit_ols_baseline(T0: np.ndarray, hsv: np.ndarray) -> dict:
    """OLS: predict HSV directly from PCA latent. d_aux=3 implicit via OLS hat."""
    # Predict each HSV channel by OLS from T0.
    Tc = T0 - T0.mean(0, keepdims=True)
    hc = hsv - hsv.mean(0, keepdims=True)
    beta, *_ = np.linalg.lstsq(Tc, hc, rcond=None)
    pred = Tc @ beta
    r2 = 1.0 - ((hc - pred) ** 2).sum(0) / (hc ** 2).sum(0).clip(min=1e-12)
    # Latent is the 3-D linear image of T0 in HSV space.
    T = pred  # (n_c, 3) — interpretable as a 3-D aux-aligned latent
    return {"T": T, "pred_hsv": pred, "r2_hsv": r2,
            "axes_kept": int(D_AUX), "tau_trace": np.zeros((1, D_AUX))}


def fit_ard_only(T0: np.ndarray, hsv: np.ndarray,
                 n_iter: int = N_ITER) -> dict:
    """ARD-only fallback: learn W (K, d_aux) by ridge with per-axis ARD prior.
    Update tau_j via posterior mean: tau_j = (alpha_0 + 1/2) / (beta_0 + 0.5 |W[:,j]|^2)."""
    rng = np.random.default_rng(33)
    n_c, K = T0.shape
    Tc = T0 - T0.mean(0, keepdims=True)
    hc = hsv - hsv.mean(0, keepdims=True)
    W = rng.normal(scale=0.05, size=(K, D_AUX))
    tau = np.ones(D_AUX)  # precisions per axis
    sigma2 = float(np.var(hc))
    tau_trace = []
    TtT = Tc.T @ Tc / n_c
    Tth = Tc.T @ hc / n_c
    for _ in range(n_iter):
        # ridge step on each output column with diag(tau) penalty
        for j in range(D_AUX):
            A = TtT + (tau[j] * sigma2 / n_c) * np.eye(K)
            W[:, j] = np.linalg.solve(A, Tth[:, j])
        # Update tau via inverse-gamma posterior mean: tau_j = K / |W[:,j]|^2
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        tau_trace.append(tau.copy())
        # update sigma2 via residual
        resid = hc - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    T = Tc @ W
    pred = T  # T already equals predicted hsv (W maps to hsv)
    r2 = 1.0 - ((hc - pred) ** 2).sum(0) / (hc ** 2).sum(0).clip(min=1e-12)
    inv_tau = 1.0 / tau
    axes_kept = int(np.count_nonzero(inv_tau > ARD_PRUNE_TAU * inv_tau.max()))
    return {"T": T, "pred_hsv": pred, "r2_hsv": r2,
            "axes_kept": axes_kept,
            "tau_trace": np.asarray(tau_trace),
            "tau": tau, "W": W}


def fit_aux_conditional_only(T0: np.ndarray, hsv: np.ndarray,
                             n_iter: int = N_ITER) -> dict:
    """AuxConditional-only: per-row precision Lambda_n diag(1/sigma_aux^2) for axes
    well-aligned with HSV. Effectively: weighted least-squares with per-row weight
    inverse proportional to HSV norm uncertainty."""
    rng = np.random.default_rng(34)
    n_c, K = T0.shape
    Tc = T0 - T0.mean(0, keepdims=True)
    hc = hsv - hsv.mean(0, keepdims=True)
    # Per-row precisions: tight (sigma_aux=0.5) on all 3 aux dims, scaled by HSV norm
    sigma_aux = 0.5
    aux_norms = np.linalg.norm(hsv - 0.5, axis=1)  # distance from neutral grey
    # higher norm -> more confident aux -> tighter precision
    w_row = 1.0 / (sigma_aux ** 2) * (1.0 + aux_norms)  # (n_c,)
    # Weighted-LS: W = (T^T W_row T)^{-1} T^T W_row hc, with also a Lambda-style
    # ridge term coming from AuxConditional pinning: + AUX_WEIGHT * I on W
    Wd = (w_row[:, None] * Tc).T @ Tc / n_c
    Wt = (w_row[:, None] * Tc).T @ hc / n_c
    A = Wd + (AUX_WEIGHT / n_c) * np.eye(K)
    W = np.linalg.solve(A, Wt)
    # iterate once more (this is closed-form for fixed Lambda, but we keep API parity)
    T = Tc @ W
    pred = T
    r2 = 1.0 - ((hc - pred) ** 2).sum(0) / (hc ** 2).sum(0).clip(min=1e-12)
    return {"T": T, "pred_hsv": pred, "r2_hsv": r2,
            "axes_kept": int(D_AUX),
            "tau_trace": np.zeros((1, D_AUX)),
            "W": W}


def fit_aux_conditional_plus_ard(T0: np.ndarray, hsv: np.ndarray,
                                 n_iter: int = N_ITER) -> dict:
    """AuxConditional + ARD: alternate W-update (weighted ridge with per-axis tau)
    and tau-update (posterior mean)."""
    rng = np.random.default_rng(35)
    n_c, K = T0.shape
    Tc = T0 - T0.mean(0, keepdims=True)
    hc = hsv - hsv.mean(0, keepdims=True)
    sigma_aux = 0.5
    aux_norms = np.linalg.norm(hsv - 0.5, axis=1)
    w_row = 1.0 / (sigma_aux ** 2) * (1.0 + aux_norms)
    W = rng.normal(scale=0.05, size=(K, D_AUX))
    tau = np.ones(D_AUX)
    sigma2 = float(np.var(hc))
    tau_trace = []
    WTW = (w_row[:, None] * Tc).T @ Tc / n_c
    WTh = (w_row[:, None] * Tc).T @ hc / n_c
    for _ in range(n_iter):
        for j in range(D_AUX):
            A = WTW + ((tau[j] * sigma2 + AUX_WEIGHT) / n_c) * np.eye(K)
            W[:, j] = np.linalg.solve(A, WTh[:, j])
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        tau_trace.append(tau.copy())
        resid = hc - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    T = Tc @ W
    pred = T
    r2 = 1.0 - ((hc - pred) ** 2).sum(0) / (hc ** 2).sum(0).clip(min=1e-12)
    inv_tau = 1.0 / tau
    axes_kept = int(np.count_nonzero(inv_tau > ARD_PRUNE_TAU * inv_tau.max()))
    return {"T": T, "pred_hsv": pred, "r2_hsv": r2,
            "axes_kept": axes_kept,
            "tau_trace": np.asarray(tau_trace),
            "tau": tau, "W": W}


# -----------------------------------------------------------------------------
def per_axis_hsv_corr(T: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """Best |Pearson| between each latent axis and any HSV channel.
    Returns (d_aux, 3) full matrix |corr(T_j, hsv_k)|."""
    Tc = T - T.mean(0, keepdims=True)
    hc = hsv - hsv.mean(0, keepdims=True)
    Tn = Tc / (Tc.std(0, keepdims=True) + 1e-12)
    Hn = hc / (hc.std(0, keepdims=True) + 1e-12)
    return np.abs(Tn.T @ Hn / Tn.shape[0])  # (d_aux, 3)


def check_gamfit() -> tuple[str, str]:
    try:
        import gamfit  # type: ignore
        ver = getattr(gamfit, "__version__", "unknown")
        has_aux = hasattr(gamfit, "AuxConditionalPriorPenalty")
        has_ard = hasattr(gamfit, "ARDPenalty")
        if has_aux and has_ard:
            return ver, "gamfit_real_wrappers"
        return ver, "fallback_python_aux_prior"
    except Exception as exc:
        return f"unavailable:{exc!r}", "fallback_python_aux_prior_no_gamfit"


def main() -> None:
    t_start = time.time()
    print("[auto_exp_33] AuxConditional+ARD on cogito L40 with d_aux=3 HSV")

    ver, path_taken = check_gamfit()
    print(f"[gamfit] version={ver} path={path_taken}")

    # 1. Load X_L40 and compute per-color centroids
    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X shape={X.shape}")
    cents = per_color_centroids_mmap(X, N_TEMPLATES)
    n_c = cents.shape[0]
    print(f"[centroids] {cents.shape}")

    # 2. Project via K=64 PCA basis, take first K_PCS dims
    basis = load_pc_basis(K=64)
    Xn = (cents - basis["mu"]) / basis["sigma"]
    Z64 = Xn @ basis["Vt"].T   # (n_c, 64)
    T0 = Z64[:, :K_PCS]        # (n_c, 16)
    print(f"[pca] T0={T0.shape}")

    # 3. HSV from xkcd RGB
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    print(f"[hsv] {hsv.shape}, first color='{names[0]}', rgb={rgb[0]}, hsv={hsv[0]}")

    # 4. Fit 4 configs
    fits = {
        "ols_baseline": fit_ols_baseline(T0, hsv),
        "ard_only": fit_ard_only(T0, hsv),
        "aux_conditional_only": fit_aux_conditional_only(T0, hsv),
        "aux_conditional_plus_ard": fit_aux_conditional_plus_ard(T0, hsv),
    }

    # 5. Metrics
    report = {}
    for name, fit in fits.items():
        r2 = fit["r2_hsv"]
        corr = per_axis_hsv_corr(fit["T"], hsv)
        # per-axis max-|corr| with any HSV channel
        per_axis = corr.max(axis=1).tolist()
        report[name] = {
            "R2_hue": float(r2[0]),
            "R2_sat": float(r2[1]),
            "R2_val": float(r2[2]),
            "axes_kept": int(fit["axes_kept"]),
            "per_axis_HSV_max_corr": [float(v) for v in per_axis],
            "per_axis_HSV_corr_matrix": corr.tolist(),  # (d_aux, 3) |r|
        }
        print(f"[fit {name}] R2(h,s,v)=({r2[0]:.3f},{r2[1]:.3f},{r2[2]:.3f})"
              f" axes_kept={fit['axes_kept']} per_axis_max_corr={per_axis}")

    # 6. Best fit by R2_hue
    best_name = max(report, key=lambda k: report[k]["R2_hue"])
    print(f"[best] {best_name}  R2_hue={report[best_name]['R2_hue']:.3f}")

    # 7. Hypothesis verdicts
    best = report[best_name]
    best_aux_only = report["aux_conditional_plus_ard"]
    h_a = bool(best_aux_only["R2_hue"] >= 0.55)
    h_b = bool(fits["aux_conditional_plus_ard"]["axes_kept"] >= 3)
    h_c = bool(all(c > 0.30 for c in best_aux_only["per_axis_HSV_max_corr"]))
    hypotheses = {
        "(a)_AuxCondARD_R2_hue_ge_0.55": h_a,
        "(b)_ARD_does_not_prune_below_3": h_b,
        "(c)_all_axes_HSV_corr_gt_0.30": h_c,
    }
    print(f"[hypotheses] {hypotheses}")

    # 8. Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    cfg_names = list(report.keys())
    colors_cfg = ["#777", "#1f77b4", "#2ca02c", "#d62728"]

    # P1: per-axis HSV max-correlation per fit
    ax = axes[0, 0]
    width = 0.2
    x = np.arange(D_AUX)
    for i, n in enumerate(cfg_names):
        vals = report[n]["per_axis_HSV_max_corr"]
        ax.bar(x + (i - 1.5) * width, vals, width, label=n, color=colors_cfg[i])
    ax.axhline(0.30, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"axis {j}" for j in range(D_AUX)])
    ax.set_ylabel("max |Pearson(latent_j, HSV_k)|")
    ax.set_title("Per-axis HSV correlation by fit")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)

    # P2: R2 per HSV channel per fit
    ax = axes[0, 1]
    width = 0.2
    chan = np.arange(3)
    for i, n in enumerate(cfg_names):
        r = [report[n]["R2_hue"], report[n]["R2_sat"], report[n]["R2_val"]]
        ax.bar(chan + (i - 1.5) * width, r, width, label=n, color=colors_cfg[i])
    ax.axhline(0.55, color="k", ls=":", lw=0.8, label="0.55 threshold")
    ax.set_xticks(chan); ax.set_xticklabels(["hue", "sat", "val"])
    ax.set_ylabel("R^2")
    ax.set_title("R^2 per HSV channel")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # P3: ARD tau trace (aux_conditional_plus_ard)
    ax = axes[1, 0]
    tau_tr = fits["aux_conditional_plus_ard"]["tau_trace"]
    if tau_tr.ndim == 2 and tau_tr.shape[0] > 1:
        for j in range(tau_tr.shape[1]):
            ax.plot(tau_tr[:, j], lw=1.4, label=f"tau axis {j}")
        ax.set_yscale("log")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("ARD precision tau_j (log)")
    ax.set_title("ARD tau trace (aux_conditional_plus_ard)\n"
                 f"final tau={fits['aux_conditional_plus_ard'].get('tau', np.zeros(D_AUX))}\n"
                 f"axes_kept={fits['aux_conditional_plus_ard']['axes_kept']}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P4: recovered-hue vs ground-truth-hue scatter for best fit
    ax = axes[1, 1]
    pred = fits[best_name]["pred_hsv"]
    gt_hue = hsv[:, 0]
    pr_hue = pred[:, 0] + hsv[:, 0].mean()  # pred is centered; add back mean
    sc_colors = np.clip(rgb, 0, 1)
    ax.scatter(gt_hue, pr_hue, c=sc_colors, s=12, alpha=0.85, edgecolors="none")
    lo, hi = float(min(gt_hue.min(), pr_hue.min())), float(max(gt_hue.max(), pr_hue.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
    ax.set_xlabel("ground-truth hue"); ax.set_ylabel("recovered hue")
    ax.set_title(f"Best fit: {best_name}  R^2(hue)={report[best_name]['R2_hue']:.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "auto_exp_33: AuxConditional+ARD on cogito L40, d_aux=3 HSV\n"
        f"path_taken={path_taken} | hyp(a)={h_a}  hyp(b)={h_b}  hyp(c)={h_c}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    runtime_s = time.time() - t_start
    out = {
        "gamfit_version": ver,
        "path_taken": path_taken,
        "experiment": "auto_exp_33",
        "config": {
            "N_TEMPLATES": N_TEMPLATES, "K_PCS": K_PCS, "D_AUX": D_AUX,
            "N_ITER": N_ITER, "AUX_WEIGHT": AUX_WEIGHT,
            "ARD_PRUNE_TAU": ARD_PRUNE_TAU,
            "n_colors": int(n_c),
        },
        "fits": report,
        "best_fit": best_name,
        "hypotheses": hypotheses,
        "hypothesis_verdicts": {
            "a_AuxCondARD_R2_hue_ge_0.55": h_a,
            "b_ARD_does_not_prune_below_3": h_b,
            "c_per_axis_HSV_corr_gt_0.30": h_c,
        },
        "runtime_seconds": runtime_s,
        "prediction_slot_for_v0.1.121_retest": {
            "expected_path_taken": "gamfit_real_wrappers",
            "expected_R2_hue_delta": "rerun once AuxConditionalPriorPenalty + ARDPenalty "
                                    "are exported in installed gamfit; compare deltas",
            "this_run_R2_hue_best": float(report[best_name]["R2_hue"]),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] saved {OUT_JSON}")
    print(f"[runtime] {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
