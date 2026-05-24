"""auto_exp_35: AuxConditional+ARD on cogito L40 with d_aux=6 (HSV + name-features).

Builds on auto_exp_33's d_aux=3 HSV-only win (R²(hue)=0.70, each axis aligned with
one HSV channel — see project_cogito_recovery_at_d_aux_3). Scales to the FULL U_3d
decomposition per project_cogito_color_manifold_decomposition: U_3d's CV R²=0.61
decomposes as 0.32 HSV (perceptual) + 0.29 name-token (semantic).

aux = concat([hue, sat, val, monoword, modifier_count, template_sigma]) -> (n_c, 6)

Hypotheses (strict booleans):
 (a) Total R²(combined aux) >= 0.55 (mean across the 6 aux R²).
 (b) >=3 of 6 axes have max-per-axis |corr(axis, HSV)| > 0.40.
 (c) >=3 of 6 axes have max-per-axis |corr(axis, name-feature)| > 0.30.
 (d) Each axis is dominated by EITHER an HSV channel OR a name-feature
     (cleanly separated subspaces — for each axis, max-HSV-corr and
     max-name-corr should not BOTH be large).

Path: gamfit 0.1.112 still lacks AuxConditionalPriorPenalty/ARDPenalty wrappers
in installed version, so we use the same Python emulator as auto_exp_33
(closed-form weighted-LS + per-axis ARD), tagged in path_taken.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

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
OUT_PNG = RUN_DIR / "auto_exp_35.png"
OUT_JSON = RUN_DIR / "auto_exp_35.json"

N_TEMPLATES = 28
K_PCS = 16
D_AUX = 6           # 3 HSV + 3 name-features
N_ITER = 400
AUX_WEIGHT = 8.0
ARD_PRUNE_TAU = 1e-2

AUX_LABELS = ["hue", "sat", "val", "monoword", "mod_count", "template_sigma"]
HSV_IDX = [0, 1, 2]
NAME_IDX = [3, 4, 5]


# -------------------------------------------------------------------------
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


def per_color_stats_mmap(x_mmap: np.ndarray, n_t: int,
                         basis: dict, k_pcs: int) -> tuple[np.ndarray, np.ndarray]:
    """One streaming pass over X_L40.

    Returns:
      T0_pca    (n_c, k_pcs) — per-color centroid projected to PCA-K=64 then sliced
      template_sigma (n_c,)  — mean across PCA dims of std across templates of that color
    """
    n_rows, d = x_mmap.shape
    n_c = n_rows // n_t
    mu = basis["mu"]
    sigma = basis["sigma"]
    Vt = basis["Vt"]  # (64, d)

    # Process one color at a time to avoid loading whole matrix.
    T0 = np.zeros((n_c, k_pcs), dtype=np.float64)
    tsig = np.zeros(n_c, dtype=np.float64)
    block = 32  # colors per block (32 colors * 28 templates * 7168 * 8 ~ 50MB)
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s = cs * n_t
        e = ce * n_t
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)  # (block*n_t, d)
        # standardize then project to PCA-64 once for the entire chunk
        chunk = (chunk - mu) / sigma
        Z = chunk @ Vt.T  # (block*n_t, 64)
        Z = Z[:, :k_pcs]
        n_block = ce - cs
        Z = Z.reshape(n_block, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        # template std: std across templates per PCA dim, mean across dims
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def name_features(names: list[str], tsig: np.ndarray) -> np.ndarray:
    """Returns (n_c, 3): [monoword (0/1), modifier_count (int), template_sigma (float)]."""
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    return np.stack([mono, modc, tsig], axis=1)


# -------------------------------------------------------------------------
def fit_aux_conditional_plus_ard(T0: np.ndarray, aux: np.ndarray,
                                 n_iter: int = N_ITER) -> dict:
    """Mirror of auto_exp_33's fit_aux_conditional_plus_ard, generalized to D_AUX."""
    rng = np.random.default_rng(35)
    n_c, K = T0.shape
    d_aux = aux.shape[1]
    Tc = T0 - T0.mean(0, keepdims=True)
    # standardize aux per column (HSV channels in [0,1] but name-features wildly different
    # scales -> standardize so AUX_WEIGHT / sigma_aux is comparable per axis)
    aux_mu = aux.mean(0, keepdims=True)
    aux_sd = aux.std(0, keepdims=True).clip(min=1e-8)
    ac = (aux - aux_mu) / aux_sd
    # mirror auto_exp_33: per-row precision; use distance from neutral as confidence
    sigma_aux = 0.5
    aux_norms = np.linalg.norm(ac, axis=1) / np.sqrt(d_aux)
    w_row = 1.0 / (sigma_aux ** 2) * (1.0 + aux_norms)
    W = rng.normal(scale=0.05, size=(K, d_aux))
    tau = np.ones(d_aux)
    sigma2 = float(np.var(ac))
    tau_trace = []
    WTW = (w_row[:, None] * Tc).T @ Tc / n_c
    WTh = (w_row[:, None] * Tc).T @ ac / n_c
    for _ in range(n_iter):
        for j in range(d_aux):
            A = WTW + ((tau[j] * sigma2 + AUX_WEIGHT) / n_c) * np.eye(K)
            W[:, j] = np.linalg.solve(A, WTh[:, j])
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        tau_trace.append(tau.copy())
        resid = ac - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    T = Tc @ W  # predicted standardized aux
    pred = T * aux_sd + aux_mu
    # R^2 per aux channel against ORIGINAL (un-standardized) aux
    aux_centered = aux - aux_mu
    pred_centered = pred - aux_mu
    r2 = 1.0 - ((aux_centered - pred_centered) ** 2).sum(0) / \
        (aux_centered ** 2).sum(0).clip(min=1e-12)
    inv_tau = 1.0 / tau
    axes_kept = int(np.count_nonzero(inv_tau > ARD_PRUNE_TAU * inv_tau.max()))
    return {"T": T, "pred_aux": pred, "r2_aux": r2,
            "axes_kept": axes_kept,
            "tau_trace": np.asarray(tau_trace),
            "tau": tau, "W": W,
            "aux_mu": aux_mu.squeeze(), "aux_sd": aux_sd.squeeze()}


def axis_aux_corr(T: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """|Pearson| matrix (d_aux_latent, d_aux_var)."""
    Tc = T - T.mean(0, keepdims=True)
    ac = aux - aux.mean(0, keepdims=True)
    Tn = Tc / (Tc.std(0, keepdims=True) + 1e-12)
    An = ac / (ac.std(0, keepdims=True) + 1e-12)
    return np.abs(Tn.T @ An / Tn.shape[0])


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
    print("[auto_exp_35] AuxConditional+ARD on cogito L40, d_aux=6 (HSV + name)")

    ver, path_taken = check_gamfit()
    print(f"[gamfit] version={ver} path={path_taken}")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    print("[pca] basis loaded K=64")

    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}, tsig={tsig.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)
    aux = np.concatenate([hsv, namef], axis=1)  # (n_c, 6)
    print(f"[aux] aux={aux.shape}, sample row 0 ({names[0]}) -> {aux[0]}")

    fit = fit_aux_conditional_plus_ard(T0, aux)
    r2 = fit["r2_aux"]
    corr = axis_aux_corr(fit["T"], aux)  # (6, 6)
    print(f"[fit] R2 per aux:")
    for i, lbl in enumerate(AUX_LABELS):
        print(f"   {lbl:>15s}: R2={r2[i]:.3f}")
    print(f"[fit] axes_kept={fit['axes_kept']}")
    print(f"[corr] |corr| matrix (latent_axis rows x aux cols):")
    print(np.round(corr, 2))

    # Per-axis dominance: for each latent axis, which aux variable is it
    # most correlated with?
    axis_dominance = []
    per_axis_hsv_max = []
    per_axis_name_max = []
    for j in range(D_AUX):
        c = corr[j]
        best_idx = int(c.argmax())
        axis_dominance.append({
            "axis": j,
            "best_aux": AUX_LABELS[best_idx],
            "best_corr": float(c[best_idx]),
            "all_corrs": {AUX_LABELS[i]: float(c[i]) for i in range(D_AUX)},
        })
        per_axis_hsv_max.append(float(c[HSV_IDX].max()))
        per_axis_name_max.append(float(c[NAME_IDX].max()))

    # Hypotheses
    h_a = bool(r2.mean() >= 0.55)
    h_b = bool(sum(1 for v in per_axis_hsv_max if v > 0.40) >= 3)
    h_c = bool(sum(1 for v in per_axis_name_max if v > 0.30) >= 3)
    # (d) clean separation: for each axis, NOT (max-HSV > 0.40 AND max-name > 0.30)
    # I.e., axis dominated by ONE side only.
    mixed = [(per_axis_hsv_max[j] > 0.40) and (per_axis_name_max[j] > 0.30)
             for j in range(D_AUX)]
    h_d = bool(not any(mixed))

    hypotheses = {
        "a_total_R2_mean_ge_0.55": h_a,
        "b_at_least_3_axes_HSV_corr_gt_0.40": h_b,
        "c_at_least_3_axes_name_corr_gt_0.30": h_c,
        "d_each_axis_dominated_by_one_subspace": h_d,
    }
    print(f"[hypotheses] {hypotheses}")

    # ------------------------------------------------------------------
    # Plot: 4-panel
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    # P1: 6x6 |corr| heatmap (latent rows, aux cols)
    ax = axs[0, 0]
    im = ax.imshow(corr, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(D_AUX)); ax.set_xticklabels(AUX_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(D_AUX)); ax.set_yticklabels([f"axis {j}" for j in range(D_AUX)])
    ax.set_title("|Pearson(latent_axis, aux_var)|")
    for j in range(D_AUX):
        for k in range(D_AUX):
            ax.text(k, j, f"{corr[j,k]:.2f}", ha="center", va="center",
                    color="white" if corr[j,k] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)

    # P2: per-aux R^2 bars
    ax = axs[0, 1]
    bars = ax.bar(range(D_AUX), r2,
                  color=["#d62728"]*3 + ["#1f77b4"]*3)
    ax.set_xticks(range(D_AUX)); ax.set_xticklabels(AUX_LABELS, rotation=30, ha="right")
    ax.set_ylabel("R^2"); ax.set_ylim(min(0, r2.min() - 0.05), 1.0)
    ax.axhline(0.55, color="k", ls=":", lw=0.8, label="mean threshold 0.55")
    ax.axhline(r2.mean(), color="r", ls="--", lw=0.8,
               label=f"actual mean = {r2.mean():.3f}")
    ax.set_title("R^2 per aux channel (red=HSV, blue=name-features)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=8)

    # P3: axes_kept bar + dominance labels
    ax = axs[1, 0]
    domlabels = [d["best_aux"] for d in axis_dominance]
    domcorrs = [d["best_corr"] for d in axis_dominance]
    bcolors = ["#d62728" if l in {"hue","sat","val"} else "#1f77b4" for l in domlabels]
    ax.bar(range(D_AUX), domcorrs, color=bcolors)
    for j, (lbl, c) in enumerate(zip(domlabels, domcorrs)):
        ax.text(j, c + 0.01, f"{lbl}\n{c:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(D_AUX)); ax.set_xticklabels([f"axis {j}" for j in range(D_AUX)])
    ax.set_ylim(0, 1.0); ax.set_ylabel("|corr| with dominant aux")
    ax.set_title(f"Per-axis dominance | axes_kept={fit['axes_kept']}/{D_AUX}")
    ax.grid(alpha=0.3)

    # P4: recovered-hue vs ground-truth-hue scatter (using the axis most correlated with hue)
    ax = axs[1, 1]
    hue_axis = int(corr[:, 0].argmax())
    T = fit["T"]
    pr = T[:, hue_axis]
    # rescale so pr roughly matches hue range
    pr_z = (pr - pr.mean()) / (pr.std() + 1e-12)
    hue_z = (hsv[:, 0] - hsv[:, 0].mean()) / (hsv[:, 0].std() + 1e-12)
    # align sign
    if np.corrcoef(pr_z, hue_z)[0, 1] < 0:
        pr_z = -pr_z
    sc_colors = np.clip(rgb, 0, 1)
    ax.scatter(hue_z, pr_z, c=sc_colors, s=12, alpha=0.85, edgecolors="none")
    lo = float(min(hue_z.min(), pr_z.min())); hi = float(max(hue_z.max(), pr_z.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
    ax.set_xlabel("ground-truth hue (z-scored)")
    ax.set_ylabel(f"latent axis {hue_axis} (z-scored, sign-aligned)")
    ax.set_title(f"Hue recovery via axis {hue_axis}  |corr|={corr[hue_axis,0]:.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "auto_exp_35: AuxConditional+ARD on cogito L40, d_aux=6 (HSV + name)\n"
        f"path={path_taken} | (a)={h_a} (b)={h_b} (c)={h_c} (d)={h_d} | "
        f"R2_mean={r2.mean():.3f} axes_kept={fit['axes_kept']}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    runtime_s = time.time() - t_start
    out = {
        "gamfit_version": ver,
        "path_taken": path_taken,
        "experiment": "auto_exp_35",
        "config": {
            "N_TEMPLATES": N_TEMPLATES, "K_PCS": K_PCS, "D_AUX": D_AUX,
            "N_ITER": N_ITER, "AUX_WEIGHT": AUX_WEIGHT,
            "ARD_PRUNE_TAU": ARD_PRUNE_TAU,
            "n_colors": int(n_c),
            "aux_labels": AUX_LABELS,
            "sigma_aux": 0.5,
        },
        "R2_per_aux": {AUX_LABELS[i]: float(r2[i]) for i in range(D_AUX)},
        "R2_mean": float(r2.mean()),
        "axes_kept": int(fit["axes_kept"]),
        "tau_final": [float(v) for v in fit["tau"]],
        "correlation_matrix_abs": corr.tolist(),
        "axis_dominance": axis_dominance,
        "per_axis_hsv_max_corr": per_axis_hsv_max,
        "per_axis_name_max_corr": per_axis_name_max,
        "hypothesis_verdicts": hypotheses,
        "runtime_seconds": runtime_s,
        "prediction_slot_for_v0.1.121_retest": {
            "expected_path_taken": "gamfit_real_wrappers",
            "expected_R2_mean_delta": "rerun once AuxConditionalPriorPenalty + ARDPenalty "
                                     "are exported; compare deltas against this fallback",
            "this_run_R2_mean": float(r2.mean()),
            "this_run_R2_per_aux": {AUX_LABELS[i]: float(r2[i]) for i in range(D_AUX)},
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] saved {OUT_JSON}")
    print(f"[runtime] {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
