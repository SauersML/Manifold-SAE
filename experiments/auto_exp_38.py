"""auto_exp_38: PARTIAL supervision on cogito L40.

Builds on auto_exp_35 (full HSV+name supervision at d_aux=6 cleanly recovered the
U_3d 3-perceptual + 3-name-semantic decomposition). Question here: if we
supervise ONLY HSV on axes 0,1,2 and leave axes 3,4,5 FREE (no aux target), do
the free axes EMERGE to capture name-semantic structure (monoword / mod_count /
template_sigma), or do they remain noisy / isotropic?

Strategy (fallback emulator, matching auto_exp_35's path):
  - W_hsv (K, 3): weighted-LS regression of HSV onto Tc with ARD across the 3
    columns (FIXED tight precision via row-weights, σ_aux=0.5)
  - W_free (K, 3): top-3 PCs of the RESIDUAL data variance after projecting
    out the HSV-predicting subspace. ARD is applied across these 3 also via
    inverse eigenvalue ratios (kept = eig > tau_prune * eig_max).
  - aux for evaluation = concat(HSV, name-features) — name-features used ONLY
    post-hoc, NEVER touched during the fit.

Hypotheses (strict booleans):
 (a) R²(hue) >= 0.65 on supervised axes 0..2 (slight dropoff from auto_exp_33's
     0.70 acceptable for partial-supervision).
 (b) Free axes 3,4,5 DO NOT clearly align with name-features:
     max per-axis |corr(free_axis, name_feature)| < 0.40.
 (c) Free axes are roughly isotropic: ratio max_eig / min_eig of the 3x3
     covariance of axes 3,4,5 is < 3.0.

If (b) is FALSE -> bigger discovery than auto_exp_35: name-semantic subspace
unsupervisedly recoverable in cogito given the right capacity + a gauge-fixing
companion on supervised axes.
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
OUT_PNG = RUN_DIR / "auto_exp_38.png"
OUT_JSON = RUN_DIR / "auto_exp_38.json"

N_TEMPLATES = 28
K_PCS = 16
D_AUX_SUP = 3       # HSV only
D_AUX_FREE = 3      # unsupervised companion axes
D_AUX_TOTAL = D_AUX_SUP + D_AUX_FREE
N_ITER = 400
AUX_WEIGHT = 8.0
ARD_PRUNE_TAU = 1e-2
SIGMA_AUX = 0.5

AUX_LABELS_HSV = ["hue", "sat", "val"]
AUX_LABELS_NAME = ["monoword", "mod_count", "template_sigma"]


# ----- shared helpers (copied from auto_exp_35) ----------------------------
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


def per_color_stats_mmap(x_mmap, n_t, basis, k_pcs):
    n_rows, d = x_mmap.shape
    n_c = n_rows // n_t
    mu = basis["mu"]
    sigma = basis["sigma"]
    Vt = basis["Vt"]
    T0 = np.zeros((n_c, k_pcs), dtype=np.float64)
    tsig = np.zeros(n_c, dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s = cs * n_t
        e = ce * n_t
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Z = chunk @ Vt.T
        Z = Z[:, :k_pcs]
        n_block = ce - cs
        Z = Z.reshape(n_block, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def name_features(names, tsig):
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    return np.stack([mono, modc, tsig], axis=1)


# ----- fit -----------------------------------------------------------------
def fit_aux_supervised_hsv(T0, hsv, n_iter=N_ITER):
    """AuxConditional+ARD on JUST the HSV columns -> W_hsv (K, 3)."""
    rng = np.random.default_rng(38)
    n_c, K = T0.shape
    d_aux = hsv.shape[1]
    Tc = T0 - T0.mean(0, keepdims=True)
    aux_mu = hsv.mean(0, keepdims=True)
    aux_sd = hsv.std(0, keepdims=True).clip(min=1e-8)
    ac = (hsv - aux_mu) / aux_sd
    aux_norms = np.linalg.norm(ac, axis=1) / np.sqrt(d_aux)
    w_row = 1.0 / (SIGMA_AUX ** 2) * (1.0 + aux_norms)
    W = rng.normal(scale=0.05, size=(K, d_aux))
    tau = np.ones(d_aux)
    sigma2 = float(np.var(ac))
    WTW = (w_row[:, None] * Tc).T @ Tc / n_c
    WTh = (w_row[:, None] * Tc).T @ ac / n_c
    tau_trace = []
    for _ in range(n_iter):
        for j in range(d_aux):
            A = WTW + ((tau[j] * sigma2 + AUX_WEIGHT) / n_c) * np.eye(K)
            W[:, j] = np.linalg.solve(A, WTh[:, j])
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        tau_trace.append(tau.copy())
        resid = ac - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    T = Tc @ W
    pred = T * aux_sd + aux_mu
    aux_centered = hsv - aux_mu
    pred_centered = pred - aux_mu
    r2 = 1.0 - ((aux_centered - pred_centered) ** 2).sum(0) / \
        (aux_centered ** 2).sum(0).clip(min=1e-12)
    return {"T_sup": T, "W_sup": W, "tau_sup": tau, "r2_hsv": r2,
            "aux_mu": aux_mu.squeeze(), "aux_sd": aux_sd.squeeze(),
            "tau_trace": np.asarray(tau_trace)}


def fit_free_axes_pca(T0, W_sup, d_free=D_AUX_FREE):
    """Top-d_free PCs of Tc after projecting out the span of W_sup.

    This gives axes 3..5 the SAME functional form as the supervised axes
    (linear projections of Tc) but their directions are chosen to maximize
    residual data variance — purely unsupervised w.r.t. any aux variable.
    Apply ARD-style pruning on the residual eigenvalues to count axes_kept.
    """
    Tc = T0 - T0.mean(0, keepdims=True)
    # Project out W_sup column span via QR
    Q, _ = np.linalg.qr(W_sup)  # (K, d_sup) orthonormal
    P_perp = np.eye(W_sup.shape[0]) - Q @ Q.T
    # Residual representation in the K-dim feature space
    Tc_perp = Tc @ P_perp  # (n_c, K) — variance only in orthogonal complement
    # SVD -> top d_free directions
    U_svd, S_svd, Vt_svd = np.linalg.svd(Tc_perp, full_matrices=False)
    # Right singular vectors are directions in the K-dim feature space
    W_free = Vt_svd[:d_free].T  # (K, d_free)
    T_free = Tc @ W_free        # (n_c, d_free)
    eig = (S_svd ** 2)[:d_free] / max(Tc_perp.shape[0] - 1, 1)
    # ARD-style kept count: kept if eig > ARD_PRUNE_TAU * eig.max()
    kept_free = int(np.count_nonzero(eig > ARD_PRUNE_TAU * eig.max()))
    return {"T_free": T_free, "W_free": W_free,
            "eig_free": eig, "kept_free": kept_free,
            "all_eig_residual": (S_svd ** 2) / max(Tc_perp.shape[0] - 1, 1)}


def abs_corr_matrix(T, aux):
    Tc = T - T.mean(0, keepdims=True)
    ac = aux - aux.mean(0, keepdims=True)
    Tn = Tc / (Tc.std(0, keepdims=True) + 1e-12)
    An = ac / (ac.std(0, keepdims=True) + 1e-12)
    return np.abs(Tn.T @ An / Tn.shape[0])


def check_gamfit():
    try:
        import gamfit
        ver = getattr(gamfit, "__version__", "unknown")
        has_aux = hasattr(gamfit, "AuxConditionalPriorPenalty")
        has_ard = hasattr(gamfit, "ARDPenalty")
        if has_aux and has_ard:
            return ver, "gamfit_real_wrappers"
        return ver, "fallback_python_aux_prior"
    except Exception as exc:
        return f"unavailable:{exc!r}", "fallback_python_aux_prior_no_gamfit"


# ----- main ----------------------------------------------------------------
def main():
    t_start = time.time()
    print("[auto_exp_38] PARTIAL supervision: HSV on axes 0..2, free axes 3..5")

    ver, path_taken = check_gamfit()
    print(f"[gamfit] version={ver} path={path_taken}")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    print("[pca] basis loaded K=64")

    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)   # held-out for post-hoc eval ONLY
    print(f"[aux] hsv={hsv.shape} (supervised); namef={namef.shape} (held-out)")

    # ---- Fit supervised HSV axes
    sup = fit_aux_supervised_hsv(T0, hsv)
    r2_hsv = sup["r2_hsv"]
    print(f"[fit] supervised HSV R^2: hue={r2_hsv[0]:.3f} sat={r2_hsv[1]:.3f} "
          f"val={r2_hsv[2]:.3f}")

    # ---- Fit free axes (unsupervised PCA of residual)
    free = fit_free_axes_pca(T0, sup["W_sup"])
    print(f"[fit] free axes eig: {np.round(free['eig_free'], 4)} kept={free['kept_free']}")

    # ---- Combined latent
    T_all = np.concatenate([sup["T_sup"], free["T_free"]], axis=1)  # (n_c, 6)

    # ---- Correlations with HSV (6 axes x 3) and with name-features (6 x 3)
    corr_hsv = abs_corr_matrix(T_all, hsv)         # (6, 3)
    corr_name = abs_corr_matrix(T_all, namef)      # (6, 3)
    print("[corr] |corr(latent, HSV)|:")
    print(np.round(corr_hsv, 2))
    print("[corr] |corr(latent, name-features)|:")
    print(np.round(corr_name, 2))

    # ---- Per-axis stats
    free_axes = [3, 4, 5]
    # Max per free-axis abs-corr against each name-feature
    free_axis_max_corr_name = []
    free_axis_max_corr_name_per_feature = []
    for j_off, j in enumerate(free_axes):
        c = corr_name[j]
        free_axis_max_corr_name.append({
            "axis": j,
            "max_corr_any_name_feature": float(c.max()),
            "best_name_feature": AUX_LABELS_NAME[int(c.argmax())],
            "per_name_corr": {AUX_LABELS_NAME[i]: float(c[i]) for i in range(3)},
        })
    # The 3 numbers requested: max per-axis correlation against name-features
    free_axes_max_correlation_with_name_features = [
        float(corr_name[j].max()) for j in free_axes
    ]

    # ---- Free-axes covariance structure (isotropy check)
    Tfc = free["T_free"] - free["T_free"].mean(0, keepdims=True)
    cov_free = (Tfc.T @ Tfc) / Tfc.shape[0]   # (3, 3)
    eig_cov = np.linalg.eigvalsh(cov_free)
    eig_cov_sorted = np.sort(eig_cov)[::-1]
    iso_ratio = float(eig_cov_sorted[0] / max(eig_cov_sorted[-1], 1e-12))
    print(f"[cov] free-axes cov eigs={np.round(eig_cov_sorted, 5)} "
          f"max/min ratio={iso_ratio:.3f}")

    # ---- Hypotheses
    h_a = bool(r2_hsv[0] >= 0.65)
    h_b = bool(max(free_axes_max_correlation_with_name_features) < 0.40)
    h_c = bool(iso_ratio < 3.0)
    hypotheses = {
        "a_R2_hue_ge_0.65": h_a,
        "b_free_axes_max_name_corr_lt_0.40": h_b,
        "c_free_axes_isotropic_ratio_lt_3": h_c,
    }
    print(f"[hypotheses] {hypotheses}")
    print(f"[free-axes max-name-corr] {free_axes_max_correlation_with_name_features}")

    # ------------------------------------------------------------------
    # 4-panel plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    # P1: 6x3 HSV correlations
    ax = axs[0, 0]
    im = ax.imshow(corr_hsv, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(AUX_LABELS_HSV)
    ax.set_yticks(range(D_AUX_TOTAL))
    ax.set_yticklabels([f"axis {j}{'  [SUP]' if j < 3 else '  [FREE]'}"
                        for j in range(D_AUX_TOTAL)])
    ax.set_title("|corr(latent_axis, HSV)|  (3 supervised + 3 free)")
    for j in range(D_AUX_TOTAL):
        for k in range(3):
            ax.text(k, j, f"{corr_hsv[j,k]:.2f}", ha="center", va="center",
                    color="white" if corr_hsv[j,k] < 0.6 else "black",
                    fontsize=9)
    ax.axhline(2.5, color="red", lw=1.5, ls="--")
    fig.colorbar(im, ax=ax, shrink=0.85)

    # P2: 3x3 name-feature correlations on FREE axes only
    ax = axs[0, 1]
    name_corr_free = corr_name[free_axes]  # (3, 3)
    im2 = ax.imshow(name_corr_free, vmin=0, vmax=1.0, cmap="magma", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(AUX_LABELS_NAME, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([f"free axis {j}" for j in free_axes])
    ax.set_title("|corr(FREE axis, name-feature)|  (held-out, NEVER fit)")
    for j in range(3):
        for k in range(3):
            ax.text(k, j, f"{name_corr_free[j,k]:.2f}", ha="center", va="center",
                    color="white" if name_corr_free[j,k] < 0.6 else "black",
                    fontsize=10)
    ax.axhline(-0.5, color="cyan", lw=0)  # spacer
    fig.colorbar(im2, ax=ax, shrink=0.85)

    # P3: free-axes pairwise covariance heatmap
    ax = axs[1, 0]
    im3 = ax.imshow(cov_free, cmap="coolwarm", aspect="auto",
                    vmin=-abs(cov_free).max(), vmax=abs(cov_free).max())
    ax.set_xticks(range(3)); ax.set_xticklabels([f"axis {j}" for j in free_axes])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"axis {j}" for j in free_axes])
    for j in range(3):
        for k in range(3):
            ax.text(k, j, f"{cov_free[j,k]:.3g}", ha="center", va="center",
                    fontsize=9,
                    color="black" if abs(cov_free[j,k]) < 0.5 * abs(cov_free).max()
                    else "white")
    ax.set_title(f"FREE axes cov (eigs={np.round(eig_cov_sorted,4)} | "
                 f"max/min={iso_ratio:.2f})")
    fig.colorbar(im3, ax=ax, shrink=0.85)

    # P4: R² bars — HSV (supervised) + R²-equivalent for free axes against
    # their best name-feature (held-out)
    ax = axs[1, 1]
    # For free axes, compute "best held-out R²" — using a simple 1d OLS of
    # name-feature on that single axis (gives R² = corr²).
    free_r2_names = [float(c ** 2) for c in
                     [corr_name[j].max() for j in free_axes]]
    free_r2_labels = [f"axis {j}\n→{AUX_LABELS_NAME[int(corr_name[j].argmax())]}"
                      for j in free_axes]
    all_labels = AUX_LABELS_HSV + free_r2_labels
    all_vals = list(r2_hsv) + free_r2_names
    bcolors = ["#d62728"] * 3 + ["#9467bd"] * 3
    bars = ax.bar(range(len(all_vals)), all_vals, color=bcolors)
    ax.set_xticks(range(len(all_vals)))
    ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R^2  (HSV: supervised fit | free: corr^2 vs best name-feature)")
    ax.set_ylim(min(0, min(all_vals) - 0.05), 1.0)
    ax.axhline(0.65, color="k", ls=":", lw=0.8, label="hyp(a) hue R^2 >= 0.65")
    ax.axhline(0.16, color="g", ls=":", lw=0.8, label="hyp(b) max-corr 0.40 = R^2 0.16")
    for b, v in zip(bars, all_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_title("R^2: supervised HSV (red) vs free-axes best name-feature corr^2 (purple)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        "auto_exp_38: PARTIAL supervision (HSV on 0..2, FREE 3..5) on cogito L40\n"
        f"path={path_taken} | (a)={h_a} (b)={h_b} (c)={h_c} | "
        f"R^2(hue)={r2_hsv[0]:.3f} | free max-name-corr={np.round(free_axes_max_correlation_with_name_features,3)} | "
        f"iso ratio={iso_ratio:.2f}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    runtime_s = time.time() - t_start
    out = {
        "gamfit_version": ver,
        "path_taken": path_taken,
        "experiment": "auto_exp_38",
        "config": {
            "N_TEMPLATES": N_TEMPLATES, "K_PCS": K_PCS,
            "D_AUX_SUP": D_AUX_SUP, "D_AUX_FREE": D_AUX_FREE,
            "D_AUX_TOTAL": D_AUX_TOTAL, "N_ITER": N_ITER,
            "AUX_WEIGHT": AUX_WEIGHT, "ARD_PRUNE_TAU": ARD_PRUNE_TAU,
            "SIGMA_AUX": SIGMA_AUX,
            "n_colors": int(n_c),
        },
        "R2_hsv_supervised": {AUX_LABELS_HSV[i]: float(r2_hsv[i]) for i in range(3)},
        "free_axes_max_correlation_with_name_features":
            free_axes_max_correlation_with_name_features,
        "free_axis_detail": free_axis_max_corr_name,
        "free_axes_covariance_eigs": [float(v) for v in eig_cov_sorted],
        "free_axes_isotropy_ratio_max_over_min": iso_ratio,
        "corr_hsv_all_axes": corr_hsv.tolist(),
        "corr_name_all_axes": corr_name.tolist(),
        "tau_sup_final": [float(v) for v in sup["tau_sup"]],
        "eig_free_top3": [float(v) for v in free["eig_free"]],
        "eig_residual_all": [float(v) for v in free["all_eig_residual"]],
        "hypothesis_verdicts": hypotheses,
        "runtime_seconds": runtime_s,
        "prediction_slot_for_v0.1.121_retest": {
            "expected_path_taken": "gamfit_real_wrappers",
            "expected_behavior_change":
                "with real AuxConditional+ARD wrappers, free axes could potentially "
                "be allocated by ARD pruning instead of by hard PCA of the residual; "
                "compare free-axis name-feature corrs across the two paths to test "
                "whether the gauge-discovery story is method-robust.",
            "this_run_free_axes_max_name_corr":
                free_axes_max_correlation_with_name_features,
            "this_run_R2_hue": float(r2_hsv[0]),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] saved {OUT_JSON}")
    print(f"[runtime] {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
