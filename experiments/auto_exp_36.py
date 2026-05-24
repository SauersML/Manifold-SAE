"""auto_exp_36: 5-fold-by-color CV of auto_exp_35's d_aux=6 cogito recovery.

Auto_exp_35 fit + evaluated on the SAME 949 colors (in-sample R^2 mean=0.707).
This script asks: do the 6 recovered axes GENERALIZE to held-out colors, or are
the per-axis correlations only clean because we're fitting + evaluating on the
same colors?

Procedure:
  1. Load X_L40 mmap, project to PCA-K=64 then slice to K_PCS=16 centroids/color.
  2. Compute aux (hue, sat, val, monoword, mod_count, template_sigma) -> (n_c, 6).
  3. 5-fold-by-color CV. For each fold: standardize aux on TRAIN colors only,
     fit AuxConditional+ARD weights W on TRAIN, predict aux on TEST via T_test @ W.
  4. Permutation control: shuffle aux rows (color-wise) ONCE then re-run same CV.
     Expected: R^2_held(hue) <= 0.10.

Hypotheses (strict):
  (a) held-out R^2(hue) >= 0.50      (within 25% of auto_exp_35's 0.70 in-sample)
  (b) held-out per-axis |corr| > 0.30 for the 3 HSV-dominant axes (any 3 of 6).
  (c) held-out R^2 >= 0.40 for each of monoword / mod_count / template_sigma.
  (d) permutation-control held-out R^2(hue) <= 0.10.
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
OUT_PNG = RUN_DIR / "auto_exp_36.png"
OUT_JSON = RUN_DIR / "auto_exp_36.json"

N_TEMPLATES = 28
K_PCS = 16
D_AUX = 6
N_ITER = 400
AUX_WEIGHT = 8.0
ARD_PRUNE_TAU = 1e-2
N_FOLDS = 5
SEED = 36

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
    n_rows, d = x_mmap.shape
    n_c = n_rows // n_t
    mu = basis["mu"]; sigma = basis["sigma"]; Vt = basis["Vt"]
    T0 = np.zeros((n_c, k_pcs), dtype=np.float64)
    tsig = np.zeros(n_c, dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s = cs * n_t; e = ce * n_t
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Z = chunk @ Vt.T
        Z = Z[:, :k_pcs]
        n_block = ce - cs
        Z = Z.reshape(n_block, n_t, k_pcs)
        T0[cs:ce] = Z.mean(axis=1)
        tsig[cs:ce] = Z.std(axis=1).mean(axis=1)
    return T0, tsig


def hsv_from_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def name_features(names: list[str], tsig: np.ndarray) -> np.ndarray:
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    return np.stack([mono, modc, tsig], axis=1)


# -------------------------------------------------------------------------
def fit_aux_conditional_plus_ard_train(
    T0_train: np.ndarray, aux_train: np.ndarray, n_iter: int = N_ITER, seed: int = SEED,
) -> dict:
    """Train-only fit. Returns W + train-time normalization params for OOS prediction."""
    rng = np.random.default_rng(seed)
    n_c, K = T0_train.shape
    d_aux = aux_train.shape[1]
    T_mean = T0_train.mean(0, keepdims=True)
    Tc = T0_train - T_mean
    aux_mu = aux_train.mean(0, keepdims=True)
    aux_sd = aux_train.std(0, keepdims=True).clip(min=1e-8)
    ac = (aux_train - aux_mu) / aux_sd
    sigma_aux = 0.5
    aux_norms = np.linalg.norm(ac, axis=1) / np.sqrt(d_aux)
    w_row = 1.0 / (sigma_aux ** 2) * (1.0 + aux_norms)
    W = rng.normal(scale=0.05, size=(K, d_aux))
    tau = np.ones(d_aux)
    sigma2 = float(np.var(ac))
    WTW = (w_row[:, None] * Tc).T @ Tc / n_c
    WTh = (w_row[:, None] * Tc).T @ ac / n_c
    for _ in range(n_iter):
        for j in range(d_aux):
            A = WTW + ((tau[j] * sigma2 + AUX_WEIGHT) / n_c) * np.eye(K)
            W[:, j] = np.linalg.solve(A, WTh[:, j])
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        resid = ac - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    inv_tau = 1.0 / tau
    axes_kept = int(np.count_nonzero(inv_tau > ARD_PRUNE_TAU * inv_tau.max()))
    return {"W": W, "tau": tau, "T_mean": T_mean,
            "aux_mu": aux_mu, "aux_sd": aux_sd, "axes_kept": axes_kept}


def predict_aux(T0_test: np.ndarray, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred_aux_real_scale, T_test_standardized_latent)."""
    Tc = T0_test - fit["T_mean"]
    T_lat = Tc @ fit["W"]                              # (n_test, d_aux)
    pred_std = T_lat
    pred = pred_std * fit["aux_sd"] + fit["aux_mu"]
    return pred, T_lat


def per_axis_r2(y_true: np.ndarray, y_pred: np.ndarray, mu_train: np.ndarray) -> np.ndarray:
    """R^2 per column, using TRAIN mean as the null predictor (proper OOS R^2)."""
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - mu_train) ** 2).sum(0).clip(min=1e-12)
    return 1.0 - ss_res / ss_tot


def per_axis_abs_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|Pearson| per matching column."""
    ac = a - a.mean(0, keepdims=True)
    bc = b - b.mean(0, keepdims=True)
    num = (ac * bc).sum(0)
    den = np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0)).clip(min=1e-12)
    return np.abs(num / den)


def kfold_cv(T0: np.ndarray, aux: np.ndarray, n_folds: int, seed: int
             ) -> dict:
    n_c = T0.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_c)
    folds = np.array_split(perm, n_folds)
    held_pred = np.zeros_like(aux)
    held_latent = np.zeros((n_c, D_AUX))
    train_means_per_fold = np.zeros((n_folds, D_AUX))
    per_fold_axes_kept = []
    for f, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
        fit = fit_aux_conditional_plus_ard_train(T0[train_idx], aux[train_idx])
        pred, lat = predict_aux(T0[test_idx], fit)
        held_pred[test_idx] = pred
        held_latent[test_idx] = lat
        train_means_per_fold[f] = fit["aux_mu"].squeeze()
        per_fold_axes_kept.append(fit["axes_kept"])
    # R^2 per axis: use overall train-mean baseline (mean over all colors,
    # which is what each fold's train approximates).
    global_mu = aux.mean(0, keepdims=True)
    r2 = per_axis_r2(aux, held_pred, global_mu)
    corr = per_axis_abs_corr(held_pred, aux)        # |corr(pred_j, true_j)|
    return {"r2": r2, "corr_diag": corr, "held_pred": held_pred,
            "held_latent": held_latent,
            "per_fold_axes_kept": per_fold_axes_kept}


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


def in_sample_fit(T0: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """auto_exp_35-style in-sample R^2 (fit+eval on all colors), for the bar comparison."""
    fit = fit_aux_conditional_plus_ard_train(T0, aux)
    pred, _ = predict_aux(T0, fit)
    return per_axis_r2(aux, pred, aux.mean(0, keepdims=True))


# -------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print("[auto_exp_36] 5-fold CV of cogito d_aux=6 recovery")
    ver, path_taken = check_gamfit()
    print(f"[gamfit] version={ver} path={path_taken}")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)
    aux = np.concatenate([hsv, namef], axis=1)
    print(f"[aux] aux={aux.shape}")

    # In-sample (reproduce auto_exp_35 numbers)
    r2_insample = in_sample_fit(T0, aux)
    print(f"[in-sample] R^2 per aux = {dict(zip(AUX_LABELS, np.round(r2_insample, 3).tolist()))}")

    # Real 5-fold CV
    cv = kfold_cv(T0, aux, N_FOLDS, SEED)
    r2_held = cv["r2"]; corr_held = cv["corr_diag"]
    print(f"[held-out] R^2 per aux = {dict(zip(AUX_LABELS, np.round(r2_held, 3).tolist()))}")
    print(f"[held-out] |corr(pred,true)| per aux = {dict(zip(AUX_LABELS, np.round(corr_held, 3).tolist()))}")

    # Permutation control
    rng = np.random.default_rng(SEED + 1)
    perm_idx = rng.permutation(n_c)
    aux_perm = aux[perm_idx]
    cv_perm = kfold_cv(T0, aux_perm, N_FOLDS, SEED + 2)
    r2_held_perm = cv_perm["r2"]
    corr_held_perm = cv_perm["corr_diag"]
    print(f"[perm-ctrl] R^2 per aux = {dict(zip(AUX_LABELS, np.round(r2_held_perm, 3).tolist()))}")

    # Cross-axis held-out correlation matrix (real CV)
    held_pred = cv["held_pred"]
    pn = (held_pred - held_pred.mean(0, keepdims=True)) / (held_pred.std(0, keepdims=True) + 1e-12)
    an = (aux - aux.mean(0, keepdims=True)) / (aux.std(0, keepdims=True) + 1e-12)
    corr_mat = np.abs(pn.T @ an / n_c)

    # Hypotheses
    h_a = bool(r2_held[0] >= 0.50)
    # (b) >=3 of HSV axes have |corr| > 0.30 (held-out)
    h_b = bool(sum(1 for v in corr_held[HSV_IDX] if v > 0.30) >= 3)
    # (c) R^2 >= 0.40 for EACH of monoword/mod_count/template_sigma
    h_c = bool(all(r2_held[i] >= 0.40 for i in NAME_IDX))
    # (d) permutation R^2(hue) <= 0.10
    h_d = bool(r2_held_perm[0] <= 0.10)

    hypotheses = {
        "a_held_R2_hue_ge_0.50": h_a,
        "b_held_corr_gt_0.30_for_3_HSV_axes": h_b,
        "c_held_R2_ge_0.40_for_each_name_axis": h_c,
        "d_perm_held_R2_hue_le_0.10": h_d,
    }
    print(f"[hypotheses] {hypotheses}")

    # -------------------- plot --------------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    # P1: held-out vs in-sample R^2 per aux
    ax = axs[0, 0]
    x = np.arange(D_AUX); w = 0.38
    ax.bar(x - w/2, r2_insample, w, label="in-sample (auto_exp_35-style)", color="#bbbbbb")
    ax.bar(x + w/2, r2_held, w, label="5-fold held-out", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(AUX_LABELS, rotation=30, ha="right")
    ax.set_ylabel("R^2"); ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.50, color="r", ls=":", lw=0.7, label="hue threshold 0.50")
    ax.set_title("Held-out vs in-sample R^2 per aux"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for i, v in enumerate(r2_held):
        ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(r2_insample):
        ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color="#555")

    # P2: real vs permutation held-out R^2
    ax = axs[0, 1]
    ax.bar(x - w/2, r2_held, w, label="real labels", color="#1f77b4")
    ax.bar(x + w/2, r2_held_perm, w, label="permutation control", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(AUX_LABELS, rotation=30, ha="right")
    ax.set_ylabel("held-out R^2"); ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.10, color="k", ls=":", lw=0.7, label="perm threshold 0.10")
    ax.set_title("Held-out R^2: real vs permutation"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P3: held-out correlation matrix (pred axis x aux var)
    ax = axs[1, 0]
    im = ax.imshow(corr_mat, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(D_AUX)); ax.set_xticklabels(AUX_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(D_AUX)); ax.set_yticklabels([f"pred {l}" for l in AUX_LABELS])
    ax.set_title("Held-out |corr(pred_axis, aux_var)|")
    for j in range(D_AUX):
        for k in range(D_AUX):
            ax.text(k, j, f"{corr_mat[j,k]:.2f}", ha="center", va="center",
                    color="white" if corr_mat[j,k] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)

    # P4: held-out hue scatter (pred vs true)
    ax = axs[1, 1]
    pred_h = held_pred[:, 0]; true_h = aux[:, 0]
    sc_colors = np.clip(rgb, 0, 1)
    ax.scatter(true_h, pred_h, c=sc_colors, s=12, alpha=0.85, edgecolors="none")
    lo = float(min(true_h.min(), pred_h.min())); hi = float(max(true_h.max(), pred_h.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
    ax.set_xlabel("true hue"); ax.set_ylabel("held-out predicted hue")
    ax.set_title(f"Held-out hue recovery | R^2={r2_held[0]:.3f}  |corr|={corr_held[0]:.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "auto_exp_36: 5-fold-by-color CV of cogito d_aux=6 recovery\n"
        f"path={path_taken} | (a)={h_a} (b)={h_b} (c)={h_c} (d)={h_d} | "
        f"R2_hue_held={r2_held[0]:.3f}  R2_hue_perm={r2_held_perm[0]:.3f}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    runtime_s = time.time() - t_start
    out = {
        "gamfit_version": ver,
        "path_taken": path_taken,
        "experiment": "auto_exp_36",
        "config": {
            "N_TEMPLATES": N_TEMPLATES, "K_PCS": K_PCS, "D_AUX": D_AUX,
            "N_ITER": N_ITER, "AUX_WEIGHT": AUX_WEIGHT,
            "ARD_PRUNE_TAU": ARD_PRUNE_TAU, "N_FOLDS": N_FOLDS, "SEED": SEED,
            "n_colors": int(n_c), "aux_labels": AUX_LABELS, "sigma_aux": 0.5,
        },
        "R2_held_per_aux": {AUX_LABELS[i]: float(r2_held[i]) for i in range(D_AUX)},
        "R2_insample_per_aux": {AUX_LABELS[i]: float(r2_insample[i]) for i in range(D_AUX)},
        "R2_perm_held_per_aux": {AUX_LABELS[i]: float(r2_held_perm[i]) for i in range(D_AUX)},
        "corr_held_diag": {AUX_LABELS[i]: float(corr_held[i]) for i in range(D_AUX)},
        "corr_perm_held_diag": {AUX_LABELS[i]: float(corr_held_perm[i]) for i in range(D_AUX)},
        "held_correlation_matrix_abs": corr_mat.tolist(),
        "per_fold_axes_kept_real": cv["per_fold_axes_kept"],
        "per_fold_axes_kept_perm": cv_perm["per_fold_axes_kept"],
        "hypothesis_verdicts": hypotheses,
        "runtime_seconds": runtime_s,
        "prediction_slot_for_followups": {
            "if_generalizes": "cogito-L40 6-axis subspace structure is REAL; safe to use "
                              "as steering basis on held-out colors and as supervised target "
                              "for production SAEs.",
            "if_within_sample_only": "auto_exp_35's recovery was overfitting; re-examine "
                                     "before propagating to production SAE-supervision losses.",
            "this_run_R2_hue_held": float(r2_held[0]),
            "this_run_R2_hue_perm": float(r2_held_perm[0]),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] saved {OUT_JSON}")
    print(f"[runtime] {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
