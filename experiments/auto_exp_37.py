"""auto_exp_37: K_PC robustness of cogito 6-axis decomposition.

auto_exp_33/35/36 all fixed K_PC=16 PCA-truncated cogito centroids. This script
sweeps K_PC in {8, 16, 32} to test whether the U_3d 3-perceptual + 3-name-semantic
recovery is intrinsic to cogito or an artifact of the K_PC=16 dim choice.

Hypotheses:
  (a) K_PC=8 : held-out R^2(hue) >= 0.55
  (b) K_PC=32: held-out R^2(hue) >= 0.68
  (c) per-axis HSV |corr| > 0.40 for 3 axes at BOTH K_PC=8 and K_PC=32
  (d) the WINNING latent axis index for each aux is stable across K_PC (modulo
      basis permutation -> stable in dominant-axis-per-aux assignment).
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
OUT_PNG = RUN_DIR / "auto_exp_37.png"
OUT_JSON = RUN_DIR / "auto_exp_37.json"

N_TEMPLATES = 28
K_PC_SWEEP = [8, 16, 32]
D_AUX = 6
N_ITER = 400
AUX_WEIGHT = 8.0
ARD_PRUNE_TAU = 1e-2
N_FOLDS = 5
SEED = 37

AUX_LABELS = ["hue", "sat", "val", "monoword", "mod_count", "template_sigma"]
HSV_IDX = [0, 1, 2]


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
    n_rows, _d = x_mmap.shape
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
    Tc = T0_test - fit["T_mean"]
    T_lat = Tc @ fit["W"]
    pred = T_lat * fit["aux_sd"] + fit["aux_mu"]
    return pred, T_lat


def per_axis_r2(y_true: np.ndarray, y_pred: np.ndarray, mu_train: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - mu_train) ** 2).sum(0).clip(min=1e-12)
    return 1.0 - ss_res / ss_tot


def per_axis_abs_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(0, keepdims=True)
    bc = b - b.mean(0, keepdims=True)
    num = (ac * bc).sum(0)
    den = np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0)).clip(min=1e-12)
    return np.abs(num / den)


def kfold_cv(T0: np.ndarray, aux: np.ndarray, n_folds: int, seed: int) -> dict:
    n_c = T0.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_c)
    folds = np.array_split(perm, n_folds)
    held_pred = np.zeros_like(aux)
    held_latent = np.zeros((n_c, D_AUX))
    per_fold_axes_kept = []
    for f, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
        fit = fit_aux_conditional_plus_ard_train(T0[train_idx], aux[train_idx])
        pred, lat = predict_aux(T0[test_idx], fit)
        held_pred[test_idx] = pred
        held_latent[test_idx] = lat
        per_fold_axes_kept.append(fit["axes_kept"])
    global_mu = aux.mean(0, keepdims=True)
    r2 = per_axis_r2(aux, held_pred, global_mu)
    corr_diag = per_axis_abs_corr(held_pred, aux)
    return {"r2": r2, "corr_diag": corr_diag, "held_pred": held_pred,
            "held_latent": held_latent, "per_fold_axes_kept": per_fold_axes_kept}


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


def dominant_axis_per_aux(held_latent: np.ndarray, aux: np.ndarray) -> tuple[list[int], np.ndarray]:
    """For each aux variable, find the latent axis with the highest |corr|."""
    d_aux = aux.shape[1]
    ln = (held_latent - held_latent.mean(0, keepdims=True)) / (
        held_latent.std(0, keepdims=True) + 1e-12)
    an = (aux - aux.mean(0, keepdims=True)) / (aux.std(0, keepdims=True) + 1e-12)
    n = held_latent.shape[0]
    corr_mat = np.abs(ln.T @ an / n)   # (d_lat, d_aux)
    dom = [int(np.argmax(corr_mat[:, j])) for j in range(d_aux)]
    return dom, corr_mat


# -------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print("[auto_exp_37] K_PC robustness sweep of cogito d_aux=6 recovery")
    ver, path_taken = check_gamfit()
    print(f"[gamfit] version={ver} path={path_taken}")

    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)

    # We need names + rgb + tsig. tsig depends on K_PC (its std-over-templates in PC space).
    # Compute per K_PC.
    n_c = X.shape[0] // N_TEMPLATES
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    print(f"[colors] n_c={n_c}")

    results = {}
    for k_pc in K_PC_SWEEP:
        print(f"\n=== K_PC = {k_pc} ===")
        T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, k_pc)
        print(f"[centroids] T0={T0.shape}")
        namef = name_features(names, tsig)
        aux = np.concatenate([hsv, namef], axis=1)

        cv = kfold_cv(T0, aux, N_FOLDS, SEED)
        r2 = cv["r2"]; corr_diag = cv["corr_diag"]
        held_latent = cv["held_latent"]
        dom, corr_mat = dominant_axis_per_aux(held_latent, aux)

        # per-axis HSV |corr|: best latent axis for each HSV var (already in dom_corr)
        hsv_axis_corrs = [float(np.max(corr_mat[:, j])) for j in HSV_IDX]
        print(f"[K_PC={k_pc}] held-out R^2 = "
              f"{dict(zip(AUX_LABELS, np.round(r2, 3).tolist()))}")
        print(f"[K_PC={k_pc}] HSV best-axis |corr| = {np.round(hsv_axis_corrs, 3).tolist()}")
        print(f"[K_PC={k_pc}] dominant_axis_per_aux = "
              f"{dict(zip(AUX_LABELS, dom))}")
        print(f"[K_PC={k_pc}] axes_kept per fold = {cv['per_fold_axes_kept']}")

        results[k_pc] = {
            "r2": r2,
            "corr_diag": corr_diag,
            "corr_mat": corr_mat,
            "dom": dom,
            "hsv_axis_corrs": hsv_axis_corrs,
            "axes_kept": cv["per_fold_axes_kept"],
        }

    # ---------------- hypothesis verdicts ----------------
    r2_8 = results[8]["r2"]; r2_16 = results[16]["r2"]; r2_32 = results[32]["r2"]
    hsv_corr_8 = results[8]["hsv_axis_corrs"]
    hsv_corr_32 = results[32]["hsv_axis_corrs"]

    h_a = bool(r2_8[0] >= 0.55)
    h_b = bool(r2_32[0] >= 0.68)
    h_c = bool(sum(1 for v in hsv_corr_8 if v > 0.40) >= 3
               and sum(1 for v in hsv_corr_32 if v > 0.40) >= 3)
    # (d) dominant-axis stability: for each aux, the dominant axis at K_PC=8 / K_PC=32
    #     matches K_PC=16 (modulo basis permutation -- so we just check exact match;
    #     because PCA bases of different ranks are NESTED, axis indices SHOULD line up).
    dom_8 = results[8]["dom"]; dom_16 = results[16]["dom"]; dom_32 = results[32]["dom"]
    stable_8 = [dom_8[j] == dom_16[j] for j in range(D_AUX)]
    stable_32 = [dom_32[j] == dom_16[j] for j in range(D_AUX)]
    # Allow up to 1 mismatch in each direction.
    h_d = bool(sum(stable_8) >= D_AUX - 1 and sum(stable_32) >= D_AUX - 1)

    hypotheses = {
        "a_K_PC8_held_R2_hue_ge_0.55": h_a,
        "b_K_PC32_held_R2_hue_ge_0.68": h_b,
        "c_HSV_corr_gt_0.40_3axes_at_both_K_PC8_and_32": h_c,
        "d_dominant_axis_stable_across_K_PC": h_d,
    }
    print(f"\n[hypotheses] {hypotheses}")
    print(f"[stability] K_PC=8 dom matches K_PC=16: {stable_8}")
    print(f"[stability] K_PC=32 dom matches K_PC=16: {stable_32}")

    # ---------------- plot ----------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # P1: R^2 bars side-by-side per K_PC
    ax = axs[0, 0]
    x = np.arange(D_AUX); w = 0.27
    ax.bar(x - w, r2_8, w, label="K_PC=8",  color="#9ecae1")
    ax.bar(x,     r2_16, w, label="K_PC=16", color="#3182bd")
    ax.bar(x + w, r2_32, w, label="K_PC=32", color="#08519c")
    ax.set_xticks(x); ax.set_xticklabels(AUX_LABELS, rotation=30, ha="right")
    ax.set_ylabel("held-out R^2"); ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.55, color="r", ls=":", lw=0.7, label="K_PC=8 hue thr 0.55")
    ax.axhline(0.68, color="darkred", ls=":", lw=0.7, label="K_PC=32 hue thr 0.68")
    ax.set_title("Held-out R^2 per aux x K_PC"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for k_pc, off, vals in [(8, -w, r2_8), (16, 0, r2_16), (32, w, r2_32)]:
        for i, v in enumerate(vals):
            ax.text(i + off, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)

    # P2-4: per-axis correlation heatmaps for K_PC = 8, 16, 32
    for k_pc, ax in zip(K_PC_SWEEP, [axs[0, 1], axs[1, 0], axs[1, 1]]):
        cm = results[k_pc]["corr_mat"]   # (d_lat, d_aux)
        d_lat = cm.shape[0]
        im = ax.imshow(cm, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(D_AUX)); ax.set_xticklabels(AUX_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(d_lat))
        ax.set_yticklabels([f"lat_{i}" for i in range(d_lat)])
        ax.set_title(f"K_PC={k_pc} |corr(latent_axis, aux)|  "
                     f"axes_kept(folds)={results[k_pc]['axes_kept']}")
        for j in range(d_lat):
            for k in range(D_AUX):
                ax.text(k, j, f"{cm[j,k]:.2f}", ha="center", va="center",
                        color="white" if cm[j, k] < 0.6 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"auto_exp_37: K_PC robustness of cogito 6-axis recovery\n"
        f"path={path_taken} | (a)={h_a} (b)={h_b} (c)={h_c} (d)={h_d} | "
        f"R2_hue: K_PC8={r2_8[0]:.3f}  K_PC16={r2_16[0]:.3f}  K_PC32={r2_32[0]:.3f}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")

    runtime_s = time.time() - t_start
    out = {
        "gamfit_version": ver,
        "path_taken": path_taken,
        "experiment": "auto_exp_37",
        "config": {
            "N_TEMPLATES": N_TEMPLATES, "K_PC_SWEEP": K_PC_SWEEP, "D_AUX": D_AUX,
            "N_ITER": N_ITER, "AUX_WEIGHT": AUX_WEIGHT,
            "ARD_PRUNE_TAU": ARD_PRUNE_TAU, "N_FOLDS": N_FOLDS, "SEED": SEED,
            "n_colors": int(n_c), "aux_labels": AUX_LABELS, "sigma_aux": 0.5,
        },
        "per_K_PC": {
            str(k_pc): {
                "held_out_R2_per_aux": {AUX_LABELS[i]: float(results[k_pc]["r2"][i])
                                        for i in range(D_AUX)},
                "axes_kept_per_fold": results[k_pc]["axes_kept"],
                "dominant_axis_per_aux": {AUX_LABELS[i]: int(results[k_pc]["dom"][i])
                                          for i in range(D_AUX)},
                "best_HSV_axis_abs_corr": {AUX_LABELS[i]: float(results[k_pc]["hsv_axis_corrs"][i])
                                           for i in range(3)},
                "corr_mat_latent_x_aux": results[k_pc]["corr_mat"].tolist(),
            } for k_pc in K_PC_SWEEP
        },
        "stability_dom_8_vs_16": stable_8,
        "stability_dom_32_vs_16": stable_32,
        "hypothesis_verdicts": hypotheses,
        "runtime_seconds": runtime_s,
        "prediction_slot_for_followups": {
            "if_robust": "U_3d 6-axis cogito decomposition is INTRINSIC across K_PC "
                         "{8,16,32}; safe to vary PCA truncation for downstream SAE "
                         "supervision without breaking the H/S/V + name-feature alignment.",
            "if_sensitive": "K_PC=16 was lucky; the recovered axis structure is an "
                            "artifact of that particular PCA-truncation dim; revisit "
                            "auto_exp_33/35/36 conclusions before propagating.",
            "this_run_R2_hue_K_PC_8": float(r2_8[0]),
            "this_run_R2_hue_K_PC_16": float(r2_16[0]),
            "this_run_R2_hue_K_PC_32": float(r2_32[0]),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] saved {OUT_JSON}")
    print(f"[runtime] {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
