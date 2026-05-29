"""auto_exp_48: RAW per-prompt residuals vs centroid HSV recovery on cogito L40.

All prior cogito experiments (auto_exp_33..47) collapsed each color's 28 prompts
to a single centroid (n=949 rows). This experiment asks the inverse question:
does fitting the HSV-supervised projection on the FULL 26572 per-prompt rows
recover HSV with HIGHER or LOWER R^2 than centroiding?

  - Higher R^2 raw  -> the template variation contains additional color signal
    that the centroid average smooths out (noise is signal).
  - Lower R^2 raw   -> template variation is genuine nuisance noise that the
    centroid correctly averages out (centroiding is the right preprocessing).

Methodology (mirrors auto_exp_38's HSV-only fit at d_aux_sup=3):
  - mmap X_L40.npy (26572 x 7168), PCA to K=16 with cached basis Vt[:16].
  - Build per-row HSV truth: row i belongs to color (i // 28), HSV via xkcd.
  - 5-fold CV BY COLOR (all 28 templates of each color held together)
    to avoid leakage. n_test_colors ~= 949/5 ~= 190.
  - For each fold: fit W_hsv (K, 3) via the same penalized weighted-LS pass
    used in auto_exp_38, then evaluate R^2 on the held-out rows.
  - Report mean R^2 over folds for (hue, sat, val).

Compare to auto_exp_38 centroid R^2 (in-sample, on 949 centroids):
    hue=0.700  sat=0.657  val=0.719
(Note: 38 is in-sample; we additionally compute an in-sample raw R^2 on all
26572 rows for an apples-to-apples R^2 comparison, plus the held-out CV R^2.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore

ROOT = Path("/Users/user/Manifold-SAE")
RUN_DIR = ROOT / "runs"
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_NPZ = RUN_DIR / "auto_exp_48_results.npz"
OUT_JSON = RUN_DIR / "auto_exp_48_results.json"

N_TEMPLATES = 28
K_PCS = 16
D_AUX_SUP = 3
N_ITER = 400
AUX_WEIGHT = 8.0
SIGMA_AUX = 0.5
N_FOLDS = 5
AUX_LABELS_HSV = ["hue", "sat", "val"]

# Centroid R^2 from auto_exp_38 (memo / project_cogito_recovery_at_d_aux_3.md)
CENTROID_R2 = {"hue": 0.700, "sat": 0.657, "val": 0.719}


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


def hsv_from_rgb(rgb):
    out = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        out[i] = mcolors.rgb_to_hsv(c)
    return out


def project_to_pcs_chunked(x_mmap, basis, k_pcs, chunk=512):
    """Stream-PCA the full mmap'd X to (n_rows, k_pcs). Peak mem <= chunk*D*8."""
    n_rows, d = x_mmap.shape
    mu = basis["mu"]
    sigma = basis["sigma"]
    Vt_k = basis["Vt"][:k_pcs]  # (k, D)
    Z = np.empty((n_rows, k_pcs), dtype=np.float64)
    for s in range(0, n_rows, chunk):
        e = min(s + chunk, n_rows)
        block = np.asarray(x_mmap[s:e], dtype=np.float64)
        block = (block - mu) / sigma
        Z[s:e] = block @ Vt_k.T
    return Z


def fit_aux_supervised_hsv(T_train, hsv_train, n_iter=N_ITER, seed=48):
    """Replicates auto_exp_38's HSV-only fit. Returns W (K,3), aux_mu, aux_sd,
    and T_train_mean for centering."""
    rng = np.random.default_rng(seed)
    n, K = T_train.shape
    d_aux = hsv_train.shape[1]
    T_mean = T_train.mean(0, keepdims=True)
    Tc = T_train - T_mean
    aux_mu = hsv_train.mean(0, keepdims=True)
    aux_sd = hsv_train.std(0, keepdims=True).clip(min=1e-8)
    ac = (hsv_train - aux_mu) / aux_sd
    aux_norms = np.linalg.norm(ac, axis=1) / np.sqrt(d_aux)
    w_row = 1.0 / (SIGMA_AUX ** 2) * (1.0 + aux_norms)
    W = rng.normal(scale=0.05, size=(K, d_aux))
    tau = np.ones(d_aux)
    sigma2 = float(np.var(ac))
    WTW = (w_row[:, None] * Tc).T @ Tc / n
    WTh = (w_row[:, None] * Tc).T @ ac / n
    for _ in range(n_iter):
        for j in range(d_aux):
            A = WTW + ((tau[j] * sigma2 + AUX_WEIGHT) / n) * np.eye(K)
            W[:, j] = np.linalg.solve(A, WTh[:, j])
        w2 = (W ** 2).sum(0)
        tau = K / np.maximum(w2, 1e-8)
        resid = ac - Tc @ W
        sigma2 = float((resid ** 2).mean()) + 1e-8
    return {"W": W, "T_mean": T_mean, "aux_mu": aux_mu, "aux_sd": aux_sd}


def eval_r2(fit, T_eval, hsv_eval):
    """Project T_eval and compute per-axis R^2 vs hsv_eval."""
    W = fit["W"]
    Tc = T_eval - fit["T_mean"]
    pred_std = Tc @ W                  # in standardized aux units
    pred = pred_std * fit["aux_sd"] + fit["aux_mu"]
    aux_mean = hsv_eval.mean(0, keepdims=True)
    ss_res = ((hsv_eval - pred) ** 2).sum(0)
    ss_tot = ((hsv_eval - aux_mean) ** 2).sum(0).clip(min=1e-12)
    return 1.0 - ss_res / ss_tot


def main():
    t0 = time.time()
    print("[auto_exp_48] RAW per-prompt residuals vs centroid HSV (cogito L40)")

    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape} dtype={X.dtype}")
    n_rows = X.shape[0]
    n_colors = n_rows // N_TEMPLATES
    assert n_colors * N_TEMPLATES == n_rows, "expected rows = n_colors * 28"

    basis = load_pc_basis(K=64)
    print(f"[pca] basis loaded, using K={K_PCS}")

    print(f"[project] streaming PCA to ({n_rows}, {K_PCS})")
    Z = project_to_pcs_chunked(X, basis, K_PCS, chunk=512)
    print(f"[project] Z={Z.shape}, mean_norm={np.linalg.norm(Z, axis=1).mean():.3f}")

    # color_idx: row -> color id; map row r -> color (r // 28)
    color_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    assert color_idx.shape[0] == n_rows

    # Load xkcd colors and per-row HSV truth
    names, rgb = load_xkcd_rgb(n_colors)
    hsv_per_color = hsv_from_rgb(rgb)         # (949, 3)
    hsv_per_row = hsv_per_color[color_idx]    # (26572, 3)
    print(f"[hsv] per_color={hsv_per_color.shape} per_row={hsv_per_row.shape}")

    # ---- Centroid-based fit (in-sample, to mirror auto_exp_38 exactly) ----
    print("\n[fit] CENTROID baseline (in-sample, n=949)")
    T_centroid = Z.reshape(n_colors, N_TEMPLATES, K_PCS).mean(axis=1)
    fit_c = fit_aux_supervised_hsv(T_centroid, hsv_per_color)
    r2_centroid_in = eval_r2(fit_c, T_centroid, hsv_per_color)
    print(f"[centroid-in] R^2: hue={r2_centroid_in[0]:.3f} "
          f"sat={r2_centroid_in[1]:.3f} val={r2_centroid_in[2]:.3f}")

    # ---- Raw per-prompt fit (in-sample, n=26572) ----
    print("\n[fit] RAW per-prompt (in-sample, n=26572)")
    fit_r = fit_aux_supervised_hsv(Z, hsv_per_row)
    r2_raw_in_per_row = eval_r2(fit_r, Z, hsv_per_row)
    print(f"[raw-in/per-row] R^2: hue={r2_raw_in_per_row[0]:.3f} "
          f"sat={r2_raw_in_per_row[1]:.3f} val={r2_raw_in_per_row[2]:.3f}")
    # Also evaluate the raw fit aggregated to color centroids (predict per row,
    # average predictions per color, then per-color R^2 vs HSV)
    pred_raw = (Z - fit_r["T_mean"]) @ fit_r["W"] * fit_r["aux_sd"] + fit_r["aux_mu"]
    pred_color = np.zeros_like(hsv_per_color)
    for c in range(n_colors):
        s = c * N_TEMPLATES
        pred_color[c] = pred_raw[s:s + N_TEMPLATES].mean(0)
    ss_res = ((hsv_per_color - pred_color) ** 2).sum(0)
    ss_tot = ((hsv_per_color - hsv_per_color.mean(0, keepdims=True)) ** 2).sum(0)
    r2_raw_in_agg = 1.0 - ss_res / ss_tot.clip(min=1e-12)
    print(f"[raw-in/agg-color] R^2: hue={r2_raw_in_agg[0]:.3f} "
          f"sat={r2_raw_in_agg[1]:.3f} val={r2_raw_in_agg[2]:.3f}")

    # ---- 5-fold CV grouped by COLOR ----
    print(f"\n[cv] {N_FOLDS}-fold grouped-by-color CV")
    rng = np.random.default_rng(48)
    perm = rng.permutation(n_colors)
    folds = np.array_split(perm, N_FOLDS)

    cv_raw_r2 = []          # per-row R^2 on held-out prompts
    cv_raw_agg_r2 = []      # per-color R^2 (aggregating raw preds by color)
    cv_centroid_r2 = []     # centroid R^2 on held-out colors
    for k_fold, test_colors in enumerate(folds):
        train_colors = np.setdiff1d(perm, test_colors, assume_unique=False)
        # Row masks
        is_test_row = np.isin(color_idx, test_colors)
        is_train_row = ~is_test_row
        # Raw fit on training rows
        fit_r_cv = fit_aux_supervised_hsv(
            Z[is_train_row], hsv_per_row[is_train_row], seed=48 + k_fold)
        r2_te_row = eval_r2(fit_r_cv,
                            Z[is_test_row], hsv_per_row[is_test_row])
        # Aggregate raw predictions to test-color centroids
        pred_te = (Z[is_test_row] - fit_r_cv["T_mean"]) @ fit_r_cv["W"] \
            * fit_r_cv["aux_sd"] + fit_r_cv["aux_mu"]
        cidx_te = color_idx[is_test_row]
        pred_agg = np.zeros((len(test_colors), 3), dtype=np.float64)
        for i, c in enumerate(test_colors):
            pred_agg[i] = pred_te[cidx_te == c].mean(0)
        truth_agg = hsv_per_color[test_colors]
        ss_res_a = ((truth_agg - pred_agg) ** 2).sum(0)
        ss_tot_a = ((truth_agg - truth_agg.mean(0, keepdims=True)) ** 2).sum(0)
        r2_te_agg = 1.0 - ss_res_a / ss_tot_a.clip(min=1e-12)
        # Centroid fit on training-color centroids
        T_centroid_tr = Z[is_train_row].reshape(-1, K_PCS)
        # Build per-color centroid for train colors
        T_centroid_train = np.zeros((len(train_colors), K_PCS))
        for i, c in enumerate(train_colors):
            T_centroid_train[i] = Z[color_idx == c].mean(0)
        hsv_train_c = hsv_per_color[train_colors]
        fit_c_cv = fit_aux_supervised_hsv(T_centroid_train, hsv_train_c,
                                          seed=148 + k_fold)
        T_centroid_test = np.zeros((len(test_colors), K_PCS))
        for i, c in enumerate(test_colors):
            T_centroid_test[i] = Z[color_idx == c].mean(0)
        r2_te_c = eval_r2(fit_c_cv, T_centroid_test, hsv_per_color[test_colors])
        cv_raw_r2.append(r2_te_row)
        cv_raw_agg_r2.append(r2_te_agg)
        cv_centroid_r2.append(r2_te_c)
        print(f"[cv fold {k_fold}] n_test_c={len(test_colors):4d} "
              f"raw-row={np.round(r2_te_row,3)} "
              f"raw-agg={np.round(r2_te_agg,3)} "
              f"centroid={np.round(r2_te_c,3)}")

    cv_raw_r2 = np.asarray(cv_raw_r2)
    cv_raw_agg_r2 = np.asarray(cv_raw_agg_r2)
    cv_centroid_r2 = np.asarray(cv_centroid_r2)
    raw_mean = cv_raw_r2.mean(0)
    raw_std = cv_raw_r2.std(0)
    raw_agg_mean = cv_raw_agg_r2.mean(0)
    raw_agg_std = cv_raw_agg_r2.std(0)
    cen_mean = cv_centroid_r2.mean(0)
    cen_std = cv_centroid_r2.std(0)

    # ---- Final table ----
    print("\n" + "=" * 78)
    print(f"{'target':8s}  {'raw R^2 (held-out per-row)':28s}  "
          f"{'raw R^2 agg-color':20s}  {'centroid R^2 (CV)':18s}  "
          f"{'centroid R^2 (memo)':>20s}")
    print("=" * 78)
    for i, lab in enumerate(AUX_LABELS_HSV):
        memo = CENTROID_R2[lab]
        print(f"{lab:8s}  {raw_mean[i]:.3f} +- {raw_std[i]:.3f}              "
              f"{raw_agg_mean[i]:.3f} +- {raw_agg_std[i]:.3f}     "
              f"{cen_mean[i]:.3f} +- {cen_std[i]:.3f}    "
              f"{memo:>14.3f}")
    print("=" * 78)
    shift_vs_memo = {lab: float(raw_mean[i] - CENTROID_R2[lab])
                     for i, lab in enumerate(AUX_LABELS_HSV)}
    shift_agg_vs_memo = {lab: float(raw_agg_mean[i] - CENTROID_R2[lab])
                         for i, lab in enumerate(AUX_LABELS_HSV)}
    print(f"[shift raw-per-row - memo]   {shift_vs_memo}")
    print(f"[shift raw-agg-color - memo] {shift_agg_vs_memo}")

    # Verdict
    mean_shift_row = float(np.mean(list(shift_vs_memo.values())))
    mean_shift_agg = float(np.mean(list(shift_agg_vs_memo.values())))
    if mean_shift_agg > 0.02:
        verdict = ("RAW per-prompt fitting BEATS centroiding when predictions "
                   "are aggregated back to colors -- template variation contains "
                   "additional color signal that centroid-then-fit averages out.")
    elif mean_shift_agg < -0.02:
        verdict = ("CENTROIDING WINS -- per-prompt template variation is noise "
                   "that hurts the supervised fit; averaging is the right "
                   "preprocessing for HSV recovery.")
    else:
        verdict = ("NEUTRAL -- raw and centroid recovery are within +-0.02 R^2 "
                   "of each other; centroiding is a near-lossless reduction "
                   "for HSV recovery.")
    print(f"[verdict] mean shift (raw-agg vs memo) = {mean_shift_agg:+.3f}")
    print(f"[verdict] {verdict}")

    out = {
        "experiment": "auto_exp_48",
        "config": {
            "N_TEMPLATES": N_TEMPLATES, "K_PCS": K_PCS,
            "D_AUX_SUP": D_AUX_SUP, "N_ITER": N_ITER,
            "AUX_WEIGHT": AUX_WEIGHT, "SIGMA_AUX": SIGMA_AUX,
            "N_FOLDS": N_FOLDS, "n_colors": int(n_colors),
            "n_rows": int(n_rows),
        },
        "r2_centroid_in_sample": {AUX_LABELS_HSV[i]: float(r2_centroid_in[i])
                                  for i in range(3)},
        "r2_raw_in_sample_per_row": {AUX_LABELS_HSV[i]: float(r2_raw_in_per_row[i])
                                     for i in range(3)},
        "r2_raw_in_sample_agg_color": {AUX_LABELS_HSV[i]: float(r2_raw_in_agg[i])
                                       for i in range(3)},
        "cv_raw_per_row_r2_mean": {AUX_LABELS_HSV[i]: float(raw_mean[i]) for i in range(3)},
        "cv_raw_per_row_r2_std":  {AUX_LABELS_HSV[i]: float(raw_std[i]) for i in range(3)},
        "cv_raw_agg_color_r2_mean": {AUX_LABELS_HSV[i]: float(raw_agg_mean[i]) for i in range(3)},
        "cv_raw_agg_color_r2_std":  {AUX_LABELS_HSV[i]: float(raw_agg_std[i]) for i in range(3)},
        "cv_centroid_r2_mean": {AUX_LABELS_HSV[i]: float(cen_mean[i]) for i in range(3)},
        "cv_centroid_r2_std":  {AUX_LABELS_HSV[i]: float(cen_std[i]) for i in range(3)},
        "centroid_r2_from_memo_auto_exp_38": CENTROID_R2,
        "shift_raw_per_row_minus_memo": shift_vs_memo,
        "shift_raw_agg_color_minus_memo": shift_agg_vs_memo,
        "mean_shift_raw_per_row":  mean_shift_row,
        "mean_shift_raw_agg_color": mean_shift_agg,
        "verdict": verdict,
        "runtime_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    np.savez(OUT_NPZ,
             cv_raw_per_row_r2=cv_raw_r2,
             cv_raw_agg_color_r2=cv_raw_agg_r2,
             cv_centroid_r2=cv_centroid_r2,
             r2_centroid_in_sample=r2_centroid_in,
             r2_raw_in_sample_per_row=r2_raw_in_per_row,
             r2_raw_in_sample_agg_color=r2_raw_in_agg)
    print(f"[json] saved {OUT_JSON}")
    print(f"[npz]  saved {OUT_NPZ}")
    print(f"[runtime] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
