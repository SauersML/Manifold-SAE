"""auto_exp_06: intrinsic-dimension elbow for unsupervised manifold fits.

Production uses U_3d.  Is 3 the right intrinsic dimensionality of the L40
color manifold?  This experiment sweeps d ∈ {2, 3, 4, 5, 6} for the
unsupervised Duchon manifold fit and compares held-out R² to find the
elbow.

  - d=2 should underfit (color geometry isn't a flat sheet).
  - d=3 is the production hypothesis (RGB / Lch dimensionality).
  - d=4..6 add capacity; if held-out R² keeps climbing the manifold has
    extra latent structure beyond color; if it plateaus or drops (more
    nullspace + harder alternation) the elbow confirms 3.

Method (cheap — NO server calls, NO new harvests):
  1. Reuse cached residuals (/runs/COLOR_COGITO_L40/X_L40.npy).
  2. Reproduce the standard 954×64 centroid PCA target.
  3. 5-fold color-grouped CV (matches color_manifold_gam, auto_exp_04/05).
  4. For each d, fit U_d on train centroids, predict held-out centroid
     Z via the alternation's project step, score macro R² across 64 PCs.
  5. Save macro R² ± fold std, per-fold values, and effective parameter
     budget (n_centers per fit).  Plot d vs R² with error bars and a
     vertical dashed line at the elbow (max R² or |slope| < threshold).

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_06_dim_elbow.{json,png}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_exp_06_dim_elbow.json"
OUT_PNG = OUT_DIR / "auto_exp_06_dim_elbow.png"

N_TEMPLATES = 28
N_PCS = 64
N_FOLDS = 5
DIMS = [2, 3, 4, 5]  # d=6 dropped: gamfit Duchon kernel needs 2(p+s) > d


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    # ---- Per-color centroid, standardize, top-64 PCs (matches auto_exp_04/05).
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    Z = centroids_n @ V_topK.T          # (954, 64)
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    folds = cmg.kfold_color_indices(n_colors, N_FOLDS, seed=0)
    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=N_FOLDS,
                      lattice_per_side=5, init_log_lambda=0.0,
                      output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    # ---- Sweep d ----
    results: dict[int, dict] = {}
    for d in DIMS:
        print(f"\n=== U_{d}d ===", flush=True)
        macro_r2s, per_pc_r2s = [], []
        n_centers = None
        t0 = time.time()
        for fold_i, (train_c, test_c) in enumerate(folds):
            tr_Z = Z[train_c]
            te_Z = Z[test_c]
            try:
                fit = cmg.fit_unsupervised_manifold(tr_Z, d=d, cfg=cfg,
                                                     n_iters=12, verbose=False)
                _, te_pred = cmg.predict_unsupervised(te_Z, fit, d)
            except Exception as exc:
                print(f"  [fold {fold_i}] FAILED: {exc}", flush=True)
                continue
            n_centers = fit["centers"].shape[0]

            ss_res = ((te_Z - te_pred) ** 2).sum(0)
            ss_tot = ((te_Z - te_Z.mean(0, keepdims=True)) ** 2).sum(0)
            per_pc = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
            macro = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
            macro_r2s.append(float(macro))
            per_pc_r2s.append(per_pc)
            print(f"  [fold {fold_i}]  n_test={len(test_c):3d}  "
                  f"n_centers={n_centers:4d}  macro R²={macro:+.4f}", flush=True)

        per_pc_mean = (np.nanmean(np.stack(per_pc_r2s, 0), axis=0).tolist()
                        if per_pc_r2s else [float("nan")] * N_PCS)
        results[d] = {
            "macro_r2_mean": float(np.mean(macro_r2s)) if macro_r2s else float("nan"),
            "macro_r2_std": float(np.std(macro_r2s)) if macro_r2s else float("nan"),
            "per_fold_macro_r2": macro_r2s,
            "per_pc_r2_mean": per_pc_mean,
            "n_centers": int(n_centers) if n_centers else None,
            "elapsed_s": time.time() - t0,
        }
        print(f"  U_{d}d  ->  macro R² = {results[d]['macro_r2_mean']:+.4f} "
              f"± {results[d]['macro_r2_std']:.4f}  "
              f"({results[d]['elapsed_s']:.1f}s)", flush=True)

    # ---- Elbow detection: argmax + first d where slope drops below 25% of
    #      the 2->3 jump.
    ds_arr = np.array(DIMS)
    r2_mean = np.array([results[d]["macro_r2_mean"] for d in DIMS])
    r2_std = np.array([results[d]["macro_r2_std"] for d in DIMS])
    argmax_d = int(ds_arr[int(np.nanargmax(r2_mean))])
    slopes = np.diff(r2_mean)
    elbow_d = int(ds_arr[0] + 1)
    if len(slopes) >= 2 and slopes[0] > 1e-6:
        thresh = 0.25 * slopes[0]
        for i in range(1, len(slopes)):
            if slopes[i] < thresh:
                elbow_d = int(ds_arr[i])
                break
        else:
            elbow_d = int(ds_arr[-1])

    summary = {
        "config": {
            "harvest": str(HARVEST),
            "n_colors": int(n_colors),
            "n_templates": N_TEMPLATES,
            "n_pcs": N_PCS,
            "n_folds": N_FOLDS,
            "dims_swept": DIMS,
        },
        "explained_variance_ratio_topK": evr.tolist(),
        "argmax_d": argmax_d,
        "elbow_d": elbow_d,
        "slopes_dR2": slopes.tolist(),
        "results": {str(d): results[d] for d in DIMS},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [1, 1.4]})

    # Left: macro R² vs d with error bars
    axes[0].errorbar(ds_arr, r2_mean, yerr=r2_std, marker="o",
                      capsize=4, color="#4060a0", linewidth=1.6,
                      label="U_d held-out macro R²")
    axes[0].axvline(elbow_d, color="#c06040", linestyle="--", linewidth=1.2,
                     label=f"elbow d* = {elbow_d}")
    axes[0].axvline(argmax_d, color="#40a060", linestyle=":", linewidth=1.2,
                     label=f"argmax d = {argmax_d}")
    for d, r, st in zip(DIMS, r2_mean, r2_std):
        axes[0].text(d, r + 0.005, f"{r:.3f}", ha="center", fontsize=8)
    axes[0].set_xlabel("latent dimensionality d")
    axes[0].set_ylabel("held-out macro R²  (5-fold color-grouped CV, 64 PCs)")
    axes[0].set_title("U_d intrinsic-dimension elbow  (cogito L40 centroids)")
    axes[0].set_xticks(DIMS)
    axes[0].grid(linestyle=":", alpha=0.4)
    axes[0].legend(fontsize=8, loc="lower right")

    # Right: per-PC R² curves overlaid for each d
    K = N_PCS
    colors_d = plt.cm.viridis(np.linspace(0.05, 0.85, len(DIMS)))
    for d, c in zip(DIMS, colors_d):
        pc = np.array(results[d]["per_pc_r2_mean"])
        axes[1].plot(np.arange(K), pc, label=f"U_{d}d", color=c, linewidth=1.4)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("PC index (centroid PCA basis, ordered by EVR)")
    axes[1].set_ylabel("held-out R²")
    axes[1].set_title("Per-PC held-out R² by manifold dimensionality")
    axes[1].grid(linestyle=":", alpha=0.4)
    axes[1].legend(fontsize=8, ncol=len(DIMS), loc="lower left")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
