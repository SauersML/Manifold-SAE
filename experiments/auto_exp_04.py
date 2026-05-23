"""auto_exp_04: raw per-prompt residuals vs per-color centroids.

Every GAM in color_manifold_gam.py is fit on the 954-row per-color
*centroid* matrix (the mean across 28 templates). That averaging buys
SNR but throws away ~28× of the rows. Question: does fitting the same
spec on the 26572 raw per-prompt residuals — keeping color-grouped
5-fold CV so the held-out R² is still cross-color — give different /
higher held-out R² on the same 64 PCs?

Three competing hypotheses:

  (H1) Centroid wins. Per-template noise is genuinely orthogonal to
       the color subspace at L40; averaging is the right denoiser, and
       the raw fit will be lower R² because the noise dominates SS_tot.

  (H2) Raw wins. The smooth has more rows to fit, so it can capture
       finer color structure (sub-color variation correlated with RGB)
       that the centroid washes out.

  (H3) Raw and centroid agree on the *RGB-driven* component but raw
       reveals additional template-driven variance the centroid hides.
       (Show this by comparing per-PC R² on the same 64 PCs of the
       centroid space — we project per-prompt residuals through the
       *centroid* PCA basis so the targets are comparable.)

Setup
-----
* Reuse cached residuals: /Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy
  (26572 × 7168). No HTTP. Cogito server is not touched.
* Targets: top-64 PCs computed on the per-color centroid (matches the
  existing GAM run's basis exactly so PC indices are comparable).
* Two training regimes per fold:
    - Centroid: 954 rows, train mask = train colors.
    - Raw:     26572 rows, train mask = all prompts of train colors.
* Evaluation always on the same held-out colors, on the SAME per-prompt
  rows (test set = 28 prompts per held-out color). This makes the two
  regimes directly comparable in R² units.
* Headline specs (small zoo, fast):
    L_lin_rgb        — linear baseline
    L_lin_lab        — perceptually-uniform linear baseline
    L_joint_rgb      — 3D Duchon (RGB)
    L_joint_lab      — 3D Duchon (Lab)
    L_lch_with_cyclic_h — cyclic-hue + Lc Duchon (best spec on centroid)

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_04_raw_vs_centroid.{json,png}
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Reuse helpers from color_manifold_gam — same fit primitives, same color
# conversions, same CV split.
sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_exp_04_raw_vs_centroid.json"
OUT_PNG = OUT_DIR / "auto_exp_04_raw_vs_centroid.png"

N_TEMPLATES = 28
N_PCS = 64
N_FOLDS = 5

SPECS = [
    "L_lin_rgb",
    "L_lin_lab",
    "L_joint_rgb",
    "L_joint_lab",
    "L_lch_with_cyclic_h",
]


def build_coords(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_rgb, X_hsv4) with X_hsv4 = (cos2πh, sin2πh, s, v)."""
    # cmg.rgb_to_hsv_arr expects 0..255 ints; build from rgb01 directly.
    import colorsys
    hsv = np.zeros((rgb01.shape[0], 3), dtype=np.float64)
    for i in range(rgb01.shape[0]):
        hsv[i] = colorsys.rgb_to_hsv(*rgb01[i])
    X_hsv4 = np.stack([
        np.cos(2 * np.pi * hsv[:, 0]),
        np.sin(2 * np.pi * hsv[:, 0]),
        hsv[:, 1], hsv[:, 2],
    ], axis=1)
    return rgb01, X_hsv4


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load cached residuals ----
    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)             # (26572, 7168)
    N, D = X.shape
    assert N % N_TEMPLATES == 0, f"N={N} not divisible by {N_TEMPLATES}"
    n_colors = N // N_TEMPLATES
    print(f"[load] X={X.shape}  n_colors={n_colors}  n_templates={N_TEMPLATES}", flush=True)

    # Color/template index per row (rows are color-major: c0t0, c0t1, ..., c1t0, ...)
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)

    # ---- Ground-truth color axes (matches color_manifold_gam) ----
    colors = cmg.load_xkcd_colors()
    assert len(colors) == n_colors, f"xkcd colors={len(colors)} != harvest colors={n_colors}"
    rgb01 = np.array([[r, g, b] for _, r, g, b in colors], dtype=np.float64) / 255.0
    X_rgb_per_color, X_hsv_per_color = build_coords(rgb01)
    X_rgb_per_prompt = X_rgb_per_color[c_idx]
    X_hsv_per_prompt = X_hsv_per_color[c_idx]

    # ---- Centroid (954, D) and per-prompt residual standardization ----
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)

    # Per-dim standardize using centroid stats (matches the existing GAM run).
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma
    X_n = (X - mu) / sigma                                 # (N, D)

    # ---- Top-64 PCs of centroid space (same target basis for both regimes) ----
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]                                    # (K, D)
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    Z_centroid = centroids_n @ V_topK.T                    # (954, K)
    Z_prompt = X_n @ V_topK.T                              # (26572, K)
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    # ---- 5-fold CV split BY COLOR ----
    folds = cmg.kfold_color_indices(n_colors, N_FOLDS, seed=0)

    # Minimal Config — only the fields cmg.fit_and_predict reads.
    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=N_FOLDS,
                      lattice_per_side=5, init_log_lambda=0.0,
                      output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    # ---- Main loop ----
    results: dict[str, dict] = {}
    for spec in SPECS:
        print(f"\n=== spec={spec} ===", flush=True)
        results[spec] = {"centroid": {}, "raw": {}}

        for regime in ("centroid", "raw"):
            macro_r2s, per_pc_r2s = [], []
            t0 = time.time()
            for fold_i, (train_c, test_c) in enumerate(folds):
                # Per-prompt test rows for these held-out colors.
                test_row_mask = np.isin(c_idx, test_c)
                test_rows = np.where(test_row_mask)[0]
                test_Z_prompt = Z_prompt[test_rows]
                test_X_rgb = X_rgb_per_prompt[test_rows]
                test_X_hsv = X_hsv_per_prompt[test_rows]

                if regime == "centroid":
                    tr_X_rgb = X_rgb_per_color[train_c]
                    tr_X_hsv = X_hsv_per_color[train_c]
                    tr_Z = Z_centroid[train_c]
                else:  # raw
                    train_row_mask = np.isin(c_idx, train_c)
                    train_rows = np.where(train_row_mask)[0]
                    tr_X_rgb = X_rgb_per_prompt[train_rows]
                    tr_X_hsv = X_hsv_per_prompt[train_rows]
                    tr_Z = Z_prompt[train_rows]

                try:
                    _, te_pred = cmg.fit_and_predict(
                        spec, tr_X_rgb, tr_X_hsv, tr_Z,
                        test_X_rgb, test_X_hsv, test_Z_prompt, cfg,
                    )
                except Exception as exc:
                    print(f"  [fold {fold_i}] {regime} FAILED: {exc}", flush=True)
                    continue

                # macro R² across all K PCs, per-PC R² vector
                ss_res = ((test_Z_prompt - te_pred) ** 2).sum(0)
                ss_tot = ((test_Z_prompt - test_Z_prompt.mean(0, keepdims=True)) ** 2).sum(0)
                per_pc = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
                macro = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
                macro_r2s.append(float(macro))
                per_pc_r2s.append(per_pc)
                print(f"  [fold {fold_i}] {regime:8s}  ntrain={len(tr_Z):5d}  "
                      f"macro R²={macro:+.4f}", flush=True)

            per_pc_mean = (np.nanmean(np.stack(per_pc_r2s, 0), axis=0).tolist()
                            if per_pc_r2s else [float("nan")] * N_PCS)
            results[spec][regime] = {
                "macro_r2_mean": float(np.mean(macro_r2s)) if macro_r2s else float("nan"),
                "macro_r2_std": float(np.std(macro_r2s)) if macro_r2s else float("nan"),
                "per_fold_macro_r2": macro_r2s,
                "per_pc_r2_mean": per_pc_mean,
                "elapsed_s": time.time() - t0,
            }
            print(f"  {regime:8s} -> macro R²={results[spec][regime]['macro_r2_mean']:+.4f}  "
                  f"({results[spec][regime]['elapsed_s']:.1f}s)", flush=True)

    # ---- Save ----
    summary = {
        "config": {"harvest": str(HARVEST), "n_colors": n_colors,
                    "n_templates": N_TEMPLATES, "n_pcs": N_PCS,
                    "n_folds": N_FOLDS, "specs": SPECS},
        "explained_variance_ratio_topK": evr.tolist(),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})

    # Left: macro R² centroid vs raw, grouped bar
    macro_c = [results[s]["centroid"]["macro_r2_mean"] for s in SPECS]
    macro_r = [results[s]["raw"]["macro_r2_mean"] for s in SPECS]
    x = np.arange(len(SPECS))
    w = 0.38
    axes[0].bar(x - w/2, macro_c, w, label="centroid (954)", color="#4060a0")
    axes[0].bar(x + w/2, macro_r, w, label="raw (26572)", color="#c06040")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SPECS, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("held-out macro R² (per-prompt test)")
    axes[0].set_title("Per-prompt vs per-color-centroid fits  (cogito L40, 64 PCs)")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].legend(fontsize=8)
    for xi, (a, b) in enumerate(zip(macro_c, macro_r)):
        axes[0].text(xi - w/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
        axes[0].text(xi + w/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)

    # Right: per-PC R² difference (raw - centroid) for each spec
    K = N_PCS
    for s in SPECS:
        pc_c = np.array(results[s]["centroid"]["per_pc_r2_mean"])
        pc_r = np.array(results[s]["raw"]["per_pc_r2_mean"])
        axes[1].plot(np.arange(K), pc_r - pc_c, label=s, linewidth=1.4, alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("PC index (centroid PCA basis, ordered by EVR)")
    axes[1].set_ylabel("ΔR²  =  raw − centroid")
    axes[1].set_title("Per-PC gain (positive = raw fit captures more on that PC)")
    axes[1].grid(linestyle=":", alpha=0.4)
    axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
