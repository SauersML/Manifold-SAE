"""auto_38: R^2 as a function of the number of templates available (T-sweep).

Idea (yyyy) from the queue.

Question
--------
How many distinct prompt phrasings does the cogito L40 residual *actually
need* to recover the per-color manifold? If the publishable R^2 is built
from averaging T=28 templates per color, what fraction of that R^2
survives if we only had, say, T=4 templates per color?

This is load-bearing for LLM-applicability: in any real downstream
intervention you won't have 28 hand-curated phrasings per token. A
fast-decaying T -> R^2 curve says the geometry the regressor exploits
is template-average artefact; a quick saturation says T=4-8 is enough
and downstream use is realistic.

Method (cheap, no server, no harvest)
-------------------------------------
* Load cached residuals (26572 x 7168) from COLOR_COGITO_L40/X_L40.npy.
* Build the per-color centroid + 64-PC target basis using **all 28**
  templates (this fixes the target Z so different T values are
  evaluated against the *same* yardstick — only the per-color training
  signal changes).
* For each T in {4, 8, 16, 28}:
    - Repeat with 3 seeded random template subsets of size T.
    - For each subset, recompute per-color centroids using ONLY those T
      templates, project onto the fixed 64-PC basis to get Z_centroid(T).
    - Run standard 5-fold color-OOD CV with the five SPECS, predicting
      Z_centroid(T) on held-out colors.
* Plot macro held-out R^2 vs T per spec, with min/max bands across
  seeds.
* No Gaussian RBF; uses linear/ridge/joint/Duchon specs from cmg.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_38.{json,png}
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
OUT_JSON = OUT_DIR / "auto_38.json"
OUT_PNG = OUT_DIR / "auto_38.png"

N_TEMPLATES = 28
N_PCS = 64
N_FOLDS = 5
SEED = 0
T_VALUES = [4, 8, 16, 28]
N_SEEDS = 3

SPECS = [
    "L_lin_rgb",
    "L_lin_lab",
    "L_joint_rgb",
    "L_joint_lab",
    "L_lch_with_cyclic_h",
]


def build_coords(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    assert N % N_TEMPLATES == 0
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    colors = cmg.load_xkcd_colors()
    assert len(colors) == n_colors
    rgb01 = np.array([[r, g, b] for _, r, g, b in colors], dtype=np.float64) / 255.0
    X_rgb_per_color, X_hsv_per_color = build_coords(rgb01)

    # ---- Reference centroids using ALL templates -> fixed PCA target basis ----
    centroids_all = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids_all[ci] = X[c_idx == ci].mean(0)
    mu = centroids_all.mean(0, keepdims=True)
    sigma = centroids_all.std(0, keepdims=True).clip(min=1e-6)
    centroids_all_n = (centroids_all - mu) / sigma
    Cc = centroids_all_n - centroids_all_n.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}  (fixed yardstick)", flush=True)

    # Pre-normalize all prompt rows once
    X_n = (X - mu) / sigma

    # Pre-compute color folds (fixed across T)
    c_folds = cmg.kfold_color_indices(n_colors, N_FOLDS, seed=SEED)
    print(f"[folds:color] {N_FOLDS} folds  sizes={[len(te) for _, te in c_folds]}", flush=True)

    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=N_FOLDS,
                     lattice_per_side=5, init_log_lambda=0.0,
                     output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    # ---- For each T, each seed: build Z_centroid(T) and run color-OOD CV ----
    results: dict = {}
    for T in T_VALUES:
        results[str(T)] = {spec: {"per_seed_macro_r2": []} for spec in SPECS}
        for si in range(N_SEEDS):
            rng = np.random.default_rng(1000 * SEED + 17 * T + si)
            t_subset = np.sort(rng.choice(N_TEMPLATES, size=T, replace=False))
            row_mask = np.isin(t_idx, t_subset)
            # Per-color centroid using ONLY this template subset
            cent_T = np.zeros((n_colors, D), dtype=np.float64)
            for ci in range(n_colors):
                rows = (c_idx == ci) & row_mask
                cent_T[ci] = X[rows].mean(0)
            cent_T_n = (cent_T - mu) / sigma
            Z_color = cent_T_n @ V_topK.T  # (n_colors, K)
            print(f"\n=== T={T}  seed={si}  templates={t_subset.tolist()} ===", flush=True)
            for spec in SPECS:
                t0 = time.time()
                fold_r2 = []
                for fi, (tr_c, te_c) in enumerate(c_folds):
                    try:
                        _, te_pred = cmg.fit_and_predict(
                            spec,
                            X_rgb_per_color[tr_c], X_hsv_per_color[tr_c], Z_color[tr_c],
                            X_rgb_per_color[te_c], X_hsv_per_color[te_c], Z_color[te_c],
                            cfg,
                        )
                    except Exception as exc:
                        print(f"  [T={T} s={si} {spec} fold {fi}] FAILED: {exc}", flush=True)
                        continue
                    te_Z = Z_color[te_c]
                    ss_res = ((te_Z - te_pred) ** 2).sum(0)
                    ss_tot = ((te_Z - te_Z.mean(0, keepdims=True)) ** 2).sum(0)
                    macro = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
                    fold_r2.append(float(macro))
                macro_mean = float(np.mean(fold_r2)) if fold_r2 else float("nan")
                results[str(T)][spec]["per_seed_macro_r2"].append(macro_mean)
                print(f"  T={T:2d} s={si} {spec:22s}  macro R^2={macro_mean:+.4f}  "
                      f"[{time.time()-t0:.1f}s]", flush=True)

    # Aggregate
    for T in T_VALUES:
        for spec in SPECS:
            arr = np.array(results[str(T)][spec]["per_seed_macro_r2"], dtype=np.float64)
            results[str(T)][spec]["mean"] = float(np.nanmean(arr))
            results[str(T)][spec]["min"] = float(np.nanmin(arr))
            results[str(T)][spec]["max"] = float(np.nanmax(arr))
            results[str(T)][spec]["std"] = float(np.nanstd(arr))

    summary = {
        "config": {"harvest": str(HARVEST), "n_colors": n_colors,
                   "n_templates_total": N_TEMPLATES, "n_pcs": N_PCS,
                   "n_folds": N_FOLDS, "specs": SPECS, "seed": SEED,
                   "T_values": T_VALUES, "n_seeds_per_T": N_SEEDS},
        "explained_variance_ratio_topK": evr.tolist(),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    colors_p = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#d35400"]
    for spec, col in zip(SPECS, colors_p):
        means = [results[str(T)][spec]["mean"] for T in T_VALUES]
        lo = [results[str(T)][spec]["min"] for T in T_VALUES]
        hi = [results[str(T)][spec]["max"] for T in T_VALUES]
        ax.plot(T_VALUES, means, "-o", color=col, label=spec, linewidth=1.6, markersize=6)
        ax.fill_between(T_VALUES, lo, hi, color=col, alpha=0.15, linewidth=0)
        for T, m in zip(T_VALUES, means):
            ax.text(T, m + 0.004, f"{m:.3f}", ha="center", fontsize=7, color=col)
    ax.set_xticks(T_VALUES)
    ax.set_xlabel("number of templates per color used to build training centroid (T)")
    ax.set_ylabel("held-out macro R^2  (color-OOD 5-fold, fixed 64-PC target from all 28T)")
    ax.set_title("R^2 vs. templates available  (cogito L40, "
                 f"{N_SEEDS} seeded template subsets per T, bands=min/max)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
