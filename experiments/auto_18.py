"""auto_18: supervised R² vs target-PC subspace dimension (idea rr).

Question
--------
For a fixed supervised predictor (linear or joint-RGB Duchon GAM), how
much of the centroid manifold can be explained when we *restrict the
target* to the top-k PCs for k = 1, 2, 3, 4, 8, 16, 32, 64?

The original results.json reports macro R² over all 64 PCs only — but
that mixes "easy" high-variance directions with "noisy" tail PCs. If a
linear-in-RGB map already explains the top-3 PCs nearly perfectly while
collapsing on PCs 16-64, then the headline R² ≈ 0.10 for L_lin_rgb is
*misleadingly pessimistic*: the model's color geometry IS basically
linear in RGB, but only on its top variance-explaining axes. This is
the supervised analog of the unsupervised U_pca_kd curve already in
results.json (which trivially hits R²=1 at k=64 because it's just the
projection of the target).

The plot answers: "what fraction of variance in the *first k* PCs is
captured by each supervised spec?"

Specs compared (cheap supervised baselines + the headline GAM):
  L_lin_rgb         — 4-param affine in RGB
  L_lin_hsv         — period-aware HSV
  L_lin_lab         — CIELAB linear
  L_joint_rgb       — 3D Duchon over RGB (the headline joint GAM)
  L_joint_rgb_with_hue — best spec on the full target

For each k ∈ {1,2,3,4,8,16,32,64}: 5-fold color CV, macro R² across the
*first k* PCs only (variance-weighted via single sum-of-squares).

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_18.{png,json}
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
OUT_JSON = OUT_DIR / "auto_18.json"
OUT_PNG = OUT_DIR / "auto_18.png"

N_TEMPLATES = 28
N_PCS_MAX = 64
KS = [1, 2, 3, 4, 8, 16, 32, 64]
N_FOLDS = 5
SPECS = [
    "L_lin_rgb",
    "L_lin_hsv",
    "L_lin_lab",
    "L_joint_rgb",
    "L_joint_rgb_with_hue",
]


def macro_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_res = ((Y_true - Y_pred) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


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
    print(f"[load] X={X.shape} n_colors={n_colors}", flush=True)

    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)

    colors = cmg.load_xkcd_colors()
    assert len(colors) == n_colors
    rgb01 = np.array([[r, g, b] for _, r, g, b in colors], dtype=np.float64) / 255.0
    X_rgb, X_hsv = build_coords(rgb01)

    # Per-color centroids → standardize → top-64 PCs
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s_sv, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS_MAX]
    Z_all = centroids_n @ V_topK.T  # (n_colors, 64)
    evr = (s_sv[:N_PCS_MAX] ** 2 / (s_sv ** 2).sum())
    print(f"[pca] top-{N_PCS_MAX} EVR sum = {evr.sum():.3f}", flush=True)
    print(f"[pca] EVR k=1..8: " + ", ".join(f"{v:.3f}" for v in evr[:8]), flush=True)

    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS_MAX, n_folds=N_FOLDS,
                      lattice_per_side=5, init_log_lambda=0.0,
                      output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    folds = cmg.kfold_color_indices(n_colors, N_FOLDS, seed=0)

    # For each spec: collect per-fold held-out predictions on the FULL 64 PCs,
    # then we can slice to first-k PCs and compute R² without refitting.
    spec_test_pred: dict[str, np.ndarray] = {}  # (n_colors, 64) — out-of-fold preds
    print(f"\n[fit] training {len(SPECS)} specs × {N_FOLDS} folds...", flush=True)
    for spec in SPECS:
        Y_pred = np.full_like(Z_all, np.nan)
        t0 = time.time()
        for fi, (tr_idx, te_idx) in enumerate(folds):
            try:
                _, te_pred = cmg.fit_and_predict(
                    spec,
                    X_rgb[tr_idx], X_hsv[tr_idx], Z_all[tr_idx],
                    X_rgb[te_idx], X_hsv[te_idx], Z_all[te_idx],
                    cfg,
                )
                Y_pred[te_idx] = te_pred
            except Exception as exc:
                print(f"  [{spec} fold={fi}] FAILED: {exc}", flush=True)
        spec_test_pred[spec] = Y_pred
        full_r2 = macro_r2(Z_all, Y_pred)
        print(f"  {spec:28s} all-64 macro R² = {full_r2:+.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # For each spec × k: compute macro R² restricted to first-k PCs
    results: dict[str, dict] = {}
    for spec in SPECS:
        Y = spec_test_pred[spec]
        per_k: list[float] = []
        for k in KS:
            r2 = macro_r2(Z_all[:, :k], Y[:, :k])
            per_k.append(r2)
        results[spec] = {"k": KS, "r2_first_k_pcs": per_k}
        print(f"[r2-first-k] {spec:28s} " +
              " ".join(f"k={k}:{r:+.3f}" for k, r in zip(KS, per_k)),
              flush=True)

    # Also: per-PC R² for the headline spec, plotted as a small bar inset
    head = "L_joint_rgb_with_hue"
    Y_head = spec_test_pred[head]
    per_pc_r2 = []
    for j in range(N_PCS_MAX):
        yt = Z_all[:, j]; yp = Y_head[:, j]
        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        per_pc_r2.append(float(1.0 - ss_res / max(ss_tot, 1e-12)))

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": int(n_colors),
            "n_templates": N_TEMPLATES, "n_pcs_max": N_PCS_MAX,
            "ks": KS, "n_folds": N_FOLDS, "specs": SPECS,
        },
        "evr_topK": evr.tolist(),
        "per_spec": results,
        "per_pc_r2_headline": {head: per_pc_r2},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ===== Plot =====
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                    gridspec_kw={"width_ratios": [1.5, 1]})

    colors_plot = {
        "L_lin_rgb":            "#7f7f7f",
        "L_lin_hsv":            "#9b59b6",
        "L_lin_lab":            "#1abc9c",
        "L_joint_rgb":          "#2c6fbb",
        "L_joint_rgb_with_hue": "#c0392b",
    }
    for spec in SPECS:
        r2s = results[spec]["r2_first_k_pcs"]
        ax.plot(KS, r2s, marker="o", linewidth=1.7,
                color=colors_plot.get(spec), label=spec)

    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.3, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("k (= # of leading target PCs)")
    ax.set_ylabel("held-out macro R²  (target = first-k PCs of centroid)")
    ax.set_title("Supervised R² vs target-PC subspace dimension\n"
                 "(cogito L40, 5-fold color CV)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    # Right panel: per-PC R² bar for headline spec; cumulative EVR overlay
    xs = np.arange(N_PCS_MAX)
    ax2.bar(xs, per_pc_r2, color=colors_plot[head], alpha=0.85,
            label=f"per-PC R² ({head})")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("PC index (0 = highest-variance)")
    ax2.set_ylabel("per-PC held-out R²", color=colors_plot[head])
    ax2.set_title("Where does the headline spec actually predict?\n"
                  "(per-PC R² vs PC index)")
    ax2.grid(linestyle=":", alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(xs, np.cumsum(evr), color="black", linewidth=1.3,
             linestyle="--", label="cumulative EVR")
    ax3.set_ylabel("cumulative EVR", color="black")
    ax3.set_ylim(0, 1.02)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
