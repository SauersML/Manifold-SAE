"""auto_17: sample-efficiency curve (idea pp).

Question: how many xkcd color anchors does the best GAM spec actually need
to reach its asymptotic R² on cogito L40? If the manifold is genuinely
low-dim (3-ish), we should saturate quickly; if R² keeps climbing all the
way to the full 954 colors, the spec is exploiting many fine-grained
anchor-specific corrections and won't transfer cleanly to a held-out
"never-seen" color regime.

Setup
-----
* Best spec on this run: L_joint_rgb_with_hue (centroid macro R² ≈ 0.2392).
  We use this as the headline; L_joint_lab and L_joint_rgb as cheaper
  sanity-check curves.
* Target: top-64 PCs of the per-color centroid (exact same basis as the
  existing GAM run -> R² is on the same scale as results.json).
* For each holdout fraction f in {0.10, 0.20, ..., 0.80}:
    - For each seed s in 0..4:
        - Random color split: ceil(f*n_colors) test colors, rest train.
        - Fit the spec on the TRAIN centroids.
        - Compute macro R² across all 64 PCs on the TEST centroids.
* Plot mean ± std of macro R² vs number of TRAIN colors, one curve per
  spec, with the published-on-full-data R² from results.json as a
  horizontal reference line for L_joint_rgb_with_hue.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_17.png (+ auto_17.json).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_17.json"
OUT_PNG = OUT_DIR / "auto_17.png"

N_TEMPLATES = 28
N_PCS = 64
HOLDOUT_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
SEEDS = list(range(5))
SPECS = [
    "L_joint_rgb_with_hue",   # headline (best on centroid)
    "L_joint_rgb",
    "L_joint_lab",
    "L_lin_rgb",              # linear baseline so the saturation pattern is visible
]
HEADLINE = "L_joint_rgb_with_hue"


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


def macro_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_res = ((Y_true - Y_pred) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


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

    # Per-color centroids
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)

    # Standardize using centroid stats (matches existing GAM run)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma

    # Top-64 PCs of centroids (target basis)
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s_sv, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    Z_all = centroids_n @ V_topK.T              # (n_colors, K)
    print(f"[pca] top-{N_PCS} EVR sum = {(s_sv[:N_PCS]**2 / (s_sv**2).sum()).sum():.3f}",
          flush=True)

    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=5,
                      lattice_per_side=5, init_log_lambda=0.0,
                      output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    # Reference R² from the existing results.json (5-fold CV, 80/20 by color)
    ref = json.load(open(OUT_DIR / "results.json"))
    ref_r2 = {s: ref["per_layer"]["L40"]["specs"][s]["r2_macro_mean"] for s in SPECS}

    out: dict[str, dict] = {}
    for spec in SPECS:
        out[spec] = {"frac": [], "n_train": [], "r2_mean": [], "r2_std": [],
                      "r2_all": []}
        print(f"\n=== spec={spec} ===", flush=True)

        for f in HOLDOUT_FRACS:
            n_test = max(2, int(math.ceil(f * n_colors)))
            n_train = n_colors - n_test
            r2s: list[float] = []
            t0 = time.time()
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * seed + int(f * 100))
                perm = rng.permutation(n_colors)
                test_c = perm[:n_test]
                train_c = perm[n_test:]

                tr_X_rgb = X_rgb[train_c]
                tr_X_hsv = X_hsv[train_c]
                tr_Z = Z_all[train_c]
                te_X_rgb = X_rgb[test_c]
                te_X_hsv = X_hsv[test_c]
                te_Z = Z_all[test_c]

                try:
                    _, te_pred = cmg.fit_and_predict(
                        spec, tr_X_rgb, tr_X_hsv, tr_Z,
                        te_X_rgb, te_X_hsv, te_Z, cfg,
                    )
                    r2 = macro_r2(te_Z, te_pred)
                except Exception as exc:
                    print(f"  [f={f:.2f} seed={seed}] FAILED: {exc}", flush=True)
                    r2 = float("nan")
                r2s.append(r2)

            arr = np.array(r2s, dtype=np.float64)
            out[spec]["frac"].append(float(f))
            out[spec]["n_train"].append(int(n_train))
            out[spec]["r2_mean"].append(float(np.nanmean(arr)))
            out[spec]["r2_std"].append(float(np.nanstd(arr)))
            out[spec]["r2_all"].append(arr.tolist())
            print(f"  holdout={f:.2f} n_train={n_train:4d}  "
                  f"R²={np.nanmean(arr):+.4f} ± {np.nanstd(arr):.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": n_colors,
            "n_templates": N_TEMPLATES, "n_pcs": N_PCS,
            "holdout_fracs": HOLDOUT_FRACS, "seeds": SEEDS,
            "specs": SPECS,
        },
        "reference_r2_from_results_json": ref_r2,
        "results": out,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))

    colors_plot = {
        "L_joint_rgb_with_hue": "#c0392b",
        "L_joint_rgb":          "#2c6fbb",
        "L_joint_lab":          "#27ae60",
        "L_lin_rgb":            "#7f7f7f",
    }
    for spec in SPECS:
        nt = np.array(out[spec]["n_train"])
        m = np.array(out[spec]["r2_mean"])
        sd = np.array(out[spec]["r2_std"])
        ax.errorbar(nt, m, yerr=sd, marker="o", capsize=3,
                    color=colors_plot.get(spec, None),
                    label=f"{spec}", linewidth=1.6)

    # 5-fold CV reference line for headline
    ax.axhline(ref_r2[HEADLINE], color=colors_plot[HEADLINE],
                linestyle=":", linewidth=1.0,
                label=f"{HEADLINE} 5-fold ref ({ref_r2[HEADLINE]:+.3f})")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("# training colors (out of {})".format(n_colors))
    ax.set_ylabel("held-out macro R² (centroid targets, 64 PCs)")
    ax.set_title("Sample-efficiency curve: held-out color R² vs # train anchors\n"
                  "(cogito L40, random color splits, 5 seeds)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
