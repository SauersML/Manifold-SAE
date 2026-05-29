"""auto_36: (hhhh) Per-color R² distribution split by RGB octant.

For each of 949 xkcd colors, compute a per-color R² on the 16-d target
(top-16 PCs of L40 cogito residuals) under 5-fold color-grouped CV:

    R²_c = 1 - ||ŷ_c - y_c||² / ||y_c - ȳ||²

with ȳ = mean target over all colors. Each color contributes one scalar
(its held-out prediction error normalised by its distance from the global
target mean). We then bin colors into 8 RGB octants (R<0.5 / G<0.5 / B<0.5)
and look at the per-color R² distribution per octant for:

  - M_rgb_finer_grid   (best supervised, non-degenerate spec)
  - L_joint_rgb        (headline linear-baseline GAM)

Question: which corner(s) of RGB space does the LM represent most
predictably? Are some octants (e.g. saturated primaries, near-grey,
black corner) systematically easier than others, and does the advantage
of the supervised top spec over the linear baseline concentrate in any
particular octant?

Hard-constraint compliant: PCA (precomputed Vt_topK), Duchon-via-cmg-spec
(no length_scale), ridge baselines via gamfit's REML. No Gaussian RBF,
no t-SNE/UMAP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
import color_manifold_gam as cmg  # noqa: E402

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"
OUT = RUN_DIR / "auto_36.png"
SIDE = RUN_DIR / "auto_36.json"

TOP_K = 16
SPECS = ["M_rgb_finer_grid", "L_joint_rgb"]


def octant_label(rgb_row: np.ndarray) -> int:
    """8 octants: bit 0=R>=0.5, bit 1=G>=0.5, bit 2=B>=0.5 -> 0..7."""
    return int((rgb_row[0] >= 0.5)) | (int(rgb_row[1] >= 0.5) << 1) | (
        int(rgb_row[2] >= 0.5) << 2
    )


OCTANT_NAMES = [
    "K  (R-G-B-)", "R  (R+G-B-)", "G  (R-G+B-)", "Y  (R+G+B-)",
    "B  (R-G-B+)", "M  (R+G-B+)", "C  (R-G+B+)", "W  (R+G+B+)",
]
# Representative swatches (centre of each octant) for the title strip:
OCTANT_SWATCH = np.array([
    [0.25, 0.25, 0.25], [0.75, 0.25, 0.25],
    [0.25, 0.75, 0.25], [0.75, 0.75, 0.25],
    [0.25, 0.25, 0.75], [0.75, 0.25, 0.75],
    [0.25, 0.75, 0.75], [0.75, 0.75, 0.75],
])


def main() -> None:
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    Vt = np.asarray(d["per_layer"]["L40"]["Vt_topK"], dtype=np.float64)
    mu = np.asarray(d["per_layer"]["L40"]["mu"], dtype=np.float64)
    sigma = np.asarray(d["per_layer"]["L40"]["sigma"], dtype=np.float64)
    K = Vt.shape[0]
    print(f"[meta] K_pcs={K}  TOP_K={TOP_K}")

    X_full = np.load(HARVEST, mmap_mode="r")
    n_rows, D = X_full.shape
    n_c = n_rows // n_t
    assert n_c * n_t == n_rows
    print(f"[data] X={X_full.shape}  n_colors={n_c}")

    # Per-color centroid (template-averaged).
    per_color = np.zeros((n_c, D), dtype=np.float64)
    counts = np.zeros(n_c, dtype=np.int64)
    block = 2048
    for s in range(0, n_rows, block):
        e = min(s + block, n_rows)
        chunk = np.asarray(X_full[s:e], dtype=np.float64)
        idx = np.arange(s, e) // n_t
        for ci in np.unique(idx):
            m = idx == ci
            per_color[ci] += chunk[m].sum(axis=0)
            counts[ci] += int(m.sum())
    per_color /= counts[:, None]

    Xn = (per_color - mu) / np.maximum(sigma, 1e-6)
    Z_full = (Xn - Xn.mean(0, keepdims=True)) @ Vt.T  # (n_c, K)
    Z16 = Z_full[:, :TOP_K]

    colors = cmg.load_xkcd_colors()[:n_c]
    rgb = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64) / 255.0
    hsv = cmg.rgb_to_hsv_arr(rgb * 255.0)
    X_rgb = rgb
    X_hsv = np.stack([
        np.cos(2 * np.pi * hsv[:, 0]),
        np.sin(2 * np.pi * hsv[:, 0]),
        hsv[:, 1],
        hsv[:, 2],
    ], axis=1)

    cfg = cmg.Config()
    folds = cmg.kfold_color_indices(n_c, cfg.n_folds)

    # Assemble out-of-fold predictions for every color, per spec.
    Z_pred_oof = {spec: np.zeros_like(Z16) for spec in SPECS}
    for f_idx, (tr, te) in enumerate(folds):
        print(f"\n[fold {f_idx}] train={len(tr)} test={len(te)}")
        for spec in SPECS:
            _, Z_te_pred = cmg.fit_and_predict(
                spec, X_rgb[tr], X_hsv[tr], Z16[tr],
                X_rgb[te], X_hsv[te], Z16[te], cfg,
            )
            Z_pred_oof[spec][te] = Z_te_pred
            err = np.mean((Z_te_pred - Z16[te]) ** 2)
            print(f"  {spec:25s} MSE_te={err:.4f}")

    # Per-color R². Normaliser: distance of y_c from the GLOBAL target mean
    # (same denominator across colors -> directly comparable scalars).
    ybar = Z16.mean(0, keepdims=True)
    denom = np.sum((Z16 - ybar) ** 2, axis=1)  # (n_c,)
    # Guard against ~0 (a color sitting exactly at the mean would be unstable).
    denom = np.maximum(denom, 1e-9)
    per_color_r2 = {}
    for spec in SPECS:
        sse = np.sum((Z_pred_oof[spec] - Z16) ** 2, axis=1)
        per_color_r2[spec] = 1.0 - sse / denom

    # Octant assignment.
    octants = np.array([octant_label(r) for r in rgb], dtype=int)
    counts_by_oct = np.bincount(octants, minlength=8)
    print("\n[octants] counts:", dict(zip(OCTANT_NAMES,
                                            counts_by_oct.tolist())))

    # Aggregate stats per octant per spec.
    stats = {spec: {"mean": np.zeros(8), "median": np.zeros(8),
                     "n": counts_by_oct.tolist()} for spec in SPECS}
    for spec in SPECS:
        for o in range(8):
            m = octants == o
            if m.sum() == 0:
                continue
            vals = per_color_r2[spec][m]
            stats[spec]["mean"][o] = float(vals.mean())
            stats[spec]["median"][o] = float(np.median(vals))

    print("\n=== Mean per-color R² by octant ===")
    print(f"{'octant':16s} {'n':>4s}  " +
          "  ".join(f"{s:>22s}" for s in SPECS) + "   advantage")
    advantages = np.zeros(8)
    for o in range(8):
        msg = f"{OCTANT_NAMES[o]:16s} {counts_by_oct[o]:4d}  "
        vals = []
        for s in SPECS:
            vals.append(stats[s]["mean"][o])
            msg += f"{stats[s]['mean'][o]:+22.4f}  "
        adv = vals[0] - vals[1]
        advantages[o] = adv
        msg += f"{adv:+.4f}"
        print(msg)

    # Save sidecar.
    SIDE.write_text(json.dumps({
        "SPECS": SPECS, "TOP_K": TOP_K,
        "octant_names": OCTANT_NAMES,
        "n_per_octant": counts_by_oct.tolist(),
        "mean_r2": {s: stats[s]["mean"].tolist() for s in SPECS},
        "median_r2": {s: stats[s]["median"].tolist() for s in SPECS},
        "advantage_top_minus_baseline": advantages.tolist(),
        "overall_mean_r2": {s: float(per_color_r2[s].mean()) for s in SPECS},
        "overall_median_r2": {s: float(np.median(per_color_r2[s])) for s in SPECS},
    }, indent=2))
    print(f"[saved] {SIDE}")

    # =================================================================
    # Plot
    # =================================================================
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.35, 1.0, 1.0],
                          hspace=0.45, wspace=0.30)

    # Top row: octant swatches as a colorbar-strip.
    for o in range(8):
        col = o % 4
        row_off = 0
        ax = fig.add_subplot(gs[row_off, col]) if o < 4 else None
        if ax is None:
            continue
        # We use a 2xN swatch panel: top row octants 0..3, then row label.
        ax.imshow(np.tile(OCTANT_SWATCH[o:o+1], (1, 1, 1)).reshape(1, 1, 3),
                  aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{OCTANT_NAMES[o]}\nn={counts_by_oct[o]}",
                     fontsize=9)

    # Boxplots per octant for both specs.
    ax_box = fig.add_subplot(gs[1, :])
    bw = 0.36
    positions = np.arange(8)
    data_top = [per_color_r2[SPECS[0]][octants == o] for o in range(8)]
    data_base = [per_color_r2[SPECS[1]][octants == o] for o in range(8)]
    bp_top = ax_box.boxplot(data_top, positions=positions - bw / 2,
                              widths=bw, patch_artist=True,
                              boxprops=dict(facecolor="#d62728", alpha=0.6),
                              medianprops=dict(color="black"),
                              flierprops=dict(marker=".", ms=3, alpha=0.5))
    bp_base = ax_box.boxplot(data_base, positions=positions + bw / 2,
                               widths=bw, patch_artist=True,
                               boxprops=dict(facecolor="#1f77b4", alpha=0.6),
                               medianprops=dict(color="black"),
                               flierprops=dict(marker=".", ms=3, alpha=0.5))
    # Mark octant swatches as tick markers using small color rectangles.
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels([f"{OCTANT_NAMES[o]}\n(n={counts_by_oct[o]})"
                              for o in range(8)], fontsize=8)
    ax_box.axhline(0, color="k", lw=0.7)
    ax_box.set_ylabel("per-color R² (top-16 PCs, OOF)")
    ax_box.set_title("Per-color R² distribution by RGB octant\n"
                     "red = M_rgb_finer_grid (best supervised) · "
                     "blue = L_joint_rgb (linear baseline)")
    ax_box.grid(axis="y", alpha=0.3)
    # Custom legend
    from matplotlib.patches import Patch
    ax_box.legend(handles=[Patch(facecolor="#d62728", alpha=0.6,
                                    label=SPECS[0]),
                              Patch(facecolor="#1f77b4", alpha=0.6,
                                    label=SPECS[1])],
                   loc="lower right", fontsize=9)

    # Bottom-left: mean R² per octant bars.
    ax_bar = fig.add_subplot(gs[2, :2])
    means_top = stats[SPECS[0]]["mean"]
    means_base = stats[SPECS[1]]["mean"]
    ax_bar.bar(positions - bw / 2, means_top, bw, color="#d62728",
                label=SPECS[0])
    ax_bar.bar(positions + bw / 2, means_base, bw, color="#1f77b4",
                label=SPECS[1])
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([OCTANT_NAMES[o].split()[0] for o in range(8)],
                             fontsize=9)
    ax_bar.axhline(0, color="k", lw=0.5)
    ax_bar.set_ylabel("mean per-color R²")
    ax_bar.set_title("Mean per-color R² by octant")
    ax_bar.legend(fontsize=9, loc="lower right")
    ax_bar.grid(axis="y", alpha=0.3)
    for o in range(8):
        for x, val in ((o - bw / 2, means_top[o]),
                          (o + bw / 2, means_base[o])):
            ax_bar.text(x, val + (0.01 if val >= 0 else -0.04),
                          f"{val:+.2f}", ha="center", fontsize=7)

    # Bottom-right: advantage (TOP - BASELINE) per octant.
    ax_adv = fig.add_subplot(gs[2, 2:])
    colors_bar = ["#d62728" if v > 0 else "#1f77b4" for v in advantages]
    ax_adv.bar(positions, advantages, color=colors_bar)
    ax_adv.set_xticks(positions)
    ax_adv.set_xticklabels([OCTANT_NAMES[o].split()[0] for o in range(8)],
                              fontsize=9)
    ax_adv.axhline(0, color="k", lw=0.5)
    ax_adv.set_ylabel(f"mean R²({SPECS[0]}) − mean R²({SPECS[1]})")
    ax_adv.set_title("Supervised advantage per octant\n"
                     "(positive = best spec wins this corner of RGB)")
    ax_adv.grid(axis="y", alpha=0.3)
    for o, v in enumerate(advantages):
        ax_adv.text(o, v + (0.003 if v >= 0 else -0.012),
                       f"{v:+.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "(hhhh) Per-color R² by RGB octant — top-16 PC target, 5-fold "
        "color-grouped CV.  Overall mean R²: "
        + " · ".join(f"{s}={per_color_r2[s].mean():+.3f}" for s in SPECS),
        fontsize=12, y=0.995,
    )
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
