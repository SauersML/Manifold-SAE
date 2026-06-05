"""Visualize color_manifold_gam.py supervised results.

Two panels:

  A) Bar chart of held-out R²_macro per supervised spec (linear / additive /
     joint), with std error bars. The broken L_joint_hsv result is clipped
     so it doesn't dominate the y-axis; an annotation flags it.

  B) Per-PC R² heatmap. Rows = specs, columns = top-64 PCs. Reveals which
     PCs each spec is fitting — useful to see if e.g. linear-HSV captures
     PCs 0-2 well but fails on higher PCs.

Reads /Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json
(override via RESULTS_JSON env var).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SUPERVISED_SPECS = [
    # Linear baselines
    "L_lin_rgb", "L_lin_hsv", "L_lin_lab", "L_lin_oklab", "L_lin_lch",
    "L_lin_luminance",
    # Polynomial baselines
    "L_poly_rgb", "L_poly_hsv", "L_poly_lab", "L_poly_oklab", "L_poly_lch",
    "L_poly3_rgb", "L_poly3_hsv", "L_poly3_lab", "L_poly4_hsv",
    # Sanity baseline
    "L_const_mean",
    # Additive 1D smooths
    "L_add_rgb", "L_add_hsv", "L_perceptual_add", "L_hue_polyharmonic",
    # Joint multi-d smooths
    "L_joint_rgb", "L_joint_hsv", "L_joint_lab", "L_joint_rgb_with_hue",
    "L_joint_oklab", "L_joint_oklab_with_h",
    "L_lch_with_cyclic_h", "L_lab_with_cyclic_hue", "L_chroma_lum_2d",
    "L_tensor_bspline_rgb", "L_kernel_rbf_rgb", "L_rgb_lab_combo",
    # Cyclic B-spline (non-Duchon periodic) + combinations
    "L_cyclic_hue", "L_cyclic_hue_plus_lin_v", "L_cyclic_hue_plus_bspline_v",
    "L_cyclic_hue_plus_bspline_s_v", "L_cyclic_hue_plus_lin_rgb",
    "L_cyclic_hue_plus_joint_rgb",
    # Manifold-shaped smooths
    "M_cyl_hue_val", "M_torus_hue_sat", "M_torus_hue_val",
    "M_sphere_hueval", "M_sphere_plus_chroma", "M_hsv_cone",
    "M_hsv_bicone", "M_chroma_disk", "M_chroma_disk_plus_L", "M_rgb_finer_grid",
    # Nonparametric baselines
    "N_knn_rgb_k5", "N_knn_rgb_k10", "N_knn_rgb_k20", "N_knn_rgb_k30",
    "N_knn_hsv_k10",
    "N_knn_lab_k5", "N_knn_lab_k10", "N_knn_lab_k20", "N_knn_lab_k30",
    "N_knn_oklab_k10",
    # Unsupervised manifolds — no GT axes, discovers its own latent t
    "U_1d", "U_2d", "U_3d", "U_4d", "U_5d",
    "U_pca_2d", "U_pca_3d", "U_pca_4d", "U_pca_8d",
    "U_pca_16d", "U_pca_24d", "U_pca_32d", "U_pca_48d",
    "U_pca8_smooth", "U_pca16_smooth",
    "U_pca_add_3d", "U_pca_add_8d", "U_pca_add_16d",
    "U_pca_pairs_4d", "U_pca_pairs_8d",
    "U_3d_pca_init", "U_3d_multistart", "U_pca3_tensor",
    "U_nmf_8d", "U_nmf_16d",
    "U_centroid_kde_smooth_3d", "U_pca_centered_8d_smooth",
    "U_kmeans_10", "U_kmeans_30", "U_kmeans_50",
    "U_loop_1d",
    "U_pca3_duchon_joint", "U_pca4_duchon_joint", "U_pca6_duchon_joint",
    "U_pca8_duchon_add1d", "U_pca16_duchon_add1d",
    "U_pca24_duchon_add1d", "U_pca32_duchon_add1d", "U_pca48_duchon_add1d",
    "U_pca16_duchon_pairs",
    "U_pca6_duchon_triples", "U_pca12_duchon_triples",
]

SPEC_FAMILY = {
    "L_lin_rgb": "linear", "L_lin_hsv": "linear",
    "L_lin_lab": "linear", "L_lin_oklab": "linear",
    "L_lin_lch": "linear", "L_lin_luminance": "linear",
    "L_poly_rgb": "polynomial", "L_poly_hsv": "polynomial",
    "L_poly_lab": "polynomial", "L_poly_oklab": "polynomial",
    "L_poly_lch": "polynomial", "L_poly3_rgb": "polynomial",
    "L_poly3_hsv": "polynomial", "L_poly3_lab": "polynomial",
    "L_poly4_hsv": "polynomial",
    "L_const_mean": "linear",
    "L_add_rgb": "additive", "L_add_hsv": "additive",
    "L_perceptual_add": "additive", "L_hue_polyharmonic": "additive",
    "L_joint_rgb": "joint", "L_joint_hsv": "joint",
    "L_joint_lab": "joint", "L_joint_oklab": "joint",
    "L_joint_rgb_with_hue": "joint", "L_joint_oklab_with_h": "joint",
    "L_lch_with_cyclic_h": "joint", "L_lab_with_cyclic_hue": "joint",
    "L_chroma_lum_2d": "joint",
    "L_tensor_bspline_rgb": "joint", "L_kernel_rbf_rgb": "joint",
    "L_rgb_lab_combo": "joint",
    "L_cyclic_hue": "cyclic", "L_cyclic_hue_plus_lin_v": "cyclic",
    "L_cyclic_hue_plus_bspline_v": "cyclic",
    "L_cyclic_hue_plus_bspline_s_v": "cyclic",
    "L_cyclic_hue_plus_lin_rgb": "cyclic",
    "L_cyclic_hue_plus_joint_rgb": "cyclic",
    "M_cyl_hue_val": "manifold", "M_torus_hue_sat": "manifold",
    "M_torus_hue_val": "manifold", "M_sphere_hueval": "manifold",
    "M_sphere_plus_chroma": "manifold", "M_hsv_cone": "manifold",
    "M_hsv_bicone": "manifold", "M_chroma_disk": "manifold",
    "M_chroma_disk_plus_L": "manifold", "M_rgb_finer_grid": "manifold",
    "N_knn_rgb_k5": "nonparametric", "N_knn_rgb_k10": "nonparametric",
    "N_knn_rgb_k20": "nonparametric", "N_knn_rgb_k30": "nonparametric",
    "N_knn_hsv_k10": "nonparametric",
    "N_knn_lab_k5": "nonparametric", "N_knn_lab_k10": "nonparametric",
    "N_knn_lab_k20": "nonparametric", "N_knn_lab_k30": "nonparametric",
    "N_knn_oklab_k10": "nonparametric",
    "U_1d": "unsupervised", "U_2d": "unsupervised",
    "U_3d": "unsupervised", "U_4d": "unsupervised",
    "U_5d": "unsupervised",
    "U_pca_2d": "unsupervised", "U_pca_3d": "unsupervised",
    "U_pca_4d": "unsupervised", "U_pca_8d": "unsupervised",
    "U_pca_16d": "unsupervised", "U_pca_24d": "unsupervised",
    "U_pca_32d": "unsupervised", "U_pca_48d": "unsupervised",
    "U_pca8_smooth": "unsupervised", "U_pca16_smooth": "unsupervised",
    "U_pca_add_3d": "unsupervised", "U_pca_add_8d": "unsupervised",
    "U_pca_add_16d": "unsupervised",
    "U_pca_pairs_4d": "unsupervised", "U_pca_pairs_8d": "unsupervised",
    "U_3d_pca_init": "unsupervised", "U_3d_multistart": "unsupervised",
    "U_pca3_tensor": "unsupervised",
    "U_nmf_8d": "unsupervised", "U_nmf_16d": "unsupervised",
    "U_centroid_kde_smooth_3d": "unsupervised",
    "U_pca_centered_8d_smooth": "unsupervised",
    "U_kmeans_10": "unsupervised", "U_kmeans_30": "unsupervised",
    "U_kmeans_50": "unsupervised", "U_loop_1d": "unsupervised",
    "U_pca3_duchon_joint": "unsupervised", "U_pca4_duchon_joint": "unsupervised",
    "U_pca6_duchon_joint": "unsupervised",
    "U_pca8_duchon_add1d": "unsupervised", "U_pca16_duchon_add1d": "unsupervised",
    "U_pca24_duchon_add1d": "unsupervised", "U_pca32_duchon_add1d": "unsupervised",
    "U_pca48_duchon_add1d": "unsupervised",
    "U_pca16_duchon_pairs": "unsupervised",
    "U_pca6_duchon_triples": "unsupervised", "U_pca12_duchon_triples": "unsupervised",
}

FAMILY_COLOR = {
    "linear":        "#cfdee9",
    "polynomial":    "#a4cae0",
    "additive":      "#7baed1",
    "joint":         "#4f93bf",
    "cyclic":        "#a684c2",
    "manifold":      "#356d96",
    "nonparametric": "#b8a989",
    "unsupervised":  "#d68a4f",
}


def main() -> int:
    results_path = Path(os.environ.get(
        "RESULTS_JSON",
        "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json",
    ))
    out_dir = results_path.parent

    data = json.loads(results_path.read_text())
    per_layer = data["per_layer"]
    layer_key = next(iter(per_layer))           # cogito only has one layer
    layer_data = per_layer[layer_key]
    spec_results = layer_data["specs"]
    n_pcs = len(layer_data["explained_variance_ratio_topK"])
    src = data["config"].get("harvest_from") or data["config"].get("model_name")

    # Pull out supervised values
    spec_r2 = {}
    spec_std = {}
    spec_per_pc = {}
    for s in SUPERVISED_SPECS:
        if s not in spec_results or "error" in spec_results[s]:
            spec_r2[s], spec_std[s] = float("nan"), float("nan")
            spec_per_pc[s] = [float("nan")] * n_pcs
            continue
        spec_r2[s] = spec_results[s]["r2_macro_mean"]
        spec_std[s] = spec_results[s]["r2_macro_std"]
        spec_per_pc[s] = spec_results[s]["r2_per_pc_mean"]

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(17, 14),
        gridspec_kw={"height_ratios": [1.0, 1.5]},
    )

    # --- Panel A: bar chart of R²_macro (sorted descending) ---------------
    spec_order = sorted(
        SUPERVISED_SPECS,
        key=lambda s: spec_r2[s] if np.isfinite(spec_r2[s]) else -1e9,
        reverse=True,
    )
    xs = np.arange(len(spec_order))
    bar_vals_raw = np.array([spec_r2[s] for s in spec_order])
    bar_stds_raw = np.array([spec_std[s] for s in spec_order])
    # Clip super-negative outliers (e.g. L_joint_hsv = -5.2) at -0.5 so
    # the chart stays readable. Annotate the true value on those bars.
    Y_LOW = -0.5
    bar_vals_clip = np.clip(bar_vals_raw, Y_LOW, None)
    bar_stds_clip = np.where(np.abs(bar_vals_raw - bar_vals_clip) > 1e-9, 0, bar_stds_raw)
    bar_colors = [FAMILY_COLOR[SPEC_FAMILY[s]] for s in spec_order]
    bars = axA.bar(xs, bar_vals_clip, yerr=bar_stds_clip, color=bar_colors,
                    edgecolor="black", linewidth=0.5, capsize=4, zorder=2)
    axA.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=1)
    axA.set_ylim(Y_LOW * 1.05, max(0.2, float(np.nanmax(bar_vals_clip)) * 1.4))
    axA.set_xticks(xs)
    axA.set_xticklabels(spec_order, rotation=30, ha="right", fontsize=9)
    axA.set_ylabel("held-out R²_macro (5-fold CV by color)", fontsize=11)
    axA.set_title("supervised GAM zoo  ·  R² across specs", fontsize=12)
    axA.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for i, (raw, clip) in enumerate(zip(bar_vals_raw, bar_vals_clip)):
        if abs(raw - clip) > 1e-9:
            axA.annotate(
                f"true: {raw:+.2f}\n(clipped)",
                xy=(i, Y_LOW), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, color="#aa2222",
            )
        else:
            axA.annotate(
                f"{raw:+.3f}",
                xy=(i, clip), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8,
            )
    # legend for families
    from matplotlib.patches import Patch
    axA.legend(
        handles=[Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", label=f)
                  for f in ("linear", "polynomial", "additive", "joint",
                             "cyclic", "manifold", "nonparametric",
                             "unsupervised")],
        loc="upper right", fontsize=8, frameon=True, ncol=2,
    )

    # --- Panel B: per-PC R² heatmap (same sort order as bar chart) --------
    H = np.array([spec_per_pc[s] for s in spec_order])
    im = axB.imshow(H, aspect="auto", cmap="RdBu_r",
                    vmin=-max(0.2, np.nanmax(np.abs(H[np.isfinite(H)]))),
                    vmax=+max(0.2, np.nanmax(np.abs(H[np.isfinite(H)]))))
    axB.set_yticks(np.arange(len(spec_order)))
    axB.set_yticklabels(spec_order, fontsize=9)
    axB.set_xlabel(f"principal component index (1..{n_pcs})", fontsize=11)
    axB.set_title("per-PC held-out R² (red = spec captures this PC, blue = worse than mean)",
                  fontsize=12)
    plt.colorbar(im, ax=axB, shrink=0.7, label="R² (clipped to [-max, +max])")

    plt.suptitle(
        f"cogito L40  ·  supervised GAM zoo  ·  source: {Path(str(src)).name}",
        fontsize=12, y=1.005,
    )
    plt.tight_layout()
    out_path = out_dir / "supervised_gam_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
