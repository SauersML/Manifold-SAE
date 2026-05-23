"""Clean R² ranking plot — supervised + unsupervised together, sorted, with
clearly-broken fits (R² < -0.05) dropped. Just the bar chart, no heatmap."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_gam_results import SUPERVISED_SPECS, SPEC_FAMILY, FAMILY_COLOR


BROKEN_THRESHOLD = -0.05      # below this = numerical instability, drop


HUMAN_NAME = {
    # Linear baselines
    "L_lin_rgb":             "Linear in RGB",
    "L_lin_hsv":             "Linear in HSV (periodic hue)",
    "L_lin_lab":             "Linear in CIE-Lab",
    "L_lin_oklab":           "Linear in Oklab",
    "L_lin_lch":             "Linear in CIE-Lch (perceptual cylindrical)",
    "L_lin_luminance":       "Linear in luminance only",
    # Polynomial
    "L_poly_rgb":            "Quadratic polynomial in RGB",
    "L_poly_hsv":            "Quadratic polynomial in HSV (periodic hue)",
    "L_poly_lab":            "Quadratic polynomial in CIE-Lab",
    "L_poly_oklab":          "Quadratic polynomial in Oklab",
    "L_poly_lch":            "Quadratic polynomial in CIE-Lch",
    "L_poly3_rgb":           "Cubic polynomial in RGB",
    "L_poly3_hsv":           "Cubic polynomial in HSV (periodic hue)",
    "L_poly3_lab":           "Cubic polynomial in CIE-Lab",
    "L_poly4_hsv":           "Quartic polynomial in HSV (periodic hue)",
    # Sanity
    "L_const_mean":          "Constant baseline (predict training mean)",
    # Additive 1D smooths
    "L_add_rgb":             "Additive B-splines per R/G/B axis",
    "L_add_hsv":             "Additive B-splines per HSV axis",
    # Joint multi-d smooths
    "L_joint_rgb":           "3D thin-plate spline in RGB",
    "L_joint_hsv":           "4D thin-plate spline in HSV (periodic hue)",
    "L_joint_lab":           "3D thin-plate spline in CIE-Lab",
    "L_joint_rgb_with_hue":  "3D thin-plate spline (RGB) + circular hue",
    "L_tensor_bspline_rgb":  "Tensor-product B-spline in RGB",
    "L_joint_oklab":         "3D thin-plate spline in Oklab",
    "L_joint_oklab_with_h":  "3D thin-plate spline (Oklab) + circular hue",
    "L_lch_with_cyclic_h":   "2D thin-plate spline (L, C) + cyclic hue",
    "L_lab_with_cyclic_hue": "3D thin-plate spline (Lab) + cyclic hue",
    "L_perceptual_add":      "Additive 1D smooths on lightness, chroma, cyclic hue",
    "L_hue_polyharmonic":    "Cyclic hue + 1D smooths on lightness and chroma",
    "L_chroma_lum_2d":       "2D thin-plate spline on chroma and lightness (no hue)",
    "L_kernel_rbf_rgb":      "Gaussian RBF kernel ridge in RGB",
    "L_rgb_lab_combo":       "3D thin-plate spline (RGB) + 3D thin-plate spline (Lab)",
    # Cyclic B-spline + combinations
    "L_cyclic_hue":                    "Cyclic B-spline on hue",
    "L_cyclic_hue_plus_lin_v":         "Cyclic hue + linear value",
    "L_cyclic_hue_plus_bspline_v":     "Cyclic hue + B-spline on value",
    "L_cyclic_hue_plus_bspline_s_v":   "Cyclic hue + B-splines on saturation and value",
    "L_cyclic_hue_plus_lin_rgb":       "Cyclic hue + linear RGB",
    "L_cyclic_hue_plus_joint_rgb":     "Cyclic hue + 3D thin-plate spline (RGB)",
    # Manifolds (topology priors)
    "M_cyl_hue_val":         "Cylinder (hue circular, value linear)",
    "M_torus_hue_sat":       "Torus (hue and saturation both circular)",
    "M_torus_hue_val":       "Torus (hue and value both circular)",
    "M_sphere_hueval":       "Sphere (Runge: hue=longitude, value=latitude)",
    "M_sphere_plus_chroma":  "Sphere + chroma additive",
    "M_hsv_cone":            "HSV cone (saturation as radius, value as height)",
    "M_hsv_bicone":          "HSV bicone (saturation vanishes at both lightness extremes)",
    "M_chroma_disk":         "2D chromaticity disk (chroma × hue, no lightness)",
    "M_chroma_disk_plus_L":  "Chromaticity disk + 1D smooth on lightness",
    "M_rgb_finer_grid":      "3D thin-plate spline (RGB) with 7³=343 centers",
    # Nonparametric
    "N_knn_rgb_k5":          "k-nearest neighbors in RGB (k=5)",
    "N_knn_rgb_k10":         "k-nearest neighbors in RGB (k=10)",
    "N_knn_rgb_k20":         "k-nearest neighbors in RGB (k=20)",
    "N_knn_rgb_k30":         "k-nearest neighbors in RGB (k=30)",
    "N_knn_hsv_k10":         "k-nearest neighbors in HSV (k=10)",
    "N_knn_lab_k5":          "k-nearest neighbors in CIE-Lab (k=5)",
    "N_knn_lab_k10":         "k-nearest neighbors in CIE-Lab (k=10)",
    "N_knn_lab_k20":         "k-nearest neighbors in CIE-Lab (k=20)",
    "N_knn_lab_k30":         "k-nearest neighbors in CIE-Lab (k=30)",
    "N_knn_oklab_k10":       "k-nearest neighbors in Oklab (k=10)",
    # Unsupervised — nonlinear alternation
    "U_1d":                  "Unsupervised 1D nonlinear alternation",
    "U_2d":                  "Unsupervised 2D nonlinear alternation",
    "U_3d":                  "Unsupervised 3D nonlinear alternation",
    "U_4d":                  "Unsupervised 4D nonlinear alternation",
    "U_5d":                  "Unsupervised 5D nonlinear alternation",
    "U_6d":                  "Unsupervised 6D nonlinear alternation",
    "U_8d":                  "Unsupervised 8D nonlinear alternation",
    "U_3d_pca_init":         "Unsupervised 3D nonlinear (PCA init)",
    "U_3d_multistart":       "Unsupervised 3D nonlinear (5-restart)",
    # Linear PCA
    "U_pca_2d":              "Linear PCA reconstruction (top 2 PCs)",
    "U_pca_3d":              "Linear PCA reconstruction (top 3 PCs)",
    "U_pca_4d":              "Linear PCA reconstruction (top 4 PCs)",
    "U_pca_8d":              "Linear PCA reconstruction (top 8 PCs)",
    "U_pca_16d":             "Linear PCA reconstruction (top 16 PCs)",
    "U_pca_24d":             "Linear PCA reconstruction (top 24 PCs)",
    "U_pca_32d":             "Linear PCA reconstruction (top 32 PCs)",
    "U_pca_48d":             "Linear PCA reconstruction (top 48 PCs)",
    # PCA hybrids
    "U_pca8_smooth":         "PCA (top 8) + Duchon smooth",
    "U_pca16_smooth":        "PCA (top 16) + Duchon smooth",
    "U_pca_add_3d":          "PCA (top 3) + additive 1D B-splines",
    "U_pca_add_8d":          "PCA (top 8) + additive 1D B-splines",
    "U_pca_add_16d":         "PCA (top 16) + additive 1D B-splines",
    # Duchon-additive (no B-splines)
    "U_pca8_duchon_add1d":   "PCA (top 8) + additive 1D Duchon",
    "U_pca16_duchon_add1d":  "PCA (top 16) + additive 1D Duchon",
    "U_pca24_duchon_add1d":  "PCA (top 24) + additive 1D Duchon",
    "U_pca32_duchon_add1d":  "PCA (top 32) + additive 1D Duchon",
    "U_pca48_duchon_add1d":  "PCA (top 48) + additive 1D Duchon",
    # Duchon-joint / pairs / triples
    "U_pca3_duchon_joint":   "PCA (top 3) + joint 3D Duchon",
    "U_pca4_duchon_joint":   "PCA (top 4) + joint 4D Duchon",
    "U_pca6_duchon_joint":   "PCA (top 6) + joint 6D Duchon",
    "U_pca16_duchon_pairs":  "PCA (top 16) + 2D Duchon on PC pairs",
    "U_pca6_duchon_triples": "PCA (top 6) + 3D Duchon on PC triples",
    "U_pca12_duchon_triples":"PCA (top 12) + 3D Duchon on PC triples",
    "U_pca8_duchon_finer_centers": "PCA (top 8) + Duchon with k-means-200 centers",
    "U_pca_pairs_4d":        "PCA (top 4) + 2D Duchon on PC pairs",
    "U_pca_pairs_8d":        "PCA (top 8) + 2D Duchon on PC pairs",
    "U_pca3_tensor":         "PCA (top 3) + tensor B-spline",
    "U_pca_centered_8d_smooth": "PCA (top 8) + Duchon with k-means centers",
    # Other unsupervised
    "U_nmf_8d":              "Non-negative matrix factorization (8 components)",
    "U_nmf_16d":             "Non-negative matrix factorization (16 components)",
    "U_kmeans_10":           "k-means 10 clusters + cluster-mean prediction",
    "U_kmeans_30":           "k-means 30 clusters + cluster-mean prediction",
    "U_kmeans_50":           "k-means 50 clusters + cluster-mean prediction",
    "U_loop_1d":             "Unsupervised 1D periodic loop (forced S¹ topology)",
    "U_centroid_kde_smooth_3d": "Distance-weighted kernel smoother",
}


def main() -> int:
    results_path = Path(os.environ.get(
        "RESULTS_JSON",
        "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json",
    ))
    data = json.loads(results_path.read_text())
    spec_results = next(iter(data["per_layer"].values()))["specs"]
    src = data["config"].get("harvest_from") or data["config"].get("model_name")

    rows = []
    for s in SUPERVISED_SPECS:
        r = spec_results.get(s, {})
        if "error" in r:
            continue
        r2 = r.get("r2_macro_mean", float("nan"))
        if not np.isfinite(r2) or r2 < BROKEN_THRESHOLD:
            continue
        rows.append((s, r2, r.get("r2_macro_std", 0.0)))
    rows.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]
    r2s   = np.array([r[1] for r in rows])
    stds  = np.array([r[2] for r in rows])
    fams  = [SPEC_FAMILY[s] for s in names]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(rows))))
    xs = np.arange(len(rows))
    ax.barh(
        xs, r2s, xerr=stds, height=0.7,
        color=[FAMILY_COLOR[f] for f in fams],
        edgecolor="black", linewidth=0.4, capsize=3,
    )
    ax.set_yticks(xs)
    ax.set_yticklabels([HUMAN_NAME.get(s, s) for s in names], fontsize=10)
    ax.invert_yaxis()                                      # winner on top
    ax.set_xlabel("held-out R²_macro  (5-fold CV by color)", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Inline R² value at each bar end
    xmax = max(0.5, float(r2s.max()) * 1.18)
    ax.set_xlim(min(BROKEN_THRESHOLD, float(r2s.min()) - 0.02), xmax)
    for i, (v, s) in enumerate(zip(r2s, stds)):
        ax.annotate(f"{v:+.3f}", xy=(v, i),
                    xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=9)

    families_used = sorted(set(fams), key=lambda f:
                            ["linear","polynomial","additive","joint",
                             "cyclic","manifold","nonparametric",
                             "unsupervised"].index(f))
    ax.legend(
        handles=[Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", label=f)
                  for f in families_used],
        loc="lower right", fontsize=9, frameon=True, ncol=1,
    )

    n_colors = data["per_layer"][next(iter(data["per_layer"]))].get(
        "n_colors", "n/a") if False else 186     # fallback hard-coded for the snapshot
    ax.set_title(
        f"cogito L40 — model ranking by held-out R²\n"
        f"{len(rows)} of {len(SUPERVISED_SPECS)} specs shown   ·   "
        f"source: {Path(str(src)).name}",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = results_path.parent / "gam_clean_ranking.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
