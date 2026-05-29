"""auto_61: Hue-rotation symmetry test of the color->Z map.

Idea (jjjjjjj)
--------------
If the cogito-L40 color manifold respected a global hue-rotation
symmetry, rotating every xkcd color's RGB by +/-30 degrees in hue
(HSV space) would correspond to a smooth, possibly rigid motion of
the predicted residual Z in the 64-PC target basis. To test this we:

  1. Fit a per-color regressor F: RGB -> Z_centroid using ridge on
     a polynomial-RGB feature lift (no Gaussian RBF, no Duchon
     length_scale -- just plain ridge in feature space).
  2. Predict Z for the original RGB colors (Z_orig).
  3. Rotate each xkcd color hue by +30 and -30 degrees in HSV,
     convert back to RGB, predict Z_rot+, Z_rot-.
  4. Compare:
       - per-color displacement ||Z_rot - Z_orig||
       - normalized vs the per-color local scale (mean dist to 5-NN
         in Z space) and vs the global Z std
       - cosine alignment between +30 and -30 displacements (should be
         opposite if F is locally hue-linear; uncorrelated if hue is
         re-encoded nonlinearly)
       - a simple Procrustes/affine check: best linear A mapping
         Z_orig -> Z_rot+; residual variance unexplained tells us how
         far from a global linear hue action F is.
  5. Compare to a hue-preserving control: rotate saturation by a
     scalar (S -> 0.7 S) instead of hue.

Pure numpy + matplotlib. PCA/centroid basis pulled from results.json
(same basis used by all published headline R^2 numbers).

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_61.{json,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import colorsys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
X_PATH  = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_PNG = RUN_DIR / "auto_61.png"
OUT_JSON = RUN_DIR / "auto_61.json"

N_TEMPLATES = 28
N_PCS = 64
RIDGE_ALPHA = 1e-2


def poly_feats(rgb: np.ndarray) -> np.ndarray:
    """Degree-3 monomial lift of RGB in [0,1]^3 + HSV (R G B S V plus periodic hue)."""
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hsv = np.stack([np.array(colorsys.rgb_to_hsv(*c)) for c in rgb], axis=0)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    cosH = np.cos(2 * np.pi * H)
    sinH = np.sin(2 * np.pi * H)
    feats = [
        np.ones_like(R), R, G, B,
        R*R, G*G, B*B, R*G, R*B, G*B,
        R*R*R, G*G*G, B*B*B, R*G*B,
        S, V, cosH, sinH,
        S*cosH, S*sinH, V*cosH, V*sinH,
    ]
    return np.stack(feats, axis=1)


def ridge_fit(Phi: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    d = Phi.shape[1]
    A = Phi.T @ Phi + alpha * np.eye(d)
    B = Phi.T @ Y
    return np.linalg.solve(A, B)


def rotate_hue(rgb: np.ndarray, deg: float) -> np.ndarray:
    out = np.zeros_like(rgb)
    dh = deg / 360.0
    for i in range(rgb.shape[0]):
        h, s, v = colorsys.rgb_to_hsv(rgb[i, 0], rgb[i, 1], rgb[i, 2])
        h = (h + dh) % 1.0
        out[i] = colorsys.hsv_to_rgb(h, s, v)
    return out


def shrink_sat(rgb: np.ndarray, factor: float) -> np.ndarray:
    out = np.zeros_like(rgb)
    for i in range(rgb.shape[0]):
        h, s, v = colorsys.rgb_to_hsv(rgb[i, 0], rgb[i, 1], rgb[i, 2])
        out[i] = colorsys.hsv_to_rgb(h, s * factor, v)
    return out


def knn_local_scale(Z: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance to k nearest neighbors (excluding self) for each row."""
    n = Z.shape[0]
    sq = (Z * Z).sum(1)
    out = np.zeros(n)
    chunk = 256
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        d2 = sq[i:j, None] + sq[None, :] - 2 * (Z[i:j] @ Z.T)
        d2 = np.maximum(d2, 0)
        d2[np.arange(j - i), np.arange(i, j)] = np.inf
        # partial sort
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        nn = np.take_along_axis(d2, idx, axis=1)
        out[i:j] = np.sqrt(nn).mean(1)
    return out


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {RESULTS}")
    res = json.loads(RESULTS.read_text())
    pl = res["per_layer"]["L40"]
    Vt = np.asarray(pl["Vt_topK"], dtype=np.float32)
    mu = np.asarray(pl["mu"], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(pl["sigma"], dtype=np.float32).reshape(1, -1)
    K, D = Vt.shape

    n_t = len(res["templates"])
    n_c = len(res["color_axes_per_color_index"]["R"])
    N = n_c * n_t
    print(f"[layout] n_colors={n_c} n_templates={n_t} K={K} D={D}")

    # Project X to Z = top-K PCs
    print(f"[load] X {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape[0] >= N, (X.shape, N)
    Z = np.zeros((N, K), dtype=np.float32)
    chunk = 2048
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xc = np.asarray(X[i:j], dtype=np.float32)
        Xc = (Xc - mu) / sigma
        Z[i:j] = Xc @ Vt.T

    # Per-color centroid (28-template average) in PC basis
    Z = Z.reshape(n_c, n_t, K)
    Z_centroid = Z.mean(1)  # (n_c, K)
    print(f"[centroids] {Z_centroid.shape}")

    # Colors: use the axis arrays in results.json (these are 0..1 floats).
    R = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    G = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    B = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    rgb = np.stack([R, G, B], axis=1)
    assert rgb.shape == (n_c, 3)

    # Fit ridge F: poly(RGB,HSV) -> Z_centroid
    Phi = poly_feats(rgb)
    W = ridge_fit(Phi, Z_centroid.astype(np.float64), RIDGE_ALPHA)
    Z_pred_orig = Phi @ W
    # In-sample R^2 (just sanity)
    ss_res = ((Z_centroid - Z_pred_orig) ** 2).sum()
    ss_tot = ((Z_centroid - Z_centroid.mean(0, keepdims=True)) ** 2).sum()
    in_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"[ridge] in-sample macro R^2 = {in_r2:.4f}  (sanity)")

    # Predictions under perturbations
    rgb_p30 = rotate_hue(rgb, +30.0)
    rgb_m30 = rotate_hue(rgb, -30.0)
    rgb_sat = shrink_sat(rgb, 0.7)

    Z_p30 = poly_feats(rgb_p30) @ W
    Z_m30 = poly_feats(rgb_m30) @ W
    Z_sat = poly_feats(rgb_sat) @ W

    # Displacements per color (in PC units)
    dp = Z_p30 - Z_pred_orig
    dm = Z_m30 - Z_pred_orig
    ds = Z_sat - Z_pred_orig

    norm_p = np.linalg.norm(dp, axis=1)
    norm_m = np.linalg.norm(dm, axis=1)
    norm_s = np.linalg.norm(ds, axis=1)

    # Reference scales
    z_std_pc = Z_pred_orig.std(0)               # per-PC std of predictions
    z_global_scale = float(np.linalg.norm(z_std_pc))  # ~total-std
    knn5 = knn_local_scale(Z_pred_orig, k=5)    # per-color 5-NN scale

    rel_p = norm_p / z_global_scale
    rel_m = norm_m / z_global_scale
    rel_s = norm_s / z_global_scale
    knn_rel_p = norm_p / np.maximum(knn5, 1e-9)
    knn_rel_m = norm_m / np.maximum(knn5, 1e-9)
    knn_rel_s = norm_s / np.maximum(knn5, 1e-9)

    # Cosine alignment between +30 and -30 displacements per color
    eps = 1e-12
    cos_pm = (dp * dm).sum(1) / np.maximum(np.linalg.norm(dp, axis=1) *
                                           np.linalg.norm(dm, axis=1), eps)

    # Affine/linear test: best A so that Z_pred_orig @ A ~ Z_p30
    # Solve via lstsq with ridge for numerical stability
    A_lstsq, *_ = np.linalg.lstsq(Z_pred_orig, Z_p30, rcond=None)
    Z_p30_lin = Z_pred_orig @ A_lstsq
    ss_res_lin = ((Z_p30 - Z_p30_lin) ** 2).sum()
    ss_tot_p30 = ((Z_p30 - Z_p30.mean(0, keepdims=True)) ** 2).sum()
    linear_r2 = 1.0 - ss_res_lin / max(ss_tot_p30, 1e-12)
    print(f"[linear-hue-action] R^2 of best linear A (Z_orig->Z_+30) = {linear_r2:.4f}")

    # Summary
    summary = {
        "config": {
            "n_colors": int(n_c), "n_templates": int(n_t), "n_pcs": int(K),
            "ridge_alpha": RIDGE_ALPHA, "rotation_deg": 30.0,
            "sat_factor": 0.7,
        },
        "in_sample_macro_r2": float(in_r2),
        "linear_hue_action_R2": float(linear_r2),
        "median_rel_disp_p30_global": float(np.median(rel_p)),
        "median_rel_disp_m30_global": float(np.median(rel_m)),
        "median_rel_disp_sat_global": float(np.median(rel_s)),
        "median_knn5_rel_disp_p30": float(np.median(knn_rel_p)),
        "median_knn5_rel_disp_m30": float(np.median(knn_rel_m)),
        "median_knn5_rel_disp_sat": float(np.median(knn_rel_s)),
        "median_cos_pm30": float(np.median(cos_pm)),
        "frac_cos_negative": float((cos_pm < 0).mean()),
        "z_global_scale": float(z_global_scale),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[done] -> {OUT_JSON}")
    print(json.dumps(summary, indent=2))

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) histograms: hue +/-30 vs saturation shrink, displacement / global scale
    ax = axes[0, 0]
    bins = np.linspace(0, max(rel_p.max(), rel_m.max(), rel_s.max()) * 1.02, 40)
    ax.hist(rel_p, bins=bins, alpha=0.55, color="#c04060", label="hue +30 deg",
            edgecolor="black", lw=0.3)
    ax.hist(rel_m, bins=bins, alpha=0.55, color="#4060c0", label="hue -30 deg",
            edgecolor="black", lw=0.3)
    ax.hist(rel_s, bins=bins, alpha=0.55, color="#60a060",
            label="sat x 0.7 (control)", edgecolor="black", lw=0.3)
    ax.set_xlabel(r"$\|F(R')-F(R)\|_2$  /  global $\|\sigma_Z\|$")
    ax.set_ylabel("# colors")
    ax.set_title("Predicted-Z displacement under RGB perturbations")
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)

    # (b) hue displacement vs HSV.S (saturation) — flat colors should move less
    ax = axes[0, 1]
    hsv = np.stack([np.array(colorsys.rgb_to_hsv(*c)) for c in rgb], axis=0)
    sat = hsv[:, 1]
    sc = ax.scatter(sat, rel_p, c=hsv[:, 2], cmap="cividis", s=12, alpha=0.85)
    ax.set_xlabel("HSV saturation (original color)")
    ax.set_ylabel(r"$\|\Delta Z_{+30}\| / \|\sigma_Z\|$")
    ax.set_title("Hue-rotation displacement vs saturation\n(low-sat colors should move less)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85); cbar.set_label("HSV value (lightness)")
    ax.grid(linestyle=":", alpha=0.4)

    # (c) cos(+30, -30) histogram -- should pile near -1 if F acts linearly on hue
    ax = axes[1, 0]
    ax.hist(cos_pm, bins=40, color="#806080", edgecolor="black", lw=0.4)
    ax.axvline(np.median(cos_pm), color="black", ls="--", lw=1,
               label=f"median = {np.median(cos_pm):.3f}")
    ax.axvline(-1.0, color="crimson", ls=":", lw=1,
               label="ideal linear action (-1)")
    ax.axvline(0.0, color="grey", ls=":", lw=1)
    ax.set_xlabel(r"$\cos(\Delta Z_{+30}, \Delta Z_{-30})$")
    ax.set_ylabel("# colors")
    ax.set_title("Cosine alignment of +30 and -30 displacements")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linestyle=":", alpha=0.4)

    # (d) knn-relative displacement (how many local 'hops' a 30 deg rotation is)
    ax = axes[1, 1]
    bins = np.linspace(0, np.percentile(knn_rel_p, 99) * 1.05, 40)
    ax.hist(knn_rel_p, bins=bins, alpha=0.55, color="#c04060", label="hue +30",
            edgecolor="black", lw=0.3)
    ax.hist(knn_rel_m, bins=bins, alpha=0.55, color="#4060c0", label="hue -30",
            edgecolor="black", lw=0.3)
    ax.hist(knn_rel_s, bins=bins, alpha=0.55, color="#60a060", label="sat x 0.7",
            edgecolor="black", lw=0.3)
    ax.axvline(1.0, color="black", ls="--", lw=1,
               label="= one 5-NN hop in Z")
    ax.set_xlabel(r"$\|\Delta Z\| / \mathrm{mean\,5NN\,dist}$")
    ax.set_ylabel("# colors")
    ax.set_title("Displacement in 'local hops' (Z-space 5-NN scale)")
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(
        "auto_61 -- Hue-rotation symmetry of the cogito-L40 color->Z map\n"
        f"linear-hue-action R^2 = {linear_r2:.3f},  median cos(+30,-30) = "
        f"{np.median(cos_pm):.3f},  median |dZ_+30|/sigma = {np.median(rel_p):.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[done] -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
