"""Visualize the fitted surface for the top supervised specs.

For each spec we:
  1. Refit on ALL 293 colors (no holdout — we want the full surface)
  2. Predict the residual centroid for a dense grid of inputs
     (RGB cube, HSV wheel, etc.) — typically 30³ grid points
  3. Project both the grid predictions AND the true cogito centroids
     into a shared 2D PCA basis (using the true centroids' PCA)
  4. Overlay them in a single plot, coloring the grid points by their
     INPUT color (so we see how the RGB/HSV cube maps onto cogito's
     residual manifold) and showing the true centroids as larger
     markers

The result: for each spec, a picture of the LEARNED MAPPING from
ground-truth color-space to cogito-residual-space, viewed in PCA-2D.

Specs covered:
  - L_lin_rgb        — linear surface for comparison
  - L_joint_rgb      — 3D Duchon spline in RGB (best supervised joint)
  - L_joint_lab      — 3D Duchon in CIE-Lab
  - L_poly_hsv       — degree-2 polynomial in HSV-periodic
  - M_hsv_cone       — HSV cone manifold
  - M_sphere_hueval  — Runge sphere via gamfit.sphere_basis
"""

from __future__ import annotations

import colorsys
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


N_T = 28


def load_xkcd_colors():
    from plot_color_geometry import load_xkcd_colors as _f
    return _f()


def load_harvest(p: Path) -> np.ndarray:
    from plot_color_geometry import load_harvest as _f
    return _f(p)


def build_centroids_and_pca(snapshot_path: Path) -> dict:
    """Load harvest, build per-color centroids, fit PCA-2D for projection."""
    X_full = load_harvest(snapshot_path)
    n_colors = X_full.shape[0] // N_T
    X_full = X_full[: n_colors * N_T]
    colors = load_xkcd_colors()[:n_colors]
    rgb = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64) / 255.0
    names = [c[0] for c in colors]
    centroids = np.zeros((n_colors, X_full.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X_full[ci * N_T:(ci + 1) * N_T].mean(0)
    # Standardize, then PCA-2D for projection
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Xn = (centroids - mu) / sigma
    Xc = Xn - Xn.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    K = 64
    pc_basis = Vt.T[:, :2]                       # (D, 2) — for projection
    Z_truth = Xc @ pc_basis                       # (n_colors, 2)
    # PCA-64 used by GAM zoo to fit (same as analysis script)
    Z_64 = Xc @ Vt.T[:, :K]
    return {
        "n_colors": n_colors,
        "rgb_per_color": rgb,
        "names": names,
        "centroids_norm": Xn,
        "mu": mu, "sigma": sigma,
        "Vt": Vt, "S": S,
        "pc_basis_2d": pc_basis,                  # (D, 2)
        "Z_truth_2d": Z_truth,
        "Z_64": Z_64,                             # (n_colors, K=64)
    }


def hsv_to_rgb_arr(h, s, v):
    out = np.zeros((len(h), 3))
    for i in range(len(h)):
        out[i] = colorsys.hsv_to_rgb(float(h[i]), float(s[i]), float(v[i]))
    return out


# ---------------------------------------------------------------------------
# Model fits — refit on ALL training colors (no CV) for surface display
# ---------------------------------------------------------------------------


def _ridge_fit(Phi, Y, alpha):
    """Plain ridge: B = (Phi'Phi + α I)^-1 Phi' Y."""
    PtP = Phi.T @ Phi
    A = PtP + alpha * np.eye(PtP.shape[0])
    PtY = Phi.T @ Y
    return np.linalg.solve(A, PtY)


def _reml_fit(Phi, Y, P):
    """Use the same gamfit-routed REML fit as the analysis script."""
    from color_manifold_gam import reml_fit
    B, _ = reml_fit(Phi, Y, P, init_log_lambda=0.0)
    return B


def fit_predict_spec(spec: str, rgb_train, hsv_train, Z_train,
                     rgb_grid, hsv_grid) -> np.ndarray:
    """Refit on ALL training data, predict on grid. Returns (n_grid, K) preds."""
    from color_manifold_gam import (
        duchon_basis_radial, lattice_centers, bspline_1d_basis, rgb_to_lab,
        _poly_features_degree2,
    )
    if spec == "L_lin_rgb":
        Phi_tr = np.concatenate([rgb_train, np.ones((rgb_train.shape[0], 1))], axis=1)
        Phi_gr = np.concatenate([rgb_grid, np.ones((rgb_grid.shape[0], 1))], axis=1)
        W = _ridge_fit(Phi_tr, Z_train, alpha=1.0)
        return Phi_gr @ W
    if spec == "L_joint_rgb":
        centers = lattice_centers(5)
        Phi_tr, P = duchon_basis_radial(rgb_train, centers)
        Phi_gr, _ = duchon_basis_radial(rgb_grid, centers)
        B = _reml_fit(Phi_tr, Z_train, P)
        return Phi_gr @ B
    if spec == "L_joint_lab":
        lab_tr = rgb_to_lab(rgb_train); lab_gr = rgb_to_lab(rgb_grid)
        lab_min = lab_tr.min(0); lab_max = lab_tr.max(0)
        ax = [np.linspace(lab_min[d], lab_max[d], 5) for d in range(3)]
        L_g, A_g, B_g = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([L_g.flatten(), A_g.flatten(), B_g.flatten()], axis=1)
        Phi_tr, P = duchon_basis_radial(lab_tr, centers)
        Phi_gr, _ = duchon_basis_radial(lab_gr, centers)
        B = _reml_fit(Phi_tr, Z_train, P)
        return Phi_gr @ B
    if spec == "L_poly_hsv":
        Phi_tr = _poly_features_degree2(hsv_train)
        Phi_gr = _poly_features_degree2(hsv_grid)
        W = _ridge_fit(Phi_tr, Z_train, alpha=1.0)
        return Phi_gr @ W
    if spec == "M_hsv_cone":
        def to_cone(X):
            h = (np.arctan2(X[:, 1], X[:, 0]) / (2*np.pi)) % 1.0
            s, v = X[:, 2], X[:, 3]
            r = s * v
            return np.stack([r * np.cos(2*np.pi*h), r * np.sin(2*np.pi*h), v], axis=1)
        co_tr = to_cone(hsv_train); co_gr = to_cone(hsv_grid)
        ang = np.linspace(0, 2*np.pi, 6, endpoint=False)
        rad = np.linspace(0.1, 1.0, 3)
        val = np.linspace(0.1, 1.0, 3)
        centers = []
        for v in val:
            for r in rad:
                for a in ang:
                    centers.append([r*np.cos(a), r*np.sin(a), v])
        for v in val: centers.append([0.0, 0.0, v])
        centers = np.array(centers)
        Phi_tr, P = duchon_basis_radial(co_tr, centers)
        Phi_gr, _ = duchon_basis_radial(co_gr, centers)
        B = _reml_fit(Phi_tr, Z_train, P)
        return Phi_gr @ B
    raise ValueError(spec)


# ---------------------------------------------------------------------------
# Build grids over input spaces
# ---------------------------------------------------------------------------


def rgb_grid(n_per_axis: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Dense RGB cube grid. Returns (rgb_grid (M, 3), display_rgb (M, 3))."""
    ax = np.linspace(0.05, 0.95, n_per_axis)
    R, G, B = np.meshgrid(ax, ax, ax, indexing="ij")
    rgb = np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)
    return rgb, rgb.copy()


def hsv_grid(n_h: int = 16, n_s: int = 4, n_v: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HSV grid → returns (hsv4_periodic (M, 4), hsv_raw (M, 3), display_rgb (M, 3))."""
    hs = np.linspace(0, 1, n_h, endpoint=False)
    ss = np.linspace(0.0, 1.0, n_s)
    vs = np.linspace(0.05, 0.95, n_v)
    H, S, V = np.meshgrid(hs, ss, vs, indexing="ij")
    h = H.flatten(); s = S.flatten(); v = V.flatten()
    hsv4 = np.stack([np.cos(2*np.pi*h), np.sin(2*np.pi*h), s, v], axis=1)
    rgb = hsv_to_rgb_arr(h, s, v)
    return hsv4, np.stack([h, s, v], axis=1), rgb


# ---------------------------------------------------------------------------
# One panel per spec
# ---------------------------------------------------------------------------


SPECS_TO_PLOT = [
    ("L_lin_rgb",     "Linear in RGB",          "rgb"),
    ("L_joint_rgb",   "3D Duchon spline (RGB)", "rgb"),
    ("L_joint_lab",   "3D Duchon spline (Lab)", "rgb"),    # input still rgb, fit converts to lab
    ("L_poly_hsv",    "Quadratic in HSV",       "hsv"),
    ("M_hsv_cone",    "HSV cone manifold",      "hsv"),
]


def render_panel(ax, ctx: dict, spec: str, label: str, grid_kind: str):
    rgb_per = ctx["rgb_per_color"]
    Z_truth = ctx["Z_truth_2d"]
    Z_64 = ctx["Z_64"]
    pc_basis = ctx["pc_basis_2d"]
    Vt = ctx["Vt"]
    mu = ctx["mu"]
    sigma = ctx["sigma"]

    # Build the input grid
    if grid_kind == "rgb":
        rgb_g, disp = rgb_grid(n_per_axis=10)
        hsv_g = None
        hsv4_g = None
    else:
        hsv4_g, hsv_g, disp = hsv_grid()
        rgb_g = disp

    # The supervised fit was on the PCA-64 of standardized centroids. Refit
    # spec on full training data (Z_64), predict for the grid in Z_64 space,
    # then map Z_64 → standardized residual via Vt[:64], then project to 2D.
    hsv4_train_full = np.stack([
        np.cos(2*np.pi*np.array([colorsys.rgb_to_hsv(*c)[0] for c in rgb_per])),
        np.sin(2*np.pi*np.array([colorsys.rgb_to_hsv(*c)[0] for c in rgb_per])),
        np.array([colorsys.rgb_to_hsv(*c)[1] for c in rgb_per]),
        np.array([colorsys.rgb_to_hsv(*c)[2] for c in rgb_per]),
    ], axis=1)
    Z_grid_pred_64 = fit_predict_spec(
        spec, rgb_per, hsv4_train_full, Z_64,
        rgb_g, hsv4_g if hsv4_g is not None else hsv4_train_full[: rgb_g.shape[0]] * 0,
    )
    # Map Z_64 → standardized residual D-dim (using PCA-64 inverse)
    # Z_64 = Xn_centered @ Vt[:64].T  ⇒  Xn_centered ≈ Z_64 @ Vt[:64]
    grid_resid_norm = Z_grid_pred_64 @ Vt[:64]
    Z_grid_2d = grid_resid_norm @ pc_basis

    # Plot: grid predictions as small dots colored by their input color
    ax.scatter(Z_grid_2d[:, 0], Z_grid_2d[:, 1], c=disp, s=18,
                edgecolors="none", alpha=0.55)
    # True centroids as larger circles with black outline
    ax.scatter(Z_truth[:, 0], Z_truth[:, 1], c=rgb_per, s=110,
                edgecolors="black", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f5f5f5")
    ax.set_title(label, fontsize=11)


def main() -> int:
    snapshot = Path(os.environ.get(
        "HARVEST_PATH",
        "/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.snapshot.npz",
    ))
    out_dir = Path(os.environ.get(
        "OUTPUT_DIR",
        "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40",
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] {snapshot}", flush=True)
    ctx = build_centroids_and_pca(snapshot)
    print(f"[shape] n_colors={ctx['n_colors']}", flush=True)

    n_specs = len(SPECS_TO_PLOT)
    cols = 3; rows = (n_specs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for ax, (spec, label, kind) in zip(axes, SPECS_TO_PLOT):
        try:
            render_panel(ax, ctx, spec, label, kind)
        except Exception as exc:
            ax.text(0.5, 0.5, f"{spec} failed:\n{type(exc).__name__}",
                    ha="center", va="center", fontsize=9, transform=ax.transAxes)
            ax.set_title(label, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n_specs:]:
        ax.set_visible(False)

    plt.suptitle(
        f"Fitted surfaces — predicted residual lay-out of input color space, projected to top-2 PCs\n"
        f"small dots = grid predictions (colored by input)   ·   large circles = true cogito centroids\n"
        f"n_colors = {ctx['n_colors']}",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    out_path = out_dir / "fitted_surfaces.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
