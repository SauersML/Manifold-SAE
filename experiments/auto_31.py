"""auto_31: partial-dependence of PC1, PC2, PC3 on each RGB axis (idea zzz).

For the cogito L40 per-color centroid manifold, we fit a 3D Duchon smooth
    f_k : (R, G, B) -> PC_k(centroid)
for k in {1, 2, 3} via gamfit's `duchon_basis` + REML λ-selection (no
length_scale, no Gaussian RBF). We then visualize the *partial* dependence
of each f_k on each axis A in {R, G, B} by sweeping A over [0, 1] while
fixing the other two channels at three slice values {0.25, 0.50, 0.75}.

Grid: 9 panels (3 PCs x 3 axes). Each panel has 3 curves (slice values of
the *other-two* channels held jointly fixed at (v, v) for v in {.25, .5, .75}).

This is a clean readout of which RGB axis drives each top PC and whether
there is interaction structure: parallel curves => additive; fanning =>
interaction. Compare with linear (ridge) partial dependence as a baseline
to show how much of the curvature is genuinely nonlinear.

No RBF. Tools used: numpy (PCA via SVD), sklearn-free ridge via closed-form,
gamfit Duchon. Centroids only — fast.

Output:
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_31.png
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_31.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import color_manifold_gam as cmg  # noqa: E402

COGITO_DIR = ROOT / "runs" / "COLOR_COGITO_L40"
OUT_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
OUT_PNG = OUT_DIR / "auto_31.png"
OUT_JSON = OUT_DIR / "auto_31.json"

N_TEMPLATES = 28
N_PCS = 3                  # only top-3 PCs (RGB has rank ~3 by construction)
LATTICE_PER_SIDE = 5       # 125 Duchon centers in [0,1]^3 (matches main run)
SLICE_VALUES = (0.25, 0.50, 0.75)
SWEEP_N = 80               # grid points along the swept axis
AXIS_NAMES = ("R", "G", "B")


def _fit_duchon_3d(X_rgb_tr: np.ndarray, Z_tr: np.ndarray,
                    X_rgb_te: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    centers = cmg.lattice_centers(LATTICE_PER_SIDE)        # (125, 3)
    Phi_tr, P = cmg.duchon_basis_radial(X_rgb_tr, centers) # train design
    B, log_lam = cmg.reml_fit(Phi_tr, Z_tr, P, init_log_lambda=0.0)
    Phi_te, _ = cmg.duchon_basis_radial(X_rgb_te, centers)
    Z_pred = Phi_te @ B
    return Z_pred, B, {"centers": centers, "log_lambda": log_lam}


def _ridge_fit_predict(X_tr: np.ndarray, Z_tr: np.ndarray,
                        X_te: np.ndarray, alpha: float = 1e-3
                        ) -> tuple[np.ndarray, np.ndarray]:
    # Add intercept, closed-form ridge that does NOT penalize the intercept.
    n_tr = X_tr.shape[0]
    Xa = np.concatenate([np.ones((n_tr, 1)), X_tr], axis=1)
    p = Xa.shape[1]
    A = Xa.T @ Xa
    A[1:, 1:] += alpha * np.eye(p - 1)
    coef = np.linalg.solve(A, Xa.T @ Z_tr)                  # (p, R)
    Xa_te = np.concatenate([np.ones((X_te.shape[0], 1)), X_te], axis=1)
    return Xa_te @ coef, coef


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load cached residuals + xkcd RGB ----
    X = np.load(COGITO_DIR / "X_L40.npy").astype(np.float64)
    n_rows, d = X.shape
    n_colors = n_rows // N_TEMPLATES
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)
    colors = cmg.load_xkcd_colors()
    assert len(colors) == n_colors
    rgb01 = np.array([[r, g, b] for _, r, g, b in colors], dtype=np.float64) / 255.0

    # ---- per-color centroid -> standardize -> top-3 PCs ----
    Xr = X.reshape(n_colors, N_TEMPLATES, d)
    centroids = Xr.mean(axis=1)                              # (n_colors, D)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroids - mu) / sigma
    Cn -= Cn.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Cn, full_matrices=False)
    V = Vt[:N_PCS].T                                          # (D, 3)
    Z = Cn @ V                                                # (n_colors, 3)
    evr = (S[:N_PCS] ** 2) / (S ** 2).sum()
    print(f"[pca] top-{N_PCS} EVR = {evr.tolist()}", flush=True)

    # ---- fit on ALL colors (we want the shape of the surface, not held-out R²;
    # held-out R² for these specs is already covered in earlier auto_*) ----
    Z_pred_duchon, B, fit_meta = _fit_duchon_3d(rgb01, Z, rgb01)
    Z_pred_lin, lin_coef = _ridge_fit_predict(rgb01, Z, rgb01, alpha=1e-3)

    # In-sample R² per PC, just for context on each panel.
    def per_pc_r2(Z_true: np.ndarray, Z_hat: np.ndarray) -> list[float]:
        ss_res = ((Z_true - Z_hat) ** 2).sum(0)
        ss_tot = ((Z_true - Z_true.mean(0, keepdims=True)) ** 2).sum(0)
        return (1.0 - ss_res / np.maximum(ss_tot, 1e-12)).tolist()

    r2_duchon = per_pc_r2(Z, Z_pred_duchon)
    r2_lin = per_pc_r2(Z, Z_pred_lin)
    print(f"[r2 duchon] {r2_duchon}", flush=True)
    print(f"[r2 ridge ] {r2_lin}", flush=True)

    # ---- build partial-dependence panels ----
    sweep = np.linspace(0.0, 1.0, SWEEP_N)
    # `pd_duchon[(pc, swept_axis)][slice_idx]` -> array of length SWEEP_N
    pd_duchon: dict = {}
    pd_lin: dict = {}
    centers = fit_meta["centers"]

    for swept in range(3):
        other_axes = [a for a in range(3) if a != swept]
        for slice_i, v in enumerate(SLICE_VALUES):
            grid = np.zeros((SWEEP_N, 3), dtype=np.float64)
            grid[:, swept] = sweep
            grid[:, other_axes[0]] = v
            grid[:, other_axes[1]] = v
            # Duchon
            Phi_g, _ = cmg.duchon_basis_radial(grid, centers)
            pd_d = Phi_g @ B                                  # (SWEEP_N, 3)
            # Ridge linear (with intercept)
            Xa_g = np.concatenate([np.ones((SWEEP_N, 1)), grid], axis=1)
            pd_l = Xa_g @ lin_coef                            # (SWEEP_N, 3)
            for pc in range(N_PCS):
                pd_duchon.setdefault((pc, swept), []).append(pd_d[:, pc])
                pd_lin.setdefault((pc, swept), []).append(pd_l[:, pc])

    # ---- plot 3x3 grid: rows=PCs, cols=swept axis ----
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), sharex=True)
    palette = plt.get_cmap("viridis")
    slice_colors = [palette(0.15), palette(0.50), palette(0.85)]

    for pc in range(N_PCS):
        for swept in range(3):
            ax = axes[pc, swept]
            curves_d = pd_duchon[(pc, swept)]
            curves_l = pd_lin[(pc, swept)]
            for s_i, (v, c) in enumerate(zip(SLICE_VALUES, slice_colors)):
                ax.plot(sweep, curves_d[s_i], color=c, lw=2.0,
                         label=f"others={v:.2f}  (Duchon)")
                ax.plot(sweep, curves_l[s_i], color=c, lw=1.0, ls="--",
                         alpha=0.85,
                         label=f"others={v:.2f}  (linear)")
            # Scatter the real centroids in this PC, colored by the swept axis
            # value of their own RGB (rugplot context).
            ax.scatter(rgb01[:, swept], Z[:, pc], s=6, c="0.55", alpha=0.35,
                        zorder=0)
            ax.axhline(0, color="black", lw=0.4, alpha=0.6)
            ax.grid(linestyle=":", alpha=0.4)
            if pc == 0:
                ax.set_title(f"swept {AXIS_NAMES[swept]}  "
                              f"(other two = R̄/Ḡ/B̄ less '{AXIS_NAMES[swept]}')",
                              fontsize=10)
            if swept == 0:
                ax.set_ylabel(f"PC{pc + 1}   "
                                f"(EVR={evr[pc]:.2f}, "
                                f"in-sample R²: Duchon={r2_duchon[pc]:.3f}, "
                                f"lin={r2_lin[pc]:.3f})", fontsize=9)
            if pc == 2:
                ax.set_xlabel(f"{AXIS_NAMES[swept]} ∈ [0, 1]")

    # one global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
                frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        "auto_31  ·  RGB → top-3 PC partial dependence  "
        "(cogito L40 per-color centroid, 3D Duchon, 125 centers, REML λ)\n"
        "Solid = Duchon ƒ(R,G,B); dashed = linear-ridge baseline.  "
        "Each curve fixes the *other two* channels at the slice value shown; "
        "non-parallel solid curves indicate RGB×RGB interactions the LM "
        "embeds in that PC.",
        fontsize=11)
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_PNG}", flush=True)

    # ---- save numerical summary ----
    summary = {
        "n_colors": int(n_colors),
        "n_pcs": int(N_PCS),
        "lattice_per_side": int(LATTICE_PER_SIDE),
        "slice_values": list(SLICE_VALUES),
        "sweep_n": int(SWEEP_N),
        "evr_top3": evr.tolist(),
        "in_sample_r2_duchon": r2_duchon,
        "in_sample_r2_ridge_linear": r2_lin,
        "duchon_log_lambda": float(fit_meta["log_lambda"]),
        "partial_dependence_duchon": {
            f"pc{pc + 1}_swept_{AXIS_NAMES[swept]}": {
                f"others={v:.2f}": pd_duchon[(pc, swept)][s_i].tolist()
                for s_i, v in enumerate(SLICE_VALUES)
            }
            for pc in range(N_PCS) for swept in range(3)
        },
        "partial_dependence_linear": {
            f"pc{pc + 1}_swept_{AXIS_NAMES[swept]}": {
                f"others={v:.2f}": pd_lin[(pc, swept)][s_i].tolist()
                for s_i, v in enumerate(SLICE_VALUES)
            }
            for pc in range(N_PCS) for swept in range(3)
        },
        "sweep_grid": sweep.tolist(),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[save] {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
