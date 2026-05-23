"""
auto_29: cross-template R² heatmap (idea rrr).

For each ordered pair (A, B) of templates, build per-color centroids of cogito
L40 activations restricted to template A (Y_A) and template B (Y_B), project
onto a shared global PCA basis (top-K), then with color-grouped k-fold CV
fit a ridge regressor RGB -> Y_A on train colors and score the predictions
against Y_B on held-out colors (macro-R² across the K PCs).

A symmetric heatmap (diagonal = within-template predictability, off-diagonal =
how well one template's color geometry transfers to another) reveals whether
templates share a common color manifold or carve it up differently.

No Gaussian RBF / radial bases — only PCA + linear Ridge (closed-form).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40"
COGITO_DIR = ROOT / "runs" / "COLOR_COGITO_L40"
OUT_PNG = RUN_DIR / "auto_29.png"
OUT_JSON = RUN_DIR / "auto_29.json"

N_TEMPLATES = 28
N_PCS = 16        # shared latent dim for cross-template scoring
N_FOLDS = 5
RIDGE_ALPHA = 1.0
SEED = 0


def load_xkcd_rgb(n_c: int) -> np.ndarray:
    """Mirror the harvest-order xkcd colors list. We only need RGB ∈ [0,1]."""
    cache = ROOT / "experiments" / "xkcd_colors.txt"
    rgb = []
    with open(cache) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            hexstr = parts[1].lstrip("#")
            r = int(hexstr[0:2], 16)
            g = int(hexstr[2:4], 16)
            b = int(hexstr[4:6], 16)
            rgb.append((r / 255.0, g / 255.0, b / 255.0))
    arr = np.asarray(rgb, dtype=np.float64)
    assert arr.shape[0] >= n_c, f"xkcd cache only has {arr.shape[0]} colors, need {n_c}"
    return arr[:n_c]


def main() -> None:
    print(f"[load] {COGITO_DIR / 'X_L40.npy'}", flush=True)
    X = np.load(COGITO_DIR / "X_L40.npy")  # (N, D)
    n_rows, d = X.shape
    n_c = n_rows // N_TEMPLATES
    n_rows_used = n_c * N_TEMPLATES
    X = X[:n_rows_used].astype(np.float32)
    print(f"[data] N={n_rows_used} = {n_c} colors × {N_TEMPLATES} templates  D={d}", flush=True)

    rgb = load_xkcd_rgb(n_c)  # (n_c, 3)
    print(f"[data] rgb features: {rgb.shape}", flush=True)

    # Reshape to (n_c, n_t, D). Row layout in harvest is colors-outer, templates-inner.
    Xc = X.reshape(n_c, N_TEMPLATES, d)

    # Global per-color centroid -> top-K PCA basis (shared target space).
    centroid_global = Xc.mean(axis=1)                # (n_c, D)
    mu = centroid_global.mean(0, keepdims=True)
    sigma = centroid_global.std(0, keepdims=True).clip(min=1e-6)
    Cn = ((centroid_global - mu) / sigma).astype(np.float64)
    Cn = Cn - Cn.mean(0, keepdims=True)
    # Top-K right singular vectors of Cn define the shared subspace.
    _, _, Vt = np.linalg.svd(Cn, full_matrices=False)
    V = Vt[:N_PCS].T  # (D, K)
    print(f"[pca] shared basis K={N_PCS}", flush=True)

    # Per-template projections: Y[t] = ((X_t - mu) / sigma) @ V  → (n_c, K)
    Y = np.empty((N_TEMPLATES, n_c, N_PCS), dtype=np.float64)
    for t in range(N_TEMPLATES):
        Xt = Xc[:, t, :].astype(np.float64)          # (n_c, D)
        Xtn = (Xt - mu) / sigma
        Xtn = Xtn - Xtn.mean(0, keepdims=True)
        Y[t] = Xtn @ V

    # Cross-template R² via color-grouped K-fold.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(n_c)))

    R2 = np.zeros((N_TEMPLATES, N_TEMPLATES), dtype=np.float64)
    for A in range(N_TEMPLATES):
        for B in range(N_TEMPLATES):
            # For each fold: fit Ridge(rgb_tr -> Y[A][tr]); predict on rgb_te;
            # score against Y[B][te]. Macro-R² = 1 - SS_res / SS_tot, where
            # SS_tot uses the held-out target's own mean per PC.
            ss_res_per_pc = np.zeros(N_PCS)
            ss_tot_per_pc = np.zeros(N_PCS)
            for tr, te in folds:
                model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
                model.fit(rgb[tr], Y[A][tr])
                Yhat = model.predict(rgb[te])         # predictions live in shared basis
                Yte = Y[B][te]
                ss_res_per_pc += ((Yte - Yhat) ** 2).sum(0)
                ss_tot_per_pc += ((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(0)
            r2_per_pc = 1.0 - ss_res_per_pc / np.maximum(ss_tot_per_pc, 1e-12)
            R2[A, B] = r2_per_pc.mean()
        print(f"  [row {A:2d}] diag={R2[A, A]:.3f}  off-mean={np.delete(R2[A], A).mean():.3f}", flush=True)

    diag = np.diag(R2)
    off = R2.copy(); np.fill_diagonal(off, np.nan)
    transfer_gap = diag - np.nanmean(off, axis=1)

    # ----- plot -----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    vmax = float(np.nanpercentile(R2, 99))
    vmin = float(np.nanpercentile(R2, 1))
    im = ax.imshow(R2, cmap="RdBu_r", vmin=-vmax if vmin < 0 else 0, vmax=vmax, aspect="equal")
    ax.set_xlabel("test template (B)")
    ax.set_ylabel("train template (A)")
    ax.set_title(
        f"Cross-template R² (Ridge α={RIDGE_ALPHA}, RGB→top-{N_PCS} PC shared basis,\n"
        f"{N_FOLDS}-fold color-grouped CV, cogito L40, n_c={n_c})"
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="macro R²")

    ax2 = axes[1]
    order = np.argsort(-diag)
    ax2.barh(np.arange(N_TEMPLATES), diag[order], color="steelblue", label="within-template R² (A→A)")
    ax2.barh(np.arange(N_TEMPLATES), np.nanmean(off, axis=1)[order], color="lightcoral",
             alpha=0.7, label="mean cross-template R² (A→B≠A)")
    ax2.set_yticks(np.arange(N_TEMPLATES))
    ax2.set_yticklabels([f"t{t:02d}" for t in order], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("R²")
    ax2.set_title("templates ranked by within-template R²")
    ax2.axvline(0, color="k", lw=0.5)
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    print(f"[save] {OUT_PNG}", flush=True)

    off_hi = R2.copy(); np.fill_diagonal(off_hi, -np.inf)
    off_lo = R2.copy(); np.fill_diagonal(off_lo, np.inf)
    bp = np.unravel_index(np.argmax(off_hi), R2.shape)
    wp = np.unravel_index(np.argmin(off_lo), R2.shape)
    summary = {
        "n_colors": int(n_c),
        "n_templates": int(N_TEMPLATES),
        "n_pcs": int(N_PCS),
        "ridge_alpha": float(RIDGE_ALPHA),
        "n_folds": int(N_FOLDS),
        "diag_mean": float(diag.mean()),
        "off_mean": float(np.nanmean(off)),
        "best_template_idx": int(np.argmax(diag)),
        "worst_template_idx": int(np.argmin(diag)),
        "best_offdiag_pair": [int(bp[0]), int(bp[1]), float(R2[bp])],
        "worst_offdiag_pair": [int(wp[0]), int(wp[1]), float(R2[wp])],
        "R2_matrix": R2.tolist(),
        "diag": diag.tolist(),
        "transfer_gap": transfer_gap.tolist(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {OUT_JSON}", flush=True)
    print(f"[summary] diag mean={diag.mean():.3f}  off mean={np.nanmean(off):.3f}  "
          f"best pair={summary['best_offdiag_pair']}  worst={summary['worst_offdiag_pair']}",
          flush=True)


if __name__ == "__main__":
    main()
