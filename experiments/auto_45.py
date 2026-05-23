"""
auto_45 — Idea (ttttt): do specs disagree on WHICH colors are hardest?

For each color c we have ~28 template repeats projected into the saved K=64
PC space (the Z used by color_manifold_gam). We refit a small basket of
*cheap* fitting specs in 5-fold CV (folds over colors, exactly like the
production run) and record:
    R²_c(s) = 1 - SSres_c(s) / SStot_c   (per-color R² for spec s)

Question: across the basket of specs, is there a stable "these colors are
hard" pattern (specs agree, high cross-spec Spearman, low residual variance
of R² rankings) — or do different fitting families fail on different
colors (specs disagree)?

Basket (no Gaussian RBF, no length_scale on Duchon):
    L_lin_rgb        — 3 features, linear baseline
    L_lin_lab        — perceptually-uniform linear
    L_lin_luminance  — 1D lightness only
    L_poly2_rgb      — degree-2 polynomial in RGB
    L_pca_ridge      — ridge on top-32 PCs of the per-color centroid input
                       (linear in the latent representation itself)
    N_knn_lab_k10    — k-NN regression in Lab (k=10)
    U_duchon_rgb     — Duchon thin-plate in RGB (no length_scale)
    U_duchon_lab     — Duchon thin-plate in Lab

Plot (single figure, 4 panels):
  (a) Spec×Spec Spearman of per-color R² heatmap.
  (b) Per-color R² scatter for two maximally-disagreeing specs (lowest
      Spearman pair) with worst & best colors labeled.
  (c) For each color the std-across-specs of R²; show top-12 swatches with
      highest std (specs disagree most) and the 12 with lowest std (specs
      agree). Plotted as colored squares with name & σ_R².
  (d) Distribution of per-color R² stratified by spec (violin / strip).

Headline number we want: median pairwise Spearman across specs, and the
list of colors that look hard to one family but easy to another.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
import color_manifold_gam as cmg  # noqa: E402

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"
OUT = RUN_DIR / "auto_45.png"

N_FOLDS = 5
SEED = 0


def ridge_fit(Phi, Y, alpha=1.0):
    K = Phi.shape[1]
    A = Phi.T @ Phi + alpha * np.eye(K)
    return np.linalg.solve(A, Phi.T @ Y)


def poly2(X):
    # X is (N,3) in [0,1]
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    cols = [a, b, c, a * a, b * b, c * c, a * b, a * c, b * c, np.ones_like(a)]
    return np.stack(cols, axis=1)


def duchon_fit_predict(X_tr, Y_tr, X_te, per_side=5, init_log_lam=0.0):
    """Thin-plate Duchon in 3D; no length_scale."""
    mn = X_tr.min(0); mx = X_tr.max(0)
    ax = [np.linspace(mn[d], mx[d], per_side) for d in range(3)]
    G = np.meshgrid(*ax, indexing="ij")
    centers = np.stack([g.flatten() for g in G], axis=1)
    Phi_tr, P = cmg.duchon_basis_radial(X_tr, centers)
    Phi_te, _ = cmg.duchon_basis_radial(X_te, centers)
    B, _ = cmg.reml_fit(Phi_tr, Y_tr, P, init_log_lam)
    return Phi_tr @ B, Phi_te @ B


def per_color_r2(Y_true, Y_pred, mean_global):
    """Per-row R² using a global Z-mean baseline (color is one row of Z)."""
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=1)
    ss_tot = np.sum((Y_true - mean_global) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def main():
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    Vt = np.asarray(d["per_layer"]["L40"]["Vt_topK"], dtype=np.float64)
    mu = np.asarray(d["per_layer"]["L40"]["mu"], dtype=np.float64)
    sigma = np.asarray(d["per_layer"]["L40"]["sigma"], dtype=np.float64)
    K = Vt.shape[0]

    R_axis = np.asarray(d["color_axes_per_color_index"]["R"], dtype=np.float64)
    G_axis = np.asarray(d["color_axes_per_color_index"]["G"], dtype=np.float64)
    B_axis = np.asarray(d["color_axes_per_color_index"]["B"], dtype=np.float64)
    n_c = R_axis.size
    RGB = np.stack([R_axis, G_axis, B_axis], axis=1)  # already in [0,1]
    print(f"[meta] n_c={n_c}, n_t={n_t}, K={K}")

    # Build per-color centroid in PC space (the regression target Z).
    X_full = np.load(HARVEST, mmap_mode="r")
    n_rows, D = X_full.shape
    assert n_rows == n_c * n_t, f"{n_rows} vs {n_c}*{n_t}"
    per_color = np.zeros((n_c, D), dtype=np.float64)
    block = 4096
    for s in range(0, n_rows, block):
        e = min(s + block, n_rows)
        chunk = np.asarray(X_full[s:e], dtype=np.float64)
        idx = np.arange(s, e) // n_t
        for ci in np.unique(idx):
            per_color[ci] += chunk[idx == ci].sum(axis=0)
    per_color /= n_t
    Xn = (per_color - mu) / np.maximum(sigma, 1e-6)
    Z = (Xn - Xn.mean(0, keepdims=True)) @ Vt.T  # (n_c, K)
    print(f"[meta] Z {Z.shape}, ||Z||_F={np.linalg.norm(Z):.2f}")

    # Lab features (perceptual)
    LAB = cmg.rgb_to_lab(RGB)
    LAB_n = (LAB - LAB.mean(0)) / (LAB.std(0) + 1e-9)
    RGB_n = (RGB - 0.5)  # centered

    # Folds (over colors)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_c)
    folds = np.array_split(perm, N_FOLDS)
    mean_global = Z.mean(0, keepdims=True)

    specs = [
        "L_lin_rgb", "L_lin_lab", "L_lin_lum", "L_poly2_rgb",
        "N_knn_lab_k10", "U_duchon_rgb", "U_duchon_lab",
    ]

    Z_pred_all = {s: np.zeros_like(Z) for s in specs}

    for k in range(N_FOLDS):
        te = folds[k]; tr = np.setdiff1d(np.arange(n_c), te)
        print(f"[fold {k+1}/{N_FOLDS}] train={tr.size} test={te.size}")
        Ztr, Zte = Z[tr], Z[te]
        rgb_tr, rgb_te = RGB[tr], RGB[te]
        lab_tr, lab_te = LAB[tr], LAB[te]

        # L_lin_rgb
        Ph_tr = np.concatenate([rgb_tr, np.ones((tr.size, 1))], axis=1)
        Ph_te = np.concatenate([rgb_te, np.ones((te.size, 1))], axis=1)
        W = ridge_fit(Ph_tr, Ztr, 1.0)
        Z_pred_all["L_lin_rgb"][te] = Ph_te @ W

        # L_lin_lab
        Ph_tr = np.concatenate([lab_tr, np.ones((tr.size, 1))], axis=1)
        Ph_te = np.concatenate([lab_te, np.ones((te.size, 1))], axis=1)
        W = ridge_fit(Ph_tr, Ztr, 1.0)
        Z_pred_all["L_lin_lab"][te] = Ph_te @ W

        # L_lin_lum
        lum_tr = (0.299*rgb_tr[:,0]+0.587*rgb_tr[:,1]+0.114*rgb_tr[:,2])[:,None]
        lum_te = (0.299*rgb_te[:,0]+0.587*rgb_te[:,1]+0.114*rgb_te[:,2])[:,None]
        Ph_tr = np.concatenate([lum_tr, np.ones((tr.size,1))],axis=1)
        Ph_te = np.concatenate([lum_te, np.ones((te.size,1))],axis=1)
        W = ridge_fit(Ph_tr, Ztr, 1.0)
        Z_pred_all["L_lin_lum"][te] = Ph_te @ W

        # L_poly2_rgb
        Ph_tr = poly2(rgb_tr); Ph_te = poly2(rgb_te)
        W = ridge_fit(Ph_tr, Ztr, 1.0)
        Z_pred_all["L_poly2_rgb"][te] = Ph_te @ W

        # N_knn_lab_k10
        knn = KNeighborsRegressor(n_neighbors=10, weights="distance")
        knn.fit(lab_tr, Ztr)
        Z_pred_all["N_knn_lab_k10"][te] = knn.predict(lab_te)

        # U_duchon_rgb
        _, pred = duchon_fit_predict(rgb_tr, Ztr, rgb_te, per_side=5)
        Z_pred_all["U_duchon_rgb"][te] = pred

        # U_duchon_lab
        _, pred = duchon_fit_predict(lab_tr, Ztr, lab_te, per_side=5)
        Z_pred_all["U_duchon_lab"][te] = pred

    # Per-color R² for each spec (one scalar per color).
    R2 = {s: per_color_r2(Z, Z_pred_all[s], mean_global) for s in specs}
    # Macro means for sanity
    for s in specs:
        print(f"  macro R² {s}: median={np.median(R2[s]):.3f} mean={R2[s].mean():.3f}")

    # ===================== analysis =====================
    R2_mat = np.stack([R2[s] for s in specs], axis=1)  # (n_c, S)
    S = len(specs)

    # Pairwise Spearman of per-color R²
    rho = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            r, _ = spearmanr(R2_mat[:, i], R2_mat[:, j])
            rho[i, j] = r
    iu = np.triu_indices(S, k=1)
    median_rho = float(np.median(rho[iu]))
    min_rho = float(rho[iu].min())
    min_pair = (iu[0][np.argmin(rho[iu])], iu[1][np.argmin(rho[iu])])
    print(f"[spearman] median off-diag={median_rho:.3f} min={min_rho:.3f} "
          f"pair=({specs[min_pair[0]]},{specs[min_pair[1]]})")

    # Per-color std across specs (disagreement)
    sd_per_color = R2_mat.std(axis=1)
    mean_per_color = R2_mat.mean(axis=1)
    order_disagree = np.argsort(-sd_per_color)
    order_agree = np.argsort(sd_per_color)

    # Colors metadata
    colors = cmg.load_xkcd_colors()[:n_c]
    names = [c[0] for c in colors]
    rgb_disp = np.stack([np.array(c[1:], dtype=float)/255.0 for c in colors])

    # ===================== plot =====================
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.1, 1.0],
                          width_ratios=[1, 1, 1.05], hspace=0.45, wspace=0.35)

    # (a) Spearman heatmap
    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    axA.set_xticks(range(S)); axA.set_xticklabels(specs, rotation=45, ha="right", fontsize=8)
    axA.set_yticks(range(S)); axA.set_yticklabels(specs, fontsize=8)
    for i in range(S):
        for j in range(S):
            axA.text(j, i, f"{rho[i,j]:.2f}", ha="center", va="center",
                     fontsize=6, color="black")
    axA.set_title(f"(a) Pairwise Spearman of per-color R²\nmedian off-diag={median_rho:.2f}")
    fig.colorbar(im, ax=axA, fraction=0.046)

    # (b) Disagreeing-pair scatter
    axB = fig.add_subplot(gs[0, 1])
    i, j = min_pair
    xi, yj = R2_mat[:, i], R2_mat[:, j]
    axB.scatter(xi, yj, c=rgb_disp, s=12, edgecolor="grey", linewidth=0.2)
    lim = (min(xi.min(), yj.min()) - 0.05, max(xi.max(), yj.max()) + 0.05)
    axB.plot(lim, lim, "k--", linewidth=0.7, alpha=0.5)
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_xlabel(f"R² · {specs[i]}"); axB.set_ylabel(f"R² · {specs[j]}")
    axB.set_title(f"(b) Most-disagreeing spec pair  ρ={min_rho:.2f}")
    # Label the points that disagree most
    diff = xi - yj
    for k in np.argsort(-np.abs(diff))[:6]:
        axB.annotate(names[k], (xi[k], yj[k]), fontsize=7,
                     xytext=(3, 2), textcoords="offset points")

    # (c) per-color R² mean vs std-across-specs
    axC = fig.add_subplot(gs[0, 2])
    axC.scatter(mean_per_color, sd_per_color, c=rgb_disp, s=10,
                edgecolor="grey", linewidth=0.2)
    axC.set_xlabel("mean R² across specs"); axC.set_ylabel("σ R² across specs (disagreement)")
    axC.set_title("(c) per-color mean vs disagreement")
    for k in order_disagree[:6]:
        axC.annotate(names[k], (mean_per_color[k], sd_per_color[k]),
                     fontsize=7, xytext=(3,2), textcoords="offset points")

    # (d) Top-12 most-disagreed swatches
    axD = fig.add_subplot(gs[1, :])
    axD.set_xlim(0, 12); axD.set_ylim(-0.05, 1.7); axD.axis("off")
    axD.set_title("(d) 12 colors where specs DISAGREE most (σ R² across specs)", loc="left")
    for col, ci in enumerate(order_disagree[:12]):
        axD.add_patch(mpatches.Rectangle((col + 0.05, 0.7), 0.9, 0.6,
                                         facecolor=rgb_disp[ci], edgecolor="black"))
        axD.text(col + 0.5, 1.42, names[ci], ha="center", va="bottom", fontsize=8)
        axD.text(col + 0.5, 0.55, f"σ={sd_per_color[ci]:.2f}\nμ={mean_per_color[ci]:.2f}",
                 ha="center", va="top", fontsize=7)
        # tiny per-spec bars
        for si, s in enumerate(specs):
            v = R2_mat[ci, si]
            bx = col + 0.05 + 0.9 * si / S
            bw = 0.9 / S - 0.01
            bh = 0.25 * max(0.0, min(1.0, v))
            axD.add_patch(mpatches.Rectangle((bx, 0.05), bw, bh,
                                             facecolor="steelblue", edgecolor="none"))

    # (e) Bottom-12 most-agreed
    axE = fig.add_subplot(gs[2, :])
    axE.set_xlim(0, 12); axE.set_ylim(-0.05, 1.7); axE.axis("off")
    axE.set_title("(e) 12 colors where specs AGREE most (low σ R² across specs)", loc="left")
    for col, ci in enumerate(order_agree[:12]):
        axE.add_patch(mpatches.Rectangle((col + 0.05, 0.7), 0.9, 0.6,
                                         facecolor=rgb_disp[ci], edgecolor="black"))
        axE.text(col + 0.5, 1.42, names[ci], ha="center", va="bottom", fontsize=8)
        axE.text(col + 0.5, 0.55, f"σ={sd_per_color[ci]:.2f}\nμ={mean_per_color[ci]:.2f}",
                 ha="center", va="top", fontsize=7)
        for si, s in enumerate(specs):
            v = R2_mat[ci, si]
            bx = col + 0.05 + 0.9 * si / S
            bw = 0.9 / S - 0.01
            bh = 0.25 * max(0.0, min(1.0, v))
            axE.add_patch(mpatches.Rectangle((bx, 0.05), bw, bh,
                                             facecolor="darkorange", edgecolor="none"))

    fig.suptitle(
        "auto_45 · do specs disagree on which colors are hard?  "
        f"[median pairwise Spearman of per-color R² = {median_rho:.2f}]",
        fontsize=12,
    )
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[done] wrote {OUT}")

    # console summary
    print("\n[top-10 disagreement colors]")
    for k in order_disagree[:10]:
        per_spec = ", ".join(f"{s}={R2_mat[k,i]:+.2f}" for i,s in enumerate(specs))
        print(f"  {names[k]:30s} σ={sd_per_color[k]:.3f}  μ={mean_per_color[k]:+.2f}  | {per_spec}")
    print("\n[top-10 agreement colors]")
    for k in order_agree[:10]:
        print(f"  {names[k]:30s} σ={sd_per_color[k]:.3f}  μ={mean_per_color[k]:+.2f}")


if __name__ == "__main__":
    main()
