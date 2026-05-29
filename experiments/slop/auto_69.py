"""auto_69.py — Gauge-FIXED canonical manifold via HSV-anchored Duchon.

Fit Z ≈ f(h, s, v) where (h, s, v) is HSV — externally observable, no
gauge freedom. The natural choice is gamfit's mixed-periodic Duchon
with periodic_per_axis=[True, False, False] on a 128-center lattice,
but the current gamfit (CPU) build hard-pins the mixed-periodic kernel
to p=1, s=0 which violates the admissibility 2(p+s) > d in d=3 (and
d=2). We therefore build the same function space as a TENSOR PRODUCT
of two valid Duchon factors — still pure Duchon, scale-free, no
length_scale, no B-splines, no Gaussian RBF:

    Φ(h, s, v)  =  Φ_h(h) ⊗ Φ_sv(s, v)

  • Φ_h: 1D periodic Duchon m=2 on hue (8 centers on the circle).
  • Φ_sv: 2D non-periodic Duchon m=3 on (s, v) (16 centers, 4×4 lattice).

Total 8 × 16 = 128 basis columns, matching the spec's 8×4×4 center
budget. The roughness penalty is the Kronecker sum
   P = P_h ⊗ I_sv  +  I_h ⊗ P_sv
which penalises hue-curvature and (s,v)-curvature additively (a single
λ scales the joint penalty; REML picks it).

KEY QUESTION: does the HSV-anchored 3D Duchon match U_3d's reported
train R² (~0.74)? If yes, U_3d's apparent extra R² is gauge slack.
If no, U_3d captures non-HSV color geometry.

No Gaussian RBF, no Duchon length_scale, no B-splines.
"""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors
from color_filter_list import filter_colors
from _pca_basis import load_pc_basis, project

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
N_FOLDS = 5
K_PC = 64
N_H, N_S, N_V = 8, 4, 4  # 128 centers total


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def r2_per_column(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    out = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return out


def hue_centers(n_h=N_H):
    return np.linspace(0.0, 1.0, n_h, endpoint=False).reshape(-1, 1)


def sv_centers(n_s=N_S, n_v=N_V):
    s = np.linspace(0.0, 1.0, n_s)
    v = np.linspace(0.0, 1.0, n_v)
    S, V = np.meshgrid(s, v, indexing="ij")
    return np.stack([S.ravel(), V.ravel()], axis=1)


def hsv_tensor_basis(points, h_ctr, sv_ctr):
    """Tensor product of (1D periodic Duchon m=2 on h) ⊗
    (2D non-periodic Duchon m=3 on s,v).

    Returns
    -------
    Phi : (N, K_h * K_sv) row-wise Kronecker product Φ_h ⊠ Φ_sv
    P   : (K, K) Kronecker-sum penalty  P_h ⊗ I_sv + I_h ⊗ P_sv
    """
    import gamfit
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    h_pts = pts[:, 0:1]
    sv_pts = pts[:, 1:3]

    Phi_h = np.asarray(
        gamfit.duchon_basis(h_pts, h_ctr, m=2, periodic_per_axis=[True])
    )
    P_h = np.asarray(
        gamfit.duchon_function_norm_penalty(h_ctr, m=2, periodic_per_axis=[True])
    )
    Phi_sv = np.asarray(gamfit.duchon_basis(sv_pts, sv_ctr, m=3))
    P_sv = np.asarray(gamfit.duchon_function_norm_penalty(sv_ctr, m=3))

    K_h = Phi_h.shape[1]
    K_sv = Phi_sv.shape[1]
    # row-wise Kronecker product (Khatri-Rao on rows):
    # Phi[i, a*K_sv + b] = Phi_h[i, a] * Phi_sv[i, b]
    Phi = (Phi_h[:, :, None] * Phi_sv[:, None, :]).reshape(pts.shape[0], K_h * K_sv)

    I_h = np.eye(K_h)
    I_sv = np.eye(K_sv)
    P = np.kron(P_h, I_sv) + np.kron(I_h, P_sv)
    # ensure symmetric
    P = 0.5 * (P + P.T)
    return Phi, P


def reml_fit_one(Phi, Y, P, init_lambda=None):
    import gamfit
    out = gamfit.gaussian_reml_fit(Phi, Y, P, init_lambda=init_lambda)
    return np.asarray(out["coefficients"]), float(np.log(float(out["lambda"])))


def main():
    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X = np.load(cache, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    print(f"[hsv-duchon] X (mmap) shape={X.shape}, n_raw={n_raw}")

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    N = len(kept)
    print(f"[hsv-duchon] {N} filtered colors")

    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    evr_top = float(basis["evr"].sum())
    print(f"[hsv-duchon] Z shape={Z.shape}, top-{K_PC} EVR cum = {evr_top:.3f}")

    h_ctr = hue_centers(N_H)
    sv_ctr = sv_centers(N_S, N_V)
    n_centers_total = h_ctr.shape[0] * sv_ctr.shape[0]
    print(f"[hsv-duchon] centers h={h_ctr.shape[0]} × sv={sv_ctr.shape[0]} "
          f"= {n_centers_total} total ({N_H}×{N_S}×{N_V})")
    pts = np.stack([hue, sat, val], axis=1)

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    # --- 5-fold CV of HSV-Duchon -> Z ---
    print("\n[cv] HSV-Duchon (periodic on h, m=2) -> Z_top64")
    preds = np.zeros_like(Z)
    ll_folds = []
    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        Phi_tr, P = hsv_tensor_basis(pts[tr], h_ctr, sv_ctr)
        Phi_te, _ = hsv_tensor_basis(pts[te], h_ctr, sv_ctr)
        B, ll = reml_fit_one(Phi_tr, Z[tr], P)
        preds[te] = Phi_te @ B
        ll_folds.append(ll)
        print(f"   fold {f}: log_lambda = {ll:+.2f}")
    r2_cv = float(r2_macro(Z, preds))
    r2_pc_cv = r2_per_column(Z, preds)
    print(f"\n[cv] HSV-Duchon macro R^2 = {r2_cv:+.4f}")
    print(f"     mean log_lambda      = {np.mean(ll_folds):+.2f}")

    # --- in-sample fit for slice/visualization ---
    Phi_full, P_full = hsv_tensor_basis(pts, h_ctr, sv_ctr)
    B_full, ll_full = reml_fit_one(Phi_full, Z, P_full)
    Z_hat_train = Phi_full @ B_full
    r2_train = float(r2_macro(Z, Z_hat_train))
    print(f"[train] HSV-Duchon train R^2 = {r2_train:+.4f}  (log_lambda={ll_full:+.2f})")

    # --- 2D slice: γ(h, s=0.7, v=0.7) through Z's top-3 PCs ---
    n_dense = 360
    h_dense = np.linspace(0, 1, n_dense, endpoint=False)
    pts_slice = np.stack(
        [h_dense, 0.7 * np.ones(n_dense), 0.7 * np.ones(n_dense)], axis=1
    )
    Phi_slice, _ = hsv_tensor_basis(pts_slice, h_ctr, sv_ctr)
    gamma_z = Phi_slice @ B_full  # shape (360, K_PC)
    # Top-3 PCs of Z (Z is already projected onto canonical PCA basis;
    # those columns ARE the principal components sorted by EVR.)
    gamma_top3 = gamma_z[:, :3]
    Z_top3 = Z[:, :3]

    # comparison anchors
    r2_hue_only = 0.251         # auto_66 hue-only Duchon
    r2_knn_lab = 0.246          # auto_41
    r2_u3d_train_ref = 0.74     # auto_exp_11 reference (gauge-dependent)

    # ---- Render ----
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3)

    # (a) Bar: HSV-Duchon vs ceilings
    ax1 = fig.add_subplot(gs[0, 0])
    names = ["HSV-Duchon\n(CV)", "Hue-only\nDuchon (CV)", "kNN-Lab\n(CV)", "U_3d\n(train, gauge)"]
    vals = [r2_cv, r2_hue_only, r2_knn_lab, r2_u3d_train_ref]
    colors = ["navy", "steelblue", "green", "firebrick"]
    bars = ax1.bar(names, vals, color=colors, edgecolor="black")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:+.3f}",
                 ha="center", fontsize=9)
    ax1.axhline(0.0, color="black", lw=0.5)
    ax1.set_ylabel("macro R²")
    ax1.set_title("(a) HSV-Duchon vs ceilings\n(8×4×4 centers, mixed-periodic m=2)")
    ax1.set_ylim(0, max(vals) * 1.15)
    ax1.grid(alpha=0.3, axis="y")

    # (b) per-PC R² heatmap (CV) -> bar
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(K_PC, 32)
    cols = np.arange(n_show)
    pc_vals = r2_pc_cv[:n_show]
    ax2.bar(cols, pc_vals, color="navy", edgecolor="black", lw=0.3)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("PC index")
    ax2.set_ylabel("CV R² (per PC)")
    ax2.set_title(f"(b) per-PC CV R² of HSV-Duchon fit (top {n_show})")
    ax2.grid(alpha=0.3, axis="y")

    # (b2) heatmap visualization of per-PC R² across folds
    ax2b = fig.add_subplot(gs[0, 2])
    # build per-fold per-PC R² matrix
    per_fold_r2 = np.zeros((N_FOLDS, K_PC))
    for f in range(N_FOLDS):
        te = fold == f
        if te.sum() < 2:
            per_fold_r2[f] = np.nan
            continue
        ss_res = ((Z[te] - preds[te]) ** 2).sum(0)
        ss_tot = ((Z[te] - Z[te].mean(0, keepdims=True)) ** 2).sum(0)
        per_fold_r2[f] = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    im = ax2b.imshow(per_fold_r2[:, :n_show], aspect="auto", cmap="RdBu_r",
                     vmin=-0.5, vmax=0.5, interpolation="nearest")
    ax2b.set_xlabel("PC index")
    ax2b.set_ylabel("fold")
    ax2b.set_title("(c) per-fold per-PC R² heatmap")
    plt.colorbar(im, ax=ax2b, shrink=0.7)

    # (d) 2D slice γ(h, 0.7, 0.7) in PC1-PC2
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(Z_top3[:, 0], Z_top3[:, 1], c=rgb, s=14, edgecolor="black", linewidth=0.2,
                alpha=0.7, label="data Z[:, 0:2]")
    ax3.plot(gamma_top3[:, 0], gamma_top3[:, 1], "k-", lw=2, alpha=0.7,
             label="γ(h, s=0.7, v=0.7)")
    for i in range(0, n_dense, 15):
        h_rgb = colorsys.hsv_to_rgb(h_dense[i], 0.7, 0.7)
        ax3.plot(gamma_top3[i, 0], gamma_top3[i, 1], "o", color=h_rgb, ms=8,
                 mec="black", mew=0.6)
    ax3.set_xlabel("PC1 of Z")
    ax3.set_ylabel("PC2 of Z")
    ax3.set_title("(d) γ(h, 0.7, 0.7) closed loop in PC1-PC2")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # (e) PC1-PC3
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(Z_top3[:, 0], Z_top3[:, 2], c=rgb, s=14, edgecolor="black", linewidth=0.2,
                alpha=0.7)
    ax4.plot(gamma_top3[:, 0], gamma_top3[:, 2], "k-", lw=2, alpha=0.7)
    for i in range(0, n_dense, 15):
        h_rgb = colorsys.hsv_to_rgb(h_dense[i], 0.7, 0.7)
        ax4.plot(gamma_top3[i, 0], gamma_top3[i, 2], "o", color=h_rgb, ms=8,
                 mec="black", mew=0.6)
    ax4.set_xlabel("PC1 of Z")
    ax4.set_ylabel("PC3 of Z")
    ax4.set_title("(e) γ(h, 0.7, 0.7) in PC1-PC3")
    ax4.grid(alpha=0.3)

    # (f) PC2-PC3
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(Z_top3[:, 1], Z_top3[:, 2], c=rgb, s=14, edgecolor="black", linewidth=0.2,
                alpha=0.7)
    ax5.plot(gamma_top3[:, 1], gamma_top3[:, 2], "k-", lw=2, alpha=0.7)
    for i in range(0, n_dense, 15):
        h_rgb = colorsys.hsv_to_rgb(h_dense[i], 0.7, 0.7)
        ax5.plot(gamma_top3[i, 1], gamma_top3[i, 2], "o", color=h_rgb, ms=8,
                 mec="black", mew=0.6)
    ax5.set_xlabel("PC2 of Z")
    ax5.set_ylabel("PC3 of Z")
    ax5.set_title("(f) γ(h, 0.7, 0.7) in PC2-PC3")
    ax5.grid(alpha=0.3)

    fig.suptitle(
        f"HSV-anchored 3D Duchon (gauge-fixed) · {N} xkcd colors · cogito L40\n"
        f"CV R²={r2_cv:+.3f} vs hue-only={r2_hue_only:.3f}, kNN-Lab={r2_knn_lab:.3f}, "
        f"U_3d train={r2_u3d_train_ref:.2f}",
        fontsize=12,
    )
    plt.tight_layout()
    out = OUT_DIR / "auto_69.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")

    (OUT_DIR / "auto_69.json").write_text(json.dumps({
        "n_colors": int(N),
        "K_PC": K_PC,
        "evr_top_K_PC": evr_top,
        "n_centers_h": N_H,
        "n_centers_s": N_S,
        "n_centers_v": N_V,
        "n_centers_total": int(n_centers_total),
        "hsv_duchon_cv_r2_macro": r2_cv,
        "hsv_duchon_train_r2_macro": r2_train,
        "hsv_duchon_train_log_lambda": ll_full,
        "hsv_duchon_cv_mean_log_lambda": float(np.mean(ll_folds)),
        "hsv_duchon_cv_log_lambda_per_fold": [float(x) for x in ll_folds],
        "per_pc_cv_r2": [float(x) for x in r2_pc_cv],
        "reference_hue_only_cv_r2": r2_hue_only,
        "reference_knn_lab_cv_r2": r2_knn_lab,
        "reference_u3d_train_r2": r2_u3d_train_ref,
        "gap_to_u3d_train": r2_u3d_train_ref - r2_cv,
        "gap_to_hue_only": r2_cv - r2_hue_only,
    }, indent=2))
    print(f"[saved] {OUT_DIR / 'auto_69.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
