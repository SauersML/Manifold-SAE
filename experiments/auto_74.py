"""auto_74.py — Identifiable residual color axes via conditional ICA (iVAE-lite).

Goal: discover the 2–4 "extra" color axes in cogito's per-color centroid
manifold that are NOT predictable from RGB (HSV). Multiple intrinsic-dim
estimators (auto_73) put the manifold at ~5–7 dim. The supervised
hue+sv decomposition (auto_67) explains 3 of those (CV R^2 ≈ 0.32 on
Z_top64). After projecting away that ceiling, the residual carries 2–4
axes of per-color signal — candidates for "name-semantic" axes (cf.
auto_52: 'celadon' tight, 'red' diffuse).

Identifiability principle (Hyvärinen & Khemakhem, iVAE 2019/2020):
if z has an auxiliary observable u and p(z|u) factorizes with
sufficiently varying conditionals across u, then learned z is identifiable
up to per-component monotone+permutation. Here:
    aux u = RGB triple
    z     = residual latent (Z_top16 with hue+sv partialled out)
We use the simplest-viable identifiable factorization: FastICA on
Z_residual, then compute a modulation index MI_k = Var(E[s|RGB-bin])
/ E[Var(s|RGB-bin)] per component; nontrivial MI ⇒ identifiable factor.

NO Gaussian RBF, NO Duchon length_scale, NO B-splines.
"""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.linalg as sla
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import FastICA

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from plot_color_geometry import load_xkcd_colors
from color_filter_list import filter_colors
from _pca_basis import load_pc_basis, project

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
X_PATH = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
K_PC = 16            # focus on the color-signal envelope
HUE_CENTERS = 40
SV_GRID = 6
N_FOLDS = 5
RGB_BINS = 2         # 2x2x2 = 8 octants for MI
N_ICA = 6            # request up to 6 ICs; we'll inspect top-K by MI


# ---------------------------------------------------------------------------
# Bases (identical spec to auto_67)
# ---------------------------------------------------------------------------
def hue_basis(hue01, n_centers=HUE_CENTERS):
    import gamfit
    centers = np.linspace(0.0, 1.0, n_centers, endpoint=False).reshape(-1, 1)
    pts = np.asarray(hue01, dtype=np.float64).reshape(-1, 1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, periodic_per_axis=[True])
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, periodic_per_axis=[True])
    )
    return Phi, P


def sv_basis(sat, val, grid=SV_GRID):
    import gamfit
    g = np.linspace(0.0, 1.0, grid)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
    pts = np.stack([np.asarray(sat, float), np.asarray(val, float)], axis=1)
    Phi = np.asarray(
        gamfit.duchon_basis(pts, centers, m=2, nullspace_order="degree2")
    )
    P = np.asarray(
        gamfit.duchon_function_norm_penalty(centers, m=2, nullspace_order="degree2")
    )
    return Phi, P


def reml_fit(Phi, Y, P):
    import gamfit
    out = gamfit.gaussian_reml_fit(Phi, Y, P)
    return np.asarray(out["coefficients"]), float(out["lambda"])


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def twoNN(Z, top_frac=0.10):
    tree = cKDTree(Z)
    dists, _ = tree.query(Z, k=3)
    r1 = dists[:, 1]; r2 = dists[:, 2]
    mask = (r1 > 0) & (r2 > 0)
    mu = r2[mask] / r1[mask]
    mu_sorted = np.sort(mu)
    n = mu_sorted.size
    F = (np.arange(1, n + 1) - 0.5) / n
    x = np.log(mu_sorted); y = np.log(1.0 - F)
    cutoff = max(int(n * (1 - top_frac)), int(0.5 * n))
    x = x[:cutoff]; y = y[:cutoff]
    return -float((x * y).sum() / (x * x).sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load centroids + filter ---
    X = np.load(X_PATH, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    print(f"[load] X mmap shape={X.shape}, n_raw={n_raw}")

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    template_std = np.zeros(n_raw, dtype=np.float64)  # capture template-variability per color
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        block = np.asarray(X[rows], dtype=np.float64)
        centroids[ci] = block.mean(0)
        template_std[ci] = float(block.std(axis=0).mean())  # mean per-feature std
    del X

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    template_std = template_std[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    names = [n for n, *_ in kept]
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    N = len(kept)
    print(f"[load] N={N} filtered colors")

    # --- 2. Z = top-16 PCs ---
    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    print(f"[pca] Z shape={Z.shape}  EVR_top{K_PC}={float(basis['evr'].sum()):.3f}")

    # --- 3. RGB-explainable part: same spec as auto_67 ---
    Phi_h, P_h = hue_basis(hue)
    Phi_sv, P_sv = sv_basis(sat, val)
    Kh, Ksv = Phi_h.shape[1], Phi_sv.shape[1]
    Phi_joint = np.concatenate([Phi_h, Phi_sv], axis=1)
    P_joint = sla.block_diag(P_h, P_sv)

    # CV preds (for honest R^2 reporting)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    pred_hue = np.zeros_like(Z)
    pred_joint = np.zeros_like(Z)
    for f in range(N_FOLDS):
        tr = fold != f; te = ~tr
        Bh, _ = reml_fit(Phi_h[tr], Z[tr], P_h)
        pred_hue[te] = Phi_h[te] @ Bh
        Bj, _ = reml_fit(Phi_joint[tr], Z[tr], P_joint)
        pred_joint[te] = Phi_joint[te] @ Bj
    r2_hue = r2_macro(Z, pred_hue)
    r2_joint = r2_macro(Z, pred_joint)
    print(f"[ceiling] R^2 hue={r2_hue:+.4f}  joint(hue+sv)={r2_joint:+.4f}")

    # In-sample joint fit to define Z_residual (we want a clean partial-out,
    # not a held-out one, for downstream ICA)
    B_full, lam_full = reml_fit(Phi_joint, Z, P_joint)
    Z_pred_in = Phi_joint @ B_full
    Z_res = Z - Z_pred_in
    print(f"[residual] λ={lam_full:.3g}  ||Z_res||/||Z||="
          f"{np.linalg.norm(Z_res) / np.linalg.norm(Z):.3f}")

    # --- 4. Intrinsic dim of Z_residual ---
    d_res_twonn = twoNN(Z_res)
    print(f"[twoNN(Z_res)] d̂ = {d_res_twonn:.2f}")

    # --- 5. ICA on Z_residual ---
    n_ica = min(N_ICA, Z_res.shape[1])
    ica = FastICA(
        n_components=n_ica,
        whiten="unit-variance",
        random_state=0,
        max_iter=2000,
        tol=1e-5,
    )
    S = ica.fit_transform(Z_res)  # (N, n_ica)
    # Order ICs by variance fraction of reconstructed Z_res
    A = ica.mixing_  # (K_PC, n_ica)
    var_explained_per_ic = np.zeros(n_ica)
    Z_res_var = float((Z_res ** 2).sum())
    for k in range(n_ica):
        rec_k = np.outer(S[:, k], A[:, k])
        var_explained_per_ic[k] = float((rec_k ** 2).sum()) / Z_res_var
    order = np.argsort(-var_explained_per_ic)
    S = S[:, order]
    A = A[:, order]
    var_explained_per_ic = var_explained_per_ic[order]
    # Standardize each component (sign-fix: positive correlate w/ name-length tie-break)
    S = (S - S.mean(0)) / S.std(0)
    print(f"[ICA] n_components={n_ica}  var fractions = "
          f"{var_explained_per_ic.round(3).tolist()}")

    # --- 6. Modulation index per IC ---
    # Bin RGB into a 2x2x2 grid (8 octants) → aux observable u
    bins = np.minimum((rgb * RGB_BINS).astype(int), RGB_BINS - 1)
    bin_id = bins[:, 0] * RGB_BINS * RGB_BINS + bins[:, 1] * RGB_BINS + bins[:, 2]
    uniq = np.unique(bin_id)
    MI = np.zeros(n_ica)
    bin_means_per_ic = {}
    for k in range(n_ica):
        s = S[:, k]
        means = np.array([s[bin_id == u].mean() for u in uniq])
        within_var = np.array([s[bin_id == u].var() for u in uniq])
        # E[Var(s|u)] weighted by bin size
        wts = np.array([(bin_id == u).sum() for u in uniq], dtype=float)
        wts /= wts.sum()
        Es_var_u = float((within_var * wts).sum())
        Var_Es_u = float(((means - (means * wts).sum()) ** 2 * wts).sum())
        MI[k] = Var_Es_u / max(Es_var_u, 1e-12)
        bin_means_per_ic[k] = means
    print(f"[MI] MI_k = {MI.round(3).tolist()}")

    # --- 7. Characterize: spearman vs candidate predictors ---
    # Predictors:
    #   lightness (HSV val), saturation, name length chars, name length words,
    #   is_monoword (1 if single word), is_basic (red/orange/yellow/green/blue/
    #   purple/pink/brown/black/white/grey), template_std,
    #   residual_hardness (||Z_res_row||)
    BASIC = {"red", "orange", "yellow", "green", "blue", "purple", "pink",
             "brown", "black", "white", "grey", "gray"}
    name_chars = np.array([len(n) for n in names], dtype=float)
    name_words = np.array([len(n.split()) for n in names], dtype=float)
    is_mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names])
    is_basic = np.array(
        [1.0 if n.strip().lower() in BASIC else 0.0 for n in names]
    )
    res_hardness = np.linalg.norm(Z_res, axis=1)
    predictors = {
        "lightness": val,
        "saturation": sat,
        "name_chars": name_chars,
        "name_words": name_words,
        "is_monoword": is_mono,
        "is_basic": is_basic,
        "template_std": template_std,
        "residual_hardness": res_hardness,
    }
    pred_names = list(predictors.keys())
    SP = np.zeros((n_ica, len(pred_names)))
    SP_p = np.zeros_like(SP)
    for k in range(n_ica):
        for j, pname in enumerate(pred_names):
            rho, p = spearmanr(S[:, k], predictors[pname])
            SP[k, j] = rho
            SP_p[k, j] = p

    # Top correlate per IC
    top_correlate = []
    for k in range(n_ica):
        j_best = int(np.argmax(np.abs(SP[k])))
        top_correlate.append((pred_names[j_best], float(SP[k, j_best])))
    print("[top correlate per IC]")
    for k, (pname, rho) in enumerate(top_correlate):
        print(f"   IC{k}  MI={MI[k]:5.2f}  var={var_explained_per_ic[k]:5.3f}  "
              f"→ {pname}  ρ={rho:+.3f}")

    # --- 8. R^2 stacking: hue + sv + IC-components on Z_top16 ---
    # CV add identified IC components (use the top-3 by MI) as additive
    # predictors. To preserve identifiability of the residual loadings,
    # we use IN-SAMPLE S as features (since S is derived from full Z_res,
    # it's a deterministic function of Z; using it CV-wise means we re-fit
    # ICA per fold — too noisy with N≈886). Report both:
    #   (a) in-sample R^2 of [Phi_joint | S_topK] regression
    #   (b) CV R^2 where S_topK is computed in-sample on training fold's
    #       Z_res then projected onto test fold via the linear map
    #       Z_res_test @ pinv(A_train)
    K_use_ic = min(3, n_ica)
    ic_by_mi = np.argsort(-MI)[:K_use_ic]
    S_top = S[:, ic_by_mi]
    A_top = A[:, ic_by_mi]

    # (a) In-sample stacked R^2
    Phi_stack = np.concatenate([Phi_joint, S_top], axis=1)
    # No penalty on IC columns (they're already a low-dim id'd basis):
    P_stack = sla.block_diag(P_joint, np.zeros((K_use_ic, K_use_ic)))
    B_stack, lam_stack = reml_fit(Phi_stack, Z, P_stack)
    Z_pred_stack_in = Phi_stack @ B_stack
    r2_stack_in = r2_macro(Z, Z_pred_stack_in)

    # (b) CV stacked R^2 (re-fit ICA per fold on Z_res_train)
    pred_stack_cv = np.zeros_like(Z)
    for f in range(N_FOLDS):
        tr = fold != f; te = ~tr
        Bj, _ = reml_fit(Phi_joint[tr], Z[tr], P_joint)
        Z_res_tr = Z[tr] - Phi_joint[tr] @ Bj
        Z_res_te = Z[te] - Phi_joint[te] @ Bj
        ica_f = FastICA(n_components=n_ica, whiten="unit-variance",
                        random_state=0, max_iter=2000, tol=1e-5)
        S_tr = ica_f.fit_transform(Z_res_tr)
        A_f = ica_f.mixing_
        # MI on training fold to choose top-K_use_ic
        bin_id_tr = bin_id[tr]
        uniq_tr = np.unique(bin_id_tr)
        S_tr_std = (S_tr - S_tr.mean(0)) / S_tr.std(0)
        MI_f = np.zeros(n_ica)
        for k in range(n_ica):
            s = S_tr_std[:, k]
            means = np.array([s[bin_id_tr == u].mean() for u in uniq_tr])
            within_var = np.array([s[bin_id_tr == u].var() for u in uniq_tr])
            wts = np.array([(bin_id_tr == u).sum() for u in uniq_tr], float)
            wts /= wts.sum()
            Es_var_u = float((within_var * wts).sum())
            Var_Es_u = float(((means - (means * wts).sum()) ** 2 * wts).sum())
            MI_f[k] = Var_Es_u / max(Es_var_u, 1e-12)
        sel = np.argsort(-MI_f)[:K_use_ic]
        # Project test residuals onto IC basis via linear unmixing
        # FastICA: S = (X - mean) @ unmixing_.T  (since whiten='unit-variance')
        S_te = (Z_res_te - ica_f.mean_) @ ica_f.components_.T
        # Standardize using training fold stats
        s_mean = S_tr.mean(0); s_std = S_tr.std(0).clip(min=1e-12)
        S_tr_std = (S_tr - s_mean) / s_std
        S_te_std = (S_te - s_mean) / s_std
        S_tr_top = S_tr_std[:, sel]
        S_te_top = S_te_std[:, sel]

        Phi_stack_tr = np.concatenate([Phi_joint[tr], S_tr_top], axis=1)
        Phi_stack_te = np.concatenate([Phi_joint[te], S_te_top], axis=1)
        P_stack_f = sla.block_diag(P_joint, np.zeros((K_use_ic, K_use_ic)))
        Bs, _ = reml_fit(Phi_stack_tr, Z[tr], P_stack_f)
        pred_stack_cv[te] = Phi_stack_te @ Bs

    r2_stack_cv = r2_macro(Z, pred_stack_cv)
    print(f"[stacked R^2] in-sample={r2_stack_in:+.4f}   "
          f"CV (re-fit ICA per fold) = {r2_stack_cv:+.4f}")
    print(f"[reference]  U_3d CV ceiling (auto_67 family) = 0.608")

    # --- 9. PLOT ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.2, 1.0])

    # (1) variance-explained scree of ICs on Z_residual
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(n_ica), var_explained_per_ic, color="#1f77b4",
           edgecolor="black")
    ax.set_xticks(range(n_ica))
    ax.set_xticklabels([f"IC{k}" for k in range(n_ica)])
    ax.set_ylabel("var frac of Z_residual")
    ax.set_title(f"(1) ICA scree on Z_residual  (twoNN d̂={d_res_twonn:.2f})")
    for k, v in enumerate(var_explained_per_ic):
        ax.text(k, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (2) MI per IC
    ax = fig.add_subplot(gs[0, 1])
    colors_bar = ["#2ca02c" if k in ic_by_mi else "#888" for k in range(n_ica)]
    ax.bar(range(n_ica), MI, color=colors_bar, edgecolor="black")
    ax.set_xticks(range(n_ica))
    ax.set_xticklabels([f"IC{k}" for k in range(n_ica)])
    ax.set_ylabel("MI = Var(E[s|u]) / E[Var(s|u)]")
    ax.set_title(f"(2) Modulation index  (RGB-octant bins, "
                 f"u dim = {len(uniq)})\n"
                 f"green = top-{K_use_ic} chosen for stacking")
    for k, m in enumerate(MI):
        ax.text(k, m + 0.005 * max(MI), f"{m:.2f}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (3) R^2 stacking bar
    ax = fig.add_subplot(gs[0, 2])
    bar_names = ["hue γ\n(CV)", "hue+sv\n(CV)",
                 f"+ top-{K_use_ic} IC\n(CV refit)", "U_3d\nceiling"]
    bar_vals = [r2_hue, r2_joint, r2_stack_cv, 0.608]
    bar_colors = ["#d62728", "#1f77b4", "#2ca02c", "#888"]
    ax.bar(bar_names, bar_vals, color=bar_colors, edgecolor="black")
    for i, v in enumerate(bar_vals):
        ax.text(i, v + 0.01, f"{v:+.3f}", ha="center", fontsize=9)
    ax.axhline(0.608, color="green", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("CV macro R²  (Z_top16)")
    ax.set_title(f"(3) R² stacking  (in-sample stacked = {r2_stack_in:+.3f})")
    ax.grid(alpha=0.3, axis="y")

    # (4) Top/bottom-12 swatches for top-3 ICs (3 rows of 2 strips each)
    for row, k in enumerate(ic_by_mi):
        s = S[:, k]
        top_idx = np.argsort(-s)[:12]
        bot_idx = np.argsort(s)[:12]
        # top
        ax = fig.add_subplot(gs[1 + (row // 3), row % 3])
        # Actually layout: use gs[1, :] split into 2 subplots per IC stacked
        # Simpler: build a swatch panel manually below in gs[2, :]
        ax.axis("off")
    # Re-do swatch layout: one row gs[1, :] = swatch grid (3 ICs × 2 strips)
    for r in range(1, 2):
        for c in range(3):
            # Clear placeholders from above
            pass

    # Custom swatch layout: gs[2, :] becomes a 6-strip grid (3 ICs × 2)
    swatch_axes = []
    for col_i in range(3):
        ax_t = fig.add_subplot(gs[1, col_i])
        ax_b = fig.add_subplot(gs[2, col_i])
        swatch_axes.append((ax_t, ax_b))

    def draw_swatches(ax, idxs, scores, title):
        ax.set_xlim(0, len(idxs))
        ax.set_ylim(0, 1)
        for i, ii in enumerate(idxs):
            ax.add_patch(Rectangle((i, 0.35), 1, 0.65, color=tuple(rgb[ii])))
            ax.text(i + 0.5, 0.27, names[ii][:14], ha="center", va="top",
                    fontsize=6.5, rotation=40)
            ax.text(i + 0.5, 0.08, f"{scores[ii]:+.2f}", ha="center", va="top",
                    fontsize=6.5, color="black")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)

    for col_i, k in enumerate(ic_by_mi):
        s = S[:, k]
        top_idx = np.argsort(-s)[:12]
        bot_idx = np.argsort(s)[:12]
        ax_t, ax_b = swatch_axes[col_i]
        pname, rho = top_correlate[k]
        draw_swatches(
            ax_t, top_idx, s,
            f"IC{k}  TOP12  (MI={MI[k]:.2f}, var={var_explained_per_ic[k]:.2f})\n"
            f"top corr: {pname} ρ={rho:+.2f}",
        )
        draw_swatches(
            ax_b, bot_idx, s,
            f"IC{k}  BOTTOM12",
        )

    # (5) Spearman heatmap: ICs × predictors
    ax = fig.add_subplot(gs[3, :2])
    vmax = float(np.abs(SP).max())
    im = ax.imshow(SP, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n_ica))
    ax.set_yticklabels([f"IC{k}" for k in range(n_ica)])
    ax.set_xticks(range(len(pred_names)))
    ax.set_xticklabels(pred_names, rotation=35, ha="right")
    for k in range(n_ica):
        for j in range(len(pred_names)):
            star = "*" if SP_p[k, j] < 0.01 else ""
            ax.text(j, k, f"{SP[k, j]:+.2f}{star}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("(5) Spearman ρ: IC × candidate predictors  "
                 "(* = p < 0.01)")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    # (6) summary text
    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    txt = (
        f"N = {N} filtered xkcd colors\n"
        f"Z = top-{K_PC} cogito L40 PCs\n"
        f"  EVR = {float(basis['evr'].sum()):.3f}\n\n"
        f"ceiling (auto_67 spec, CV):\n"
        f"  hue γ      : {r2_hue:+.3f}\n"
        f"  hue+sv     : {r2_joint:+.3f}\n\n"
        f"residual Z_res = Z − γ(hue) − g(sat,val)\n"
        f"  ||res||/||Z||  = "
        f"{np.linalg.norm(Z_res) / np.linalg.norm(Z):.3f}\n"
        f"  TwoNN d̂      = {d_res_twonn:.2f}\n\n"
        f"FastICA on Z_res, n_components={n_ica}\n"
        f"  RGB-aux bins   = {RGB_BINS}³ = {RGB_BINS**3}\n\n"
        f"per-IC (MI, top correlate):\n" +
        "\n".join(
            f"  IC{k}  MI={MI[k]:4.2f}  → {top_correlate[k][0]:<16s} "
            f"ρ={top_correlate[k][1]:+.2f}"
            for k in range(n_ica)
        ) +
        f"\n\nstacked CV R² = {r2_stack_cv:+.3f}\n"
        f"  (vs U_3d ceiling 0.608)\n"
        f"in-sample stacked = {r2_stack_in:+.3f}\n"
    )
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=8,
            va="top", ha="left", transform=ax.transAxes)

    fig.suptitle(
        f"auto_74 · iVAE-lite identifiable residual color axes  ·  "
        f"FastICA on Z_top{K_PC} − γ(hue) − g(sat,val)  ·  cogito L40",
        fontsize=13,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "auto_74.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_png}")

    # --- 10. JSON payload ---
    payload = {
        "n_colors": int(N),
        "K_PC": K_PC,
        "evr_top_K_PC": float(basis["evr"].sum()),
        "hue_centers": HUE_CENTERS,
        "sv_grid": SV_GRID,
        "n_ica": int(n_ica),
        "rgb_bins": RGB_BINS,
        "ceiling_cv": {
            "hue": float(r2_hue),
            "hue_plus_sv": float(r2_joint),
        },
        "residual": {
            "frob_ratio": float(np.linalg.norm(Z_res) / np.linalg.norm(Z)),
            "twoNN_d_hat": float(d_res_twonn),
        },
        "ica": {
            "var_fraction_per_ic": var_explained_per_ic.tolist(),
            "modulation_index": MI.tolist(),
            "selected_top_K_by_MI": [int(i) for i in ic_by_mi],
        },
        "spearman_predictors": pred_names,
        "spearman_rho": SP.tolist(),
        "spearman_p": SP_p.tolist(),
        "top_correlate_per_ic": [
            {"ic": int(k), "predictor": top_correlate[k][0],
             "rho": float(top_correlate[k][1]),
             "MI": float(MI[k])} for k in range(n_ica)
        ],
        "stacked_r2": {
            "in_sample": float(r2_stack_in),
            "cv_refit_ica_per_fold": float(r2_stack_cv),
            "U_3d_ceiling_reference": 0.608,
        },
        "top12_per_top_ic": {
            f"IC{int(k)}": {
                "top12": [names[int(i)] for i in np.argsort(-S[:, k])[:12]],
                "bottom12": [names[int(i)] for i in np.argsort(S[:, k])[:12]],
            } for k in ic_by_mi
        },
        "notes": (
            "iVAE-lite: condition on aux observable u=RGB. Partial out RGB via "
            "the auto_67 supervised spec (additive periodic-Duchon hue + 2D "
            "degree2-Duchon sat/val), then FastICA on Z_residual. Identifiable "
            "components are those with nontrivial modulation index MI = "
            "Var(E[s|u]) / E[Var(s|u)] over RGB octants. CV stacked R^2 "
            "re-fits ICA on each training fold and projects test residuals "
            "via the learned unmixing matrix. No Gaussian RBF, no Duchon "
            "length_scale, no B-splines."
        ),
    }
    (OUT_DIR / "auto_74.json").write_text(json.dumps(payload, indent=2))
    print(f"[saved] {OUT_DIR / 'auto_74.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
