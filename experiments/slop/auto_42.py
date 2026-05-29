"""auto_42 [idea kkkkk]: alignment of L_lin_rgb's predicted hyperplane
normals with PCA directions per PC.

results.json says L_lin_rgb (linear in R,G,B) hits macro R² ~0.10 with
considerable spread across PCs. What it *is* doing geometrically:

  ẑ_j(c)  =  α_j·R + β_j·G + γ_j·B + b_j        for PC j

So in the 7168-D activation space the RGB-explainable direction for
PC j is

  d_j  =  Vt.T @ w_j      with  w_j = (α_j, β_j, γ_j)

But there is something stronger to ask. The *RGB-explainable subspace*
in the 7168-D activation space has only three independent directions
(one per RGB axis):

  u_R = Vt.T @ W[R, :]     u_G = Vt.T @ W[G, :]     u_B = Vt.T @ W[B, :]

  S_RGB = span(u_R, u_G, u_B)     dim ≤ 3

For each PC direction v_j = Vt[j] (a unit vector in 7168-D), measure

  align_j  =  ‖ Proj_{S_RGB}(v_j) ‖         (in [0,1])

That tells us: of the variance PC j captures, what fraction lies in
the *exact 3-D plane that L_lin_rgb can reach*. We then decompose
align_j² into per-axis contributions (after orthonormalising S_RGB)
to see whether PCs lean toward the R, G, or B coordinate of the
fitted hyperplane.

Constraints respected: no Gaussian RBF, no Duchon length_scale, only
linear ridge + PCA.

Procedure
---------
1. Reload centroid matrix as auto_41 does (same n_c=949, same Vt/μ/σ).
2. Project to PCs:  Z = Xn @ Vt.T,  shape (n_c, 64).
3. Fit ridge L_lin_rgb on the *full* data (no CV needed: this is a
   structural diagnostic, not a held-out R²):
       Phi = [R, G, B, 1] ;   W (4 × 64)
4. Take W_xyz = W[:3]  (3 × 64).  Lift to activation space:
       U = Vt.T @ W_xyz.T              # (D, 3)
5. Orthonormalise U via QR:  U = Q R~  (Q has orthonormal columns).
6. For each PC j: g_j = Q.T @ Vt[j]  (3-vector); align_j = ‖g_j‖.
   Per-axis "preference" of PC j toward (R, G, B) basis directions
   in the lifted space is the renormalised |R~ @ g_j| (since the
   original axes are R~^{-1} times Q-coords).

7. Plot four panels:
     (a) bar plot  align_j  per PC (top 32 PCs); annotate macro
         alignment and held-out R² from results.json.
     (b) scatter  align_j   vs  per-PC R² (held-out)  — should be
         strongly correlated; we report Spearman ρ.
     (c) stacked bars: per-axis (R/G/B) contribution fractions for
         the top-12 PCs (most aligned).
     (d) cumulative explained variance × alignment²:
              C_k = Σ_{j≤k} EVR_j · align_j²
         which is the *amount of total activation variance that lies
         in the RGB-explainable 3-D subspace, accumulated through
         the top-k PCs*. Also plot Σ_{j≤k} EVR_j for reference.

This is a structural counterpart to per-PC R²: R² says "did ridge
learn it?", alignment says "could it ever?".
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from plot_color_geometry import load_xkcd_colors, load_harvest  # noqa: E402
from color_filter_list import filter_colors  # noqa: E402

N_T = 28
RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN / "auto_42.png"
RESULTS = RUN / "results.json"
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
TOP_PCS_BAR = 32
TOP_PCS_STACK = 12


def ridge_fit(Phi, Z, lam=1e-4):
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    B = Phi.T @ Z
    return np.linalg.solve(A, B)


def main():
    with open(RESULTS) as f:
        res = json.load(f)
    L = res["per_layer"]["L40"]
    Vt = np.asarray(L["Vt_topK"], dtype=np.float64)     # (K, D)
    mu = np.asarray(L["mu"], dtype=np.float64)
    sigma = np.asarray(L["sigma"], dtype=np.float64)
    evr = np.asarray(L["explained_variance_ratio_topK"], dtype=np.float64)
    r2_per_pc = np.asarray(
        L["specs"]["L_lin_rgb"]["r2_per_pc_mean"], dtype=np.float64)
    r2_macro = L["specs"]["L_lin_rgb"]["r2_macro_mean"]
    K, D = Vt.shape

    X_full = load_harvest(HARVEST)
    n_raw = X_full.shape[0] // N_T
    X_full = X_full[: n_raw * N_T]
    centroids_all = X_full.reshape(n_raw, N_T, -1).mean(1)
    colors_all = load_xkcd_colors()[:n_raw]
    _, kept = filter_colors(colors_all)
    centroids = centroids_all[kept]
    colors = [colors_all[i] for i in kept]
    n_c = len(colors)
    assert centroids.shape[1] == D

    Xn = (centroids - mu[None, :]) / np.maximum(sigma[None, :], 1e-8)
    Z = Xn @ Vt.T                                      # (n_c, K)
    rgb = np.array([(r, g, b) for _, r, g, b in colors],
                   dtype=np.float64) / 255.0           # in [0,1]
    print(f"[auto_42] n_c={n_c}  K={K}  D={D}", flush=True)

    # --- fit L_lin_rgb on full data ---
    Phi = np.concatenate([rgb, np.ones((n_c, 1))], 1)  # (n_c, 4)
    W = ridge_fit(Phi, Z, lam=1e-4)                    # (4, K)
    W_xyz = W[:3]                                      # (3, K)

    # --- lift to activation space; orthonormalise ---
    U = Vt.T @ W_xyz.T                                 # (D, 3)
    Q, R_tri = np.linalg.qr(U, mode="reduced")         # Q (D,3), R_tri (3,3)
    # PC vectors are *rows* of Vt; each is unit-norm because PCA basis.
    # Coordinates of v_j in Q basis:
    G = Vt @ Q                                         # (K, 3)
    align = np.linalg.norm(G, axis=1)                  # (K,)
    align = np.clip(align, 0.0, 1.0)

    # Per-axis (R, G, B) preference for each PC.
    # Recover RGB-axis directions in Q-coords: u_axis = Q R_tri[:,axis].
    # So G @ R_tri  gives, for each PC j, the *dot products* with the
    # un-normalised RGB-direction vectors.  Take absolute values and
    # renormalise across the 3 axes so each PC sums to 1 (only meaningful
    # for PCs with non-trivial alignment).
    axis_dots = np.abs(G @ R_tri)                      # (K, 3)
    axis_norms = np.linalg.norm(R_tri, axis=0)         # (3,) ≈ ‖u_axis‖
    axis_pref = axis_dots / (axis_norms[None, :] + 1e-12)
    row_sum = axis_pref.sum(1, keepdims=True)
    axis_pref_norm = axis_pref / (row_sum + 1e-12)

    print(f"[auto_42] sing.values(U) = {np.linalg.svd(U, compute_uv=False)}",
          flush=True)
    print(f"[auto_42] mean alignment (top-32 PCs) = "
          f"{align[:32].mean():.3f}", flush=True)
    print(f"[auto_42] EVR-weighted alignment²    = "
          f"{(evr * align ** 2).sum():.4f}", flush=True)
    rho, p_rho = spearmanr(align, r2_per_pc)
    print(f"[auto_42] Spearman(align, r2_per_pc) = "
          f"{rho:+.3f}  (p={p_rho:.2e})", flush=True)

    # --- plot ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # (a) bar plot align_j top-32
    ax = fig.add_subplot(gs[0, 0])
    idx = np.arange(TOP_PCS_BAR)
    bars = ax.bar(idx, align[:TOP_PCS_BAR],
                  color=plt.cm.viridis(align[:TOP_PCS_BAR]),
                  edgecolor="black", lw=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5, label="ceiling = 1.0")
    ax.set_xlabel("PC index")
    ax.set_ylabel(r"alignment $\|\mathrm{Proj}_{S_{RGB}}\,v_j\|$")
    ax.set_title(f"(a) per-PC alignment with the RGB-explainable 3-D subspace\n"
                 f"top-{TOP_PCS_BAR} PCs · mean align = "
                 f"{align[:TOP_PCS_BAR].mean():.3f}",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # (b) scatter align vs per-PC R²
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(align, r2_per_pc,
                    c=evr, s=24 + 600 * evr, cmap="plasma",
                    edgecolor="black", lw=0.3)
    for j in np.argsort(-evr)[:8]:
        ax.annotate(f"PC{j}", (align[j], r2_per_pc[j]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"alignment$_j$")
    ax.set_ylabel(r"per-PC held-out $R^2$ (from results.json)")
    ax.set_title(f"(b) alignment vs per-PC $R^2$  ·  "
                 f"Spearman ρ = {rho:+.3f}",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("EVR$_j$ (size also)", fontsize=8)

    # (c) stacked bars: per-axis preference for top-12 most-aligned PCs
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(-align)[:TOP_PCS_STACK]
    x = np.arange(TOP_PCS_STACK)
    bottom = np.zeros(TOP_PCS_STACK)
    axis_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    axis_names = ["R", "G", "B"]
    for a in range(3):
        # raw axis_pref (not normalised) scaled by align² so heights sum
        # to align² (= variance fraction in the subspace).
        # We display normalised within-subspace proportions, scaled by align.
        h = axis_pref_norm[order, a] * align[order]
        ax.bar(x, h, bottom=bottom,
               color=axis_colors[a], edgecolor="black", lw=0.4,
               label=axis_names[a])
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{j}" for j in order], rotation=45, fontsize=8)
    ax.set_ylabel("alignment × axis fraction")
    ax.set_title(f"(c) within-subspace RGB-axis preference\n"
                 f"top-{TOP_PCS_STACK} most-aligned PCs",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, title="RGB axis", loc="upper right")

    # (d) cumulative EVR × alignment²
    ax = fig.add_subplot(gs[1, 1])
    cum_align_var = np.cumsum(evr * align ** 2)
    cum_evr = np.cumsum(evr)
    k_axis = np.arange(1, K + 1)
    ax.plot(k_axis, cum_evr, "-", color="gray", lw=1.6,
            label=r"$\sum_{j\leq k}$ EVR$_j$  (total)")
    ax.plot(k_axis, cum_align_var, "-o", color="#1f77b4", ms=3, lw=1.4,
            label=r"$\sum_{j\leq k}$ EVR$_j \cdot$ align$_j^2$  (in $S_{RGB}$)")
    ax.fill_between(k_axis, 0, cum_align_var, alpha=0.15, color="#1f77b4")
    ax.axhline(cum_align_var[-1], color="#1f77b4", ls=":", lw=0.7)
    ax.text(K, cum_align_var[-1], f"  {cum_align_var[-1]:.3f}",
            color="#1f77b4", va="center", fontsize=9)
    ax.set_xlabel("k  (number of PCs)")
    ax.set_ylabel("cumulative explained variance fraction")
    ax.set_title("(d) cumulative variance in the RGB-reachable subspace",
                 fontsize=10)
    ax.set_xlim(1, K)
    ax.set_ylim(0, max(cum_evr[-1], 1.0) * 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"auto_42 · alignment of L_lin_rgb hyperplane normals with PCA\n"
        f"cogito L40 · n_c={n_c} · K={K} PCs · "
        f"L_lin_rgb macro $R^2$ = {r2_macro:+.3f} "
        f"(top-{TOP_PCS_BAR} mean align = {align[:TOP_PCS_BAR].mean():.3f}, "
        f"EVR-weighted align² = {(evr*align**2).sum():.3f})  [idea kkkkk]",
        fontsize=12, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
