"""auto_28: residual that U_3d cannot capture.

Idea (uuu, fresh — not covered by auto_01..27 or auto_exp_04..08):

The d=3 unsupervised manifold U_3d gives per-color latent coordinates
T ∈ R^(949, 3) but the saved results don't include a Z-space reconstruction
of those latents — only Spearman correlations to known axes. We ask:

  *What direction in Z space does U_3d fail to capture?*

Procedure:
  1. Reconstruct per-color centroids Z ∈ R^(949, 64) from raw X_L40 + the
     saved (mu, sigma, Vt_topK) — verified to match the GAM run.
  2. Smooth-fit Z ~ f(T) with a thin-plate / radial-basis ridge using a
     small lattice of inducing centers (the same family the GAM uses), get
     Z_hat. Choose ridge α by 5-fold CV macro R² on Z (EVR-weighted).
  3. Residual R = Z - Z_hat (949, 64). Macro R² = 1 - var(R)/var(Z).
  4. SVD(R) — its top 3 singular vectors are the orthogonal directions in
     centroid space that *no smooth function of U_3d* can explain. Project
     residuals back onto these vectors → 3 scalar per-color residual axes.
  5. Plot:
     (a) per-PC EVR captured by U_3d (bars: EVR explained vs EVR residual),
     (b) cumulative EVR captured vs intrinsic dim of residual SVD,
     (c)-(e) scatter of each color's true RGB plotted in (resid_1, resid_2),
         (resid_2, resid_3), (resid_1, resid_3), with marker color = true
         sRGB. Clusters in residual space ⇒ a coherent perceptual direction
         that U_3d misses.

Output: PNG + JSON with macro R², per-PC R², residual singular values,
and the top-3 residual axes' Spearman correlations against the 7 known
color axes (R,G,B,hue,sat,value,luminance).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST_X = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN / "results.json"
OUT_PNG = RUN / "auto_28.png"
OUT_JSON = RUN / "auto_28.json"

N_TEMPLATES = 28
LATTICE_PER_SIDE = 6              # 6^3 = 216 centers — matches GAM's grid scale
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
N_FOLDS = 5
RNG_SEED = 0


# ---------------------------------------------------------------- helpers
def _build_centers(d: int, per_side: int) -> np.ndarray:
    """Uniform lattice on [0,1]^d (T was rescaled to unit cube)."""
    g = np.linspace(0.0, 1.0, per_side)
    grids = np.meshgrid(*([g] * d), indexing="ij")
    return np.stack([G.ravel() for G in grids], axis=1)


def _gaussian_features(T: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """RBF feature map (N, M); add constant column for intercept."""
    # (N, 1, d) - (1, M, d) -> (N, M)
    D2 = ((T[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    Phi = np.exp(-D2 / (2.0 * sigma * sigma))
    return np.concatenate([np.ones((T.shape[0], 1)), Phi], axis=1)


def _ridge_predict(Phi_tr, Y_tr, Phi_te, alpha):
    """Multi-output ridge: B = (Phi^T Phi + αI)^-1 Phi^T Y."""
    M = Phi_tr.shape[1]
    A = Phi_tr.T @ Phi_tr
    A.flat[:: M + 1] += alpha
    B = np.linalg.solve(A, Phi_tr.T @ Y_tr)
    return Phi_te @ B, B


def _macro_r2(Y, Y_hat, evr):
    """EVR-weighted macro R²."""
    ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum(0)
    ss_res = ((Y - Y_hat) ** 2).sum(0)
    per_pc = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return float((per_pc * evr).sum() / evr.sum()), per_pc


def main() -> None:
    print("[load] results.json")
    d_res = json.loads(RESULTS.read_text())
    L = d_res["per_layer"]["L40"]
    mu = np.array(L["mu"], dtype=np.float64)              # (7168,)
    sigma = np.array(L["sigma"], dtype=np.float64)        # (7168,)
    Vt = np.array(L["Vt_topK"], dtype=np.float64)         # (64, 7168)
    evr = np.array(L["explained_variance_ratio_topK"], dtype=np.float64)  # (64,)
    T = np.array(L["unsupervised_full_data"]["d=3"]["T"], dtype=np.float64)  # (949, 3)

    print("[load] X_L40")
    X = np.load(HARVEST_X)                                # (26572, 7168)
    n_c = X.shape[0] // N_TEMPLATES
    assert n_c == T.shape[0], f"n_c={n_c} != T rows {T.shape[0]}"

    print("[recon] per-color centroids → Z")
    per_color = X.reshape(n_c, N_TEMPLATES, -1).mean(axis=1).astype(np.float64)
    Xn = (per_color - mu) / sigma
    Xc = Xn - Xn.mean(0, keepdims=True)
    Z = Xc @ Vt.T                                         # (949, 64)

    # ----- choose RBF sigma & ridge α by 5-fold CV macro R² -----
    centers = _build_centers(3, LATTICE_PER_SIDE)         # (216, 3)
    # Median pairwise distance among centers as a sane bandwidth scale.
    dc = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    sigma_rbf = float(np.median(dc[dc > 0])) * 0.7
    print(f"[rbf]   centers={centers.shape[0]}  sigma_rbf={sigma_rbf:.3f}")

    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(n_c)
    fold_id = np.zeros(n_c, dtype=int)
    for k, idx in enumerate(np.array_split(perm, N_FOLDS)):
        fold_id[idx] = k

    Phi = _gaussian_features(T, centers, sigma_rbf)       # (949, 217)

    best_alpha, best_macro = None, -np.inf
    for alpha in RIDGE_ALPHAS:
        Z_hat = np.zeros_like(Z)
        for k in range(N_FOLDS):
            te = (fold_id == k); tr = ~te
            yhat, _ = _ridge_predict(Phi[tr], Z[tr], Phi[te], alpha)
            Z_hat[te] = yhat
        macro, _ = _macro_r2(Z, Z_hat, evr)
        print(f"  alpha={alpha:>7.3g}  cv_macro_r2={macro:+.4f}")
        if macro > best_macro:
            best_macro, best_alpha = macro, alpha
    print(f"[cv]    best_alpha={best_alpha}  cv_macro_r2={best_macro:+.4f}")

    # ----- full-data fit at best α (this defines U_3d's reach into Z) -----
    Z_hat_full, _ = _ridge_predict(Phi, Z, Phi, best_alpha)
    macro_full, per_pc_full = _macro_r2(Z, Z_hat_full, evr)
    R = Z - Z_hat_full                                    # (949, 64) residual
    print(f"[full]  in-sample macro_r2={macro_full:+.4f}  "
          f"residual_var_frac={(R.var(0).sum()/Z.var(0).sum()):.4f}")

    # ----- SVD of residual: top-3 directions U_3d misses -----
    Rc = R - R.mean(0, keepdims=True)
    U_r, S_r, Vt_r = np.linalg.svd(Rc, full_matrices=False)
    # Per-PC variance fractions of residual vs original Z
    resid_var_per_pc = R.var(0)
    z_var_per_pc = Z.var(0)
    frac_resid_per_pc = resid_var_per_pc / np.maximum(z_var_per_pc, 1e-12)

    # Project residual onto top-3 residual axes  (≡ U_r[:, :3] * S_r[:3])
    R_proj = U_r[:, :3] * S_r[:3]                          # (949, 3)

    # Cumulative variance captured by k-dim residual subspace
    resid_total = (S_r ** 2).sum()
    cum_frac = np.cumsum(S_r ** 2) / resid_total

    # Spearman of each residual axis vs each known color axis
    color_axes = d_res["color_axes_per_color_index"]
    axis_names = list(color_axes.keys())
    sp_table = {}
    for k in range(3):
        sp_table[f"resid_{k+1}"] = {
            ax: float(spearmanr(R_proj[:, k], color_axes[ax]).statistic)
            for ax in axis_names
        }

    # True sRGB for each color (clip to [0,1] for plotting)
    rgb = np.stack(
        [np.array(color_axes["R"]), np.array(color_axes["G"]),
         np.array(color_axes["B"])], axis=1
    )
    rgb = np.clip(rgb, 0.0, 1.0)

    # --------------------------------------------------------------- figure
    fig = plt.figure(figsize=(17.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # (a) per-PC EVR: captured by U_3d vs left in residual
    ax = fig.add_subplot(gs[0, 0])
    k_show = 32
    xs = np.arange(k_show)
    cap = np.clip(per_pc_full[:k_show], 0, 1) * evr[:k_show]
    res = (1.0 - np.clip(per_pc_full[:k_show], 0, 1)) * evr[:k_show]
    ax.bar(xs, cap, color="#3b8bba", label="EVR explained by U_3d")
    ax.bar(xs, res, bottom=cap, color="#d65f5f", label="EVR residual")
    ax.set_xlabel("PC index"); ax.set_ylabel("EVR fraction")
    ax.set_title(
        f"(a) per-PC EVR captured vs residual\n"
        f"macro R² = {macro_full:+.3f}  (CV: {best_macro:+.3f}, α={best_alpha})",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    # (b) cumulative residual variance captured by residual-SVD subspace
    ax = fig.add_subplot(gs[0, 1])
    k_show2 = min(40, len(cum_frac))
    ax.plot(np.arange(1, k_show2 + 1), cum_frac[:k_show2], "o-",
            color="#444", ms=4)
    for k_mark, c in [(1, "#d65f5f"), (3, "#cc8800"), (10, "#3b8bba")]:
        ax.axvline(k_mark, color=c, ls=":", alpha=0.7,
                   label=f"k={k_mark}: {cum_frac[k_mark-1]*100:.1f}% of resid")
    ax.set_xlabel("residual-SVD rank k")
    ax.set_ylabel("cum. fraction of residual variance")
    ax.set_title("(b) intrinsic dim of what U_3d misses", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) spearman table of top-3 residual axes vs known color axes
    ax = fig.add_subplot(gs[0, 2])
    mat = np.array([[sp_table[f"resid_{k+1}"][a] for a in axis_names]
                    for k in range(3)])
    vlim = float(np.abs(mat).max())
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(axis_names)))
    ax.set_xticklabels(axis_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(["resid_1", "resid_2", "resid_3"])
    for i in range(3):
        for j, a in enumerate(axis_names):
            ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(mat[i, j]) > 0.5 * vlim else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ")
    ax.set_title("(c) what known axes do residual axes track?", fontsize=10)

    # (d-f) scatter projections of residual coords, marker = true sRGB
    pairs = [(0, 1, "(d) resid_1 vs resid_2"),
             (1, 2, "(e) resid_2 vs resid_3"),
             (0, 2, "(f) resid_1 vs resid_3")]
    for i, (a, b, ttl) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, i])
        ax.scatter(R_proj[:, a], R_proj[:, b], c=rgb, s=14,
                   edgecolors="black", linewidths=0.15)
        ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
        ax.axvline(0, color="grey", lw=0.6, alpha=0.5)
        ax.set_xlabel(f"residual axis {a+1}")
        ax.set_ylabel(f"residual axis {b+1}")
        ax.set_title(ttl + "  (marker = true sRGB)", fontsize=10)

    fig.suptitle(
        "auto_28: what U_3d cannot reach in centroid space (residual SVD)",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[save] {OUT_PNG}")

    OUT_JSON.write_text(json.dumps({
        "n_colors": int(n_c),
        "n_templates": N_TEMPLATES,
        "n_pcs": int(Z.shape[1]),
        "rbf": {"centers": int(centers.shape[0]),
                "sigma_rbf": sigma_rbf,
                "alpha_grid": RIDGE_ALPHAS,
                "best_alpha": float(best_alpha),
                "cv_macro_r2": best_macro,
                "full_macro_r2": macro_full},
        "per_pc_r2_u3d": per_pc_full.tolist(),
        "frac_residual_var_per_pc": frac_resid_per_pc.tolist(),
        "residual_singular_values": S_r.tolist(),
        "cum_residual_var_fraction": cum_frac.tolist(),
        "spearman_residual_axes_vs_known": sp_table,
        "note": (
            "U_3d fitted with a 6³=216-center Gaussian-RBF ridge on T∈[0,1]^3, "
            "ridge α chosen by 5-fold CV on EVR-weighted macro R². Residual "
            "axes are the leading SVD directions of (Z - Z_hat). Strong "
            "Spearman ρ ⇒ U_3d systematically misses that known color axis."
        ),
    }, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
