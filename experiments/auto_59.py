"""
auto_59 — (hhhhhhh) Residual covariance matrix of the top supervised spec.

Question
--------
The headline supervised spec (L_joint_rgb_with_hue) achieves R^2 macro mean
~0.24 across the top-64 PCs of L40 per-color centroids.  What is the
*structure* of the 64-dim residual that ridge-on-RGB+hue cannot explain?
Is the leftover error white noise across PCs, or does it lie in a
low-rank, structured subspace (suggesting a missing latent axis)?

We:
  1. Reshape X_L40 (26572 x 7168) -> (949 colors x 28 templates x 7168)
     and take per-color centroids.
  2. Project centroids onto the top-64 PCs from results.json (matching the
     spec's PCA target).
  3. Fit the best supervised spec L_joint_rgb_with_hue (features:
     [1, R, G, B, R*G, R*B, G*B, R*G*B, sin(2*pi*H), cos(2*pi*H)]) per-PC
     via 5-fold OOF Ridge.  These are all allow-listed primitives —
     linear/ridge only.  No Gaussian RBF, no Duchon.
  4. Compute the residual covariance C_res (64x64), residual correlation
     matrix, and compare to the raw target covariance C_tgt.
  5. Eigendecompose C_res: how concentrated is the leftover variance?
     What is the effective rank?  How does the top residual eigenvector
     project onto RGB-derived directions in PC space?

Outputs
-------
PNG  : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_59.png  (6 panels)
JSON : runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_59.json

Allowed primitives only: PCA, linear/ridge, k-fold OOF.  No Gaussian RBF.
No Duchon length_scale.
"""
from __future__ import annotations

import json
import colorsys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
DATA_X = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"
OUT_PNG = RUN_DIR / "auto_59.png"
OUT_JSON = RUN_DIR / "auto_59.json"

N_FOLDS = 5
RIDGE_ALPHA = 1.0
SPEC = "L_joint_rgb_with_hue"


def joint_rgb_hue_features(rgb: np.ndarray) -> np.ndarray:
    """[1, R, G, B, R*G, R*B, G*B, R*G*B, sin(2*pi*H), cos(2*pi*H)]"""
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    H = np.empty(len(rgb))
    for i, (r, g, b) in enumerate(rgb):
        H[i], _, _ = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    cols = [
        np.ones_like(R), R, G, B,
        R * G, R * B, G * B, R * G * B,
        np.sin(2 * np.pi * H), np.cos(2 * np.pi * H),
    ]
    return np.column_stack(cols)


def oof_ridge_per_pc(F: np.ndarray, Y: np.ndarray,
                     n_folds: int, alpha: float, seed: int = 0) -> np.ndarray:
    """Return out-of-fold predictions of shape Y.shape.

    Standardizes features (zero-mean, unit-std) per fold and fits Ridge with
    intercept so the constant 1 column is harmless / redundant but ridge alpha
    behaves sensibly.
    """
    n = F.shape[0]
    pred = np.zeros_like(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        mu_f = F[tr].mean(axis=0)
        sd_f = F[tr].std(axis=0)
        sd_f = np.where(sd_f < 1e-12, 1.0, sd_f)
        Ftr = (F[tr] - mu_f) / sd_f
        Fte = (F[te] - mu_f) / sd_f
        # RidgeCV per fold picks alpha via inner LOO-GCV (allow-listed
        # ridge primitive).  We pass the input alpha as a sensible mid-grid
        # anchor and add a wide grid around it.
        alphas = np.logspace(-3, 3, 13)
        model = RidgeCV(alphas=alphas, fit_intercept=True)
        model.fit(Ftr, Y[tr])
        pred[te] = model.predict(Fte)
    return pred


def r2_per_col(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    ss_res = ((Y - Yhat) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def r2_pooled(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """Match the GAM script: 1 - sum(ss_res)/sum(ss_tot) across all elements."""
    ss_res = float(((Y - Yhat) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def effective_rank(eigvals: np.ndarray) -> float:
    """exp of spectral entropy (participation ratio of eigenvalues)."""
    p = np.clip(eigvals, 1e-12, None)
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> None:
    print(f"[load] {RESULTS}", flush=True)
    res = json.load(RESULTS.open())
    pl = res["per_layer"]["L40"]
    templates = res["templates"]
    n_t = len(templates)
    Vt = np.asarray(pl["Vt_topK"], dtype=np.float64)        # (K, D)
    mu = np.asarray(pl["mu"], dtype=np.float64)              # (D,)
    sigma = np.asarray(pl["sigma"], dtype=np.float64)        # (D,)
    sigma = np.where(sigma < 1e-6, 1e-6, sigma)
    K = Vt.shape[0]
    print(f"[pca] K={K}  D={Vt.shape[1]}")

    Rax = np.asarray(res["color_axes_per_color_index"]["R"], dtype=np.float64)
    Gax = np.asarray(res["color_axes_per_color_index"]["G"], dtype=np.float64)
    Bax = np.asarray(res["color_axes_per_color_index"]["B"], dtype=np.float64)
    rgb = np.column_stack([Rax, Gax, Bax])
    n_c = rgb.shape[0]
    print(f"[colors] n_colors={n_c}")

    print(f"[load] {DATA_X}", flush=True)
    X = np.load(DATA_X, mmap_mode="r")
    assert X.shape[0] == n_c * n_t, (X.shape, n_c, n_t)
    # per-color centroid in raw 7168-D space
    print("[centroid] computing per-color mean over templates...")
    X_ct = np.array(X[: n_c * n_t], dtype=np.float32).reshape(n_c, n_t, -1)
    C = X_ct.mean(axis=1).astype(np.float64)                 # (C, D)
    # Reproduce the GAM pipeline normalization: per-dim z-score, then PCA.
    Xn = (C - mu) / sigma
    Xn = Xn - Xn.mean(axis=0, keepdims=True)
    Y = Xn @ Vt.T                                            # (C, K)
    print(f"[project] Y shape={Y.shape}  per-PC var range "
          f"[{Y.var(0).min():.3g}, {Y.var(0).max():.3g}]")

    F = joint_rgb_hue_features(rgb)
    print(f"[features] {SPEC} dim={F.shape[1]}")

    Yhat = oof_ridge_per_pc(F, Y, N_FOLDS, RIDGE_ALPHA, seed=0)
    r2 = r2_per_col(Y, Yhat)
    r2_macro_uniform = float(np.mean(r2))
    r2_macro_pooled = r2_pooled(Y, Yhat)
    print(f"[fit] OOF R^2 (unweighted per-PC mean)={r2_macro_uniform:.4f}  "
          f"R^2 (pooled SS / variance-weighted)={r2_macro_pooled:.4f}  "
          f"(json reports {pl['specs'][SPEC]['r2_macro_mean']:.4f})")
    r2_macro = r2_macro_pooled

    residual = Y - Yhat                                       # (C, K)
    print(f"[resid] shape={residual.shape}  ||resid||_F={np.linalg.norm(residual):.2f}")

    # Covariance matrices (K x K)
    C_tgt = np.cov(Y, rowvar=False)
    C_res = np.cov(residual, rowvar=False)
    # Correlation form (easier to read structure)
    d = np.sqrt(np.diag(C_res))
    R_corr = C_res / np.outer(np.maximum(d, 1e-12), np.maximum(d, 1e-12))
    np.fill_diagonal(R_corr, 1.0)

    # Spectra
    eig_tgt = np.linalg.eigvalsh(C_tgt)[::-1]
    eig_res, eigvec_res = np.linalg.eigh(C_res)
    eig_res = eig_res[::-1]
    eigvec_res = eigvec_res[:, ::-1]                          # columns sorted desc
    eff_rank_res = effective_rank(eig_res)
    eff_rank_tgt = effective_rank(eig_tgt)
    print(f"[spectrum] eff_rank residual={eff_rank_res:.2f}  "
          f"target={eff_rank_tgt:.2f}  (K={K})")

    # Off-diagonal magnitude of residual correlation
    offdiag = R_corr[np.triu_indices(K, k=1)]
    print(f"[corr] residual off-diag |r|: mean={np.mean(np.abs(offdiag)):.3f}  "
          f"max={np.max(np.abs(offdiag)):.3f}  "
          f"frac>|0.2|={(np.abs(offdiag) > 0.2).mean():.3f}")

    # Project top-3 residual eigenvectors back onto the colors:
    # scores[i, j] = residual[i] @ eigvec_res[:, j]
    scores = residual @ eigvec_res[:, :3]                     # (C, 3)
    # And see how those scores correlate with RGB / hue / sat / val
    H = np.array([colorsys.rgb_to_hsv(*c)[0] for c in rgb])
    S = np.array([colorsys.rgb_to_hsv(*c)[1] for c in rgb])
    V = np.array([colorsys.rgb_to_hsv(*c)[2] for c in rgb])
    cands = {
        "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
        "sin(2piH)": np.sin(2 * np.pi * H),
        "cos(2piH)": np.cos(2 * np.pi * H),
        "S": S, "V": V, "L=mean(RGB)": rgb.mean(1),
    }
    print("[interp] |corr| of top-3 residual eigen-scores with RGB/HSV:")
    interp_table = {}
    for j in range(3):
        row = {k: float(np.corrcoef(v, scores[:, j])[0, 1]) for k, v in cands.items()}
        interp_table[f"resid_eig{j+1}"] = row
        best = max(row.items(), key=lambda kv: abs(kv[1]))
        print(f"  eig{j+1}: var={eig_res[j]:.3g}  "
              f"best |corr| -> {best[0]}={best[1]:+.3f}")

    # ----- plot -----------------------------------------------------------
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.05, right=0.98, top=0.91, bottom=0.07)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # (A) target covariance (log abs)
    im = axA.imshow(np.abs(C_tgt) + 1e-9, cmap="magma",
                    norm=LogNorm(vmin=1e-3, vmax=np.abs(C_tgt).max()))
    axA.set_title(f"|target covariance|  (PCA top-{K}, log scale)\n"
                  f"eff. rank = {eff_rank_tgt:.1f}")
    axA.set_xlabel("PC index"); axA.set_ylabel("PC index")
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)

    # (B) residual covariance (log abs)
    im = axB.imshow(np.abs(C_res) + 1e-9, cmap="magma",
                    norm=LogNorm(vmin=1e-3, vmax=max(np.abs(C_res).max(), 1e-2)))
    axB.set_title(f"|residual covariance|  after {SPEC}\n"
                  f"eff. rank = {eff_rank_res:.1f}")
    axB.set_xlabel("PC index"); axB.set_ylabel("PC index")
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.03)

    # (C) residual correlation (signed)
    vmax = max(0.05, float(np.max(np.abs(R_corr - np.eye(K)))))
    im = axC.imshow(R_corr, cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    axC.set_title(f"Residual correlation (signed)\n"
                  f"mean |off-diag|={np.mean(np.abs(offdiag)):.3f}, "
                  f"max={np.max(np.abs(offdiag)):.3f}")
    axC.set_xlabel("PC index"); axC.set_ylabel("PC index")
    fig.colorbar(im, ax=axC, fraction=0.046, pad=0.03)

    # (D) eigenvalue spectrum: target vs residual (log y, raw + cumulative)
    axD.semilogy(np.arange(1, K + 1), np.clip(eig_tgt, 1e-12, None),
                 "o-", color="#1f77b4", lw=1.5, ms=3, label="target cov")
    axD.semilogy(np.arange(1, K + 1), np.clip(eig_res, 1e-12, None),
                 "s-", color="#d62728", lw=1.5, ms=3, label="residual cov")
    axD.set_xlabel("eigen-index (descending)")
    axD.set_ylabel("eigenvalue (log)")
    axD.set_title("Eigenvalue spectra")
    axD.grid(True, alpha=0.3, which="both")
    axD.legend(fontsize=9, loc="upper right")

    # (E) cumulative variance explained
    cum_tgt = np.cumsum(eig_tgt) / eig_tgt.sum()
    cum_res = np.cumsum(eig_res) / eig_res.sum()
    axE.plot(np.arange(1, K + 1), cum_tgt, "o-", color="#1f77b4",
             lw=1.5, ms=3, label="target")
    axE.plot(np.arange(1, K + 1), cum_res, "s-", color="#d62728",
             lw=1.5, ms=3, label="residual")
    for thr in (0.5, 0.8, 0.95):
        axE.axhline(thr, color="grey", lw=0.5, ls="--")
        k_t = int(np.searchsorted(cum_tgt, thr) + 1)
        k_r = int(np.searchsorted(cum_res, thr) + 1)
        axE.text(K, thr + 0.01, f"{int(100*thr)}%: tgt k={k_t}, res k={k_r}",
                 ha="right", fontsize=8, color="grey")
    axE.set_xlabel("# eigen-components")
    axE.set_ylabel("cumulative variance share")
    axE.set_title("Cumulative variance — residual concentrates in fewer modes?")
    axE.set_ylim(0, 1.02)
    axE.grid(True, alpha=0.3)
    axE.legend(fontsize=9, loc="lower right")

    # (F) per-PC R^2 and residual std bar plot
    axF.bar(np.arange(K), r2, color="#2ca02c", alpha=0.9,
            label=f"R^2 per PC (macro mean={r2_macro:.3f})")
    axF.axhline(0, color="black", lw=0.5)
    axF.set_xlabel("PC index")
    axF.set_ylabel("OOF R^2")
    axF.set_title(f"Per-PC OOF R^2 of {SPEC}")
    axF.set_ylim(min(-0.2, float(r2.min()) - 0.05), 1.0)
    axF.grid(True, axis="y", alpha=0.3)
    axF.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"auto_59 (hhhhhhh): residual covariance structure of best supervised spec "
        f"[{SPEC}, L40, OOF 5-fold ridge]   "
        f"residual eff. rank = {eff_rank_res:.1f}/{K},  "
        f"top-1 resid eig captures {100*eig_res[0]/eig_res.sum():.1f}% of leftover var",
        fontsize=12)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    # ----- JSON summary --------------------------------------------------
    summary = {
        "idea": "hhhhhhh",
        "spec": SPEC,
        "n_colors": int(n_c),
        "K_pcs": int(K),
        "n_folds": N_FOLDS,
        "ridge_alpha": RIDGE_ALPHA,
        "r2_macro_mean_oof": r2_macro,
        "r2_macro_mean_json": float(pl["specs"][SPEC]["r2_macro_mean"]),
        "per_pc_r2": [float(x) for x in r2],
        "eig_target": [float(x) for x in eig_tgt],
        "eig_residual": [float(x) for x in eig_res],
        "effective_rank_target": eff_rank_tgt,
        "effective_rank_residual": eff_rank_res,
        "residual_top_eig_share": float(eig_res[0] / eig_res.sum()),
        "residual_top3_eig_cum_share": float(eig_res[:3].sum() / eig_res.sum()),
        "residual_offdiag_corr_abs_mean": float(np.mean(np.abs(offdiag))),
        "residual_offdiag_corr_abs_max": float(np.max(np.abs(offdiag))),
        "residual_offdiag_corr_frac_gt_0p2": float((np.abs(offdiag) > 0.2).mean()),
        "top3_resid_eigvec_correlations": interp_table,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
