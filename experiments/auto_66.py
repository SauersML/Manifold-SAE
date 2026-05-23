"""auto_66.py — Cyclic-hue smooth fit to cogito L40 via gamfit Duchon.

Uses gamfit's TRUE 1D periodic Duchon spline: ``duchon_basis(hue, centers,
m=2, periodic_per_axis=[True])`` — Bernoulli-Green function on the circle,
no B-spline knot machinery. REML-selected smoothing via gaussian_reml_fit.

Two tests:
  (a) Fit theta -> Z_top64. Sweep n_centers ∈ {6, 12, 20, 40}; report
      5-fold CV macro R^2 and the REML-selected log-lambda. Compare to
      L_lin_rgb (0.118) and kNN-Lab (0.246) ceilings from auto_41.

  (b) Fit a closed curve gamma(theta) -> T in U_3d latent space using the
      same 1D periodic Duchon. Per-color off-curve residual r_i = ||T_i -
      gamma(theta_i)||; correlate r with HSV saturation, value, and
      centroid magnitude. Tests whether the manifold factorizes as
      `hue-circle x (sat,val)` or whether sat/val are absent.

NOTE: U_3d's T is gauge-arbitrary (the loss is invariant under
T -> T o phi), so the "closed-curve in T" panel is reporting the curve
in whatever chart this random-init landed in. The pure-hue Z-fit in
panel (a) is the gauge-invariant statement — it measures the fraction
of Z's variance recoverable from the single observed coordinate (hue).

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
from _pca_basis import load_pc_basis, project, TOP_TEMPLATES as _TT

OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
N_T = 28
TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
N_FOLDS = 5
K_PC = 64
N_CENTERS_MAIN = 20  # main center count for the closed-curve fit


def r2_macro(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def cyclic_basis(t01, n_centers):
    """1D periodic Duchon m=2 spline on [0,1) via gamfit.

    Returns (Phi, P) where Phi has shape (N, K - nullity); centers are
    uniformly spaced on the circle.  No length_scale (scale-free spectrum).
    No B-splines.
    """
    import gamfit
    centers = np.linspace(0.0, 1.0, n_centers, endpoint=False).reshape(-1, 1)
    pts = np.asarray(t01, dtype=np.float64).reshape(-1, 1)
    Phi = gamfit.duchon_basis(pts, centers, m=2, periodic_per_axis=[True])
    P = gamfit.duchon_function_norm_penalty(centers, m=2, periodic_per_axis=[True])
    return np.asarray(Phi), np.asarray(P)


def reml_fit_one(Phi, Y, P, init_lambda=None):
    """Wrap gamfit.gaussian_reml_fit; returns (B coeffs, log_lambda)."""
    import gamfit
    out = gamfit.gaussian_reml_fit(Phi, Y, P, init_lambda=init_lambda)
    return np.asarray(out["coefficients"]), float(np.log(float(out["lambda"])))


def main():
    # Use mmap + canonical basis to avoid OOM (X_L40.npy is 760 MB)
    cache = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
    X = np.load(cache, mmap_mode="r")
    n_raw = X.shape[0] // N_T
    print(f"[circle] X (mmap) shape={X.shape}, n_raw={n_raw}")

    centroids = np.zeros((n_raw, X.shape[1]), dtype=np.float64)
    for ci in range(n_raw):
        rows = [ci * N_T + ti for ti in TOP_TEMPLATES]
        # explicit array copy avoids holding multiple mmap views
        centroids[ci] = np.asarray(X[rows], dtype=np.float64).mean(0)
    del X  # release mmap

    colors_all = load_xkcd_colors()[:n_raw]
    kept, kept_idx = filter_colors(colors_all)
    centroids = centroids[kept_idx]
    rgb = np.array([(r, g, b) for _, r, g, b in kept], dtype=np.float64) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    theta = hue * 2 * np.pi
    N = len(kept)
    print(f"[circle] {N} filtered colors")

    # Canonical sklearn PCA basis (cached)
    basis = load_pc_basis(K=K_PC)
    Z = project(centroids, basis)
    evr_top = float(basis["evr"].sum())
    print(f"[circle] Z shape={Z.shape}, top-{K_PC} EVR cum = {evr_top:.3f}")
    mu = basis["mu"]

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=int)
    fold[perm] = np.arange(N) % N_FOLDS

    # --- (a) 1D periodic Duchon + REML sweep over n_centers ---
    print("\n[a] gamfit 1D periodic Duchon (m=2) + REML -> Z_top64  (hue ∈ [0,1))")
    t01 = hue.astype(np.float64)  # in [0, 1)
    r2_per_k = {}
    loglam_per_k = {}
    for n_k in [6, 12, 20, 40]:
        preds = np.zeros_like(Z)
        ll_folds = []
        for f in range(N_FOLDS):
            tr = fold != f
            te = ~tr
            Phi_tr, P = cyclic_basis(t01[tr], n_k)
            Phi_te, _ = cyclic_basis(t01[te], n_k)
            B, ll = reml_fit_one(Phi_tr, Z[tr], P)
            preds[te] = Phi_te @ B
            ll_folds.append(ll)
        r2_per_k[n_k] = float(r2_macro(Z, preds))
        loglam_per_k[n_k] = float(np.mean(ll_folds))
        print(f"   n_centers={n_k:3d}  CV macro R^2 = {r2_per_k[n_k]:+.4f}   "
              f"mean log_lambda = {loglam_per_k[n_k]:+.2f}")

    # --- (b) Closed curve in U_3d latent ---
    print("\n[b] Fitting U_3d (N_iters=50, default PCA init)...")
    from color_manifold_gam import fit_unsupervised_manifold, Config, duchon_basis_radial
    cfg = Config()
    fit = fit_unsupervised_manifold(Z, d=3, cfg=cfg, n_iters=15, verbose=False)
    T = fit["T"]
    # Train R^2 of U_3d (Z reconstruction)
    Phi, _ = duchon_basis_radial(T, fit["centers"])
    Z_hat = Phi @ fit["B"]
    r2_u3d_train = r2_macro(Z, Z_hat)
    print(f"   U_3d train R^2 vs Z = {r2_u3d_train:+.3f}")

    # Closed curve gamma(theta) in 3D via gamfit cyclic basis + REML
    preds_T = np.zeros_like(T)
    for f in range(N_FOLDS):
        tr = fold != f
        te = ~tr
        Phi_tr, P = cyclic_basis(t01[tr], N_CENTERS_MAIN)
        Phi_te, _ = cyclic_basis(t01[te], N_CENTERS_MAIN)
        B, _ = reml_fit_one(Phi_tr, T[tr], P)
        preds_T[te] = Phi_te @ B
    r2_T_cv = float(r2_macro(T, preds_T))
    print(f"   closed-curve CV R^2 in T (n_centers={N_CENTERS_MAIN}) = {r2_T_cv:+.3f}")

    # In-sample fit for geometric residual
    Phi_full, P_full = cyclic_basis(t01, N_CENTERS_MAIN)
    B_full, _ = reml_fit_one(Phi_full, T, P_full)
    T_on_curve = Phi_full @ B_full
    resid = np.linalg.norm(T - T_on_curve, axis=1)
    centroid_mag = np.linalg.norm(centroids - mu, axis=1)

    from scipy.stats import spearmanr
    rho_sat = float(spearmanr(resid, sat).statistic)
    rho_val = float(spearmanr(resid, val).statistic)
    rho_mag = float(spearmanr(resid, centroid_mag).statistic)
    print(f"   resid ~ sat  : Spearman rho = {rho_sat:+.3f}")
    print(f"   resid ~ val  : Spearman rho = {rho_val:+.3f}")
    print(f"   resid ~ |cen|: Spearman rho = {rho_mag:+.3f}")

    t01_dense = np.linspace(0, 1, 360, endpoint=False)
    Phi_dense, _ = cyclic_basis(t01_dense, N_CENTERS_MAIN)
    gamma = Phi_dense @ B_full
    theta_dense = t01_dense * 2 * np.pi

    # ---- Render ----
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ks = sorted(r2_per_k.keys())
    ax1.plot(ks, [r2_per_k[k] for k in ks], "o-", lw=2, color="navy", ms=8)
    ax1.axhline(0.246, color="green", ls="--", lw=1, label="kNN-Lab ceiling (0.246)")
    ax1.axhline(0.234, color="darkgreen", ls=":", lw=1, label="L_joint_rgb_with_hue (0.234)")
    ax1.axhline(0.182, color="orange", ls="--", lw=1, label="L_cyc_hue_polysv (0.182)")
    ax1.axhline(0.118, color="red", ls="--", lw=1, label="L_lin_rgb (0.118)")
    ax1.set_xlabel("n_centers  (cyclic B-spline capacity; REML picks DoF)")
    ax1.set_ylabel("5-fold CV macro R²  (target = Z_top64)")
    ax1.set_title("(a) gamfit cyclic-Bspline + REML R²")
    ax1.set_xscale("log")
    ax1.set_xticks(ks)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    # PCA view of T + gamma
    from sklearn.decomposition import PCA
    p2 = PCA(n_components=2).fit(T)
    T2 = p2.transform(T)
    g2 = p2.transform(gamma)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(T2[:, 0], T2[:, 1], c=rgb, s=18, edgecolor="black", linewidth=0.25)
    ax2.plot(g2[:, 0], g2[:, 1], "k-", lw=2, alpha=0.6, label=f"γ(θ), n_centers={N_CENTERS_MAIN}")
    for i in range(0, 360, 15):
        h_rgb = colorsys.hsv_to_rgb(theta_dense[i] / (2 * np.pi), 1, 1)
        ax2.plot(g2[i, 0], g2[i, 1], "o", color=h_rgb, ms=8, mec="black", mew=0.6)
    ax2.set_xlabel(f"PCA1 of T ({p2.explained_variance_ratio_[0] * 100:.0f}% var)")
    ax2.set_ylabel(f"PCA2 of T ({p2.explained_variance_ratio_[1] * 100:.0f}% var)")
    ax2.set_title(f"(b) Closed Fourier curve in U_3d latent\nCV R² vs T = {r2_T_cv:+.3f}")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(resid, bins=40, color="steelblue", edgecolor="black")
    ax3.set_xlabel("Off-curve residual  ‖T − γ(θ)‖")
    ax3.set_ylabel("count")
    ax3.set_title(f"Per-color off-curve residual\nmed={np.median(resid):.3f}, max={resid.max():.3f}")

    for ax, xs, name, rho in zip(
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
        [sat, val, centroid_mag],
        ["HSV saturation", "HSV value", "‖centroid − μ‖"],
        [rho_sat, rho_val, rho_mag],
    ):
        ax.scatter(xs, resid, c=rgb, s=14, edgecolor="black", linewidth=0.2)
        ax.set_xlabel(name)
        ax.set_ylabel("‖T − γ(θ)‖")
        ax.set_title(f"resid vs {name}\nSpearman ρ = {rho:+.3f}")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"gamfit cyclic-Bspline (REML) on hue · {N} filtered xkcd colors · cogito L40",
        fontsize=13,
    )
    plt.tight_layout()
    out = OUT_DIR / "auto_66.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")

    (OUT_DIR / "auto_66.json").write_text(json.dumps({
        "n_colors": int(N),
        "K_PC": K_PC,
        "evr_top_K_PC": evr_top,
        "r2_pure_hue_per_n_centers": r2_per_k,
        "log_lambda_per_n_centers": loglam_per_k,
        "U_3d_train_r2": float(r2_u3d_train),
        "closed_curve_cv_r2_in_T": r2_T_cv,
        "n_centers_main": N_CENTERS_MAIN,
        "residual_spearman_vs_sat": rho_sat,
        "residual_spearman_vs_val": rho_val,
        "residual_spearman_vs_centroid_magnitude": rho_mag,
        "residual_median": float(np.median(resid)),
        "residual_max": float(resid.max()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
