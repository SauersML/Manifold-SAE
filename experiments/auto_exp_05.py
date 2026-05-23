"""auto_exp_05: manifold curvature ranking at each anchor color.

The U_3d unsupervised fit parameterises the L40 color manifold by latent
coordinates T ∈ [0,1]^3 and a smooth f(t) : R^3 → R^64 (top-64 PCs of the
centroid residual matrix).  This experiment asks:

  Where does the manifold *bend*?  Which xkcd colors sit at high-curvature
  spots, and which sit on near-flat patches?

Method (cheap — NO server calls, NO new fits beyond one U_3d):
  1. Load cached residuals (/runs/COLOR_COGITO_L40/X_L40.npy).
  2. Reproduce the standard 954×64 centroid PCA target used by
     color_manifold_gam.py.
  3. Fit a single U_3d (no CV — we want one global smooth).
  4. For each anchor, take its converged latent t = fit["T"][i] and
     evaluate the smooth's Hessian via central finite differences
     (h = 5e-3) on each output PC.  Per-anchor curvature scalar:

         κ_i = sqrt( Σ_p Σ_{j,k}  (∂²f_p / ∂t_j ∂t_k)² )

     i.e. the Frobenius norm of the stacked Hessians across PCs, which
     equals ‖∂²f/∂t²‖ as a tensor.  Skip anchors within h of the
     [0,1]^3 boundary (their stencils would leave the support).
  5. Rank colors by κ; save full ranking + plot (top-20, bottom-20,
     swatches + κ values) and a histogram.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_05_curvature.{json,png}

Why this is useful: high-κ colors mark regions where the *tangent*
parameterisation is locally non-linear — i.e. where a small change in t
maps to a large turn of f.  These are the points where supervised
linear baselines (L_lin_rgb, L_lin_lab) should be weakest, and where
the Duchon smooth's nonlinearity pays off most.  Inverse direction
(low κ, near-flat) tells us where the residual manifold is well
approximated by a local tangent plane — useful for picking anchor
points for tangent-steering experiments (auto_exp_06 candidate).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_exp_05_curvature.json"
OUT_PNG = OUT_DIR / "auto_exp_05_curvature.png"

N_TEMPLATES = 28
N_PCS = 64
FD_H = 5e-3        # finite-diff step; small enough to be local, large enough
                   # that the gamfit Duchon kernel (m=2) is well-conditioned.
                   # Skip any anchor within FD_H of any boundary face.


def hessian_at(t: np.ndarray, B: np.ndarray, centers: np.ndarray,
                 h: float) -> np.ndarray:
    """Central-difference 3×3 Hessian per output PC at latent t ∈ R^3.

    Returns H of shape (n_pcs, 3, 3).  Uses 6+12 evaluations of the
    Duchon basis (cheap — a few hundred centers × tens of points).
    """
    d = t.shape[0]
    assert d == 3
    n_pcs = B.shape[1]
    H = np.zeros((n_pcs, d, d), dtype=np.float64)

    # Evaluate f at the centre once (for diagonal stencil).
    pts = [t.copy()]
    for j in range(d):
        ep = t.copy(); ep[j] += h
        em = t.copy(); em[j] -= h
        pts += [ep, em]
    # Off-diagonal four-point cross stencils.
    for j in range(d):
        for k in range(j + 1, d):
            for sj in (+1, -1):
                for sk in (+1, -1):
                    p = t.copy(); p[j] += sj * h; p[k] += sk * h
                    pts.append(p)
    pts = np.asarray(pts)
    Phi, _ = cmg.duchon_basis_radial(pts, centers)
    F = Phi @ B   # (n_pts, n_pcs)

    # Diagonal:  (f(t+h_j) - 2 f(t) + f(t-h_j)) / h²
    f0 = F[0]
    idx = 1
    for j in range(d):
        fp = F[idx]; fm = F[idx + 1]; idx += 2
        H[:, j, j] = (fp - 2.0 * f0 + fm) / (h * h)
    # Off-diagonal:  (f(++) - f(+-) - f(-+) + f(--)) / (4 h²)
    for j in range(d):
        for k in range(j + 1, d):
            f_pp = F[idx]; f_pm = F[idx + 1]; f_mp = F[idx + 2]; f_mm = F[idx + 3]
            idx += 4
            v = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            H[:, j, k] = v
            H[:, k, j] = v
    return H


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)

    colors = cmg.load_xkcd_colors()
    assert len(colors) == n_colors
    rgb01 = np.array([[r, g, b] for _, r, g, b in colors], dtype=np.float64) / 255.0

    # ---- Per-color centroid, standardize, top-64 PCs (same basis as the
    # production GAM run and auto_exp_04).
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    Z = centroids_n @ V_topK.T          # (954, 64)
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    # ---- Fit one U_3d on ALL 954 centroids (no CV — we want one smooth).
    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=5,
                      lattice_per_side=5, init_log_lambda=0.0,
                      output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    print("[fit] U_3d alternation on 954 centroids ...", flush=True)
    t0 = time.time()
    fit = cmg.fit_unsupervised_manifold(Z, d=3, cfg=cfg, n_iters=15, verbose=True)
    T = fit["T"]; B = fit["B"]; centers = fit["centers"]
    print(f"[fit] done in {time.time() - t0:.1f}s  T={T.shape}  B={B.shape}  "
          f"centers={centers.shape}", flush=True)

    # Training R² sanity check
    Phi_tr, _ = cmg.duchon_basis_radial(T, centers)
    Z_hat = Phi_tr @ B
    ss_res = ((Z - Z_hat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    train_r2 = 1.0 - ss_res / ss_tot
    print(f"[fit] training R² = {train_r2:+.4f}", flush=True)

    # ---- Curvature per anchor ----
    print(f"[curv] computing Hessians  (h={FD_H})", flush=True)
    kappa = np.full(n_colors, np.nan, dtype=np.float64)
    op_norm = np.full(n_colors, np.nan, dtype=np.float64)
    skipped = 0
    t0 = time.time()
    for i in range(n_colors):
        t = T[i]
        if (t < FD_H).any() or (t > 1.0 - FD_H).any():
            skipped += 1
            continue
        H = hessian_at(t, B, centers, FD_H)
        # Frobenius norm across all PCs and axes — the ‖∂²f/∂t²‖ tensor.
        kappa[i] = float(np.sqrt((H ** 2).sum()))
        # Operator-style scalar: largest singular value of the per-PC-stacked
        # 64×9 unfolding (alternative "principal curvature magnitude").
        op_norm[i] = float(np.linalg.norm(H.reshape(N_PCS, -1), ord=2))
        if (i + 1) % 100 == 0:
            print(f"   {i + 1}/{n_colors}  elapsed {time.time() - t0:.1f}s",
                  flush=True)
    print(f"[curv] done — {skipped} anchors skipped (near boundary)", flush=True)

    valid = ~np.isnan(kappa)
    order = np.argsort(-kappa)   # high → low; NaNs sink to end
    rank = [
        {
            "rank": int(r),
            "color_idx": int(i),
            "name": colors[i][0],
            "rgb": [int(colors[i][1]), int(colors[i][2]), int(colors[i][3])],
            "t": [float(x) for x in T[i]],
            "kappa": float(kappa[i]),
            "op_norm": float(op_norm[i]),
        }
        for r, i in enumerate(order) if valid[i]
    ]

    summary = {
        "config": {
            "harvest": str(HARVEST),
            "n_colors": int(n_colors),
            "n_templates": N_TEMPLATES,
            "n_pcs": N_PCS,
            "fd_h": FD_H,
            "lattice_per_side": int(fit["centers_per_axis"]),
        },
        "train_r2_U3d": float(train_r2),
        "kappa_quantiles": {
            q: float(np.quantile(kappa[valid], q))
            for q in (0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0)
        },
        "n_skipped_boundary": int(skipped),
        "ranking": rank,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt

    valid_idx = np.where(valid)[0]
    k_valid = kappa[valid]
    top = order[:20]
    bot = [i for i in order[::-1] if valid[i]][:20]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 1])

    # (a) histogram of κ on log scale
    ax_h = fig.add_subplot(gs[0, 0])
    ax_h.hist(np.log10(k_valid + 1e-12), bins=60, color="#4060a0", edgecolor="white")
    ax_h.set_xlabel(r"$\log_{10}\,\kappa$    ($\kappa$ = $\|\partial^2 f/\partial t^2\|_F$)")
    ax_h.set_ylabel("count")
    ax_h.set_title(f"U_3d curvature distribution ({len(k_valid)} anchors)")
    ax_h.grid(linestyle=":", alpha=0.4)

    # (b) κ vs RGB lightness (luma) — does curvature correlate with luminance?
    ax_l = fig.add_subplot(gs[0, 1])
    luma = 0.2126 * rgb01[:, 0] + 0.7152 * rgb01[:, 1] + 0.0722 * rgb01[:, 2]
    ax_l.scatter(luma[valid_idx], k_valid, s=8,
                  c=rgb01[valid_idx], edgecolors="none", alpha=0.75)
    ax_l.set_xlabel("luma (BT.709)")
    ax_l.set_ylabel(r"$\kappa$")
    ax_l.set_title("curvature vs luma  (point = anchor RGB)")
    ax_l.set_yscale("log")
    ax_l.grid(linestyle=":", alpha=0.4)

    # (c) κ vs distance-to-boundary in T-space (close to boundary -> often
    #     where the smooth's nullspace dominates, low curvature).
    ax_d = fig.add_subplot(gs[0, 2])
    d_bnd = np.minimum(T.min(1), 1.0 - T.max(1))
    ax_d.scatter(d_bnd[valid_idx], k_valid, s=8,
                  c=rgb01[valid_idx], edgecolors="none", alpha=0.75)
    ax_d.set_xlabel(r"distance to $[0,1]^3$ boundary in $T$")
    ax_d.set_ylabel(r"$\kappa$")
    ax_d.set_yscale("log")
    ax_d.set_title("curvature vs latent-space boundary distance")
    ax_d.grid(linestyle=":", alpha=0.4)

    # (d) top-20 swatches
    ax_top = fig.add_subplot(gs[1, :])
    for j, i in enumerate(top):
        ax_top.add_patch(plt.Rectangle((j, 0), 1, 1, color=rgb01[i]))
        ax_top.text(j + 0.5, -0.18, colors[i][0][:14], ha="center",
                     va="top", fontsize=7, rotation=30)
        ax_top.text(j + 0.5, 1.05, f"{kappa[i]:.2g}", ha="center",
                     va="bottom", fontsize=7)
    ax_top.set_xlim(0, 20); ax_top.set_ylim(-0.6, 1.4)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_title("Top-20 highest-curvature anchors  (κ above each swatch)")

    # (e) bottom-20 swatches
    ax_bot = fig.add_subplot(gs[2, :])
    for j, i in enumerate(bot):
        ax_bot.add_patch(plt.Rectangle((j, 0), 1, 1, color=rgb01[i]))
        ax_bot.text(j + 0.5, -0.18, colors[i][0][:14], ha="center",
                     va="top", fontsize=7, rotation=30)
        ax_bot.text(j + 0.5, 1.05, f"{kappa[i]:.2g}", ha="center",
                     va="bottom", fontsize=7)
    ax_bot.set_xlim(0, 20); ax_bot.set_ylim(-0.6, 1.4)
    ax_bot.set_xticks([]); ax_bot.set_yticks([])
    ax_bot.set_title("Bottom-20 lowest-curvature anchors  (locally flat patches)")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
