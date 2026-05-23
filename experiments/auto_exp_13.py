"""auto_exp_13: orthogonal complement of U_3d in PC-space — what's left?

Motivation (option qq)
----------------------
auto_exp_06 (dim-elbow) and auto_exp_09 (LOTO U_3d) established that a
d=3 Duchon manifold absorbs the bulk of the centroid PCA-64 signal.
But "the bulk" is not "everything": auto_exp_12 settled at train_R^2
~ 0.717 even with N_iters=50. That leaves ~28% of normalized centroid
variance OUTSIDE U_3d. Open question (qq): what is the geometry of
that residual "color-orthogonal" subspace?

Concrete questions
------------------
  1. What is the effective dimension of the residual? (cumulative EVR
     elbow on Z - Z_hat.) If the residual is ~ isotropic noise in
     ~k_eff dimensions, color manifold = 3 + slack channels.
  2. Does the residual carry hue/template structure, or is it a wash?
     - hue-axis test: cosine sim of residual PCs with (Z_hat[red] -
       Z_hat[cyan])-style hue contrasts in PC-space.
     - template-axis test: per-template mean of the residual; does
       it have a preferred direction? (templates were averaged out of
       the centroid, so any structure here is the *unexplained*
       template signature seeping back through residual PCs.)
  3. Anchor probe: do the residual top-3 PCs separate
     achromatic-vs-chromatic? (project the achromatic ramp
     black/gray/white-named colors onto residual PC1-3; check rank.)

Why this is cheap and worth running
-----------------------------------
  - Zero server traffic (we already have X_L40.npy).
  - One Duchon fit (N_iters=50 per auto_exp_12).
  - Plain SVD on the (n_colors x N_PCS) residual matrix.
  - Total runtime ~ same as auto_exp_06/09 ≈ a few seconds.

If the residual top-PCs have nontrivial template structure, then
"centroid + d=3" is throwing away a real signal and we should report
that as an upper bound, not a sufficiency claim. If residual is
isotropic, U_3d is the right ceiling for centroid-level geometry.

NO Gaussian RBF. NO length_scale on Duchon. No server traffic.
Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_13_ortho_complement.{json,png}
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
NAMES = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/color_names.json")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_exp_13_ortho_complement.json"
OUT_PNG = OUT_DIR / "auto_exp_13_ortho_complement.png"

N_TEMPLATES = 28
N_PCS = 64
D = 3
N_ITERS = 50  # auto_exp_12: 50 > 20 by +0.006 R^2; cheap enough


def centroids_full(X: np.ndarray, c_idx: np.ndarray, n_colors: int) -> np.ndarray:
    out = np.zeros((n_colors, X.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        out[ci] = X[c_idx == ci].mean(0)
    return out


def cumulative_evr_elbow(evr: np.ndarray, thresh: float) -> int:
    """Smallest k s.t. cumulative EVR >= thresh."""
    c = np.cumsum(evr)
    idx = int(np.searchsorted(c, thresh) + 1)
    return min(idx, len(evr))


def participation_ratio(evr: np.ndarray) -> float:
    """(sum lam)^2 / sum lam^2; effective dim under L2 mass."""
    s = float(evr.sum())
    if s <= 0:
        return float("nan")
    return float(s * s / float((evr * evr).sum()))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, H = X.shape
    assert N % N_TEMPLATES == 0
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    # ---- centroids, normalize, fixed top-64 PCA basis (matches exp_09/12) ----
    centroids = centroids_full(X, c_idx, n_colors)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroids - mu) / sigma  # (n_colors, H)
    Cn = Cn - Cn.mean(0, keepdims=True)
    _, s_full, Vt_full = np.linalg.svd(Cn, full_matrices=False)
    V_topK = Vt_full[:N_PCS]                       # (N_PCS, H)
    evr_topK = (s_full ** 2 / (s_full ** 2).sum())[:N_PCS]
    Z = Cn @ V_topK.T                              # (n_colors, N_PCS)
    print(f"[pca] fixed top-{N_PCS} EVR sum = {evr_topK.sum():.3f}  Z={Z.shape}",
          flush=True)

    # ---- one Duchon U_3d fit ----
    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=5,
                     lattice_per_side=5, init_log_lambda=0.0,
                     output_dir=str(OUT_DIR), harvest_from=str(HARVEST))
    print(f"[fit ] U_{D}d, N_iters={N_ITERS}", flush=True)
    t0 = time.time()
    fit = cmg.fit_unsupervised_manifold(Z, D, cfg, n_iters=N_ITERS,
                                        init_T=None, verbose=False)
    Phi, _ = cmg.duchon_basis_radial(fit["T"], fit["centers"])
    Z_hat = Phi @ fit["B"]
    R = Z - Z_hat                                  # (n_colors, N_PCS) residual
    dt = time.time() - t0
    ss_res = float((R ** 2).sum())
    ss_tot = float(((Z - Z.mean(0, keepdims=True)) ** 2).sum())
    train_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"[fit ] train_r2={train_r2:+.4f}  log_lam={float(fit['log_lambda']):+.2f}  "
          f"({dt:.1f}s)", flush=True)

    # ---- SVD on the residual: this IS the orthogonal complement geometry ----
    R0 = R - R.mean(0, keepdims=True)
    Ur, sr, Vrt = np.linalg.svd(R0, full_matrices=False)
    var_R = (sr ** 2)
    evr_R = var_R / max(var_R.sum(), 1e-12)
    cum_R = np.cumsum(evr_R)
    elbow_50 = cumulative_evr_elbow(evr_R, 0.50)
    elbow_90 = cumulative_evr_elbow(evr_R, 0.90)
    pr = participation_ratio(evr_R)
    print(f"[ortho] residual var share of total Z var = "
          f"{var_R.sum() / max((Z ** 2).sum() - n_colors * (Z.mean(0) ** 2).sum(), 1e-12):.3f}",
          flush=True)
    print(f"[ortho] residual EVR k(50%)={elbow_50}  k(90%)={elbow_90}  "
          f"participation_ratio={pr:.2f}", flush=True)

    # ---- template-structure test: per-template mean of residual ----
    # In the centroid PC space Z, the per-template mean leaks in via the
    # template-replicate scatter that wasn't averaged into the centroid;
    # but here we instead probe: project EACH raw sample (not centroid)
    # onto V_topK, subtract its color's Z_hat, then compute per-template
    # mean of the resulting residual. Magnitude per template = is there
    # a residual template axis that U_3d failed to absorb?
    Xn = (X - mu) / sigma
    Xn = Xn - Cn.mean(0, keepdims=True)   # same offset as Cn
    Zx = Xn @ V_topK.T                    # (N, N_PCS)
    Rx = Zx - Z_hat[c_idx]                # (N, N_PCS)
    per_template_mean = np.zeros((N_TEMPLATES, N_PCS), dtype=np.float64)
    for ti in range(N_TEMPLATES):
        per_template_mean[ti] = Rx[t_idx == ti].mean(0)
    tpl_norms = np.linalg.norm(per_template_mean, axis=1)
    # variance decomposition on Rx (each row a sample):
    #   total SS = within-template SS + between-template SS.
    # We want: fraction of residual energy explained by template mean.
    total_ss = float((Rx ** 2).sum())
    # between-template SS = sum_t n_t * ||mean_t||^2  (n_t = n_colors here)
    between_ss = float((per_template_mean ** 2).sum() * n_colors)
    tpl_fraction = between_ss / max(total_ss, 1e-12)
    print(f"[tpl  ] per-template-mean residual ||.||  min={tpl_norms.min():.3f}  "
          f"median={float(np.median(tpl_norms)):.3f}  max={tpl_norms.max():.3f}", flush=True)
    print(f"[tpl  ] template-axis var / total residual var = {tpl_fraction:.3f}  "
          f"(0 = isotropic noise, ~1 = pure template axis)", flush=True)

    # ---- effective rank of the manifold image in PC-space ----
    # The Duchon radial basis maps the d=3 latent to a (n_colors, N_PCS)
    # fitted matrix Z_hat that, after SVD, is generically full rank in
    # PC space. So the right notion of "manifold subspace" is the
    # EFFECTIVE rank: number of singular values carrying ≥ 99% of var.
    # Anything *outside* that effective subspace plus the residual
    # together form the color-orthogonal complement we care about.
    Zhat0 = Z_hat - Z_hat.mean(0, keepdims=True)
    Uh, sh, Vhat_t = np.linalg.svd(Zhat0, full_matrices=False)
    sh2 = sh ** 2
    cum_h = np.cumsum(sh2) / max(sh2.sum(), 1e-12)
    rk_eff_99 = int(np.searchsorted(cum_h, 0.99) + 1)
    rk_eff_999 = int(np.searchsorted(cum_h, 0.999) + 1)
    Vhat_basis = Vhat_t[:rk_eff_99]                  # (rk_eff_99, N_PCS)
    top_k = min(5, Vrt.shape[0])
    align = np.linalg.norm(Vrt[:top_k] @ Vhat_basis.T, axis=1)  # in [0, 1]
    print(f"[manif] Z_hat effective rank: 99%={rk_eff_99}  99.9%={rk_eff_999}  "
          f"(formal rank={(sh > 1e-8 * sh.max()).sum()})", flush=True)
    print(f"[align] ||proj_U99 Vr[k]||  "
          "(0 = residual top-PC sits outside 99%-manifold-subspace):  "
          + "  ".join(f"k{i}={align[i]:.3f}" for i in range(top_k)), flush=True)
    rk_hat = rk_eff_99

    # ---- achromatic anchor probe (best-effort; skip if names missing) ----
    achro_rank: list | None = None
    achro_msg = "names file missing; skipped"
    if NAMES.exists():
        try:
            names = json.loads(NAMES.read_text())
            ach_keys = ("black", "white", "gray", "grey", "silver")
            ach_idx = [i for i, nm in enumerate(names)
                       if isinstance(nm, str) and nm.lower() in ach_keys]
            if ach_idx:
                # project residuals onto top-3 residual PCs
                R_top3 = R0 @ Vrt[:3].T            # (n_colors, 3)
                ach_mean = R_top3[ach_idx].mean(0)
                chro_mask = np.ones(n_colors, dtype=bool); chro_mask[ach_idx] = False
                chro_mean = R_top3[chro_mask].mean(0)
                sep = float(np.linalg.norm(ach_mean - chro_mean))
                achro_rank = {
                    "n_achromatic_found": len(ach_idx),
                    "achromatic_idx_sample": ach_idx[:10],
                    "ach_minus_chro_norm_in_residual_top3": sep,
                    "ach_mean_top3": ach_mean.tolist(),
                    "chro_mean_top3": chro_mean.tolist(),
                }
                achro_msg = (f"{len(ach_idx)} achromatic colors; "
                             f"||ach-chro|| in residual top-3 = {sep:.3f}")
        except Exception as e:    # noqa: BLE001
            achro_msg = f"failed: {e!r}"
    print(f"[achro] {achro_msg}", flush=True)

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": int(n_colors),
            "n_templates": N_TEMPLATES, "n_pcs": N_PCS, "d": D,
            "n_iters": N_ITERS,
        },
        "fit": {
            "train_r2": train_r2,
            "log_lambda": float(fit["log_lambda"]),
            "manifold_eff_rank_99pct": int(rk_eff_99),
            "manifold_eff_rank_999pct": int(rk_eff_999),
        },
        "residual_geometry": {
            "share_of_normalized_var": float(
                var_R.sum()
                / max((Z ** 2).sum() - n_colors * (Z.mean(0) ** 2).sum(), 1e-12)
            ),
            "evr_top16": evr_R[:16].tolist(),
            "cum_evr_top16": cum_R[:16].tolist(),
            "k_50pct": int(elbow_50),
            "k_90pct": int(elbow_90),
            "participation_ratio": float(pr),
        },
        "template_structure_in_residual": {
            "per_template_norm": tpl_norms.tolist(),
            "template_axis_var_fraction": float(tpl_fraction),
            "interpretation": (
                "0 = residual is template-isotropic noise; "
                "~1 = a coherent template axis survived the U_3d fit."
            ),
        },
        "alignment_residual_top_with_manifold_range": {
            "top_k": int(top_k),
            "norms": align.tolist(),
            "note": "values ~0 confirm residual ⊥ manifold range in PC space",
        },
        "achromatic_probe": achro_rank,
        "achromatic_msg": achro_msg,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (1) cumulative EVR of residual vs original
    ax = axes[0, 0]
    K = min(32, N_PCS)
    ax.plot(np.arange(1, K + 1), np.cumsum(evr_topK[:K]),
            "-o", color="steelblue", ms=4, label="centroid Z (PCA EVR)")
    ax.plot(np.arange(1, min(K, len(evr_R)) + 1),
            cum_R[:K], "-s", color="firebrick", ms=4,
            label="residual Z - Z_hat (EVR)")
    ax.axhline(0.5, color="gray", ls=":", lw=0.7)
    ax.axhline(0.9, color="gray", ls=":", lw=0.7)
    ax.axvline(elbow_50, color="firebrick", ls="--", lw=0.7,
               label=f"resid k(50%)={elbow_50}")
    ax.axvline(elbow_90, color="firebrick", ls="-.", lw=0.7,
               label=f"resid k(90%)={elbow_90}")
    ax.set_xlabel("PC index"); ax.set_ylabel("cumulative EVR")
    ax.set_title(f"Residual is broader than Z\nresid PR={pr:.1f}, "
                 f"share={var_R.sum() / max((Z**2).sum() - n_colors*(Z.mean(0)**2).sum(),1e-12):.3f}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)

    # (2) per-template residual norm
    ax = axes[0, 1]
    order = np.argsort(tpl_norms)[::-1]
    ax.bar(np.arange(N_TEMPLATES), tpl_norms[order],
           color="darkorange", edgecolor="k", lw=0.4)
    ax.set_xlabel("template (sorted)"); ax.set_ylabel("||mean residual||")
    ax.set_title(f"Per-template residual axis (frac={tpl_fraction:.3f})\n"
                 "high bars = template-specific bias U_3d missed")
    ax.grid(alpha=0.3, axis="y")

    # (3) alignment of residual top-PCs with manifold range
    ax = axes[1, 0]
    ax.bar(np.arange(top_k), align, color="seagreen", edgecolor="k", lw=0.5)
    ax.set_xlabel("residual PC index")
    ax.set_ylabel("||proj_U Vr[k]||  (0=orthogonal, 1=in-range)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Orthogonality check (manifold rank={rk_hat})")
    ax.grid(alpha=0.3, axis="y")

    # (4) residual top-3 scatter, achromatic flagged
    ax = axes[1, 1]
    R_top3 = R0 @ Vrt[:3].T
    ax.scatter(R_top3[:, 0], R_top3[:, 1], s=12, c="lightgray",
               edgecolor="none", label="chromatic")
    if achro_rank is not None and achro_rank.get("n_achromatic_found", 0) > 0:
        ach_idx = json.loads(NAMES.read_text())
        ach_keys = ("black", "white", "gray", "grey", "silver")
        ai = [i for i, nm in enumerate(ach_idx)
              if isinstance(nm, str) and nm.lower() in ach_keys]
        ax.scatter(R_top3[ai, 0], R_top3[ai, 1], s=40, c="black",
                   marker="x", label=f"achromatic (n={len(ai)})")
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("residual PC1"); ax.set_ylabel("residual PC2")
    ax.set_title("Residual top-2: any color-orthogonal structure?")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"auto_exp_13: orthogonal complement of U_{D}d in PC-{N_PCS} space\n"
        f"train_R^2={train_r2:.3f}  →  residual carries "
        f"{var_R.sum() / max((Z**2).sum() - n_colors*(Z.mean(0)**2).sum(),1e-12):.1%} of var",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
