"""auto_exp_09: leave-one-template-out U_3d jackknife.

Motivation
----------
auto_exp_08 showed template-OOD CV is catastrophic on supervised specs:
the cogito-L40 per-prompt residual mixes substantial template-specific
structure into what we have been calling "the color manifold". That
result was on *supervised* (RGB/HSV/LAB inputs) fits. Open question:
does the same template sensitivity contaminate the *unsupervised* U_3d
geometry — i.e. is the latent T (N_colors, 3) we fit on 28-template
centroids stable, or is it being yanked around by a few influential
templates?

We answer this with a leave-one-template-out (LOTO) jackknife on U_3d:
  - baseline:   fit U_3d on centroids averaged over all 28 templates
  - jackknife:  for each t in 0..27, fit U_3d on centroids averaged
                over the remaining 27 templates
  - compare:    Procrustes-align each jackknifed T to the baseline T
                and report normalized disparity. Big disparity for a
                given t means "removing this template substantially
                changes the discovered latent geometry" — that template
                disproportionately shapes U_3d.

Also reported per fold:
  - train R^2 on the 64-PC target basis (basis is fixed once, from the
    full-28 centroids, so all 28 jackknife fits live in the SAME target
    space and R^2 is directly comparable)
  - alternation iterations to convergence + final log_lambda

Cheap: 28 fits of d=3 (5 centers/axis = 125 basis fns) on ~950 rows. No
server calls, no harvest, ~minutes on CPU. NO Gaussian RBF. NO length_scale.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_09_loto_u3d.{json,png}
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
OUT_JSON = OUT_DIR / "auto_exp_09_loto_u3d.json"
OUT_PNG = OUT_DIR / "auto_exp_09_loto_u3d.png"

N_TEMPLATES = 28
N_PCS = 64
D = 3
N_ITERS = 12
SEED = 0


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    """Orthogonal Procrustes with isotropic scaling + translation.

    Aligns A -> B (same shape (n, d)). Returns (disparity, A_aligned) where
    disparity = ||B - A_aligned||_F^2 / ||B - mean(B)||_F^2  (scale-free,
    0 = perfect, 1 = predicting the mean would do as well).
    """
    A0 = A - A.mean(0, keepdims=True)
    B0 = B - B.mean(0, keepdims=True)
    nA = np.linalg.norm(A0)
    nB = np.linalg.norm(B0)
    if nA < 1e-12 or nB < 1e-12:
        return float("nan"), A.copy()
    A0n = A0 / nA
    B0n = B0 / nB
    U, _, Vt = np.linalg.svd(B0n.T @ A0n, full_matrices=False)
    R = Vt.T @ U.T
    # scale chosen to minimize ||B0 - s * A0 R||_F
    M = A0 @ R
    s = float((B0 * M).sum() / max((M * M).sum(), 1e-12))
    A_aligned = s * M + B.mean(0, keepdims=True)
    disparity = float(((B - A_aligned) ** 2).sum() / max((B0 ** 2).sum(), 1e-12))
    return disparity, A_aligned


def centroids_drop(X: np.ndarray, t_idx: np.ndarray, c_idx: np.ndarray,
                   drop_t: int | None, n_colors: int) -> np.ndarray:
    """Per-color centroid averaging over all templates != drop_t.
    drop_t=None means use all templates."""
    if drop_t is None:
        mask = np.ones(X.shape[0], dtype=bool)
    else:
        mask = (t_idx != drop_t)
    out = np.zeros((n_colors, X.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        rows = mask & (c_idx == ci)
        out[ci] = X[rows].mean(0)
    return out


def fit_u3d_train_r2(Z: np.ndarray, cfg: cmg.Config) -> dict:
    fit = cmg.fit_unsupervised_manifold(Z, D, cfg, n_iters=N_ITERS, verbose=False)
    Phi, _ = cmg.duchon_basis_radial(fit["T"], fit["centers"])
    Z_hat = Phi @ fit["B"]
    ss_res = ((Z - Z_hat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    train_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {
        "T": fit["T"],
        "train_r2": train_r2,
        "n_iters_run": len(fit["history"]),
        "final_log_lambda": float(fit["log_lambda"]),
        "final_dT": float(fit["history"][-1]["dT"]) if fit["history"] else float("nan"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, Dfull = X.shape
    assert N % N_TEMPLATES == 0
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    # ---- Build a FIXED 64-PC target basis from the full-28 centroids.
    # Every jackknife fit projects its own centroids into this same basis
    # so train R^2 values are directly comparable across folds.
    centroids_full = centroids_drop(X, t_idx, c_idx, None, n_colors)
    mu = centroids_full.mean(0, keepdims=True)
    sigma = centroids_full.std(0, keepdims=True).clip(min=1e-6)
    Cc = (centroids_full - mu) / sigma
    Cc = Cc - Cc.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] fixed top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    def project(centroids: np.ndarray) -> np.ndarray:
        return ((centroids - mu) / sigma) @ V_topK.T

    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=5,
                     lattice_per_side=5, init_log_lambda=0.0,
                     output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    # ---- Baseline U_3d on all 28 ----
    print("\n[baseline] fit U_3d on full-28 centroids", flush=True)
    Z_full = project(centroids_full)
    t0 = time.time()
    base = fit_u3d_train_r2(Z_full, cfg)
    print(f"  train_r2={base['train_r2']:+.4f}  iters={base['n_iters_run']}  "
          f"log_lam={base['final_log_lambda']:+.2f}  ({time.time()-t0:.1f}s)",
          flush=True)
    T_base = base["T"]

    # ---- Jackknife: drop one template at a time ----
    per_template: list[dict] = []
    for t in range(N_TEMPLATES):
        print(f"\n[LOTO] drop template {t}: {cmg.TEMPLATES[t][:70]!r}",
              flush=True)
        cent_t = centroids_drop(X, t_idx, c_idx, t, n_colors)
        Z_t = project(cent_t)
        t0 = time.time()
        try:
            res = fit_u3d_train_r2(Z_t, cfg)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            per_template.append({"template_idx": t,
                                  "template": cmg.TEMPLATES[t],
                                  "error": str(exc)})
            continue
        disp, T_aligned = procrustes_disparity(res["T"], T_base)
        # Also report the disparity vs the *projected centroids* shift
        # (how much did the input target Z move when we dropped t?).
        ss_res = ((Z_full - Z_t) ** 2).sum()
        ss_tot = ((Z_full - Z_full.mean(0, keepdims=True)) ** 2).sum()
        z_drift = float(ss_res / max(ss_tot, 1e-12))
        print(f"  train_r2={res['train_r2']:+.4f}  iters={res['n_iters_run']}  "
              f"log_lam={res['final_log_lambda']:+.2f}  "
              f"procrustes_disp={disp:.4e}  Z_drift={z_drift:.4e}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        per_template.append({
            "template_idx": t,
            "template": cmg.TEMPLATES[t],
            "train_r2": res["train_r2"],
            "n_iters": res["n_iters_run"],
            "final_log_lambda": res["final_log_lambda"],
            "final_dT": res["final_dT"],
            "procrustes_disparity": float(disp),
            "z_drift_norm": z_drift,
        })

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": n_colors,
            "n_templates": N_TEMPLATES, "n_pcs": N_PCS,
            "d": D, "n_iters": N_ITERS, "lattice_per_side": 5,
            "seed": SEED,
        },
        "fixed_pca_evr_topK": evr.tolist(),
        "baseline": {
            "train_r2": base["train_r2"],
            "n_iters": base["n_iters_run"],
            "final_log_lambda": base["final_log_lambda"],
            "final_dT": base["final_dT"],
        },
        "per_template": per_template,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    ok = [r for r in per_template if "procrustes_disparity" in r]
    if not ok:
        print("[warn] no successful folds, skipping plot", flush=True)
        return 0
    ok_sorted = sorted(ok, key=lambda r: r["procrustes_disparity"], reverse=True)
    idxs = [r["template_idx"] for r in ok_sorted]
    disps = [r["procrustes_disparity"] for r in ok_sorted]
    r2s = [r["train_r2"] for r in ok_sorted]
    drifts = [r["z_drift_norm"] for r in ok_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                              gridspec_kw={"width_ratios": [3, 2]})

    x = np.arange(len(ok_sorted))
    bars = axes[0].barh(x, disps, color="#a04060", alpha=0.85)
    axes[0].set_yticks(x)
    labels = [f"[t{ti:02d}] {cmg.TEMPLATES[ti][:55]}..." for ti in idxs]
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Procrustes disparity vs full-28 U_3d  (0 = identical geom)")
    axes[0].set_title(f"LOTO jackknife: which template most distorts U_3d?\n"
                       f"baseline train R^2 = {base['train_r2']:+.4f} on {N_PCS} PCs")
    axes[0].grid(axis="x", linestyle=":", alpha=0.4)
    for xi, (d, ti) in enumerate(zip(disps, idxs)):
        axes[0].text(d, xi, f"  {d:.2e}", va="center", fontsize=6)

    sc = axes[1].scatter(drifts, disps, c=r2s, cmap="viridis",
                          s=40, edgecolor="k", linewidth=0.4)
    for ti, dr, ds in zip(idxs, drifts, disps):
        axes[1].annotate(f"t{ti}", (dr, ds), fontsize=6,
                          xytext=(3, 2), textcoords="offset points")
    axes[1].set_xlabel("Z_drift   = ||Z_full - Z_drop_t||_F^2 / ||Z_full - mean||_F^2")
    axes[1].set_ylabel("Procrustes disparity of fitted T")
    axes[1].set_title("Geometry distortion vs input-shift\n"
                       "(off-diagonal points = nonlinear leverage on U_3d)")
    axes[1].grid(linestyle=":", alpha=0.4)
    cb = plt.colorbar(sc, ax=axes[1])
    cb.set_label("LOTO train R^2", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
