"""auto_exp_39: d_aux sweep extending auto_exp_38.

auto_exp_38 found that at d_aux=6 (3 HSV-supervised + 3 free), free axes
unsupervisedly recover name-semantic structure (monoword, mod_count,
template_sigma). Free-axis 4 hit |corr|=0.67 with mod_count.

QUESTION: how does this generalize as d_free changes?
  - d_aux=5 (3 sup + 2 free): does name-semantic squeeze into 2 axes?
  - d_aux=8 (3 sup + 5 free): extras = MORE structure or just noise?
  - d_aux=10 (3 sup + 7 free): same, stronger test.

For each, fit ONCE (no CV) and report:
  - Mean R^2 on supervised axes (expect ~0.69)
  - Max |corr| of each free axis vs {monoword, mod_count, template_sigma}
  - name-active count: free axes with max |corr| > 0.4 with any name feature
  - junk count: free axes with max |corr| < 0.2 across all targets

Hypothesis: name-active saturates ~3 (the latent dimensionality of the
name-semantic block); extras are junk near isotropic noise.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (  # reuse pipeline pieces
    X_PATH, N_TEMPLATES, K_PCS, AUX_LABELS_HSV, AUX_LABELS_NAME,
    per_color_stats_mmap, load_xkcd_rgb, hsv_from_rgb, name_features,
    fit_aux_supervised_hsv, abs_corr_matrix,
    ARD_PRUNE_TAU,
)

ROOT = Path("/Users/user/Manifold-SAE")
OUT_NPZ = ROOT / "runs" / "auto_exp_39_results.npz"

D_AUX_SUP = 3
D_FREE_LIST = [2, 5, 7]  # giving d_aux = 5, 8, 10
NAME_ACTIVE_THRESH = 0.40
JUNK_THRESH = 0.20


def fit_free_axes_pca_d(T0, W_sup, d_free):
    """Top-d_free PCs of residual after projecting out span(W_sup)."""
    Tc = T0 - T0.mean(0, keepdims=True)
    Q, _ = np.linalg.qr(W_sup)
    P_perp = np.eye(W_sup.shape[0]) - Q @ Q.T
    Tc_perp = Tc @ P_perp
    U_svd, S_svd, Vt_svd = np.linalg.svd(Tc_perp, full_matrices=False)
    W_free = Vt_svd[:d_free].T
    T_free = Tc @ W_free
    eig = (S_svd ** 2)[:d_free] / max(Tc_perp.shape[0] - 1, 1)
    kept = int(np.count_nonzero(eig > ARD_PRUNE_TAU * eig.max()))
    return {"T_free": T_free, "W_free": W_free,
            "eig_free": eig, "kept_free": kept}


def main():
    t_start = time.time()
    print("[auto_exp_39] d_aux sweep over free-axis count")

    # ---- Load + featurize ONCE (shared across the sweep)
    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c = T0.shape[0]
    print(f"[centroids] T0={T0.shape}")
    names, rgb = load_xkcd_rgb(n_c)
    hsv = hsv_from_rgb(rgb)
    namef = name_features(names, tsig)  # held out
    print(f"[aux] hsv={hsv.shape}; namef={namef.shape}")

    # ---- Fit supervised HSV ONCE (shared across all d_free)
    sup = fit_aux_supervised_hsv(T0, hsv)
    r2_hsv = sup["r2_hsv"]
    mean_r2_sup = float(r2_hsv.mean())
    print(f"[fit] supervised HSV R^2: hue={r2_hsv[0]:.3f} sat={r2_hsv[1]:.3f} "
          f"val={r2_hsv[2]:.3f}  mean={mean_r2_sup:.3f}")

    # ---- Sweep
    results = []
    corr_archives = {}
    for d_free in D_FREE_LIST:
        d_aux = D_AUX_SUP + d_free
        free = fit_free_axes_pca_d(T0, sup["W_sup"], d_free)
        T_all = np.concatenate([sup["T_sup"], free["T_free"]], axis=1)

        corr_hsv = abs_corr_matrix(T_all, hsv)        # (d_aux, 3)
        corr_name = abs_corr_matrix(T_all, namef)     # (d_aux, 3)
        # Free-axis stats
        free_idx = list(range(D_AUX_SUP, d_aux))
        free_corr_name = corr_name[free_idx]          # (d_free, 3)
        free_corr_hsv = corr_hsv[free_idx]            # (d_free, 3) -- diagnostic

        per_axis_max_name = free_corr_name.max(axis=1)     # (d_free,)
        per_axis_best_name = [AUX_LABELS_NAME[int(i)]
                              for i in free_corr_name.argmax(axis=1)]

        # Junk benchmarks against ALL targets (HSV + name) -- ensure they really
        # are noise, not just HSV-aligned (we projected out W_sup but check).
        per_axis_max_all = np.maximum(
            free_corr_name.max(axis=1), free_corr_hsv.max(axis=1)
        )

        n_name_active = int((per_axis_max_name > NAME_ACTIVE_THRESH).sum())
        n_junk = int((per_axis_max_all < JUNK_THRESH).sum())

        # Free-axes covariance isotropy
        Tfc = free["T_free"] - free["T_free"].mean(0, keepdims=True)
        cov_free = (Tfc.T @ Tfc) / Tfc.shape[0]
        eig_cov = np.sort(np.linalg.eigvalsh(cov_free))[::-1]
        iso_ratio = float(eig_cov[0] / max(eig_cov[-1], 1e-12))

        results.append({
            "d_aux": d_aux,
            "d_free": d_free,
            "mean_R2_sup": mean_r2_sup,
            "R2_hue": float(r2_hsv[0]),
            "free_corr_name": free_corr_name,
            "free_corr_hsv": free_corr_hsv,
            "per_axis_max_name": per_axis_max_name,
            "per_axis_best_name": per_axis_best_name,
            "per_axis_max_all": per_axis_max_all,
            "n_name_active": n_name_active,
            "n_junk": n_junk,
            "iso_ratio_free_cov": iso_ratio,
            "eig_free_residual": free["eig_free"],
        })
        corr_archives[f"d{d_aux}_corr_name"] = free_corr_name
        corr_archives[f"d{d_aux}_corr_hsv"] = free_corr_hsv

        print(f"[d_aux={d_aux} d_free={d_free}] name-active={n_name_active}  "
              f"junk={n_junk}  iso_ratio={iso_ratio:.2f}")
        print(f"    per-axis max-name-corr = {np.round(per_axis_max_name, 3)}")
        print(f"    per-axis best-name     = {per_axis_best_name}")

    # ---- Save
    np.savez(OUT_NPZ,
             d_aux_list=np.array([r["d_aux"] for r in results]),
             d_free_list=np.array([r["d_free"] for r in results]),
             mean_R2_sup=np.array([r["mean_R2_sup"] for r in results]),
             n_name_active=np.array([r["n_name_active"] for r in results]),
             n_junk=np.array([r["n_junk"] for r in results]),
             iso_ratios=np.array([r["iso_ratio_free_cov"] for r in results]),
             **corr_archives)
    print(f"[npz] saved {OUT_NPZ}")

    # ---- Final stdout table
    print()
    print("=" * 78)
    print(" auto_exp_39 RESULTS: d_aux sweep (HSV-supervised, name-features held out)")
    print("=" * 78)
    print(f" supervised HSV (shared): hue={r2_hsv[0]:.3f} sat={r2_hsv[1]:.3f} "
          f"val={r2_hsv[2]:.3f}  mean={mean_r2_sup:.3f}")
    print()
    header = f"{'d_aux':>6} {'d_free':>7} {'name-active':>12} {'junk':>6} " \
             f"{'iso(max/min)':>14} {'best free corr':>20}"
    print(header)
    print("-" * len(header))
    for r in results:
        best = float(r["per_axis_max_name"].max())
        best_ax = int(r["per_axis_max_name"].argmax())
        best_feat = r["per_axis_best_name"][best_ax]
        best_str = f"{best:.3f} ({best_feat})"
        print(f"{r['d_aux']:>6} {r['d_free']:>7} {r['n_name_active']:>12} "
              f"{r['n_junk']:>6} {r['iso_ratio_free_cov']:>14.2f} "
              f"{best_str:>20}")
    print()
    print(" per-axis detail:")
    for r in results:
        print(f"  d_aux={r['d_aux']} d_free={r['d_free']}")
        for i in range(r["d_free"]):
            mn = float(r["per_axis_max_name"][i])
            mh = float(r["free_corr_hsv"][i].max())
            corrs = ", ".join(
                f"{AUX_LABELS_NAME[k]}={r['free_corr_name'][i,k]:.2f}"
                for k in range(3)
            )
            print(f"    free axis {i}: best-name={mn:.3f} ({r['per_axis_best_name'][i]}); "
                  f"max-HSV-leak={mh:.3f}; [{corrs}]")
    print("=" * 78)
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
