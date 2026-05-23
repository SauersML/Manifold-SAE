"""auto_exp_11: U_3d basin sensitivity — PCA-init vs random-init across seeds.

Motivation
----------
The alternating Duchon fit (`fit_unsupervised_manifold`) defaults to a
PCA-3 initialization for the latent T. Open question (option jj from the
project ideas list): does the init choose the basin? That is, if we
instead initialize T from random uniform [0, 1]^3, do we converge to the
*same* manifold (up to gauge: rotation/reflection/scale) or to a
qualitatively different one with different train R^2 / different
geometry?

This is a load-bearing question for everything we have claimed about
"the discovered color manifold": if the basin is unique-ish, U_3d is a
real geometric object; if many basins exist, U_3d is just one fixed
point among many and Procrustes-disparity claims (auto_exp_09) need to
be re-read modulo seed.

Protocol (cheap, ~minutes on CPU, no server calls)
--------------------------------------------------
  1. Project 28-template centroids (n_colors ~ 949) onto a fixed top-64
     PCA basis (same recipe as auto_exp_09 so numbers are comparable).
  2. baseline: 1 PCA-init fit (deterministic).
  3. random-init: 8 fits with seeds 0..7, init_T ~ Uniform([0,1]^3).
  4. For each fit, record:
       - train R^2 in the fixed 64-PC target space
       - final log_lambda, iters to converge, final dT
       - per-iter (log_lambda, train_mse) trace
  5. Pairwise Procrustes disparity between every pair (PCA-init + 8
     random-inits = 9 fits → 36 pairs). Report:
       - mean / max disparity vs PCA-init basin
       - disparity matrix as a heatmap
       - within-cluster vs cross-cluster disparity (k-means on the
         pairwise disparity to detect multiple basins automatically)

Interpretation
--------------
  - All disparities < 1e-3 and all R^2 within 1e-3 → SAME basin, U_3d
    is essentially unique up to gauge. Strong claim.
  - Clear clustering (some pairs near zero, others orders of magnitude
    larger) → multiple basins; init matters; report basin frequencies.
  - PCA-init R^2 strictly dominates all random-inits → PCA-init is the
    best local optimum we know how to reach; random-init underfits.

NO Gaussian RBF. NO length_scale on Duchon. No server traffic.
Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_11_init_basin.{json,png}
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
OUT_JSON = OUT_DIR / "auto_exp_11_init_basin.json"
OUT_PNG = OUT_DIR / "auto_exp_11_init_basin.png"

N_TEMPLATES = 28
N_PCS = 64
D = 3
N_ITERS = 20
N_RAND_SEEDS = 8


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """Scale-free orthogonal Procrustes disparity (see auto_exp_09)."""
    A0 = A - A.mean(0, keepdims=True)
    B0 = B - B.mean(0, keepdims=True)
    nA = np.linalg.norm(A0)
    nB = np.linalg.norm(B0)
    if nA < 1e-12 or nB < 1e-12:
        return float("nan")
    A0n = A0 / nA
    B0n = B0 / nB
    U, _, Vt = np.linalg.svd(B0n.T @ A0n, full_matrices=False)
    R = Vt.T @ U.T
    M = A0 @ R
    s = float((B0 * M).sum() / max((M * M).sum(), 1e-12))
    A_aligned = s * M + B.mean(0, keepdims=True)
    return float(((B - A_aligned) ** 2).sum() / max((B0 ** 2).sum(), 1e-12))


def centroids_full(X: np.ndarray, t_idx: np.ndarray, c_idx: np.ndarray,
                    n_colors: int) -> np.ndarray:
    out = np.zeros((n_colors, X.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        rows = (c_idx == ci)
        out[ci] = X[rows].mean(0)
    return out


def fit_with_init(Z: np.ndarray, cfg: cmg.Config,
                   init_T: np.ndarray | None) -> dict:
    fit = cmg.fit_unsupervised_manifold(
        Z, D, cfg, n_iters=N_ITERS, init_T=init_T, verbose=False,
    )
    Phi, _ = cmg.duchon_basis_radial(fit["T"], fit["centers"])
    Z_hat = Phi @ fit["B"]
    ss_res = ((Z - Z_hat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0, keepdims=True)) ** 2).sum()
    train_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {
        "T": fit["T"],
        "train_r2": train_r2,
        "final_log_lambda": float(fit["log_lambda"]),
        "n_iters_run": len(fit["history"]),
        "final_dT": float(fit["history"][-1]["dT"]) if fit["history"] else float("nan"),
        "trace_train_mse": [float(h["train_mse"]) for h in fit["history"]],
        "trace_log_lambda": [float(h["log_lambda"]) for h in fit["history"]],
        "trace_dT": [float(h["dT"]) for h in fit["history"]],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, _ = X.shape
    assert N % N_TEMPLATES == 0
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    centroids = centroids_full(X, t_idx, c_idx, n_colors)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cc = (centroids - mu) / sigma
    Cc = Cc - Cc.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    Z = ((centroids - mu) / sigma) @ V_topK.T
    print(f"[pca] fixed top-{N_PCS} EVR sum = {evr.sum():.3f}  Z={Z.shape}",
          flush=True)

    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=5,
                     lattice_per_side=5, init_log_lambda=0.0,
                     output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    runs: list[dict] = []

    # ---- PCA-init baseline ----
    print("\n[run 0] PCA-init baseline", flush=True)
    t0 = time.time()
    base = fit_with_init(Z, cfg, init_T=None)
    print(f"  train_r2={base['train_r2']:+.4f}  iters={base['n_iters_run']}  "
          f"log_lam={base['final_log_lambda']:+.2f}  dT={base['final_dT']:.2e}  "
          f"({time.time()-t0:.1f}s)", flush=True)
    runs.append({"label": "pca_init", "seed": None, **base})

    # ---- Random-init runs ----
    for seed in range(N_RAND_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        init_T = rng.uniform(0.0, 1.0, size=(Z.shape[0], D))
        print(f"\n[run {seed+1}] random-init seed={seed}", flush=True)
        t0 = time.time()
        try:
            res = fit_with_init(Z, cfg, init_T=init_T)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            runs.append({"label": f"rand_{seed}", "seed": seed, "error": str(exc)})
            continue
        print(f"  train_r2={res['train_r2']:+.4f}  iters={res['n_iters_run']}  "
              f"log_lam={res['final_log_lambda']:+.2f}  dT={res['final_dT']:.2e}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        runs.append({"label": f"rand_{seed}", "seed": seed, **res})

    ok = [r for r in runs if "T" in r]
    Ts = [np.asarray(r["T"]) for r in ok]
    labels = [r["label"] for r in ok]
    n = len(Ts)

    # ---- Pairwise Procrustes disparity matrix ----
    D_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            D_mat[i, j] = procrustes_disparity(Ts[i], Ts[j])
    print("\n[procrustes] pairwise disparity matrix:", flush=True)
    print("           " + "  ".join(f"{l[:7]:>7}" for l in labels), flush=True)
    for i, li in enumerate(labels):
        row = "  ".join(f"{D_mat[i,j]:7.1e}" for j in range(n))
        print(f"  {li[:9]:>9}  {row}", flush=True)

    # vs PCA-init
    vs_pca = D_mat[0, 1:]
    r2s = np.array([r["train_r2"] for r in ok])
    r2_pca = r2s[0]
    r2_rand = r2s[1:]
    print(f"\n[vs PCA-init]  disparity  mean={vs_pca.mean():.3e}  "
          f"max={vs_pca.max():.3e}  min={vs_pca.min():.3e}", flush=True)
    print(f"[R^2]  pca={r2_pca:+.4f}  rand mean={r2_rand.mean():+.4f}  "
          f"min={r2_rand.min():+.4f}  max={r2_rand.max():+.4f}", flush=True)

    # crude basin detection: cluster rows of D_mat
    off = D_mat[np.triu_indices(n, k=1)]
    med = float(np.median(off))
    same_basin = (D_mat < med).astype(int)
    # connected components
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i+1, n):
            if same_basin[i, j]:
                union(i, j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    n_basins = len(comps)
    print(f"\n[basins]  n_components @ disparity<median({med:.2e}) = {n_basins}",
          flush=True)
    for root, members in comps.items():
        print(f"    basin: {[labels[m] for m in members]}", flush=True)

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": int(n_colors),
            "n_templates": N_TEMPLATES, "n_pcs": N_PCS, "d": D,
            "n_iters": N_ITERS, "n_rand_seeds": N_RAND_SEEDS,
        },
        "fixed_pca_evr_topK_sum": float(evr.sum()),
        "runs": [{k: v for k, v in r.items() if k != "T"} for r in runs],
        "labels": labels,
        "disparity_matrix": D_mat.tolist(),
        "vs_pca_init": {
            "mean": float(vs_pca.mean()),
            "max": float(vs_pca.max()),
            "min": float(vs_pca.min()),
            "per_seed": vs_pca.tolist(),
        },
        "r2": {
            "pca_init": float(r2_pca),
            "rand_mean": float(r2_rand.mean()),
            "rand_min": float(r2_rand.min()),
            "rand_max": float(r2_rand.max()),
            "rand_per_seed": r2_rand.tolist(),
        },
        "basin_detection": {
            "threshold_median_offdiag": med,
            "n_basins": n_basins,
            "basins": [[labels[m] for m in members] for members in comps.values()],
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              gridspec_kw={"width_ratios": [3, 2, 3]})

    # (1) disparity heatmap (log scale)
    M = D_mat.copy()
    np.fill_diagonal(M, np.nan)
    im = axes[0].imshow(np.log10(M + 1e-12), cmap="magma", aspect="auto")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title(f"log10 pairwise Procrustes disparity\n"
                       f"({n_basins} basin(s) at thr={med:.1e})")
    cb = plt.colorbar(im, ax=axes[0])
    cb.set_label("log10 disparity", fontsize=8)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            axes[0].text(j, i, f"{D_mat[i,j]:.1e}", ha="center", va="center",
                          fontsize=5,
                          color="white" if np.log10(D_mat[i,j]+1e-12) < -2 else "black")

    # (2) R^2 bars
    colors = ["#3060a0"] + ["#a04060"] * (n - 1)
    axes[1].bar(range(n), r2s, color=colors, alpha=0.85)
    axes[1].axhline(r2_pca, color="#3060a0", ls="--", lw=1,
                     label=f"PCA-init R^2 = {r2_pca:+.4f}")
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("train R^2 on 64 PCs")
    axes[1].set_title("Fit quality per init")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].legend(fontsize=8, loc="lower right")
    for i, v in enumerate(r2s):
        axes[1].text(i, v, f" {v:+.4f}", ha="center", va="bottom",
                      fontsize=6, rotation=90)

    # (3) train_mse traces (alternation curves)
    for r in ok:
        ls = "-" if r["label"] == "pca_init" else ":"
        lw = 2.0 if r["label"] == "pca_init" else 1.0
        col = "#3060a0" if r["label"] == "pca_init" else None
        axes[2].plot(r["trace_train_mse"], ls=ls, lw=lw, color=col,
                      label=r["label"], alpha=0.85)
    axes[2].set_xlabel("alternation iter")
    axes[2].set_ylabel("train_mse")
    axes[2].set_yscale("log")
    axes[2].set_title("Alternation convergence per init")
    axes[2].grid(linestyle=":", alpha=0.4)
    axes[2].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
