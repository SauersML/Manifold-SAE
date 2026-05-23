"""auto_exp_12: U_3d alternation N_iters sweep — when has it converged?

Motivation
----------
The alternating Duchon fit (`fit_unsupervised_manifold`) defaults to
N_ITERS=20 across the auto_exp_* family (see auto_exp_09, auto_exp_11).
Open question (option hh): does N_iters in {5, 10, 20, 50} change the
final manifold, or has the fit fully converged by iter ~10? If iter 5
is already at-asymptote, downstream experiments can be 4x cheaper. If
iter 50 is meaningfully different from iter 20, every previous claim
that quoted train R^2 at iter 20 is underfit.

Protocol (cheap, fully offline, no server traffic)
--------------------------------------------------
  1. Project 28-template centroids (n_colors ~ 949) onto a fixed top-64
     PCA basis. Identical recipe to auto_exp_09 and auto_exp_11 so the
     comparison is clean.
  2. Run the alternating Duchon fit with PCA-init for n_iters in
     {5, 10, 20, 50}. PCA-init is deterministic, so a single fit per
     setting is sufficient (no seed averaging needed).
  3. For each setting, record:
       - final train R^2 in the fixed 64-PC target space
       - final log_lambda
       - final dT  (latent-step norm at last iteration)
       - per-iter (train_mse, log_lambda, dT) trace
  4. Procrustes-align T_(5), T_(10), T_(50) to T_(20) (our default).
     Report disparity → does the latent geometry actually move after
     iter 5?

Interpretation
--------------
  - R^2(5) ≈ R^2(50) and disparity(T_5, T_20) < 1e-3 → iter 5 is
    already at asymptote; the default of 20 is wasteful, future runs
    can use 10.
  - R^2(50) >> R^2(20) → we are systematically underfit and should
    raise the default. Re-read all previous auto_exp_* R^2 numbers.
  - R^2 plateaus but T keeps moving → gauge drift; the geometry is
    not unique even at the same loss value, and convergence-by-loss
    is not convergence-by-geometry.

NO Gaussian RBF. NO length_scale on Duchon. No server traffic.
Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_12_niters.{json,png}
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
OUT_JSON = OUT_DIR / "auto_exp_12_niters.json"
OUT_PNG = OUT_DIR / "auto_exp_12_niters.png"

N_TEMPLATES = 28
N_PCS = 64
D = 3
NITER_GRID = [5, 10, 20, 50]
REF_NITER = 20  # all Procrustes comparisons target this


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """Scale-free orthogonal Procrustes disparity (see auto_exp_09/11)."""
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


def centroids_full(X: np.ndarray, c_idx: np.ndarray, n_colors: int) -> np.ndarray:
    out = np.zeros((n_colors, X.shape[1]), dtype=np.float64)
    for ci in range(n_colors):
        out[ci] = X[c_idx == ci].mean(0)
    return out


def fit_with_niters(Z: np.ndarray, cfg: cmg.Config, n_iters: int) -> dict:
    fit = cmg.fit_unsupervised_manifold(
        Z, D, cfg, n_iters=n_iters, init_T=None, verbose=False,
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
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    centroids = centroids_full(X, c_idx, n_colors)
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
    for n_iters in NITER_GRID:
        print(f"\n[run] n_iters={n_iters}", flush=True)
        t0 = time.time()
        res = fit_with_niters(Z, cfg, n_iters=n_iters)
        dt = time.time() - t0
        print(f"  train_r2={res['train_r2']:+.4f}  iters_run={res['n_iters_run']}  "
              f"log_lam={res['final_log_lambda']:+.2f}  dT={res['final_dT']:.2e}  "
              f"({dt:.1f}s)", flush=True)
        res["wall_seconds"] = dt
        res["n_iters_requested"] = n_iters
        runs.append(res)

    # locate ref
    ref_idx = NITER_GRID.index(REF_NITER)
    T_ref = np.asarray(runs[ref_idx]["T"])
    r2_ref = runs[ref_idx]["train_r2"]

    print(f"\n[procrustes vs N_iters={REF_NITER}]", flush=True)
    disparities: list[float] = []
    for n_iters, r in zip(NITER_GRID, runs):
        if n_iters == REF_NITER:
            disparities.append(0.0)
            continue
        d = procrustes_disparity(np.asarray(r["T"]), T_ref)
        disparities.append(d)
        print(f"  N_iters={n_iters:>3}: disparity={d:.3e}  "
              f"dR2={r['train_r2'] - r2_ref:+.4e}", flush=True)

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": int(n_colors),
            "n_templates": N_TEMPLATES, "n_pcs": N_PCS, "d": D,
            "niter_grid": NITER_GRID, "ref_niter": REF_NITER,
        },
        "fixed_pca_evr_topK_sum": float(evr.sum()),
        "runs": [
            {k: v for k, v in r.items() if k != "T"} for r in runs
        ],
        "procrustes_vs_ref": {
            "ref_niter": REF_NITER,
            "ref_train_r2": r2_ref,
            "per_niter": [
                {"n_iters": ni, "disparity": disparities[i],
                 "dR2_vs_ref": runs[i]["train_r2"] - r2_ref}
                for i, ni in enumerate(NITER_GRID)
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (1) train_mse traces; longest trace dominates x axis
    cmap = plt.get_cmap("viridis")
    colors_iter = {ni: cmap(i / max(1, len(NITER_GRID) - 1))
                    for i, ni in enumerate(NITER_GRID)}
    for ni, r in zip(NITER_GRID, runs):
        axes[0].plot(r["trace_train_mse"], "-o", ms=3,
                      color=colors_iter[ni], lw=1.4,
                      label=f"N_iters={ni}  R^2={r['train_r2']:+.4f}")
    axes[0].set_xlabel("alternation iter")
    axes[0].set_ylabel("train_mse")
    axes[0].set_yscale("log")
    axes[0].set_title("Convergence of train_mse")
    axes[0].grid(linestyle=":", alpha=0.4)
    axes[0].legend(fontsize=8)

    # (2) final R^2 + dT bars (twin axes)
    nis = [str(ni) for ni in NITER_GRID]
    r2s = [r["train_r2"] for r in runs]
    dTs = [r["final_dT"] for r in runs]
    x = np.arange(len(NITER_GRID))
    bw = 0.4
    bars1 = axes[1].bar(x - bw / 2, r2s, bw, color="#3060a0",
                         alpha=0.85, label="train R^2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nis)
    axes[1].set_xlabel("N_iters requested")
    axes[1].set_ylabel("train R^2", color="#3060a0")
    axes[1].tick_params(axis="y", labelcolor="#3060a0")
    for b, v in zip(bars1, r2s):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f" {v:+.4f}",
                      ha="center", va="bottom", fontsize=7, rotation=90,
                      color="#3060a0")
    ax1b = axes[1].twinx()
    bars2 = ax1b.bar(x + bw / 2, dTs, bw, color="#a04060",
                      alpha=0.85, label="final dT")
    ax1b.set_yscale("log")
    ax1b.set_ylabel("final dT (last-step latent move)", color="#a04060")
    ax1b.tick_params(axis="y", labelcolor="#a04060")
    axes[1].set_title("Fit quality vs convergence-witness")
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    # (3) Procrustes disparity vs ref
    axes[2].bar(x, disparities, color="#608030", alpha=0.85)
    axes[2].set_yscale("symlog", linthresh=1e-6)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(nis)
    axes[2].set_xlabel(f"N_iters requested (ref={REF_NITER})")
    axes[2].set_ylabel(f"Procrustes disparity vs T(N_iters={REF_NITER})")
    axes[2].set_title("Latent geometry shift vs ref")
    axes[2].grid(axis="y", linestyle=":", alpha=0.4)
    for xi, d in zip(x, disparities):
        axes[2].text(xi, d, f" {d:.1e}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
