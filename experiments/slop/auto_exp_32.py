"""auto_exp_32: NuclearNormPenalty rank-identification on cogito L40 centroids.

Hypothesis (from memory `project_ard_gauge_fix_doesnt_help_cogito.md` +
`project_cogito_color_manifold_decomposition`): cogito's L40 color signal is
intrinsically ~6-dim (3 perceptual + 3 name-semantic). A nuclear-norm
penalty on the centroid matrix (in the K_PC-dim PCA-projected space) should
identify an effective rank <= 6, *stably* across K_PC truncation.

Why nuclear norm: convex surrogate for rank. Penalizing sum(σ) shrinks the
small singular values to zero (soft-threshold). The number of σ that survive
above a fixed fraction of σ_max is our "effective rank" estimate.

Fallback path: installed gamfit 0.1.112 lacks NuclearNormPenalty. We use
the closed-form proximal operator on the column-centred centroid matrix
directly via SVD soft-thresholding. (Equivalent to the GAM-side smoothed-L¹
on singular values in the w → small ε limit.)

CV scheme: 5-fold over the ~949 colors. For each fold:
  - SVD-soft-threshold Y_train -> Y_train_hat with threshold τ = w
  - V_r = right singular vectors with σ_i(Y_train_hat) > 0
  - test MSE = ||Y_test - Y_test V_r V_r^T||_F^2 / ||Y_test||_F^2
This is a proper subspace-recovery CV: the discovered low-rank subspace
must capture held-out colors.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
from _pca_basis import _per_color_centroids, load_pc_basis  # noqa: E402

import gamfit  # noqa: E402

# ------------------------------- config ----------------------------------- #
RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RUN_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = RUN_DIR / "auto_exp_32.png"
OUT_JSON = RUN_DIR / "auto_exp_32.json"

HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
K_PC_SWEEP = [16, 32, 64]
W_SWEEP = [0.1, 1.0, 10.0, 100.0]
# Supplementary adaptive grid: w = α · σ_max(Y), α covering [0, 1].
# Needed because the user's absolute grid is small relative to cogito's
# σ_max (~900); only the adaptive sweep can actually drive rank → 6.
ALPHA_SWEEP = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]
SMOOTHING_EPS = 1e-6
RANK_THRESH = 0.01            # σ_i > 0.01 σ_max counts as "effective"
N_FOLDS = 5
SEED = 0

# -------------------------- nuclear-norm prox ----------------------------- #
def _soft_threshold_svd(Y: np.ndarray, tau: float, eps: float = SMOOTHING_EPS):
    """Smoothed soft-threshold of singular values.

    σ_i^new = max(σ_i - tau, 0) in the strict (eps=0) case.
    With smoothing, use σ_i * max(0, 1 - tau / sqrt(σ_i^2 + eps^2)),
    matching the closed-form gradient of the smoothed nuclear norm.
    """
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    shrink = np.maximum(0.0, 1.0 - tau / np.sqrt(s * s + eps * eps))
    s_new = s * shrink
    Y_hat = (U * s_new) @ Vt
    return Y_hat, s_new, U, Vt


def _effective_rank(s: np.ndarray, thresh: float = RANK_THRESH) -> int:
    if s.size == 0 or s.max() <= 0:
        return 0
    return int(np.sum(s > thresh * s.max()))


# --------------------------- CV on one K_PC ------------------------------- #
def cv_one_kpc(Y: np.ndarray, w: float, n_folds: int, seed: int) -> dict:
    """5-fold-by-row CV of nuclear-norm-thresholded subspace recovery."""
    n = Y.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)
    fold_mse, fold_rank = [], []
    for fi in range(n_folds):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[fj] for fj in range(n_folds) if fj != fi])
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        Y_hat, s_new, _U, Vt = _soft_threshold_svd(Y_tr, tau=w)
        r = _effective_rank(s_new)
        if r == 0:
            # degenerate: subspace is empty, predict zero
            mse = float(np.mean(Y_te ** 2))
        else:
            V_r = Vt[:r]                              # (r, K_PC)
            proj = Y_te @ V_r.T @ V_r                 # (m, K_PC)
            num = float(np.mean((Y_te - proj) ** 2))
            den = float(np.mean(Y_te ** 2)) + 1e-12
            mse = num / den
        fold_mse.append(mse)
        fold_rank.append(r)
    return {
        "mean_mse": float(np.mean(fold_mse)),
        "std_mse": float(np.std(fold_mse)),
        "mean_rank": float(np.mean(fold_rank)),
        "fold_ranks": fold_rank,
    }


# --------------------------------- main ----------------------------------- #
def main() -> None:
    t0 = time.time()
    # 1) Build centroids from mmap'd harvest. _per_color_centroids needs to
    #    read full rows; with mmap='r' the OS pages them in / out so peak RAM
    #    is ~ centroids array (949 * 7168 * 8 ≈ 54 MB) plus working buffers.
    harvest = np.load(HARVEST, mmap_mode="r")
    centroids = _per_color_centroids(harvest)  # (n_colors, 7168), float64
    n_colors = centroids.shape[0]
    print(f"centroids: {centroids.shape}  RAM~{centroids.nbytes/1e6:.1f} MB")

    # 2) Canonical K=64 basis; truncate for smaller K_PC.
    basis64 = load_pc_basis(K=64)
    Vt64 = basis64["Vt"]                       # (64, D)
    mu = basis64["mu"]; sigma = basis64["sigma"]
    Xn_full = (centroids - mu) / sigma         # standardized centroids
    Y64 = Xn_full @ Vt64.T                     # (n_colors, 64) PC scores

    # 3) Per-K_PC nuclear-norm sweep + CV.
    #    Run BOTH the user-specified absolute w grid AND a σ_max-relative
    #    α grid (since cogito's σ_max ~ 900 ≫ w_max=100, the absolute grid
    #    alone can't actually shrink rank below K_PC).
    per_k = {}
    for K_PC in K_PC_SWEEP:
        Y = Y64[:, :K_PC].copy()               # truncate
        s_raw = np.linalg.svd(Y, compute_uv=False)
        sigma_max = float(s_raw[0])

        absolute_ws = list(W_SWEEP)
        adaptive_ws = [float(a * sigma_max) for a in ALPHA_SWEEP]
        all_ws = sorted(set(absolute_ws + adaptive_ws))

        sweeps = []
        for w in all_ws:
            _, s_full, _, _ = _soft_threshold_svd(Y, tau=w)
            cv = cv_one_kpc(Y, w=w, n_folds=N_FOLDS, seed=SEED)
            sweeps.append({
                "w": w,
                "w_over_sigma_max": w / sigma_max,
                "cv_mse": cv["mean_mse"],
                "cv_mse_std": cv["std_mse"],
                "full_eff_rank": _effective_rank(s_full),
                "cv_mean_rank": cv["mean_rank"],
                "singular_values_top10": [float(x) for x in s_full[:10]],
                "from_user_grid": w in absolute_ws,
            })
        # Best-w selection via 1-SE rule: sparsest model whose CV MSE is within
        # 1 SE of the global minimum. This is the standard convex-penalty
        # criterion for rank selection (otherwise CV trivially picks w → 0
        # whenever K_PC ≤ N because V_r = I gives zero held-out MSE).
        min_cv = min(r["cv_mse"] for r in sweeps)
        min_se = next(r["cv_mse_std"] for r in sweeps if r["cv_mse"] == min_cv)
        tol = min_cv + min_se
        eligible = [r for r in sweeps if r["cv_mse"] <= tol]
        best = max(eligible, key=lambda r: r["w"])  # sparsest <=> largest w

        _, s_best, _, _ = _soft_threshold_svd(Y, tau=best["w"])
        per_k[K_PC] = {
            "sigma_max": sigma_max,
            "sweeps": sweeps,
            "best_w": best["w"],
            "best_w_over_sigma_max": best["w_over_sigma_max"],
            "best_w_selection_rule": "1-SE (sparsest within 1 SE of CV min)",
            "effective_rank": _effective_rank(s_best),
            "singular_values_top10": [float(x) for x in s_best[:10]],
            "raw_singular_values": [float(x) for x in s_raw[:min(K_PC, 20)]],
            "spectrum_at_best_w": [float(x) for x in s_best],
        }
        print(f"K_PC={K_PC:>2d}  σ_max={sigma_max:.1f}  best_w={best['w']:.2f}  "
              f"eff_rank={per_k[K_PC]['effective_rank']}  "
              f"raw_top6={[round(x,1) for x in s_raw[:6]]}")

    # 4) Hypothesis verdicts
    eff = {k: per_k[k]["effective_rank"] for k in K_PC_SWEEP}
    verdicts = {
        "a_kpc16_rank_le_6": bool(eff[16] <= 6),
        "b_kpc32_rank_le_6": bool(eff[32] <= 6),
        "c_kpc64_rank_le_6": bool(eff[64] <= 6),
        "d_stable_within_pm2": bool(max(eff.values()) - min(eff.values()) <= 2),
    }
    runtime = time.time() - t0

    # 5) Plot 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (a) singular spectra at best-w per K_PC
    ax = axes[0]
    cmap = {16: "#1f77b4", 32: "#2ca02c", 64: "#d62728"}
    for K_PC in K_PC_SWEEP:
        s = np.asarray(per_k[K_PC]["spectrum_at_best_w"])
        ax.plot(np.arange(1, len(s) + 1), s, "o-", lw=1.6, ms=4,
                color=cmap[K_PC],
                label=f"K_PC={K_PC}  best_w={per_k[K_PC]['best_w']}  r_eff={eff[K_PC]}")
        thr = RANK_THRESH * s.max()
        ax.axhline(thr, color=cmap[K_PC], lw=0.6, ls=":", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("σ_i (post soft-threshold)")
    ax.set_title("Singular spectrum @ best-w per K_PC\n(dotted = 1% σ_max effective-rank threshold)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) effective-rank bar chart
    ax = axes[1]
    xs = np.arange(len(K_PC_SWEEP))
    bars = ax.bar(xs, [eff[k] for k in K_PC_SWEEP],
                  color=[cmap[k] for k in K_PC_SWEEP], edgecolor="black")
    ax.axhline(6, color="k", lw=1.2, ls="--", label="hypothesis: rank ≤ 6")
    ax.set_xticks(xs); ax.set_xticklabels([f"K_PC={k}" for k in K_PC_SWEEP])
    ax.set_ylabel("effective rank  (#σ > 1% σ_max)")
    ax.set_title("Identified intrinsic rank vs PCA truncation")
    for b, k in zip(bars, K_PC_SWEEP):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                f"{eff[k]}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(eff.values()), 8) * 1.25)

    # (c) CV-MSE vs w
    ax = axes[2]
    for K_PC in K_PC_SWEEP:
        ws = [r["w"] for r in per_k[K_PC]["sweeps"]]
        ms = [r["cv_mse"] for r in per_k[K_PC]["sweeps"]]
        es = [r["cv_mse_std"] for r in per_k[K_PC]["sweeps"]]
        ax.errorbar(ws, ms, yerr=es, fmt="o-", lw=1.5, capsize=3,
                    color=cmap[K_PC], label=f"K_PC={K_PC}")
        bw = per_k[K_PC]["best_w"]
        bm = next(r["cv_mse"] for r in per_k[K_PC]["sweeps"] if r["w"] == bw)
        ax.plot([bw], [bm], "*", ms=14, color=cmap[K_PC], mec="black", mew=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("nuclear-norm weight w")
    ax.set_ylabel("CV held-out MSE (normalized)")
    ax.set_title("5-fold CV: subspace-recovery error vs w\n(★ = best-w)")
    ax.legend(); ax.grid(alpha=0.3)

    verdict_line = "  ".join(
        f"{k}={'PASS' if v else 'FAIL'}" for k, v in verdicts.items()
    )
    fig.suptitle("auto_exp_32: NuclearNormPenalty rank identification on cogito L40 centroids\n"
                 f"verdicts:  {verdict_line}   |   path=fallback_python_svd",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"saved {OUT_PNG}")

    # 6) JSON dump
    payload = {
        "experiment": "auto_exp_32",
        "gamfit_version": gamfit.__version__,
        "path_taken": "fallback_python_svd",
        "reason_for_fallback": "gamfit 0.1.112 lacks NuclearNormPenalty (planned v0.1.121)",
        "n_colors": int(n_colors),
        "k_pc_sweep": K_PC_SWEEP,
        "w_sweep": W_SWEEP,
        "smoothing_eps": SMOOTHING_EPS,
        "rank_threshold_frac_of_sigma_max": RANK_THRESH,
        "n_folds": N_FOLDS,
        "per_k_pc": {
            str(k): {
                "best_w": per_k[k]["best_w"],
                "effective_rank": int(per_k[k]["effective_rank"]),
                "singular_values_top10": per_k[k]["singular_values_top10"],
                "raw_singular_values_top20": per_k[k]["raw_singular_values"],
                "cv_sweeps": per_k[k]["sweeps"],
            } for k in K_PC_SWEEP
        },
        "hypothesis_verdicts": verdicts,
        "runtime_seconds": float(runtime),
        "prediction_slot_for_v0_1_121_retest": {
            "expected_effective_rank_kpc16": 6,
            "expected_effective_rank_kpc32": 6,
            "expected_effective_rank_kpc64": 6,
            "note": ("Re-run when gamfit ships NuclearNormPenalty; compare effective_rank "
                     "vs the fallback-SVD numbers above. Divergence >1 means the smoothed "
                     "GAM-side penalty differs materially from the closed-form prox."),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"saved {OUT_JSON}")
    print(f"runtime: {runtime:.2f}s")
    print(f"verdicts: {verdicts}")


if __name__ == "__main__":
    main()
