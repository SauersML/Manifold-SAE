"""auto_exp_54: does the HSV gauge-fix recipe generalize to NON-perceptual targets?

auto_exp_38 used HSV(R,G,B-derived) as the supervised gauge → recovered
R²(h,s,v) = (0.700, 0.657, 0.719).  auto_exp_53 confirmed d=3 optimal for HSV.

Question (per project_cogito_color_manifold_decomposition.md): the decomposition
of U_3d into 0.32 HSV-perceptual + 0.29 name-token-semantic suggests two
distinct subspaces.  Does the SAME RRR gauge-fix recipe work when the supervised
target is name-semantic (modifier_count, monoword, template_σ) instead of
perceptual?

Hypothesis split:
  - GENERALIZES   → R² ≥ 0.3 on at least one non-HSV target at d=1
  - HSV-SPECIFIC  → All non-HSV R² < 0.3 (and the recipe just exploits the
                    bulk-variance alignment of HSV per auto_77/81/82)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore
from auto_exp_38 import (
    X_PATH, N_TEMPLATES, K_PCS,
    per_color_stats_mmap, load_xkcd_rgb,
)
from auto_exp_53 import fit_rrr, r2_per_channel

ROOT = Path("/Users/user/Manifold-SAE")
OUT_NPZ = ROOT / "runs" / "auto_exp_54_nonhsv_gauge.npz"
MEMORY_MD = Path(
    "/Users/user/.claude/projects/-Users-user-Manifold-SAE/memory/"
    "project_cogito_recovery_at_d_aux_3.md"
)

N_FOLDS = 5
SEED = 54
RIDGE_LAM = 1.0
TARGETS = ["modifier_count", "monoword", "template_sigma"]


def compute_targets(names, tsig):
    mono = np.array([1.0 if len(n.split()) == 1 else 0.0 for n in names],
                    dtype=np.float64)
    modc = np.array([max(0, len(n.split()) - 1) for n in names], dtype=np.float64)
    return {
        "modifier_count": modc,
        "monoword": mono,
        "template_sigma": tsig.astype(np.float64),
    }


def cv_r2_single(Tc, y, d, n_folds=N_FOLDS, seed=SEED):
    """5-fold CV per scalar target (RRR with 1-col Y is just rank-1 ridge)."""
    rng = np.random.default_rng(seed)
    n = Tc.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    preds = np.zeros(n)
    Y = y[:, None]
    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[k] for k in range(n_folds) if k != f])
        mu_T = Tc[tr].mean(0, keepdims=True)
        mu_y = float(Y[tr].mean())
        fit = fit_rrr(Tc[tr] - mu_T, Y[tr] - mu_y, d, lam=RIDGE_LAM)
        T_te = (Tc[te] - mu_T) @ fit["W"]
        preds[te] = (T_te @ fit["A"]).squeeze(-1) + mu_y
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = max(((y - y.mean()) ** 2).sum(), 1e-12)
    return 1.0 - ss_res / ss_tot


def cv_r2_joint(Tc, Y, d, n_folds=N_FOLDS, seed=SEED):
    """5-fold CV per channel for joint d=3 fit."""
    rng = np.random.default_rng(seed)
    n = Tc.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    preds = np.zeros_like(Y)
    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[k] for k in range(n_folds) if k != f])
        mu_T = Tc[tr].mean(0, keepdims=True)
        mu_Y = Y[tr].mean(0, keepdims=True)
        fit = fit_rrr(Tc[tr] - mu_T, Y[tr] - mu_Y, d, lam=RIDGE_LAM)
        T_te = (Tc[te] - mu_T) @ fit["W"]
        preds[te] = T_te @ fit["A"] + mu_Y
    return r2_per_channel(Y, preds)


def main():
    t0 = time.time()
    print("[auto_exp_54] non-HSV gauge-fix generalization test")
    print(f"[data] mmap {X_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    print(f"[data] X={X.shape}")
    basis = load_pc_basis(K=64)
    T0, tsig = per_color_stats_mmap(X, N_TEMPLATES, basis, K_PCS)
    n_c, K = T0.shape
    print(f"[centroids] T0={T0.shape}")
    names, rgb = load_xkcd_rgb(n_c)
    targets = compute_targets(names, tsig)
    for k in TARGETS:
        y = targets[k]
        print(f"[target {k}] mean={y.mean():.3f} std={y.std():.3f} "
              f"min={y.min():.3f} max={y.max():.3f}")

    Tc = T0 - T0.mean(0, keepdims=True)

    # --- d=1 per-target fits
    d1_in = {}
    d1_cv = {}
    for k in TARGETS:
        y = targets[k]
        Y = y[:, None]
        Yc = Y - Y.mean(0, keepdims=True)
        fit = fit_rrr(Tc, Yc, 1, lam=RIDGE_LAM)
        pred = fit["pred"] + Y.mean(0, keepdims=True)
        r2_in = float(r2_per_channel(Y, pred)[0])
        r2_cv = float(cv_r2_single(Tc, y, 1))
        d1_in[k] = r2_in
        d1_cv[k] = r2_cv
        print(f"[d=1 {k}] R²_in={r2_in:.3f}  R²_CV={r2_cv:.3f}")

    # --- joint d=3 fit with all three targets stacked
    Y_joint = np.stack([targets[k] for k in TARGETS], axis=1)
    # Standardize columns so unequal scales don't dominate the SVD
    y_mu = Y_joint.mean(0, keepdims=True)
    y_sd = Y_joint.std(0, keepdims=True).clip(min=1e-8)
    Y_std = (Y_joint - y_mu) / y_sd
    Yc = Y_std - Y_std.mean(0, keepdims=True)
    fit3 = fit_rrr(Tc, Yc, 3, lam=RIDGE_LAM)
    pred_std = fit3["pred"]
    # back to original scale for R²
    pred = pred_std * y_sd + y_mu
    r2_joint_in = r2_per_channel(Y_joint, pred)
    r2_joint_cv_std = cv_r2_joint(Tc, Y_std, 3)  # CV on standardized space (equiv. R²)
    print(f"[joint d=3] R²_in per channel: "
          + " ".join(f"{TARGETS[i]}={r2_joint_in[i]:.3f}" for i in range(3)))
    print(f"[joint d=3] R²_CV per channel: "
          + " ".join(f"{TARGETS[i]}={r2_joint_cv_std[i]:.3f}" for i in range(3)))

    # --- Stdout table
    print()
    print("=" * 80)
    print(" auto_exp_54: gauge-fix on NON-HSV targets")
    print("=" * 80)
    hdr = f"{'target':>18} {'d=1 R²_in':>12} {'d=1 R²_CV':>12} {'joint d=3 R²_in':>17} {'joint d=3 R²_CV':>17}"
    print(hdr)
    print("-" * len(hdr))
    for i, k in enumerate(TARGETS):
        print(f"{k:>18} {d1_in[k]:>12.3f} {d1_cv[k]:>12.3f} "
              f"{r2_joint_in[i]:>17.3f} {r2_joint_cv_std[i]:>17.3f}")
    print("=" * 80)

    # --- Verdict
    max_d1_cv = max(d1_cv.values())
    mean_joint = float(np.mean(r2_joint_cv_std))
    primary = max_d1_cv >= 0.30
    secondary = mean_joint >= 0.40
    print(f"[verdict.primary]   ANY target d=1 R²_CV >= 0.30 ? {primary} "
          f"(max={max_d1_cv:.3f})")
    print(f"[verdict.secondary] joint d=3 mean R²_CV >= 0.40 ? {secondary} "
          f"(mean={mean_joint:.3f})")

    if primary:
        verdict = "GENERALIZES"
    else:
        verdict = "HSV-SPECIFIC"
    print(f"[verdict] gauge-fix recipe: {verdict}")

    # --- Save
    np.savez(
        OUT_NPZ,
        targets=np.array(TARGETS),
        d1_r2_in=np.array([d1_in[k] for k in TARGETS]),
        d1_r2_cv=np.array([d1_cv[k] for k in TARGETS]),
        joint_d3_r2_in=r2_joint_in,
        joint_d3_r2_cv=r2_joint_cv_std,
        max_d1_cv=max_d1_cv,
        mean_joint_cv=mean_joint,
        primary_verdict=primary,
        secondary_verdict=secondary,
        verdict=verdict,
        W_joint=fit3["W"],
        A_joint=fit3["A"],
        T_joint=fit3["T"],
    )
    print(f"[npz] saved {OUT_NPZ}")

    # --- Append memory note
    block = []
    block.append("\n## auto_exp_54: gauge-fix on non-HSV targets\n")
    block.append("Tests whether auto_exp_38's HSV-supervised RRR gauge-fix recipe "
                 "generalizes to NAME-SEMANTIC targets (modifier_count, monoword, "
                 "template_sigma). Same n=949 cogito-L40 centroids, K=16 PCs, "
                 "RRR with ridge λ=1.0, 5-fold CV.\n\n")
    block.append("| target | d=1 R²_in | d=1 R²_CV | joint d=3 R²_in | joint d=3 R²_CV |\n")
    block.append("|---|---|---|---|---|\n")
    for i, k in enumerate(TARGETS):
        block.append(f"| {k} | {d1_in[k]:.3f} | {d1_cv[k]:.3f} | "
                     f"{r2_joint_in[i]:.3f} | {r2_joint_cv_std[i]:.3f} |\n")
    block.append(f"\nPrimary (any target d=1 R²_CV ≥ 0.30): **{primary}** "
                 f"(max={max_d1_cv:.3f}).\n")
    block.append(f"Secondary (joint d=3 mean R²_CV ≥ 0.40): **{secondary}** "
                 f"(mean={mean_joint:.3f}).\n")
    block.append(f"\nVerdict: **{verdict}**.\n")
    if MEMORY_MD.exists():
        with open(MEMORY_MD, "a") as f:
            f.writelines(block)
        print(f"[memory] appended to {MEMORY_MD}")
    else:
        print(f"[memory] WARNING: {MEMORY_MD} missing, skipping append")

    print(f"[runtime] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
