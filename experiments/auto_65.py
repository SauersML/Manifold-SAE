"""auto_65 — idea (sssssss): per-color R² as a function of CIELab Lightness
banding (L*=0..100, 5 bins of width 20).

For every color c (n_c=949) we have ~28 cogito-L40 reps. We build a per-color
centroid in the published top-K=64 PC basis (Z_c, the same target as the
production GAM run), then fit two RGB→Z models in 5-fold CV over colors:

    A) L_lin_rgb       — linear ridge on RGB (a 3-feature baseline).
    B) U_duchon_rgb    — Duchon thin-plate on a 5×5×5 RGB lattice
                         (NO length_scale; REML λ). Same family that the
                         production run uses for L_joint_rgb.

For each held-out color we compute per-color R²:
    R²_c = 1 - ||Z_c - Ẑ_c||² / ||Z_c - mean(Z_train)||²
Then we convert RGB → CIELab and bin colors by L* into [0,20),[20,40),
[40,60),[60,80),[80,100]. For each bin we report mean ± 95% bootstrap CI
on per-color R² for both models, plus the count of colors.

This isolates whether the GAM does better on mid-lightness colors (a known
perceptual-coverage sweet spot in xkcd) or whether very dark / very light
swatches are systematically easier or harder to predict from cogito L40.

Constraints respected: PCA, ridge (linear-RGB baseline), Duchon (no
length_scale) — no Gaussian RBF, no kernel tricks.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_65.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
import color_manifold_gam as cmg  # noqa: E402

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"
OUT_PNG = RUN_DIR / "auto_65.png"
OUT_JSON = RUN_DIR / "auto_65.json"

N_FOLDS = 5
SEED = 0
PER_SIDE = 5  # 5x5x5 Duchon lattice in RGB
N_BOOT = 1000

L_EDGES = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])


def ridge_fit_predict(X_tr, Y_tr, X_te, alpha=1e-2):
    """Linear ridge regression with an intercept column."""
    Phi_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    Phi_te = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
    K = Phi_tr.shape[1]
    A = Phi_tr.T @ Phi_tr + alpha * np.eye(K)
    B = np.linalg.solve(A, Phi_tr.T @ Y_tr)
    return Phi_te @ B


def duchon_fit_predict(X_tr, Y_tr, X_te, per_side=PER_SIDE,
                       init_log_lam=0.0):
    """Thin-plate Duchon in 3D on a regular RGB lattice. No length_scale."""
    ax = [np.linspace(0.0, 1.0, per_side) for _ in range(3)]
    G = np.meshgrid(*ax, indexing="ij")
    centers = np.stack([g.flatten() for g in G], axis=1)
    Phi_tr, P = cmg.duchon_basis_radial(X_tr, centers)
    Phi_te, _ = cmg.duchon_basis_radial(X_te, centers)
    B, _ = cmg.reml_fit(Phi_tr, Y_tr, P, init_log_lam)
    return Phi_te @ B


def per_color_r2(Y_true, Y_pred, mean_baseline):
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=1)
    ss_tot = np.sum((Y_true - mean_baseline) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def bootstrap_mean_ci(x, n_boot=N_BOOT, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {RESULTS}")
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    Vt = np.asarray(d["per_layer"]["L40"]["Vt_topK"], dtype=np.float64)
    mu = np.asarray(d["per_layer"]["L40"]["mu"], dtype=np.float64)
    sigma = np.asarray(d["per_layer"]["L40"]["sigma"], dtype=np.float64)
    K = Vt.shape[0]

    R_axis = np.asarray(d["color_axes_per_color_index"]["R"], dtype=np.float64)
    G_axis = np.asarray(d["color_axes_per_color_index"]["G"], dtype=np.float64)
    B_axis = np.asarray(d["color_axes_per_color_index"]["B"], dtype=np.float64)
    n_c = R_axis.size
    RGB = np.stack([R_axis, G_axis, B_axis], axis=1)
    print(f"[meta] n_c={n_c} n_t={n_t} K={K}")

    # Per-color centroid in the same standardised top-K PC space.
    print(f"[load] {HARVEST}")
    X_full = np.load(HARVEST, mmap_mode="r")
    assert X_full.shape[0] == n_c * n_t, (X_full.shape, n_c * n_t)
    per_color = np.zeros((n_c, X_full.shape[1]), dtype=np.float64)
    block = 4096
    for s in range(0, X_full.shape[0], block):
        e = min(s + block, X_full.shape[0])
        chunk = np.asarray(X_full[s:e], dtype=np.float64)
        idx = np.arange(s, e) // n_t
        for ci in np.unique(idx):
            per_color[ci] += chunk[idx == ci].sum(axis=0)
    per_color /= n_t
    Xn = (per_color - mu) / np.maximum(sigma, 1e-6)
    Z = (Xn - Xn.mean(0, keepdims=True)) @ Vt.T
    print(f"[meta] Z {Z.shape}")

    # 5-fold CV over colors.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_c)
    folds = np.array_split(perm, N_FOLDS)
    r2_lin = np.zeros(n_c, dtype=np.float64)
    r2_duc = np.zeros(n_c, dtype=np.float64)

    for fi, te_idx in enumerate(folds):
        mask = np.ones(n_c, dtype=bool); mask[te_idx] = False
        tr_idx = np.where(mask)[0]
        X_tr, X_te = RGB[tr_idx], RGB[te_idx]
        Y_tr, Y_te = Z[tr_idx], Z[te_idx]
        mean_tr = Y_tr.mean(axis=0, keepdims=True)

        print(f"[fold {fi+1}/{N_FOLDS}] tr={tr_idx.size} te={te_idx.size}  ridge…")
        Yp_lin = ridge_fit_predict(X_tr, Y_tr, X_te, alpha=1e-2)
        print(f"[fold {fi+1}/{N_FOLDS}] duchon…")
        Yp_duc = duchon_fit_predict(X_tr, Y_tr, X_te, per_side=PER_SIDE)

        r2_lin[te_idx] = per_color_r2(Y_te, Yp_lin, mean_tr)
        r2_duc[te_idx] = per_color_r2(Y_te, Yp_duc, mean_tr)
        print(f"[fold {fi+1}] mean R²: lin={r2_lin[te_idx].mean():.4f} "
              f"duc={r2_duc[te_idx].mean():.4f}")

    # CIELab Lightness for binning.
    LAB = cmg.rgb_to_lab(RGB)
    L_star = LAB[:, 0]
    print(f"[lab] L* range [{L_star.min():.2f}, {L_star.max():.2f}] "
          f"median={np.median(L_star):.2f}")

    # Bin into 5 bands.
    band_idx = np.clip(np.digitize(L_star, L_EDGES[1:-1], right=False), 0, 4)

    bands = []
    for b in range(5):
        sel = np.where(band_idx == b)[0]
        n_b = sel.size
        rng_b = np.random.default_rng(SEED + b + 1)
        lin_mean, lin_lo, lin_hi = bootstrap_mean_ci(r2_lin[sel], rng=rng_b)
        duc_mean, duc_lo, duc_hi = bootstrap_mean_ci(r2_duc[sel], rng=rng_b)
        bands.append({
            "band": f"L*∈[{L_EDGES[b]:.0f},{L_EDGES[b+1]:.0f}{')' if b<4 else ']'}",
            "L_lo": float(L_EDGES[b]),
            "L_hi": float(L_EDGES[b+1]),
            "n": int(n_b),
            "lin_mean": lin_mean, "lin_lo": lin_lo, "lin_hi": lin_hi,
            "duc_mean": duc_mean, "duc_lo": duc_lo, "duc_hi": duc_hi,
            "gain_mean": float(duc_mean - lin_mean),
        })
        print(f"[band {b}] {bands[-1]['band']:>14s}  n={n_b:4d}  "
              f"lin={lin_mean:+.4f} [{lin_lo:+.4f},{lin_hi:+.4f}]  "
              f"duc={duc_mean:+.4f} [{duc_lo:+.4f},{duc_hi:+.4f}]  "
              f"Δ={duc_mean-lin_mean:+.4f}")

    summary = {
        "n_colors": int(n_c), "n_templates": int(n_t), "K": int(K),
        "n_folds": N_FOLDS, "per_side": PER_SIDE,
        "L_edges": L_EDGES.tolist(),
        "overall_lin_mean_r2": float(r2_lin.mean()),
        "overall_duc_mean_r2": float(r2_duc.mean()),
        "bands": bands,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] {OUT_JSON}")

    # --------------- plot ---------------
    fig = plt.figure(figsize=(13.0, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)

    centers = 0.5 * (L_EDGES[:-1] + L_EDGES[1:])

    # (a) per-band mean R² with bootstrap 95% CI.
    ax = fig.add_subplot(gs[0, 0])
    lin_m = np.array([b["lin_mean"] for b in bands])
    lin_l = np.array([b["lin_lo"]   for b in bands])
    lin_h = np.array([b["lin_hi"]   for b in bands])
    duc_m = np.array([b["duc_mean"] for b in bands])
    duc_l = np.array([b["duc_lo"]   for b in bands])
    duc_h = np.array([b["duc_hi"]   for b in bands])
    ax.errorbar(centers - 1.0, lin_m, yerr=[lin_m - lin_l, lin_h - lin_m],
                fmt="o-", color="#1f77b4", lw=1.6, capsize=4,
                label="L_lin_rgb (ridge)")
    ax.errorbar(centers + 1.0, duc_m, yerr=[duc_m - duc_l, duc_h - duc_m],
                fmt="s-", color="#d62728", lw=1.6, capsize=4,
                label="U_duchon_rgb (5×5×5)")
    ax.set_xticks(centers)
    ax.set_xticklabels([b["band"] for b in bands], rotation=15, fontsize=9)
    ax.set_ylabel("mean per-color R² (CV)")
    ax.set_xlabel("CIELab Lightness band")
    ax.set_title("(a) Mean per-color R² by L* band  (bootstrap 95% CI)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # (b) Δ R² (Duchon − linear) per band, with bootstrap CI on the diff.
    ax = fig.add_subplot(gs[0, 1])
    deltas = duc_m - lin_m
    # bootstrap on per-color diffs inside each band for honest CI
    diff_lo = np.zeros(5); diff_hi = np.zeros(5)
    for b in range(5):
        sel = np.where(band_idx == b)[0]
        diff = r2_duc[sel] - r2_lin[sel]
        rng_b = np.random.default_rng(SEED + 100 + b)
        if diff.size:
            idx = rng_b.integers(0, diff.size, size=(N_BOOT, diff.size))
            means = diff[idx].mean(axis=1)
            diff_lo[b], diff_hi[b] = np.quantile(means, [0.025, 0.975])
    counts = np.array([b["n"] for b in bands])
    bars = ax.bar(centers, deltas, width=14, color="#2ca02c", alpha=0.85,
                  edgecolor="k")
    ax.errorbar(centers, deltas,
                yerr=[deltas - diff_lo, diff_hi - deltas],
                fmt="none", ecolor="k", capsize=4, lw=1.2)
    ax.axhline(0, color="k", lw=0.7)
    for c, dlt, n in zip(centers, deltas, counts):
        ax.text(c, dlt + 0.003 * np.sign(dlt or 1), f"n={n}",
                ha="center", va="bottom" if dlt >= 0 else "top", fontsize=8)
    ax.set_xticks(centers)
    ax.set_xticklabels([b["band"] for b in bands], rotation=15, fontsize=9)
    ax.set_ylabel("Δ R²  (Duchon − linear)")
    ax.set_xlabel("CIELab Lightness band")
    ax.set_title("(b) Nonlinear gain over linear-RGB per L* band")
    ax.grid(True, axis="y", alpha=0.3)

    # (c) per-color scatter: L* vs R², colored by swatch.
    ax = fig.add_subplot(gs[1, 0])
    face = np.clip(RGB, 0, 1)
    ax.scatter(L_star, r2_duc, s=12, c=face, edgecolors="none",
               alpha=0.85, label="Duchon")
    # rolling mean curve over L*
    order = np.argsort(L_star)
    Ls = L_star[order]; rs = r2_duc[order]
    win = max(20, n_c // 25)
    kern = np.ones(win) / win
    roll = np.convolve(rs, kern, mode="same")
    ax.plot(Ls, roll, color="#222", lw=1.6, label=f"rolling mean (win={win})")
    for x in L_EDGES[1:-1]:
        ax.axvline(x, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("CIELab L*")
    ax.set_ylabel("per-color R² (Duchon)")
    ax.set_title("(c) Per-color Duchon R² vs CIELab Lightness")
    ax.set_xlim(0, 100); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # (d) histogram of L* with the 5 bands shaded; sanity check on counts.
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(L_star, bins=40, color="#888", edgecolor="white")
    for x in L_EDGES[1:-1]:
        ax.axvline(x, color="red", ls="--", lw=1.0)
    for b, c, n in zip(bands, centers, counts):
        ax.text(c, ax.get_ylim()[1] * 0.92, f"n={n}",
                ha="center", fontsize=9, color="red")
    ax.set_xlabel("CIELab L*")
    ax.set_ylabel("colors")
    ax.set_title("(d) Color count per L* band (xkcd palette)")
    ax.set_xlim(0, 100); ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"auto_65 — per-color R² by CIELab Lightness | L40 | "
        f"n_c={n_c}  K={K}  {N_FOLDS}-fold CV  |  "
        f"overall: lin={r2_lin.mean():+.4f}  duc={r2_duc.mean():+.4f}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
