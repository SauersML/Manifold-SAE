"""auto_39: Ridge alpha sweep (kk) for the best supervised RGB->PC spec.

We take the top supervised regressor family in results.json
(L_joint_rgb / L_joint_rgb_with_hue group, ~R^2 = 0.24 macro) and
re-fit it as an explicit ridge regression while sweeping alpha across
~12 decades. We answer:

  1. Where does the bias-variance optimum sit?
  2. How sharply does macro R^2 collapse on either side?
  3. Does the optimal alpha differ for high-variance vs low-variance
     PCs (per-PC alpha curves)?
  4. Is the cross-validated R^2 stable across folds (per-fold spread)?

Setup
-----
- Target: top-64 PCs of per-color (n_colors=949) centroid activations
  at L40 (matches results.json: n_pcs=64, n_templates=28).
- Features: joint RGB polynomial expansion
  [R, G, B, R^2, G^2, B^2, RG, RB, GB, RGB]    (10 features)
  plus an intercept; standardized.
- alpha grid: logspace(-6, 6, 25).
- 5-fold over color indices (matching the project default n_folds=5).
- Ridge is closed-form sklearn.linear_model.Ridge (allowed: "linear/ridge").

Output
------
runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_39.{json,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_PNG = OUT_DIR / "auto_39.png"
OUT_JSON = OUT_DIR / "auto_39.json"

N_TEMPLATES = 28
N_PCS = 64
N_FOLDS = 5
ALPHAS = np.logspace(-6, 6, 25)


def joint_rgb_features(rgb: np.ndarray) -> np.ndarray:
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return np.stack(
        [R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, R * G * B],
        axis=1,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] results.json", flush=True)
    res = json.loads(RESULTS.read_text())
    ax = res["color_axes_per_color_index"]
    R = np.asarray(ax["R"], dtype=np.float64)
    G = np.asarray(ax["G"], dtype=np.float64)
    B = np.asarray(ax["B"], dtype=np.float64)
    rgb = np.stack([R, G, B], axis=1)
    n_colors = rgb.shape[0]
    print(f"[load] n_colors={n_colors}", flush=True)

    print("[load] harvest", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    assert N == n_colors * N_TEMPLATES, f"{N} != {n_colors}*{N_TEMPLATES}"
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)

    # Per-color centroids
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)

    # PCA -> top-64 (match project pipeline: standardize, then SVD)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    Cn = (centroids - mu) / sigma
    Cc = Cn - Cn.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    Vk = Vt[:N_PCS]
    Z = Cn @ Vk.T  # (n_colors, 64)
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    # Features
    Phi = joint_rgb_features(rgb)  # (n_colors, 10)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

    # alphas x folds x pcs
    r2_per_alpha_pc = np.zeros((len(ALPHAS), N_FOLDS, N_PCS))
    macro_per_alpha_fold = np.zeros((len(ALPHAS), N_FOLDS))

    for ai, alpha in enumerate(ALPHAS):
        for fi, (tr, te) in enumerate(kf.split(np.arange(n_colors))):
            # standardize features on train
            mu_f = Phi[tr].mean(0, keepdims=True)
            sd_f = Phi[tr].std(0, keepdims=True).clip(min=1e-9)
            Ptr = (Phi[tr] - mu_f) / sd_f
            Pte = (Phi[te] - mu_f) / sd_f

            mod = Ridge(alpha=alpha, fit_intercept=True)
            mod.fit(Ptr, Z[tr])
            Zhat = mod.predict(Pte)
            # per-PC R^2 on test fold
            ss_res = ((Z[te] - Zhat) ** 2).sum(0)
            ss_tot = ((Z[te] - Z[te].mean(0, keepdims=True)) ** 2).sum(0)
            r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
            r2_per_alpha_pc[ai, fi] = r2
            # macro = mean over PCs, weighted by EVR -> use simple mean for
            # apples-to-apples with results.json "r2_macro_mean"
            macro_per_alpha_fold[ai, fi] = float(r2.mean())
        print(f"  alpha={alpha:.2e}  macro_mean={macro_per_alpha_fold[ai].mean():+.4f}  "
              f"std={macro_per_alpha_fold[ai].std():+.4f}", flush=True)

    macro_mean = macro_per_alpha_fold.mean(1)
    macro_std = macro_per_alpha_fold.std(1)
    a_star_idx = int(np.argmax(macro_mean))
    a_star = float(ALPHAS[a_star_idx])
    r2_star = float(macro_mean[a_star_idx])
    print(f"\n[opt] alpha* = {a_star:.3e}  macro R^2 = {r2_star:+.4f}  "
          f"(fold std {macro_std[a_star_idx]:+.4f})", flush=True)

    # Per-PC optimal alpha
    r2_pc = r2_per_alpha_pc.mean(1)  # (alphas, pcs)
    best_alpha_per_pc = ALPHAS[np.argmax(r2_pc, axis=0)]
    best_r2_per_pc = r2_pc.max(0)

    summary = {
        "config": {
            "harvest": str(HARVEST), "n_colors": n_colors,
            "n_templates": N_TEMPLATES, "n_pcs": N_PCS,
            "n_folds": N_FOLDS, "alphas": ALPHAS.tolist(),
            "feature_set": "joint_rgb_poly[R,G,B,R2,G2,B2,RG,RB,GB,RGB]",
        },
        "macro_r2_mean_per_alpha": macro_mean.tolist(),
        "macro_r2_std_per_alpha": macro_std.tolist(),
        "alpha_star": a_star,
        "macro_r2_star": r2_star,
        "best_alpha_per_pc": best_alpha_per_pc.tolist(),
        "best_r2_per_pc": best_r2_per_pc.tolist(),
        "evr_top64": evr.tolist(),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}", flush=True)

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) macro R^2 vs alpha with fold band
    ax = axes[0, 0]
    ax.plot(ALPHAS, macro_mean, "-o", color="#3050a0", lw=1.8, ms=4,
            label="mean over 5 folds")
    ax.fill_between(ALPHAS, macro_mean - macro_std, macro_mean + macro_std,
                     color="#3050a0", alpha=0.22, label="+/- 1 SD")
    ax.axvline(a_star, color="k", ls="--", lw=1,
                label=f"alpha* = {a_star:.2e}")
    ax.axhline(r2_star, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("ridge alpha")
    ax.set_ylabel("macro R^2 (mean over PCs, mean over folds)")
    ax.set_title(f"Ridge alpha sweep, joint-RGB poly features ({Phi.shape[1]}D)\n"
                  f"best macro R^2 = {r2_star:+.4f} at alpha={a_star:.2e}")
    ax.grid(which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower center")

    # (b) per-fold macro R^2 curves
    ax = axes[0, 1]
    for fi in range(N_FOLDS):
        ax.plot(ALPHAS, macro_per_alpha_fold[:, fi], "-", lw=1, alpha=0.7,
                label=f"fold {fi}")
    ax.set_xscale("log")
    ax.set_xlabel("ridge alpha")
    ax.set_ylabel("macro R^2")
    ax.set_title("Per-fold macro R^2 (stability check)")
    ax.grid(which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=5)

    # (c) heatmap: alpha x PC, color = R^2 (clip at -0.2..max)
    ax = axes[1, 0]
    vmax = float(r2_pc.max())
    vmin = max(-0.2, float(r2_pc.min()))
    im = ax.imshow(r2_pc, aspect="auto", origin="lower",
                    extent=[0, N_PCS, np.log10(ALPHAS[0]), np.log10(ALPHAS[-1])],
                    cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("PC index (0 = highest EVR)")
    ax.set_ylabel("log10(alpha)")
    ax.set_title("R^2 per (alpha, PC)")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("R^2 (fold-mean)")
    # overlay best-alpha-per-PC
    ax.plot(np.arange(N_PCS) + 0.5, np.log10(best_alpha_per_pc), "w.",
             ms=3, label="argmax_alpha")
    ax.legend(fontsize=8, loc="upper right")

    # (d) best R^2 per PC vs EVR
    ax = axes[1, 1]
    sc = ax.scatter(evr, best_r2_per_pc, c=np.log10(best_alpha_per_pc),
                     cmap="plasma", s=22, edgecolor="k", linewidth=0.3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("log10(best alpha for that PC)")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("EVR of PC (centroid variance share)")
    ax.set_ylabel("best CV R^2 for that PC")
    ax.set_title("Per-PC ridge best R^2 vs variance share")
    ax.grid(which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
