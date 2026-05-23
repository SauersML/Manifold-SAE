"""auto_34: (nnnn) Per-template R² for top spec vs linear baseline.

For each of the 28 prompt templates we compute how well the top-ranked
non-degenerate model (U_pca48_duchon_add1d) explains the per-(color, template)
residual at L40, compared to the headline linear/joint baseline (L_joint_rgb).

Setup
-----
- Targets are built exactly as in color_manifold_gam.main:
    per_color = mean over templates of X_L40 (color-major, n_t=28)
    Xn = (per_color - mu) / sigma                (per-dim z-score across colors)
    Z  = (Xn - Xn.mean(0)) @ Vt_topK^T           (n_colors x 64)
- 5-fold color-grouped CV: held-out colors' Z is predicted by both specs.
- Predictions Z_te_pred (n_held_colors x 64) are mapped back to residual space:
    Xn_hat = Z_te_pred @ Vt_topK
    pc_hat = Xn_hat * sigma + mu                 (n_held_colors x 7168)
- For each template t in {0..27}, the per-color "target at template t" is
  the raw residual X_L40[c*28+t] for held-out colors c. We compute the macro
  R² (across the 7168 residual dims and held-out colors) of pc_hat vs that
  target. R² is sklearn-style multivariate macro across output dims (uniform
  weighting), pooled per fold then averaged.

Interpretation
--------------
The model never sees per-template structure (it was trained on
template-averaged centroids). A template whose per-(color, t) residual is
dominated by the color-shared centroid will score high; a template that
contributes most of its own template-specific direction (or noise) will
score low. The comparison reveals whether the high-capacity Duchon model's
gains over the linear baseline transfer uniformly across the 28 prompts or
concentrate on a subset.

Hard-constraint compliant: PCA + Duchon (no length_scale) + linear/ridge
via gamfit's REML; no Gaussian RBF; no t-SNE/UMAP/kNN here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import the canonical pipeline so we use the exact same Duchon/REML path.
sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
import color_manifold_gam as cmg  # noqa: E402

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS = RUN_DIR / "results.json"
OUT = RUN_DIR / "auto_34.png"

TOP_SPEC = "U_pca48_duchon_add1d"
BASELINE = "L_joint_rgb"


def short_template(t: str, n: int = 40) -> str:
    s = t.replace("{x}", "X")
    return s if len(s) <= n else s[: n - 1] + "..."


def macro_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sklearn-style multioutput uniform_average R² (across columns)."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)
    return float(np.mean(r2))


def main() -> None:
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    Vt_topK = np.asarray(d["per_layer"]["L40"]["Vt_topK"], dtype=np.float64)  # (64, 7168)
    mu = np.asarray(d["per_layer"]["L40"]["mu"], dtype=np.float64)             # (7168,)
    sigma = np.asarray(d["per_layer"]["L40"]["sigma"], dtype=np.float64)       # (7168,)
    K = Vt_topK.shape[0]
    print(f"[meta] templates={n_t}, K_pcs={K}, D={Vt_topK.shape[1]}")

    # Harvest of per-(color, template) residuals, color-major (verified in auto_33).
    X_full = np.load(HARVEST, mmap_mode="r")
    n_rows, D = X_full.shape
    n_c = n_rows // n_t
    assert n_c * n_t == n_rows
    print(f"[data] X={X_full.shape}, n_colors={n_c}")

    # Reconstruct the per-color centroid + PCA target Z exactly as cmg.main.
    per_color = np.zeros((n_c, D), dtype=np.float64)
    block = 2048
    counts = np.zeros(n_c, dtype=np.int64)
    for s in range(0, n_rows, block):
        e = min(s + block, n_rows)
        chunk = np.asarray(X_full[s:e], dtype=np.float64)
        idx = (np.arange(s, e) // n_t)
        for ci_local in np.unique(idx):
            m = idx == ci_local
            per_color[ci_local] += chunk[m].sum(axis=0)
            counts[ci_local] += int(m.sum())
    per_color /= counts[:, None]
    print(f"[centroid] per-color row count: min={counts.min()} max={counts.max()}")
    # Sanity vs cached mu/sigma
    pc_mu = per_color.mean(0)
    pc_sigma = per_color.std(0)
    drift_mu = np.max(np.abs(pc_mu - mu))
    drift_sig = np.max(np.abs(pc_sigma - sigma)) / max(float(sigma.max()), 1e-9)
    print(f"[sanity] |mu_local - mu_json|_max = {drift_mu:.4e}, "
          f"|sigma_local - sigma_json|_rel_max = {drift_sig:.4e}")
    # Use cached mu/sigma (what Vt was built on)
    Xn = (per_color - mu) / np.maximum(sigma, 1e-6)
    Z = (Xn - Xn.mean(0, keepdims=True)) @ Vt_topK.T  # (n_c, K)

    # Build color side inputs the same way cmg.main does.
    colors = cmg.load_xkcd_colors()[:n_c]
    rgb_per_color = np.array([(r, g, b) for _, r, g, b in colors],
                              dtype=np.float64) / 255.0
    hsv_per_color = cmg.rgb_to_hsv_arr(rgb_per_color * 255.0)
    X_rgb = rgb_per_color
    X_hsv = np.stack([
        np.cos(2 * np.pi * hsv_per_color[:, 0]),
        np.sin(2 * np.pi * hsv_per_color[:, 0]),
        hsv_per_color[:, 1],
        hsv_per_color[:, 2],
    ], axis=1)

    cfg = cmg.Config()
    folds = cmg.kfold_color_indices(n_c, cfg.n_folds)

    # Storage: per-template R², one value per fold per spec.
    per_t_r2 = {TOP_SPEC: np.zeros((cfg.n_folds, n_t)),
                BASELINE: np.zeros((cfg.n_folds, n_t))}
    per_t_r2_pc = {TOP_SPEC: np.zeros((cfg.n_folds, n_t)),
                    BASELINE: np.zeros((cfg.n_folds, n_t))}  # R² on the held-out per-color centroid (no template split)
    per_fold_macro = {TOP_SPEC: [], BASELINE: []}

    for f_idx, (tr, te) in enumerate(folds):
        print(f"\n[fold {f_idx}] train={len(tr)} test={len(te)}")
        for spec in (TOP_SPEC, BASELINE):
            _, Z_te_pred = cmg.fit_and_predict(
                spec, X_rgb[tr], X_hsv[tr], Z[tr],
                X_rgb[te], X_hsv[te], Z[te], cfg,
            )
            # macro on Z (matches results.json convention)
            r2_z = macro_r2(Z[te], Z_te_pred)
            per_fold_macro[spec].append(r2_z)

            # Map prediction back to residual space.
            Xn_hat = Z_te_pred @ Vt_topK            # (|te|, D)
            pc_hat = Xn_hat * sigma + mu            # (|te|, D)

            # R² on the held-out per-color centroid itself (training target).
            r2_pc_center = macro_r2(per_color[te], pc_hat)

            for t in range(n_t):
                # held-out (color, template t) residual targets
                te_rows = te * n_t + t
                y_t = np.asarray(X_full[te_rows], dtype=np.float64)  # (|te|, D)
                # Predict y_t = pc_hat (template-agnostic prediction)
                per_t_r2[spec][f_idx, t] = macro_r2(y_t, pc_hat)
                per_t_r2_pc[spec][f_idx, t] = r2_pc_center
            print(f"  {spec:30s} R²(Z)={r2_z:+.4f}  R²(per-color centroid)={r2_pc_center:+.4f}")

    print()
    for spec in (TOP_SPEC, BASELINE):
        m = float(np.mean(per_fold_macro[spec]))
        s = float(np.std(per_fold_macro[spec]))
        print(f"[overall] {spec:30s} macro-R²(Z) {m:+.4f} ± {s:.4f}")

    # Aggregate per-template across folds (mean / std).
    mean_top = per_t_r2[TOP_SPEC].mean(0)
    std_top = per_t_r2[TOP_SPEC].std(0)
    mean_base = per_t_r2[BASELINE].mean(0)
    std_base = per_t_r2[BASELINE].std(0)
    delta = mean_top - mean_base

    # Print ranking
    order_top = np.argsort(mean_top)[::-1]
    print("\nPer-template R² (template-conditional residual, ranked by top-spec):")
    print(f"  {'t':>3} {'top R²':>9} {'base R²':>9} {'Δ':>8}  template")
    for t in order_top:
        print(f"  {t:>3d} {mean_top[t]:+9.4f} {mean_base[t]:+9.4f} "
              f"{delta[t]:+8.4f}  {templates[t]}")

    # =================================================================
    # Plot
    # =================================================================
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0],
                          height_ratios=[1.0, 0.85], hspace=0.40, wspace=0.30)

    # (A) Bar chart: per-template R² for both specs, sorted by top-spec R².
    ax = fig.add_subplot(gs[0, 0])
    order = order_top
    y = np.arange(n_t)
    bw = 0.4
    ax.barh(y - bw / 2, mean_top[order], height=bw,
            xerr=std_top[order], color="#d62728",
            label=f"{TOP_SPEC}", ecolor="#5a0e0e", capsize=2)
    ax.barh(y + bw / 2, mean_base[order], height=bw,
            xerr=std_base[order], color="#1f77b4",
            label=f"{BASELINE}", ecolor="#0c2e57", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{t:2d}] {short_template(templates[t])}" for t in order],
                       fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("R² on held-out (color, template) residual (7168-d, macro)")
    ax.set_title(f"Per-template R² of color-only prediction\n"
                 f"sorted by {TOP_SPEC}; both models trained on template-averaged centroid")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # (B) Scatter: top vs baseline per template (with diagonal)
    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(mean_base, mean_top, xerr=std_base, yerr=std_top,
                fmt="o", ms=5, color="#444", ecolor="#aaa",
                elinewidth=0.7, capsize=2)
    for t in range(n_t):
        ax.annotate(str(t), (mean_base[t], mean_top[t]),
                     fontsize=7, ha="left", va="bottom", color="#222")
    lo = float(min(mean_top.min(), mean_base.min())) - 0.005
    hi = float(max(mean_top.max(), mean_base.max())) + 0.005
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.6, label="y=x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"{BASELINE} R² (per template)")
    ax.set_ylabel(f"{TOP_SPEC} R² (per template)")
    ax.set_title("Above y=x ↔ top spec beats baseline on that template")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (C) Delta bars: top - baseline per template
    ax = fig.add_subplot(gs[1, 0])
    order_d = np.argsort(delta)[::-1]
    ax.barh(np.arange(n_t), delta[order_d],
            color=["#d62728" if v > 0 else "#1f77b4" for v in delta[order_d]])
    ax.set_yticks(np.arange(n_t))
    ax.set_yticklabels([f"[{t:2d}] {short_template(templates[t], 38)}" for t in order_d],
                       fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(f"Δ R²  =  {TOP_SPEC}  −  {BASELINE}")
    ax.set_title("Per-template improvement of top spec over linear baseline")
    ax.grid(axis="x", alpha=0.3)

    # (D) Bar chart: macro-R²(Z) and macro-R²(per-color centroid) per spec.
    ax = fig.add_subplot(gs[1, 1])
    rows = []
    for spec in (TOP_SPEC, BASELINE):
        rows.append((
            spec,
            float(np.mean(per_fold_macro[spec])),
            float(np.std(per_fold_macro[spec])),
            float(per_t_r2[spec].mean()),
            float(per_t_r2_pc[spec].mean()),
        ))
    labels = [r[0] for r in rows]
    z_means = [r[1] for r in rows]
    z_stds = [r[2] for r in rows]
    pt_means = [r[3] for r in rows]
    pc_means = [r[4] for r in rows]
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, z_means, w, yerr=z_stds, color="#9467bd",
            label="R²(Z) [training metric]", capsize=3)
    ax.bar(x, pc_means, w, color="#2ca02c",
            label="R²(per-color centroid)")
    ax.bar(x + w, pt_means, w, color="#ff7f0e",
            label="mean per-template R²")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("R²")
    ax.set_title("Headline metrics: Z-space ↔ centroid ↔ per-template")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("(nnnn) Per-template R² — top spec ({}) vs linear baseline ({})\n"
                 "L40 cogito residuals, 5-fold color-grouped CV; both models trained "
                 "on template-averaged centroid; evaluated on per-(color, template) "
                 "raw residual.".format(TOP_SPEC, BASELINE), fontsize=12, y=0.995)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nsaved {OUT}")

    # Summary numbers
    print(f"\n>>> Top-spec per-template R²: mean={mean_top.mean():+.4f}, "
          f"min={mean_top.min():+.4f}, max={mean_top.max():+.4f}")
    print(f">>> Baseline per-template R²: mean={mean_base.mean():+.4f}, "
          f"min={mean_base.min():+.4f}, max={mean_base.max():+.4f}")
    print(f">>> Mean Δ (top − base) across templates: {delta.mean():+.4f}, "
          f"min={delta.min():+.4f}, max={delta.max():+.4f}")
    print(f">>> Templates where top BEATS baseline: {int(np.sum(delta > 0))}/{n_t}")


if __name__ == "__main__":
    main()
