"""auto_35: (oooo) Does the top spec's R² flip on first 8 vs last 8 PCs of top-16?

Pick the best non-degenerate spec from results.json (U_pca48_duchon_add1d,
which excludes the trivial unsupervised PCA "predict-yourself" specs that
score 1.000), refit it twice on color-grouped 5-fold CV:

  - Target = first 8 PCs of the top-16 (high-variance PCs)
  - Target = last 8 PCs of the top-16 (low-variance PCs)

For comparison we also run L_joint_rgb (the headline linear-baseline GAM
in the project) under the same two slicings. The question: does the
ordering between the two specs flip when going from high-variance to
low-variance PCs?

We report per-PC R² (each of the 16 PCs separately) and the macro-R² over
the first 8 vs last 8.

Hard-constraint compliant: PCA + Duchon (no length_scale) + ridge via
gamfit's REML; no Gaussian RBF; no t-SNE/UMAP here.
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
OUT = RUN_DIR / "auto_35.png"

TOP_SPEC = "M_rgb_finer_grid"   # best non-degenerate, non-RBF supervised spec
BASELINE = "L_joint_rgb"
TOP_K = 16  # we compare first 8 vs last 8 within the top-16 PCs


def macro_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)
    return float(np.mean(r2))


def per_pc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    return np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)


def main() -> None:
    d = json.loads(RESULTS.read_text())
    templates = d["templates"]
    n_t = len(templates)
    Vt = np.asarray(d["per_layer"]["L40"]["Vt_topK"], dtype=np.float64)
    mu = np.asarray(d["per_layer"]["L40"]["mu"], dtype=np.float64)
    sigma = np.asarray(d["per_layer"]["L40"]["sigma"], dtype=np.float64)
    evr = np.asarray(d["per_layer"]["L40"]["explained_variance_ratio_topK"],
                     dtype=np.float64)
    K = Vt.shape[0]
    print(f"[meta] K_pcs={K}  TOP_K={TOP_K}  EVR(first 8)={evr[:8].sum():.4f}  "
          f"EVR(PC 8..15)={evr[8:16].sum():.4f}")

    X_full = np.load(HARVEST, mmap_mode="r")
    n_rows, D = X_full.shape
    n_c = n_rows // n_t
    assert n_c * n_t == n_rows
    print(f"[data] X={X_full.shape}  n_colors={n_c}")

    # Per-color centroid (template-averaged).
    per_color = np.zeros((n_c, D), dtype=np.float64)
    counts = np.zeros(n_c, dtype=np.int64)
    block = 2048
    for s in range(0, n_rows, block):
        e = min(s + block, n_rows)
        chunk = np.asarray(X_full[s:e], dtype=np.float64)
        idx = (np.arange(s, e) // n_t)
        for ci in np.unique(idx):
            m = idx == ci
            per_color[ci] += chunk[m].sum(axis=0)
            counts[ci] += int(m.sum())
    per_color /= counts[:, None]

    Xn = (per_color - mu) / np.maximum(sigma, 1e-6)
    Z_full = (Xn - Xn.mean(0, keepdims=True)) @ Vt.T  # (n_c, K)
    Z16 = Z_full[:, :TOP_K]

    colors = cmg.load_xkcd_colors()[:n_c]
    rgb = np.array([(r, g, b) for _, r, g, b in colors], dtype=np.float64) / 255.0
    hsv = cmg.rgb_to_hsv_arr(rgb * 255.0)
    X_rgb = rgb
    X_hsv = np.stack([
        np.cos(2 * np.pi * hsv[:, 0]),
        np.sin(2 * np.pi * hsv[:, 0]),
        hsv[:, 1],
        hsv[:, 2],
    ], axis=1)

    cfg = cmg.Config()
    folds = cmg.kfold_color_indices(n_c, cfg.n_folds)

    # Storage: per-fold per-PC predictions, computed by re-fitting on the
    # corresponding 8-d target slice.
    per_pc_r2_acc = {spec: {"first8": np.zeros((cfg.n_folds, 8)),
                              "last8":  np.zeros((cfg.n_folds, 8))}
                      for spec in (TOP_SPEC, BASELINE)}
    macro_acc = {spec: {"first8": [], "last8": []}
                  for spec in (TOP_SPEC, BASELINE)}

    for f_idx, (tr, te) in enumerate(folds):
        print(f"\n[fold {f_idx}] train={len(tr)} test={len(te)}")
        for spec in (TOP_SPEC, BASELINE):
            for slice_name, slc in (("first8", slice(0, 8)),
                                       ("last8",  slice(8, 16))):
                Z_tr = Z16[tr, slc]
                Z_te = Z16[te, slc]
                _, Z_te_pred = cmg.fit_and_predict(
                    spec, X_rgb[tr], X_hsv[tr], Z_tr,
                    X_rgb[te], X_hsv[te], Z_te, cfg,
                )
                m = macro_r2(Z_te, Z_te_pred)
                per_pc = per_pc_r2(Z_te, Z_te_pred)
                macro_acc[spec][slice_name].append(m)
                per_pc_r2_acc[spec][slice_name][f_idx] = per_pc
                print(f"  {spec:30s} [{slice_name}] macro R²={m:+.4f}  "
                      f"per-PC mean={per_pc.mean():+.4f}")

    # Aggregate
    summary = {}
    for spec in (TOP_SPEC, BASELINE):
        summary[spec] = {}
        for s_name in ("first8", "last8"):
            arr = np.array(macro_acc[spec][s_name])
            pp = per_pc_r2_acc[spec][s_name].mean(0)
            summary[spec][s_name] = dict(mean=float(arr.mean()),
                                          std=float(arr.std()),
                                          per_pc=pp.tolist())

    print("\n=== Macro R² summary ===")
    for spec in (TOP_SPEC, BASELINE):
        f = summary[spec]["first8"]; l = summary[spec]["last8"]
        print(f"  {spec:30s} first8 {f['mean']:+.4f} ± {f['std']:.4f} | "
              f"last8 {l['mean']:+.4f} ± {l['std']:.4f}  "
              f"(Δ = {l['mean']-f['mean']:+.4f})")

    # Did the ordering flip?
    f_top = summary[TOP_SPEC]["first8"]["mean"]
    f_base = summary[BASELINE]["first8"]["mean"]
    l_top = summary[TOP_SPEC]["last8"]["mean"]
    l_base = summary[BASELINE]["last8"]["mean"]
    flip = (f_top - f_base) * (l_top - l_base) < 0
    print(f"\n>>> first8: {TOP_SPEC} - {BASELINE} = {f_top-f_base:+.4f}")
    print(f">>> last8 : {TOP_SPEC} - {BASELINE} = {l_top-l_base:+.4f}")
    print(f">>> FLIP? {'YES' if flip else 'NO'}")

    # Save numeric summary as JSON sidecar.
    sidecar = RUN_DIR / "auto_35.json"
    sidecar.write_text(json.dumps({
        "TOP_SPEC": TOP_SPEC, "BASELINE": BASELINE, "TOP_K": TOP_K,
        "evr_first8_sum": float(evr[:8].sum()),
        "evr_last8_sum": float(evr[8:16].sum()),
        "summary": summary,
        "flip": bool(flip),
    }, indent=2))
    print(f"[saved] {sidecar}")

    # =================================================================
    # Plot
    # =================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          width_ratios=[1.3, 1.0],
                          hspace=0.38, wspace=0.30)

    # (A) Per-PC R² for both specs across all 16 PCs (mean ± std over folds).
    ax = fig.add_subplot(gs[0, :])
    pcs = np.arange(TOP_K)
    for spec, color in ((TOP_SPEC, "#d62728"), (BASELINE, "#1f77b4")):
        pp_all = np.concatenate([per_pc_r2_acc[spec]["first8"],
                                   per_pc_r2_acc[spec]["last8"]], axis=1)
        m = pp_all.mean(0); s = pp_all.std(0)
        ax.errorbar(pcs, m, yerr=s, marker="o", lw=1.5, ms=5,
                     capsize=2, color=color, label=spec)
    ax.axvline(7.5, color="k", lw=0.8, ls="--", alpha=0.6,
               label="first8 / last8 split")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(pcs)
    ax.set_xlabel("PC index (within top-16, 0=highest variance)")
    ax.set_ylabel("R² (5-fold color-grouped CV)")
    ax.set_title("Per-PC R² across the top-16 PCs — refit per slice")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # (B) Macro R² bars: first8 vs last8, both specs.
    ax = fig.add_subplot(gs[1, 0])
    labels = ["first 8 PCs\n(EVR={:.1%})".format(evr[:8].sum()),
              "last 8 of top-16\n(EVR={:.1%})".format(evr[8:16].sum())]
    x = np.arange(2)
    bw = 0.35
    top_means = [summary[TOP_SPEC]["first8"]["mean"],
                  summary[TOP_SPEC]["last8"]["mean"]]
    top_stds = [summary[TOP_SPEC]["first8"]["std"],
                 summary[TOP_SPEC]["last8"]["std"]]
    base_means = [summary[BASELINE]["first8"]["mean"],
                   summary[BASELINE]["last8"]["mean"]]
    base_stds = [summary[BASELINE]["first8"]["std"],
                  summary[BASELINE]["last8"]["std"]]
    ax.bar(x - bw / 2, top_means, bw, yerr=top_stds, color="#d62728",
            label=TOP_SPEC, capsize=4)
    ax.bar(x + bw / 2, base_means, bw, yerr=base_stds, color="#1f77b4",
            label=BASELINE, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Macro R² over PC slice")
    ax.set_title("Macro-R²: first 8 vs last 8 of top-16 PCs")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(top_means, top_stds)):
        ax.text(i - bw / 2, m + (0.005 if m >= 0 else -0.02),
                 f"{m:+.3f}", ha="center", fontsize=8)
    for i, (m, s) in enumerate(zip(base_means, base_stds)):
        ax.text(i + bw / 2, m + (0.005 if m >= 0 else -0.02),
                 f"{m:+.3f}", ha="center", fontsize=8)

    # (C) Spec-difference plot: (TOP - BASELINE) on each PC.
    ax = fig.add_subplot(gs[1, 1])
    pp_top_all = np.concatenate([per_pc_r2_acc[TOP_SPEC]["first8"],
                                   per_pc_r2_acc[TOP_SPEC]["last8"]], axis=1)
    pp_base_all = np.concatenate([per_pc_r2_acc[BASELINE]["first8"],
                                    per_pc_r2_acc[BASELINE]["last8"]], axis=1)
    diff = pp_top_all.mean(0) - pp_base_all.mean(0)
    diff_std = (pp_top_all - pp_base_all).std(0)
    colors_bar = ["#d62728" if v > 0 else "#1f77b4" for v in diff]
    ax.bar(pcs, diff, yerr=diff_std, color=colors_bar, capsize=3)
    ax.axvline(7.5, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(pcs)
    ax.set_xlabel("PC index")
    ax.set_ylabel(f"R²({TOP_SPEC}) − R²({BASELINE})")
    ax.set_title("Per-PC advantage of TOP over BASELINE\n(positive = TOP wins)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "(oooo) Refit best-spec ({}) vs baseline ({}) on first 8 vs last 8 of top-16 PCs\n"
        "L40 cogito residuals, 5-fold color-grouped CV.  "
        "Flip in macro ordering: {}".format(TOP_SPEC, BASELINE,
                                              "YES" if flip else "NO"),
        fontsize=12, y=0.995,
    )
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
