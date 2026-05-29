"""auto_exp_08: holdout-by-template-group CV (template OOD).

All prior color-manifold experiments split CV by *color*: train on a
subset of colors, evaluate on held-out colors. Templates were always
averaged into the per-color centroid, so every template appears in both
train and test (the model gets to "see" each template phrasing). That
makes the held-out R² a measure of color generalization but says
nothing about whether the fitted manifold generalizes to *prompt
phrasings* the regressor hasn't seen.

This matters for LLM applicability: a real downstream probe / steering
intervention runs on prompts whose phrasing we have not curated. If
template variation re-orients the per-color residual within the same
64-PC target space, then a centroid fit on templates {T_train} will
have systematically wrong predictions on T_test, even for colors it
"knows". That's exactly the OOD failure mode SAE-style probes hide
under averaging.

Method (cheap, no server, no harvest)
-------------------------------------
* Load cached residuals (26572 x 7168) from COLOR_COGITO_L40/X_L40.npy.
* Same 64-PC target basis as exp_04..07 (PCA on per-color centroids).
* Partition the 28 templates into 5 groups of ~6 templates each (seeded
  random shuffle; deterministic). For each fold:
    - Train: all 954 colors × 22 train-templates = 20988 rows (raw).
    - Test:  all 954 colors × 6 test-templates  = 5724 rows (raw).
  Colors are NOT held out — we want to isolate the template-OOD axis.
* For comparison we also run the standard color-holdout CV using the
  same raw-prompt regime so the two OOD axes are directly comparable
  in R^2 units on the same target basis.
* Headline specs (small, matches exp_04 zoo):
    L_lin_rgb, L_lin_lab, L_joint_rgb, L_joint_lab, L_lch_with_cyclic_h
* Output: per-spec macro R^2 and per-PC R^2 under:
    (i)  template-group CV  (color in-distribution, template OOD)
    (ii) color CV           (template in-distribution, color OOD)

Hypotheses
----------
  (H1) Template-OOD R^2 is high (>~ color-OOD R^2). Then the cogito L40
       residual is color-driven and template-invariant — averaging is
       valid and downstream LLM use should transfer to fresh phrasings.
  (H2) Template-OOD R^2 collapses. Then a chunk of what the centroid
       fit calls "color" is actually template-specific structure that
       the 28-template average smears together; the published R^2 is
       optimistic for real-world prompts.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_exp_08_template_ood.{json,png}
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
OUT_JSON = OUT_DIR / "auto_exp_08_template_ood.json"
OUT_PNG = OUT_DIR / "auto_exp_08_template_ood.png"

N_TEMPLATES = 28
N_PCS = 64
N_FOLDS = 5
SEED = 0

SPECS = [
    "L_lin_rgb",
    "L_lin_lab",
    "L_joint_rgb",
    "L_joint_lab",
    "L_lch_with_cyclic_h",
]


def template_folds(n_templates: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_t_idx, test_t_idx). Seeded random partition
    of {0..n_templates-1} into ``n_folds`` near-equal chunks."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_templates)
    splits = np.array_split(order, n_folds)
    folds = []
    for i in range(n_folds):
        test_t = np.sort(splits[i])
        train_t = np.sort(np.concatenate([splits[j] for j in range(n_folds) if j != i]))
        folds.append((train_t, test_t))
    return folds


def build_coords(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import colorsys
    hsv = np.zeros((rgb01.shape[0], 3), dtype=np.float64)
    for i in range(rgb01.shape[0]):
        hsv[i] = colorsys.rgb_to_hsv(*rgb01[i])
    X_hsv4 = np.stack([
        np.cos(2 * np.pi * hsv[:, 0]),
        np.sin(2 * np.pi * hsv[:, 0]),
        hsv[:, 1], hsv[:, 2],
    ], axis=1)
    return rgb01, X_hsv4


def run_cv(
    spec: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    row_train_mask_fn,   # (fold_i) -> bool array (N,)
    row_test_mask_fn,    # (fold_i) -> bool array (N,)
    X_rgb_per_prompt: np.ndarray,
    X_hsv_per_prompt: np.ndarray,
    Z_prompt: np.ndarray,
    cfg: cmg.Config,
    label: str,
) -> dict:
    macro_r2s, per_pc_r2s = [], []
    t0 = time.time()
    for fi in range(len(folds)):
        tr = row_train_mask_fn(fi)
        te = row_test_mask_fn(fi)
        tr_X_rgb = X_rgb_per_prompt[tr]
        tr_X_hsv = X_hsv_per_prompt[tr]
        tr_Z = Z_prompt[tr]
        te_X_rgb = X_rgb_per_prompt[te]
        te_X_hsv = X_hsv_per_prompt[te]
        te_Z = Z_prompt[te]
        try:
            _, te_pred = cmg.fit_and_predict(
                spec, tr_X_rgb, tr_X_hsv, tr_Z,
                te_X_rgb, te_X_hsv, te_Z, cfg,
            )
        except Exception as exc:
            print(f"  [fold {fi}] {label} FAILED: {exc}", flush=True)
            continue
        ss_res = ((te_Z - te_pred) ** 2).sum(0)
        ss_tot = ((te_Z - te_Z.mean(0, keepdims=True)) ** 2).sum(0)
        per_pc = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        macro = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
        macro_r2s.append(float(macro))
        per_pc_r2s.append(per_pc)
        print(f"  [fold {fi}] {label:14s}  ntrain={tr.sum():5d}  "
              f"ntest={te.sum():5d}  macro R^2={macro:+.4f}", flush=True)
    per_pc_mean = (np.nanmean(np.stack(per_pc_r2s, 0), axis=0).tolist()
                   if per_pc_r2s else [float("nan")] * N_PCS)
    return {
        "macro_r2_mean": float(np.mean(macro_r2s)) if macro_r2s else float("nan"),
        "macro_r2_std": float(np.std(macro_r2s)) if macro_r2s else float("nan"),
        "per_fold_macro_r2": macro_r2s,
        "per_pc_r2_mean": per_pc_mean,
        "elapsed_s": time.time() - t0,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)
    N, D = X.shape
    assert N % N_TEMPLATES == 0
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    colors = cmg.load_xkcd_colors()
    assert len(colors) == n_colors
    rgb01 = np.array([[r, g, b] for _, r, g, b in colors], dtype=np.float64) / 255.0
    X_rgb_per_color, X_hsv_per_color = build_coords(rgb01)
    X_rgb_per_prompt = X_rgb_per_color[c_idx]
    X_hsv_per_prompt = X_hsv_per_color[c_idx]

    # ---- centroid + standardize + 64-PC target basis ----
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    centroids_n = (centroids - mu) / sigma
    X_n = (X - mu) / sigma
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    Z_prompt = X_n @ V_topK.T
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] top-{N_PCS} EVR sum = {evr.sum():.3f}", flush=True)

    # ---- Folds ----
    t_folds = template_folds(N_TEMPLATES, N_FOLDS, seed=SEED)
    print("[folds:template] groups:")
    for fi, (tr_t, te_t) in enumerate(t_folds):
        print(f"  fold {fi}: test_templates={te_t.tolist()}", flush=True)
    c_folds = cmg.kfold_color_indices(n_colors, N_FOLDS, seed=SEED)
    print(f"[folds:color] {N_FOLDS} color folds, "
          f"sizes={[len(te) for _, te in c_folds]}", flush=True)

    # Precompute boolean masks per fold to avoid recomputing each spec
    t_train_masks = [np.isin(t_idx, tr_t) for tr_t, _ in t_folds]
    t_test_masks  = [np.isin(t_idx, te_t) for _, te_t in t_folds]
    c_train_masks = [np.isin(c_idx, tr_c) for tr_c, _ in c_folds]
    c_test_masks  = [np.isin(c_idx, te_c) for _, te_c in c_folds]

    cfg = cmg.Config(layers=(40,), n_pcs=N_PCS, n_folds=N_FOLDS,
                     lattice_per_side=5, init_log_lambda=0.0,
                     output_dir=str(OUT_DIR), harvest_from=str(HARVEST))

    results: dict[str, dict] = {}
    for spec in SPECS:
        print(f"\n=== spec={spec} ===", flush=True)
        results[spec] = {}
        print("-- template-OOD CV (color in-dist) --", flush=True)
        results[spec]["template_ood"] = run_cv(
            spec, t_folds,
            lambda fi: t_train_masks[fi], lambda fi: t_test_masks[fi],
            X_rgb_per_prompt, X_hsv_per_prompt, Z_prompt, cfg, "template_ood")
        print("-- color-OOD CV (template in-dist) --", flush=True)
        results[spec]["color_ood"] = run_cv(
            spec, c_folds,
            lambda fi: c_train_masks[fi], lambda fi: c_test_masks[fi],
            X_rgb_per_prompt, X_hsv_per_prompt, Z_prompt, cfg, "color_ood")

    summary = {
        "config": {"harvest": str(HARVEST), "n_colors": n_colors,
                   "n_templates": N_TEMPLATES, "n_pcs": N_PCS,
                   "n_folds": N_FOLDS, "specs": SPECS, "seed": SEED},
        "template_folds": [{"train": tr.tolist(), "test": te.tolist()}
                            for tr, te in t_folds],
        "explained_variance_ratio_topK": evr.tolist(),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] -> {OUT_JSON}", flush=True)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})

    macro_t = [results[s]["template_ood"]["macro_r2_mean"] for s in SPECS]
    macro_c = [results[s]["color_ood"]["macro_r2_mean"] for s in SPECS]
    std_t = [results[s]["template_ood"]["macro_r2_std"] for s in SPECS]
    std_c = [results[s]["color_ood"]["macro_r2_std"] for s in SPECS]
    x = np.arange(len(SPECS))
    w = 0.38
    axes[0].bar(x - w/2, macro_t, w, yerr=std_t, label="template OOD (color in-dist)",
                color="#c06040", capsize=3)
    axes[0].bar(x + w/2, macro_c, w, yerr=std_c, label="color OOD (template in-dist)",
                color="#4060a0", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SPECS, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("held-out macro R^2 (per-prompt test)")
    axes[0].set_title("Template-OOD vs Color-OOD generalization  (cogito L40, 64 PCs)")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].legend(fontsize=8)
    for xi, (a, b) in enumerate(zip(macro_t, macro_c)):
        axes[0].text(xi - w/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
        axes[0].text(xi + w/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)

    K = N_PCS
    for s in SPECS:
        pc_t = np.array(results[s]["template_ood"]["per_pc_r2_mean"])
        pc_c = np.array(results[s]["color_ood"]["per_pc_r2_mean"])
        axes[1].plot(np.arange(K), pc_t - pc_c, label=s, linewidth=1.4, alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("PC index (centroid PCA basis, ordered by EVR)")
    axes[1].set_ylabel("delta R^2  =  template-OOD - color-OOD")
    axes[1].set_title("Per-PC gap (positive = template variation easier than color extrap)")
    axes[1].grid(linestyle=":", alpha=0.4)
    axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
