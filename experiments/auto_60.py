"""auto_60: Bayes-optimal R^2 ceiling from per-color template variance.

Idea (iiiiiii)
--------------
Every per-prompt row decomposes as

    Z[c, t, :] = mu_c + epsilon[c, t, :]

where ``mu_c`` is the (unobservable) true per-color signal and
``epsilon[c, t, :]`` is template noise. Any regressor whose inputs are
*color only* (RGB/HSV/Lab/...) cannot do better than predicting mu_c:
the template axis is, by construction, invisible to it. So the
information-theoretic ceiling on per-prompt R^2 in the 64-PC target
basis is

    R^2_ceiling = 1 - SS_within / SS_total

where
    SS_within = sum_{c,t} || Z[c,t] - mean_t Z[c, .] ||^2
    SS_total  = sum_{c,t} || Z[c,t] - global_mean ||^2

Estimated unbiasedly from the 28 templates per color (no extra fitting,
no Gaussian RBF, no Duchon — just variance bookkeeping in the same
PCA target basis the headline R^2 numbers use). Compared against the
best per-spec macro R^2 from results.json's specs zoo so we can see how
much of the "missing" R^2 is reducible (model could do better) vs
irreducible (template noise floor).

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_60.{json,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color_manifold_gam as cmg


HARVEST = Path("/Users/user/Manifold-SAE/runs/COLOR_COGITO_L40/X_L40.npy")
RESULTS_JSON = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json")
OUT_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT_JSON = OUT_DIR / "auto_60.json"
OUT_PNG = OUT_DIR / "auto_60.png"

N_TEMPLATES = 28
N_PCS = 64


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {HARVEST}", flush=True)
    X = np.load(HARVEST).astype(np.float64)  # (N, D), N = n_colors * n_templates
    N, D = X.shape
    assert N % N_TEMPLATES == 0, f"N={N} not divisible by n_templates={N_TEMPLATES}"
    n_colors = N // N_TEMPLATES
    c_idx = np.repeat(np.arange(n_colors), N_TEMPLATES)
    t_idx = np.tile(np.arange(N_TEMPLATES), n_colors)
    print(f"[load] X={X.shape}  n_colors={n_colors}", flush=True)

    # --- Same 64-PC target basis as the published spec results: centered+std
    #     centroids -> SVD, take top K right singular vectors as basis.
    centroids = np.zeros((n_colors, D), dtype=np.float64)
    for ci in range(n_colors):
        centroids[ci] = X[c_idx == ci].mean(0)
    mu = centroids.mean(0, keepdims=True)
    sigma = centroids.std(0, keepdims=True).clip(min=1e-6)
    X_n = (X - mu) / sigma
    centroids_n = (centroids - mu) / sigma
    Cc = centroids_n - centroids_n.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    V_topK = Vt[:N_PCS]
    Z_prompt = X_n @ V_topK.T  # (N, K)
    evr = (s ** 2 / (s ** 2).sum())[:N_PCS]
    print(f"[pca] EVR top-{N_PCS} sum = {evr.sum():.3f}", flush=True)

    # --- Per-color centroid in PC basis and within/total decomposition ---
    Z_per_color_mean = np.zeros((n_colors, N_PCS), dtype=np.float64)
    for ci in range(n_colors):
        Z_per_color_mean[ci] = Z_prompt[c_idx == ci].mean(0)
    Z_color_for_prompt = Z_per_color_mean[c_idx]  # (N, K)
    Z_grand = Z_prompt.mean(0, keepdims=True)  # (1, K)

    resid_within = Z_prompt - Z_color_for_prompt   # template noise
    resid_total = Z_prompt - Z_grand               # all variability

    ss_within_pc = (resid_within ** 2).sum(axis=0)  # (K,)
    ss_total_pc = (resid_total ** 2).sum(axis=0)    # (K,)
    ss_between_pc = ss_total_pc - ss_within_pc      # = sum_c n_t * (mu_c - grand)^2

    per_pc_ceiling = 1.0 - ss_within_pc / np.maximum(ss_total_pc, 1e-12)
    macro_ceiling = 1.0 - ss_within_pc.sum() / max(ss_total_pc.sum(), 1e-12)

    # EVR-weighted ceiling: the per-PC ceilings weighted by how much of
    # the centroid-PCA variance each PC owns (closer to what we care about).
    evr_weighted_ceiling = float(np.sum(per_pc_ceiling * evr) / np.sum(evr))

    # Naive variance ratio: average within-color std vs total std per PC.
    # Useful as a per-PC noise-floor visualization.
    sd_within_pc = np.sqrt(ss_within_pc / max(N - n_colors, 1))
    sd_total_pc = np.sqrt(ss_total_pc / max(N - 1, 1))

    print(f"[ceiling] macro R^2 ceiling = {macro_ceiling:.4f}")
    print(f"[ceiling] EVR-weighted ceiling = {evr_weighted_ceiling:.4f}")
    print(f"[ceiling] per-PC: min={per_pc_ceiling.min():.3f}  "
          f"max={per_pc_ceiling.max():.3f}  median={np.median(per_pc_ceiling):.3f}")

    # --- Per-color "consistency" score: 1 - within_c / total_c (all PCs) ---
    per_color_within = np.zeros(n_colors, dtype=np.float64)
    per_color_total = np.zeros(n_colors, dtype=np.float64)
    for ci in range(n_colors):
        rows = Z_prompt[c_idx == ci]
        per_color_within[ci] = ((rows - rows.mean(0, keepdims=True)) ** 2).sum()
        per_color_total[ci] = ((rows - Z_grand) ** 2).sum()
    per_color_consistency = 1.0 - per_color_within / np.maximum(per_color_total, 1e-12)

    # --- Pull published per-spec macro R^2 to compare against the ceiling ---
    spec_macros: dict[str, float] = {}
    try:
        published = json.loads(RESULTS_JSON.read_text())
        specs_block = published["per_layer"]["L40"].get("specs", {})
        for sname, sval in specs_block.items():
            # Try common keys; tolerate missing.
            if isinstance(sval, dict):
                for k in ("macro_r2_mean", "macro_r2", "mean_macro_r2",
                          "r2", "mean_r2", "macro"):
                    if k in sval and isinstance(sval[k], (int, float)):
                        spec_macros[sname] = float(sval[k])
                        break
                else:
                    # try nested 'cv'
                    cv = sval.get("cv") if isinstance(sval, dict) else None
                    if isinstance(cv, dict) and "macro_r2_mean" in cv:
                        spec_macros[sname] = float(cv["macro_r2_mean"])
    except Exception as exc:
        print(f"[warn] could not parse published spec R^2: {exc}", flush=True)
    print(f"[published] parsed {len(spec_macros)} spec macro R^2 values")

    # --- Save JSON summary ---
    summary = {
        "config": {
            "harvest": str(HARVEST),
            "n_colors": int(n_colors),
            "n_templates": int(N_TEMPLATES),
            "n_pcs": int(N_PCS),
        },
        "macro_r2_ceiling": float(macro_ceiling),
        "evr_weighted_r2_ceiling": float(evr_weighted_ceiling),
        "per_pc_ceiling": per_pc_ceiling.tolist(),
        "per_pc_evr": evr.tolist(),
        "ss_within_pc": ss_within_pc.tolist(),
        "ss_between_pc": ss_between_pc.tolist(),
        "ss_total_pc": ss_total_pc.tolist(),
        "sd_within_pc": sd_within_pc.tolist(),
        "sd_total_pc": sd_total_pc.tolist(),
        "per_color_consistency": per_color_consistency.tolist(),
        "published_spec_macro_r2": spec_macros,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {OUT_JSON}")

    # --- Plot ---
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) per-PC ceiling vs PC index, colored by EVR
    ax = axes[0, 0]
    pcs = np.arange(N_PCS)
    sc = ax.scatter(pcs, per_pc_ceiling, c=np.log10(np.maximum(evr, 1e-6)),
                    cmap="viridis", s=24)
    ax.axhline(macro_ceiling, color="crimson", lw=1.2, ls="--",
               label=f"macro ceiling = {macro_ceiling:.3f}")
    ax.axhline(evr_weighted_ceiling, color="black", lw=1.0, ls=":",
               label=f"EVR-weighted = {evr_weighted_ceiling:.3f}")
    ax.set_xlabel("PC index (0 = largest centroid-variance direction)")
    ax.set_ylabel(r"Bayes-optimal $R^2$ ceiling (per-PC)")
    ax.set_title("Per-PC noise floor: 1 - SS_within(template) / SS_total")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower left")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("log10 EVR")

    # (b) within vs between std per-PC (log)
    ax = axes[0, 1]
    sd_between_pc = np.sqrt(np.maximum(ss_between_pc, 0) / max(N - 1, 1))
    ax.plot(pcs, sd_within_pc, "o-", color="#c06040",
            label="within-color (template) sd", ms=3, lw=1)
    ax.plot(pcs, sd_between_pc, "s-", color="#4060a0",
            label="between-color sd", ms=3, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("PC index")
    ax.set_ylabel("standard deviation (PC units)")
    ax.set_title("Signal (between-color) vs noise (within-color template) per PC")
    ax.grid(linestyle=":", alpha=0.4, which="both")
    ax.legend(fontsize=8)

    # (c) achieved vs ceiling for published specs
    ax = axes[1, 0]
    if spec_macros:
        names = list(spec_macros.keys())
        vals = [spec_macros[n] for n in names]
        order = np.argsort(vals)[::-1]
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]
        y = np.arange(len(names))
        ax.barh(y, vals, color="#4060a0", label="published macro R^2")
        ax.axvline(macro_ceiling, color="crimson", lw=1.5, ls="--",
                   label=f"ceiling = {macro_ceiling:.3f}")
        ax.axvline(evr_weighted_ceiling, color="black", lw=1.0, ls=":",
                   label=f"EVR-weighted = {evr_weighted_ceiling:.3f}")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("macro R^2 (per-prompt target basis)")
        ax.set_title("Published per-spec R^2 vs Bayes-optimal ceiling")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")
        gap = macro_ceiling - max(vals) if vals else float("nan")
        ax.text(0.02, 0.02, f"best spec = {max(vals):.3f}\n"
                            f"reducible gap = {gap:+.3f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
    else:
        ax.text(0.5, 0.5, "no per-spec macro R^2 found in results.json",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Published per-spec R^2 (missing)")

    # (d) histogram of per-color consistency (Bayes-optimal-per-color)
    ax = axes[1, 1]
    ax.hist(per_color_consistency, bins=40, color="#60a060", edgecolor="black", lw=0.4)
    ax.axvline(np.median(per_color_consistency), color="black", ls="--", lw=1,
               label=f"median = {np.median(per_color_consistency):.3f}")
    ax.axvline(macro_ceiling, color="crimson", ls=":", lw=1.2,
               label=f"global ceiling = {macro_ceiling:.3f}")
    ax.set_xlabel(r"per-color $1 - SS_{within}/SS_{total}$  (all 64 PCs)")
    ax.set_ylabel("# colors")
    ax.set_title("Per-color consistency — left tail = phrasing-fragile colors")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(
        "auto_60 — Bayes-optimal R^2 ceiling from per-color template variance  "
        "(cogito L40, 64 PCs, 28 templates × {n_colors} colors)".format(n_colors=n_colors),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[done] -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
