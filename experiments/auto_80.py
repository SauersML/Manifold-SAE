"""auto_80 — Overlay of all unsupervised methods on one chart (option b).

Fresh angle. auto_77 picked the per-family BEST and plotted r2_per_pc vs EVR.
auto_78 collapsed each spec to (bulk, tail) scalars. NEITHER shows every
unsupervised spec's per-PC R² profile on a single chart, so you can SEE which
methods cover the bulk, which mop up the tail, and which are uniformly weak.

For every U_* / N_* spec we plot r2_per_pc_mean[k] (5-fold CV, k=1..64) as a
faint line, grouped/colored by sub-family:

  - Duchon manifold (U_1d..U_8d, U_loop_1d, U_3d_multistart, U_3d_pca_init)
  - PCA-self (U_pca_*d, U_pca_centered_8d_smooth, U_pca*_smooth)
  - PCA + Duchon GAM (U_pca*_duchon_*)
  - NMF (U_nmf_*d)
  - k-means + centroid kde (U_kmeans_*, U_centroid_kde_smooth_3d)
  - kNN regressors (N_knn_*)

Overlaid black step = EVR cumulative, so you can read where the variance is.
Numbers in the legend = macro-R² for each family's best representative.

Outputs:
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_80.png
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_80.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS = Path(
    "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
)
OUT_PNG = RESULTS.parent / "auto_80.png"
OUT_JSON = RESULTS.parent / "auto_80.json"


FAMILIES = [
    # (label, color, pattern_fn)
    ("Duchon manifold (U_Nd)", "#d62728",
     lambda k: re.fullmatch(r"U_\d+d|U_loop_1d|U_3d_multistart|U_3d_pca_init", k) is not None),
    ("PCA self-predict (U_pca_*d)", "#1f77b4",
     lambda k: re.fullmatch(r"U_pca_\d+d|U_pca\d+_smooth|U_pca_centered_\d+d_smooth", k) is not None),
    ("PCA + Duchon GAM (U_pca*_duchon_*)", "#2ca02c",
     lambda k: k.startswith("U_pca") and "duchon" in k),
    ("PCA additive / pairs / tensor", "#9467bd",
     lambda k: re.fullmatch(r"U_pca_add_\d+d|U_pca_pairs_\d+d|U_pca\d+_tensor", k) is not None),
    ("NMF (U_nmf_*d)", "#ff7f0e",
     lambda k: k.startswith("U_nmf_")),
    ("k-means / centroid KDE", "#8c564b",
     lambda k: k.startswith("U_kmeans_") or k.startswith("U_centroid_")),
    ("kNN regressor (N_knn_*)", "#7f7f7f",
     lambda k: k.startswith("N_knn_")),
]


def main() -> int:
    d = json.load(open(RESULTS))
    L = d["per_layer"]["L40"]
    specs = L["specs"]
    evr = np.array(L["explained_variance_ratio_topK"], dtype=float)
    K = len(evr)
    cum_evr = np.cumsum(evr)
    ks = np.arange(1, K + 1)

    # Group specs by family — skip specs that errored (missing r2_macro_mean)
    grouped = {label: [] for label, _, _ in FAMILIES}
    skipped = []
    for sid in specs:
        if "r2_macro_mean" not in specs[sid] or "r2_per_pc_mean" not in specs[sid]:
            skipped.append(sid)
            continue
        for label, _color, match in FAMILIES:
            if match(sid):
                grouped[label].append(sid)
                break
    if skipped:
        print(f"[skip] {len(skipped)} errored specs: {skipped[:8]}{'...' if len(skipped)>8 else ''}")

    # Print summary for the report + save
    summary = {}
    for label, _color, _ in FAMILIES:
        members = grouped[label]
        best = None
        for sid in members:
            m = specs[sid]["r2_macro_mean"]
            if best is None or m > best[1]:
                best = (sid, m)
        summary[label] = {
            "n_members": len(members),
            "best_spec": best[0] if best else None,
            "best_r2_macro": best[1] if best else None,
        }
        print(f"{label:42s}  n={len(members):2d}  best={best[0] if best else '-':35s}  R²={best[1]:+.3f}" if best else label)

    # Figure: 2 panels — (a) raw per-PC curves, (b) per-PC curves with EVR weighting
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True)

    for ax, weighted in zip(axes, [False, True]):
        for label, color, _ in FAMILIES:
            members = grouped[label]
            if not members:
                continue
            # plot each member as a faint line
            best_r2_per_pc = None
            best_macro = -np.inf
            for sid in members:
                r2pc = np.array(specs[sid]["r2_per_pc_mean"], dtype=float)
                y = r2pc * evr if weighted else r2pc
                ax.plot(ks, y, color=color, alpha=0.18, linewidth=1.0)
                if specs[sid]["r2_macro_mean"] > best_macro:
                    best_macro = specs[sid]["r2_macro_mean"]
                    best_r2_per_pc = r2pc
            # bold the best per family
            y = best_r2_per_pc * evr if weighted else best_r2_per_pc
            ax.plot(ks, y, color=color, alpha=0.95, linewidth=2.2,
                    label=f"{label}  ({summary[label]['best_spec']}, R²={best_macro:+.2f})")

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("PC index k (1..64, ordered by descending EVR)", fontsize=11)
        if weighted:
            ax.set_ylabel("EVR_k · R²(PC_k)   (contribution to macro-R²)", fontsize=11)
            ax.set_title("EVR-weighted contribution per PC", fontsize=12)
        else:
            ax.set_ylabel("held-out R²(PC_k)   (5-fold CV)", fontsize=11)
            ax.set_title("Raw per-PC R²", fontsize=12)

        # twin axis for cumulative EVR
        ax2 = ax.twinx()
        ax2.step(ks, cum_evr, color="black", where="mid", linewidth=1.2, alpha=0.55,
                 linestyle="--")
        ax2.set_ylabel("cumulative EVR (dashed black)", fontsize=10)
        ax2.set_ylim(0, 1.02)

        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_xlim(0.5, K + 0.5)

    axes[0].legend(loc="upper right", fontsize=8, frameon=True,
                   title="family (best member)")
    fig.suptitle(
        "All unsupervised methods overlaid — per-PC R² on cogito-L40 color manifold\n"
        f"{sum(len(v) for v in grouped.values())} unsupervised specs · "
        "949 xkcd colors · 64 PCs (cumulative EVR overlaid)",
        fontsize=12.5,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)

    json.dump(
        {"summary_per_family": summary,
         "K": K,
         "cum_evr_at_k8": float(cum_evr[7]),
         "cum_evr_at_k16": float(cum_evr[15])},
        open(OUT_JSON, "w"), indent=2,
    )
    print(f"\n[done] {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
