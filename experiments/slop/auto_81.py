"""auto_81 — Per-PC predictability distribution across all 108 specs.

Genuinely new angle vs auto_77-80:
  - auto_77/80 plot per-PC R² lines (one per spec/family).
  - auto_78 collapses each spec to (bulk, tail) scalars.
  - auto_79 is geometry, not GAM-zoo results.

NONE of them aggregate ACROSS specs at each fixed PC index to ask:
  "How predictable is PC_k itself, across the entire 108-spec zoo?"

Output (single figure, 2 panels):
  (a) Box+strip plot of held-out R² at each PC index k=1..64, where every
      box pools all 108 specs. Reveals which PCs are universally easy
      (small box, high median) vs adversarial (wide spread or low ceiling).
  (b) Scatter: EVR_k (x) vs best-of-zoo R² achieved at PC_k (y), one dot
      per PC, colored by k. Tests whether high-variance PCs are inherently
      easier to predict from color attributes — i.e. whether cogito-L40 is
      "linear-color in the bulk, noise in the tail" or genuinely diverse.

Outputs:
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_81.png
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_81.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


RESULTS = Path(
    "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
)
OUT_PNG = RESULTS.parent / "auto_81.png"
OUT_JSON = RESULTS.parent / "auto_81.json"


def main() -> int:
    d = json.load(open(RESULTS))
    L = d["per_layer"]["L40"]
    specs = L["specs"]
    evr = np.array(L["explained_variance_ratio_topK"], dtype=float)
    K = len(evr)

    # Build (n_specs, K) matrix of per-PC R²
    # Exclude trivially self-predicting specs: U_pca_Nd with N >= K reconstruct
    # the same K PCs they're built from, so R²≡1.0 by construction. Keep only
    # specs that don't have this leakage.
    def is_trivial(sid: str) -> bool:
        import re
        m = re.fullmatch(r"U_pca_(\d+)d", sid)
        if m and int(m.group(1)) >= K:
            return True
        return False

    sids = [s for s in specs
             if "r2_per_pc_mean" in specs[s] and not is_trivial(s)]
    n_excluded = sum(1 for s in specs if is_trivial(s))
    print(f"[auto_81] excluded {n_excluded} trivially-self-predicting specs")
    M = np.array([specs[s]["r2_per_pc_mean"] for s in sids], dtype=float)
    print(f"[auto_81] n_specs={M.shape[0]}  K={M.shape[1]}")

    # Clip the box display to a reasonable range (some specs go very negative)
    M_clip = np.clip(M, -1.0, 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(16, 11),
                              gridspec_kw={"height_ratios": [1.4, 1.0]})

    # ---- Panel (a) — per-PC distribution across all specs ----
    ax = axes[0]
    ks = np.arange(1, K + 1)
    # Box plot
    bp = ax.boxplot(
        [M_clip[:, k] for k in range(K)],
        positions=ks, widths=0.65, showfliers=False, patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfdee9")
        patch.set_edgecolor("#356d96")
        patch.set_linewidth(0.7)
    for med in bp["medians"]:
        med.set_color("#b22222")
        med.set_linewidth(1.4)

    # Overlay: best spec per PC (max across specs), and worst (min, clipped)
    best_per_pc = M.max(axis=0)
    median_per_pc = np.median(M, axis=0)
    ax.plot(ks, best_per_pc, color="#1a7a1a", linewidth=1.6,
            marker="o", markersize=3, label="best spec at PC_k", zorder=5)
    ax.plot(ks, median_per_pc, color="#b22222", linewidth=1.0,
            linestyle="--", label="median across 108 specs", zorder=4)

    # Twin axis: EVR
    ax2 = ax.twinx()
    ax2.bar(ks, evr, color="black", alpha=0.10, width=0.85,
            label="EVR_k")
    ax2.set_ylabel("EVR_k (bars, faint)", fontsize=10)
    ax2.set_ylim(0, max(evr) * 1.05)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.6)
    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylim(-0.6, 1.05)
    ax.set_xlabel("PC index k (1..64, descending EVR)", fontsize=11)
    ax.set_ylabel("held-out R²(PC_k)   (5-fold CV)", fontsize=11)
    ax.set_title(
        f"Per-PC predictability — aggregated across {M.shape[0]} zoo specs\n"
        "(each box pools R²(PC_k) over every supervised + unsupervised spec)",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    # Only label every 4th tick to avoid clutter
    ax.set_xticks(ks[::4])
    ax.set_xticklabels([str(k) for k in ks[::4]])

    # ---- Panel (b) — EVR vs best-of-zoo R² per PC ----
    ax = axes[1]
    colors = cm.viridis(np.linspace(0, 1, K))
    ax.scatter(evr, best_per_pc, c=colors, s=55, edgecolor="black",
                linewidth=0.5, zorder=3)
    for k in range(K):
        if k < 8 or evr[k] > 0.02 or best_per_pc[k] < 0.0:
            ax.annotate(f"PC{k+1}", (evr[k], best_per_pc[k]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=7, alpha=0.85)
    # Reference: weighted-mean R² of the best macro spec
    best_macro_sid = max(sids, key=lambda s: specs[s]["r2_macro_mean"])
    best_macro_val = specs[best_macro_sid]["r2_macro_mean"]
    ax.axhline(best_macro_val, color="#b22222", linestyle=":",
                linewidth=1.1,
                label=f"best zoo macro-R² = {best_macro_val:+.3f} ({best_macro_sid})")

    ax.set_xscale("log")
    ax.set_xlabel("EVR_k  (log scale)", fontsize=11)
    ax.set_ylabel("best-of-zoo R²(PC_k)", fontsize=11)
    ax.set_title(
        "Does PC variance predict PC predictability?\n"
        "Dot = (EVR_k, max-over-108-specs R²) for one PC.  "
        "Color = PC index (viridis: low→high k).",
        fontsize=12,
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    # Spearman to quantify
    from scipy.stats import spearmanr
    rho_evr_best, p_evr_best = spearmanr(evr, best_per_pc)
    ax.text(
        0.02, 0.97,
        f"Spearman ρ(EVR_k, best R²_k) = {rho_evr_best:+.3f}  (p={p_evr_best:.2g})",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)

    out = {
        "n_specs": int(M.shape[0]),
        "K": int(K),
        "best_macro_spec": best_macro_sid,
        "best_macro_r2": float(best_macro_val),
        "best_per_pc": best_per_pc.tolist(),
        "median_per_pc": median_per_pc.tolist(),
        "evr": evr.tolist(),
        "spearman_evr_vs_best_r2_per_pc": {
            "rho": float(rho_evr_best),
            "p": float(p_evr_best),
        },
        # which PC has biggest spread (most "adversarial")?
        "iqr_per_pc": (np.percentile(M_clip, 75, axis=0)
                        - np.percentile(M_clip, 25, axis=0)).tolist(),
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"[done] {OUT_PNG}")
    print(f"[done] {OUT_JSON}")
    print(f"  best macro spec: {best_macro_sid}  R²={best_macro_val:+.3f}")
    print(f"  Spearman ρ(EVR, best-R²) = {rho_evr_best:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
