"""auto_24: per-PC RGB-linear-baseline explainability (idea hhh).

For each of the 64 PCs of the cogito L40 per-color centroids, compare:
  - L_lin_rgb's per-PC R^2 (the RGB linear baseline)
  - the BEST per-PC R^2 achieved by ANY other spec (the ceiling)
  - the gain (ceiling - RGB baseline)

This isolates which PCs are "RGB-explainable" (small gain) vs which
PCs encode color information that is non-linear / non-RGB (large gain
beyond RGB linear).

Cross-references with the eigenvalue spectrum (variance per PC) to
weight the importance of each PC.

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_24.{png,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN / "results.json"
OUT_PNG = RUN / "auto_24.png"
OUT_JSON = RUN / "auto_24.json"

BASELINE = "L_lin_rgb"
TOP_N_PCS_SHOW = 32  # focus on PCs that actually carry variance


def main() -> None:
    d = json.loads(RESULTS.read_text())
    L = d["per_layer"]["L40"]
    specs = L["specs"]
    evr = np.array(L["explained_variance_ratio_topK"])  # (64,)
    n_pcs = len(evr)

    baseline_r2 = np.array(specs[BASELINE]["r2_per_pc_mean"])  # (64,)

    # For each PC, find best spec (excluding baseline) and its R^2.
    best_r2 = np.full(n_pcs, -np.inf)
    best_spec = [""] * n_pcs
    # Exclude unsupervised PC-target specs (they trivially fit PC targets
    # since their latents are derived from the same PCs) and constant-mean.
    EXCLUDE_PREFIXES = ("U_", "L_const_mean")
    for sn, sd in specs.items():
        if sn == BASELINE:
            continue
        if sn.startswith(EXCLUDE_PREFIXES):
            continue
        if "r2_per_pc_mean" not in sd:
            continue
        r2pp = np.array(sd["r2_per_pc_mean"])
        for i in range(n_pcs):
            if r2pp[i] > best_r2[i]:
                best_r2[i] = r2pp[i]
                best_spec[i] = sn

    gain = best_r2 - baseline_r2

    # Weight: how much each PC matters (variance-weighted gain)
    weighted_gain = gain * evr
    weighted_base = baseline_r2 * evr

    # ----- Plot -----
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    x = np.arange(n_pcs)
    keep = slice(0, TOP_N_PCS_SHOW)

    # (a) per-PC R^2: baseline vs ceiling
    ax = axes[0]
    ax.bar(x[keep] - 0.2, baseline_r2[keep], width=0.4,
           label=f"{BASELINE} (RGB linear baseline)", color="#888")
    ax.bar(x[keep] + 0.2, best_r2[keep], width=0.4,
           label="best non-RGB-linear spec (ceiling)", color="#c44")
    ax.set_xticks(x[keep])
    ax.set_xlabel("PC index")
    ax.set_ylabel("per-PC R²")
    ax.set_title(f"Per-PC R²: RGB-linear baseline vs ceiling (top {TOP_N_PCS_SHOW} PCs)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) gain over RGB-linear baseline, annotated with winning spec
    ax = axes[1]
    colors = ["#2a8" if g > 0.05 else "#bbb" for g in gain[keep]]
    ax.bar(x[keep], gain[keep], color=colors)
    ax.set_xticks(x[keep])
    ax.set_xlabel("PC index")
    ax.set_ylabel("R² gain over RGB-linear")
    ax.set_title("Gain beyond RGB linear (green = >0.05 gain). Spec name labels for top-5 gains.")
    # label top-5 gains
    order = np.argsort(-gain[:TOP_N_PCS_SHOW])
    for j in order[:5]:
        ax.annotate(best_spec[j], xy=(j, gain[j]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3)

    # (c) variance-weighted gain + cumulative
    ax = axes[2]
    ax2 = ax.twinx()
    ax.bar(x[keep], weighted_gain[keep] * 100, color="#2a8", alpha=0.7,
           label="variance-weighted gain (%)")
    ax.bar(x[keep], weighted_base[keep] * 100, color="#888", alpha=0.5,
           bottom=weighted_gain[keep] * 100, label="variance-weighted baseline R² (%)")
    cum_total_explained = np.cumsum((baseline_r2 + gain) * evr) * 100
    cum_baseline_explained = np.cumsum(baseline_r2 * evr) * 100
    ax2.plot(x, cum_total_explained, "r-", lw=2, label="cum. variance fit by ceiling (%)")
    ax2.plot(x, cum_baseline_explained, "k--", lw=1.5, label="cum. variance fit by RGB-linear (%)")
    ax.set_xticks(x[keep])
    ax.set_xlabel("PC index")
    ax.set_ylabel("variance-weighted R² contribution (% of total var)")
    ax2.set_ylabel("cumulative variance-weighted R² (%)")
    ax.set_title("Variance-weighted contribution per PC (left), cumulative (right)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    total_baseline = float(np.sum(baseline_r2 * evr))
    total_ceiling = float(np.sum(best_r2 * evr))
    fig.suptitle(
        f"auto_24 (hhh): per-PC RGB-linear explainability  |  "
        f"variance-weighted: baseline={total_baseline:.3f}, ceiling={total_ceiling:.3f}, "
        f"gain={total_ceiling - total_baseline:.3f}",
        fontsize=11,
    )
    fig.savefig(OUT_PNG, dpi=130)

    # ----- Save JSON -----
    summary = {
        "baseline_spec": BASELINE,
        "baseline_r2_per_pc": baseline_r2.tolist(),
        "ceiling_r2_per_pc": best_r2.tolist(),
        "best_spec_per_pc": best_spec,
        "gain_per_pc": gain.tolist(),
        "explained_variance_ratio_per_pc": evr.tolist(),
        "variance_weighted_baseline_total": total_baseline,
        "variance_weighted_ceiling_total": total_ceiling,
        "variance_weighted_gain_total": total_ceiling - total_baseline,
        "top5_gain_pcs": [
            {"pc": int(j), "gain": float(gain[j]),
             "baseline_r2": float(baseline_r2[j]),
             "ceiling_r2": float(best_r2[j]),
             "best_spec": best_spec[j],
             "evr": float(evr[j])}
            for j in np.argsort(-gain)[:5]
        ],
        "top5_rgb_explainable_pcs": [
            {"pc": int(j), "baseline_r2": float(baseline_r2[j]),
             "gain": float(gain[j]), "evr": float(evr[j])}
            for j in np.argsort(-baseline_r2)[:5]
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PNG} and {OUT_JSON}")
    print(f"variance-weighted: baseline={total_baseline:.3f} ceiling={total_ceiling:.3f} "
          f"gain={total_ceiling - total_baseline:.3f}")


if __name__ == "__main__":
    main()
