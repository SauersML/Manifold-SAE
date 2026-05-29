"""auto_32: (eeee) Compare R^2 across RGB vs HSV vs CIELab feature spaces.

Reads runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json and groups specs by
(model_family, color_space) where color_space in {RGB, HSV, Lab}. For each
family present in 2+ color spaces, plot grouped bars with std error bars,
plus a per-PC R^2 trace for the three L_lin_* baselines.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
OUT = RUN_DIR / "auto_32.png"


def color_space(name: str) -> str | None:
    n = name.lower()
    # avoid matching 'lab' inside 'oklab' separately
    if "oklab" in n:
        return "OKLab"
    if re.search(r"(^|_)lab(_|$)", n):
        return "Lab"
    if "hsv" in n:
        return "HSV"
    if "rgb" in n:
        return "RGB"
    return None


def family(name: str) -> str | None:
    # strip leading 'L_'/'N_'/'M_' and color-space token
    n = name
    # known prefixes
    fam_map = [
        ("L_lin_", "lin"),
        ("L_add_", "add"),
        ("L_joint_", "joint"),
        ("L_poly_", "poly2"),
        ("L_poly3_", "poly3"),
        ("L_poly4_", "poly4"),
        ("N_knn_", "knn"),
    ]
    for pref, fam in fam_map:
        if n.startswith(pref):
            # ignore decorated variants like '_with_hue', '_with_h', 'k10', 'combo'
            tail = n[len(pref):]
            # for knn, strip _kNN suffix
            if fam == "knn":
                tail = re.sub(r"_k\d+$", "", tail)
            # only accept pure space tokens
            if tail.lower() in ("rgb", "hsv", "lab", "oklab"):
                return fam
            return None
    return None


def main() -> None:
    d = json.loads((RUN_DIR / "results.json").read_text())
    specs = d["per_layer"]["L40"]["specs"]

    # Collect (family, space) -> (r2, std)
    grid: dict[tuple[str, str], tuple[float, float]] = {}
    for n, s in specs.items():
        r2 = s.get("r2_macro_mean")
        if r2 is None:
            continue
        fam = family(n)
        sp = color_space(n)
        if fam is None or sp is None:
            continue
        grid[(fam, sp)] = (r2, s.get("r2_macro_std", 0.0))

    families = ["lin", "add", "poly2", "poly3", "poly4", "joint", "knn"]
    spaces = ["RGB", "HSV", "Lab", "OKLab"]
    space_color = {"RGB": "#d62728", "HSV": "#2ca02c", "Lab": "#1f77b4", "OKLab": "#9467bd"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.6, 1.0]})

    # --- Left: grouped bars per family per color space ---
    ax = axes[0]
    x = np.arange(len(families))
    w = 0.2
    for i, sp in enumerate(spaces):
        rs, es, present = [], [], []
        for f in families:
            v = grid.get((f, sp))
            if v is None:
                rs.append(0.0); es.append(0.0); present.append(False)
            else:
                rs.append(v[0]); es.append(v[1]); present.append(True)
        bars = ax.bar(x + (i - 1.5) * w, rs, w, yerr=es, color=space_color[sp],
                      label=sp, alpha=0.85, capsize=3, edgecolor="black", linewidth=0.4)
        for j, b in enumerate(bars):
            if not present[j]:
                b.set_alpha(0.08)
            else:
                ax.text(b.get_x() + b.get_width() / 2, rs[j] + es[j] + 0.003,
                        f"{rs[j]:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("R^2 (macro, 5-fold mean)")
    ax.set_xlabel("model family")
    ax.set_title("L40 cogito centroids: R^2 by color-space features × model family\n(fadeded bars = spec absent in results.json)")
    ax.legend(title="features", loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(0.27, max(v[0] for v in grid.values()) * 1.15))

    # --- Right: per-PC R^2 traces for L_lin_{rgb,hsv,lab} ---
    ax = axes[1]
    for sp_key, sp_label in [("rgb", "RGB"), ("hsv", "HSV"), ("lab", "Lab"), ("oklab", "OKLab")]:
        spec_name = f"L_lin_{sp_key}"
        s = specs.get(spec_name)
        if s is None or s.get("r2_per_pc_mean") is None:
            continue
        ppc = np.asarray(s["r2_per_pc_mean"])
        ax.plot(np.arange(1, len(ppc) + 1), ppc, color=space_color[sp_label],
                lw=1.5, label=f"{sp_label}  (macro={s['r2_macro_mean']:.3f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("PC index")
    ax.set_ylabel("R^2 (per PC, mean over folds)")
    ax.set_title("Per-PC R^2: ridge( features → PC_i )\nlinear-only fits, no interactions")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.suptitle("(eeee) RGB vs HSV vs CIELab features — does color space choice matter?",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")

    # Print a small summary for the report
    print("\nSummary (R^2 macro):")
    for f in families:
        row = []
        for sp in spaces:
            v = grid.get((f, sp))
            row.append(f"{sp}={v[0]:.3f}" if v else f"{sp}=  -  ")
        print(f"  {f:7s}  " + "  ".join(row))


if __name__ == "__main__":
    main()
