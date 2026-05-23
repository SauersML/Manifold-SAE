"""auto_12: joint plot of macro R² vs effective intrinsic dim across specs.

Angle (aa): for each spec, define an "effective intrinsic dim" = number of top
PCs whose mean R² is >= 0.5 (i.e. how many PCs the spec actually fits well),
plus an "R²-weighted effective rank" = sum(r2_per_pc) (continuous variant).
Then scatter macro R² vs both, colored by spec family. Reveals whether high
macro-R² specs achieve it by fitting many PCs broadly or just a few well, and
which families (U_pca*, U_pca*_duchon, L_*, M_*, etc.) trade off differently.

Reads: runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json
Writes: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_12.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path("/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40")
RESULTS = RUN_DIR / "results.json"
OUT = RUN_DIR / "auto_12.png"


def family_of(name: str) -> str:
    # First token before underscore captures L_/M_/U_ family
    m = re.match(r"([A-Za-z]+)_", name)
    head = m.group(1) if m else name
    if name.startswith("U_pca") and "duchon" in name:
        return "U_pca_duchon"
    if name.startswith("U_pca"):
        return "U_pca"
    if name.startswith("L_"):
        return "L_linpoly"
    if name.startswith("M_"):
        return "M_manifold"
    if name.startswith("U_"):
        return "U_other"
    return head


def main() -> None:
    d = json.load(open(RESULTS))
    specs = d["per_layer"]["L40"]["specs"]

    rows = []
    for name, s in specs.items():
        if "r2_macro_mean" not in s or "r2_per_pc_mean" not in s:
            continue
        r2_pc = np.asarray(s["r2_per_pc_mean"], dtype=float)
        macro = float(s["r2_macro_mean"])
        # effective dim: number of PCs with R² >= 0.5
        eff_dim_05 = int((r2_pc >= 0.5).sum())
        # continuous variant: sum of clipped per-PC R² (R²-weighted rank)
        eff_rank = float(np.clip(r2_pc, 0.0, 1.0).sum())
        rows.append((name, macro, eff_dim_05, eff_rank))

    fams = sorted({family_of(r[0]) for r in rows})
    palette = plt.cm.tab10(np.linspace(0, 1, len(fams)))
    fam_color = dict(zip(fams, palette))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, xidx, xlab, title in [
        (axes[0], 2, "# PCs with R² ≥ 0.5  (hard intrinsic dim)",
         "Macro R² vs hard effective dim"),
        (axes[1], 3, "Σ clipped per-PC R²  (R²-weighted effective rank)",
         "Macro R² vs R²-weighted rank"),
    ]:
        for fam in fams:
            xs = [r[xidx] for r in rows if family_of(r[0]) == fam]
            ys = [r[1] for r in rows if family_of(r[0]) == fam]
            ax.scatter(xs, ys, s=42, alpha=0.78, color=fam_color[fam],
                       label=fam, edgecolor="black", linewidth=0.4)
        ax.set_xlabel(xlab)
        ax.set_ylabel("macro R²")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    # Annotate the top-7 specs by macro R² on the left panel
    top = sorted(rows, key=lambda r: -r[1])[:7]
    for name, macro, eff05, _ in top:
        axes[0].annotate(name, (eff05, macro), fontsize=7,
                         xytext=(4, 3), textcoords="offset points")

    fig.suptitle(
        "auto_12 — macro R² vs effective intrinsic dim across 100 specs "
        "(Cogito L40)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")

    # Quick textual takeaway
    by_fam = {}
    for name, macro, eff05, effr in rows:
        by_fam.setdefault(family_of(name), []).append((macro, eff05, effr))
    print("family  n  macroR²_med  effDim05_med  effRank_med")
    for fam in fams:
        arr = np.asarray(by_fam[fam])
        print(f"{fam:18s} {len(arr):3d}  {np.median(arr[:,0]):.3f}  "
              f"{np.median(arr[:,1]):6.1f}  {np.median(arr[:,2]):6.2f}")


if __name__ == "__main__":
    main()
