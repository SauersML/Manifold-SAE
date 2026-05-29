"""auto_77 — Per-PC R² vs PC explained-variance, across spec families.

Fresh angle on the GAM zoo. Existing plots show macro-R² rankings of the
~100 specs and per-PC heatmaps for a few hand-picked specs. NONE answer
the structural question: "do specs explain the high-variance PCs or the
low-variance PCs?"

For each spec family (Linear, Polynomial, Joint-Duchon, Manifold M_*,
Unsupervised U_*, kNN N_*), we plot r2_per_pc_mean[k] vs EVR[k] for the
single best spec in that family, k = 1..K. A spec that "captures the
bulk" should sit high-left (big EVR, big R²); a spec that only mops up
nuisance dims sits high-right.

Outputs:
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_77.png
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_77.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS = Path(
    "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
)
OUT_PNG = RESULTS.parent / "auto_77.png"
OUT_JSON = RESULTS.parent / "auto_77.json"


def family_of(spec: str) -> str:
    if spec.startswith("U_"):
        return "Unsupervised (U_*)"
    if spec.startswith("N_"):
        return "kNN (N_*)"
    if spec.startswith("M_"):
        return "Manifold (M_*)"
    if "poly" in spec:
        return "Polynomial"
    if "joint" in spec or "tensor" in spec or "duchon" in spec.lower():
        return "Joint smooth"
    if "kernel" in spec or "rbf" in spec:
        return "Kernel"
    if "cyclic" in spec or "hue" in spec:
        return "Hue / cyclic"
    if "add" in spec:
        return "Additive"
    if spec == "L_const_mean":
        return "Const baseline"
    if spec.startswith("L_lin") or spec.startswith("L_"):
        return "Linear"
    return "Other"


COLORS = {
    "Linear":              "#1f77b4",
    "Polynomial":          "#ff7f0e",
    "Additive":            "#9467bd",
    "Joint smooth":        "#2ca02c",
    "Hue / cyclic":        "#bcbd22",
    "Kernel":              "#e377c2",
    "Manifold (M_*)":      "#17becf",
    "kNN (N_*)":           "#8c564b",
    "Unsupervised (U_*)":  "#d62728",
    "Const baseline":      "#7f7f7f",
    "Other":               "#000000",
}


def main() -> int:
    with RESULTS.open() as f:
        r = json.load(f)
    pl = r["per_layer"]["L40"]
    evr = np.asarray(pl["explained_variance_ratio_topK"], dtype=np.float64)  # (K,)
    K = evr.size
    specs = pl["specs"]

    # Pick the single best spec in each family by macro R².
    by_family: dict[str, list[tuple[str, float]]] = {}
    skipped = 0
    trivial = 0
    for sid, s in specs.items():
        if "r2_macro_mean" not in s or "r2_per_pc_mean" not in s:
            skipped += 1
            continue
        # Exclude trivial PCA-self-fits: any U_pca* with embedded dim >= 6
        # reuses the full PC basis and approaches R²=1 by construction.
        if sid.startswith("U_pca"):
            import re
            m = re.search(r"(\d+)", sid)
            if m and int(m.group(1)) >= 6:
                trivial += 1
                continue
        fam = family_of(sid)
        by_family.setdefault(fam, []).append((sid, float(s["r2_macro_mean"])))
    print(f"[load] {len(specs)} specs, {skipped} skipped (errored), "
          f"{trivial} excluded (trivial PCA self-fits d>=6)", flush=True)
    best_per_family = {fam: max(v, key=lambda kv: kv[1])
                       for fam, v in by_family.items()}

    # ---- Plot: 2 panels.
    # (1) per-PC R² vs PC index (k = 1..K), one line per family.
    # (2) per-PC R² vs EVR[k] log scale, one line per family.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    ks = np.arange(1, K + 1)
    ax1.bar(ks, evr / evr.max() * 0.5, color="lightgray",
            edgecolor="none", zorder=0, width=0.85,
            label="EVR (rescaled to [0,0.5])")

    out = {"families": {}}
    fam_order = sorted(best_per_family.keys(),
                       key=lambda f: -best_per_family[f][1])
    for fam in fam_order:
        sid, r2m = best_per_family[fam]
        pc_r2 = np.asarray(specs[sid]["r2_per_pc_mean"], dtype=np.float64)
        col = COLORS.get(fam, "#000000")
        ax1.plot(ks, pc_r2, color=col, linewidth=1.7,
                 marker="o", markersize=3.3,
                 label=f"{fam}: {sid}  (macro={r2m:+.3f})")
        ax2.scatter(evr, pc_r2, color=col, s=22 + 60 * (evr / evr.max()),
                    edgecolor="black", linewidth=0.3,
                    alpha=0.85, label=fam)
        out["families"][fam] = {
            "spec": sid,
            "macro_r2": r2m,
            "per_pc_r2": pc_r2.tolist(),
        }

    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("PC index k (top → bottom of L40 spectrum)")
    ax1.set_ylabel("held-out R²[k] per PC")
    ax1.set_xlim(0.5, K + 0.5)
    ax1.set_title(
        "Per-PC R² across spec families (best spec per family)\n"
        "Where in the spectrum does each family pull its R² from?",
        fontsize=11,
    )
    ax1.legend(fontsize=7.3, loc="upper right", frameon=True, ncol=1)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("EVR[k]  (log scale, share of L40 variance on PC k)")
    ax2.set_ylabel("held-out R²[k]")
    ax2.set_title(
        "R² vs PC eigenvalue\n"
        "high-left = explains the bulk;  high-right = nuisance mop-up",
        fontsize=11,
    )
    ax2.legend(fontsize=7.3, loc="upper left", frameon=True)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.suptitle(
        f"auto_77 — GAM zoo, per-PC R² × PC EVR  ·  cogito L40  ·  K={K}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)

    out["evr"] = evr.tolist()
    out["K"] = int(K)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[done] {OUT_PNG}", flush=True)

    # Print short text summary: variance-weighted vs uniform-mean R² per family.
    print("\nfamily summary  (best spec)")
    print(f"{'family':24s} {'spec':30s} {'macro':>7s} "
          f"{'evr-weighted':>13s} {'top-8 mean':>11s} {'tail mean':>10s}")
    w = evr / evr.sum()
    for fam in fam_order:
        sid, r2m = best_per_family[fam]
        pc = np.asarray(specs[sid]["r2_per_pc_mean"])
        ew = float((w * pc).sum())
        top = float(pc[:8].mean())
        tail = float(pc[8:].mean()) if K > 8 else float("nan")
        print(f"{fam:24s} {sid:30s} {r2m:+7.3f} "
              f"{ew:+13.3f} {top:+11.3f} {tail:+10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
