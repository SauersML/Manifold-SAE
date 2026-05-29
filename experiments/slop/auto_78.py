"""auto_78 — Bulk vs tail: EVR-weighted R² for every spec on PCs 1-8 vs 9-64.

Fresh angle. auto_77 took the per-family BEST and plotted r2_per_pc[k] vs
EVR[k]. That averages away spec-level variation. Here we put every one of
the ~108 specs on a single 2D plane:

    x = EVR-weighted R²_macro restricted to PCs 1-8     (bulk, 51% EVR)
    y = EVR-weighted R²_macro restricted to PCs 9-64    (tail, 49% EVR)

Color by family. Reveals:
  - which specs cleanly capture the bulk but ignore the tail (HSV/RGB linear)
  - which specs spread thinly across the tail (high-d unsupervised)
  - which specs sit on the diagonal (true geometry, e.g. Duchon joints)
  - the trivial U_pca_*d specs that 'predict-yourself' should pin to (1, 1)
    for d ≤ 8 / d ≥ 16 respectively — sanity check.

Outputs:
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_78.png
  runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_78.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS = Path(
    "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json"
)
OUT_PNG = RESULTS.parent / "auto_78.png"
OUT_JSON = RESULTS.parent / "auto_78.json"

BULK_END = 8        # PCs 1..8
TAIL_END = 64       # PCs 9..64


def family_of(spec: str) -> str:
    if spec.startswith("U_pca") and "duchon" not in spec and "smooth" not in spec \
            and "tensor" not in spec and "pairs" not in spec \
            and "centered" not in spec and "add" not in spec:
        return "U_pca trivial"
    if spec.startswith("U_"):
        return "Unsupervised U_*"
    if spec.startswith("N_"):
        return "kNN N_*"
    if spec.startswith("M_"):
        return "Manifold M_*"
    if "poly" in spec:
        return "Polynomial"
    if "duchon" in spec or "joint" in spec or "tensor" in spec:
        return "Joint smooth"
    if "kernel" in spec or "rbf" in spec:
        return "Kernel"
    if "cyclic" in spec or "hue" in spec.lower():
        return "Hue / cyclic"
    if "add" in spec:
        return "Additive"
    return "Linear"


FAMILY_COLORS = {
    "U_pca trivial":     "#bbbbbb",
    "Unsupervised U_*":  "#d62728",
    "kNN N_*":           "#9467bd",
    "Manifold M_*":      "#2ca02c",
    "Polynomial":        "#ff7f0e",
    "Joint smooth":      "#1f77b4",
    "Kernel":            "#17becf",
    "Hue / cyclic":      "#e377c2",
    "Additive":          "#8c564b",
    "Linear":            "#7f7f7f",
}


def weighted_r2(r2_per_pc: np.ndarray, evr: np.ndarray, lo: int, hi: int) -> float:
    r = r2_per_pc[lo:hi]
    w = evr[lo:hi]
    if w.sum() == 0:
        return float("nan")
    return float((r * w).sum() / w.sum())


def main() -> int:
    with RESULTS.open() as f:
        r = json.load(f)

    pl = r["per_layer"]["L40"]
    evr = np.array(pl["explained_variance_ratio_topK"], dtype=np.float64)
    K = evr.shape[0]
    print(f"[auto_78] K={K} top-PCs  bulk=PCs 1..{BULK_END}  tail=PCs {BULK_END+1}..{TAIL_END}")
    print(f"[auto_78] bulk EVR = {evr[:BULK_END].sum():.3f}  tail EVR = {evr[BULK_END:TAIL_END].sum():.3f}")

    rows = []
    n_err = 0
    for name, s in pl["specs"].items():
        if "r2_per_pc_mean" not in s:
            n_err += 1
            continue
        r2pc = np.array(s["r2_per_pc_mean"], dtype=np.float64)
        if r2pc.shape[0] < TAIL_END:
            continue
        x = weighted_r2(r2pc, evr, 0, BULK_END)
        y = weighted_r2(r2pc, evr, BULK_END, TAIL_END)
        rows.append({
            "spec": name,
            "family": family_of(name),
            "bulk_r2": x,
            "tail_r2": y,
            "macro_r2": float(s["r2_macro_mean"]),
        })

    print(f"[auto_78] {len(rows)} specs ({n_err} errored)")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axhline(0, color="black", lw=0.4, alpha=0.4)
    ax.axvline(0, color="black", lw=0.4, alpha=0.4)
    ax.plot([-0.1, 1.05], [-0.1, 1.05], color="black", lw=0.6, alpha=0.3, ls=":")

    # Group rows by family for legend
    by_fam = {}
    for row in rows:
        by_fam.setdefault(row["family"], []).append(row)

    for fam, items in by_fam.items():
        xs = [it["bulk_r2"] for it in items]
        ys = [it["tail_r2"] for it in items]
        ax.scatter(xs, ys, s=70, c=FAMILY_COLORS.get(fam, "#000000"),
                   edgecolors="black", linewidths=0.4, alpha=0.85, label=fam, zorder=3)

    # Annotate stand-outs: top-3 macro, top-3 bulk-only, top-3 tail-only
    rows_by_macro = sorted(rows, key=lambda r: -r["macro_r2"])[:5]
    rows_by_bulk = sorted(rows, key=lambda r: -r["bulk_r2"])[:5]
    rows_by_tail = sorted(rows, key=lambda r: -r["tail_r2"])[:5]
    rows_by_bulk_only = sorted(rows, key=lambda r: -(r["bulk_r2"] - r["tail_r2"]))[:3]

    annot_seen = set()
    for it in rows_by_macro + rows_by_bulk[:3] + rows_by_tail[:3] + rows_by_bulk_only:
        if it["spec"] in annot_seen:
            continue
        annot_seen.add(it["spec"])
        ax.annotate(it["spec"], (it["bulk_r2"], it["tail_r2"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points", alpha=0.85)

    ax.set_xlabel(f"EVR-weighted R²  on bulk PCs 1..{BULK_END}  (Σ EVR = {evr[:BULK_END].sum():.2f})",
                  fontsize=11)
    ax.set_ylabel(f"EVR-weighted R²  on tail PCs {BULK_END+1}..{TAIL_END}  (Σ EVR = {evr[BULK_END:TAIL_END].sum():.2f})",
                  fontsize=11)
    ax.set_title(
        "auto_78 — Bulk vs tail: every GAM-zoo spec on a single plane\n"
        f"cogito L40 · n_specs = {len(rows)} · dotted line = bulk-R² = tail-R² (uniform explainer)",
        fontsize=12,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, frameon=True, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT_PNG}")

    with OUT_JSON.open("w") as f:
        json.dump({
            "bulk_end_pc": BULK_END,
            "tail_end_pc": TAIL_END,
            "bulk_evr": float(evr[:BULK_END].sum()),
            "tail_evr": float(evr[BULK_END:TAIL_END].sum()),
            "rows": rows,
        }, f, indent=2)
    print(f"[done] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
