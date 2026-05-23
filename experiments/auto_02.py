"""auto_02: capacity-scaling curves for unsupervised manifold methods.

For every unsupervised spec in the GAM zoo (U_*) we extract its
effective latent dimensionality and its 5-fold held-out R²_macro,
then group by family (raw-PCA, multi-penalty Duchon add1d / pairs /
triples, GAM-init joint U_Nd, k-means anchors, NMF anchors, …) and
plot R² vs. dim on one chart. Reference horizontals: the best
supervised linear-RGB spec, best joint-color spec, and best
manifold/parametric spec — these are the "ceilings" each
unsupervised family is climbing toward.

Question: which unsupervised parameterisation is the most
data-efficient per latent dimension, and where does each family
saturate?

Output: runs/COLOR_MANIFOLD_GAM_COGITO_L40/auto_02.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "results.json"
OUT = ROOT / "runs" / "COLOR_MANIFOLD_GAM_COGITO_L40" / "auto_02.png"


def parse_dim(name: str) -> int | None:
    """Best-effort extraction of latent dim from a U_* spec name."""
    m = re.search(r"(\d+)d?(?:_|$)", name)
    if m:
        return int(m.group(1))
    return None


# --- group definitions (regex -> label, color, marker) ------------------
GROUPS = [
    (r"^U_pca_\d+d$",                   "raw PCA",                    "#1f77b4", "o"),
    (r"^U_pca_add_\d+d$",               "PCA + additive smooth",      "#17becf", "v"),
    (r"^U_pca_pairs_\d+d$",             "PCA + pair tensor smooth",   "#9467bd", "P"),
    (r"^U_pca\d+_duchon_add1d$",        "Duchon additive 1-D",        "#2ca02c", "s"),
    (r"^U_pca\d+_duchon_joint$",        "Duchon joint",               "#bcbd22", "D"),
    (r"^U_pca\d+_duchon_pairs$",        "Duchon pairs",               "#8c564b", "X"),
    (r"^U_pca\d+_duchon_triples$",      "Duchon triples",             "#e377c2", "*"),
    (r"^U_\d+d$",                       "GAM-joint U_Nd (full opt)",  "#d62728", "^"),
    (r"^U_nmf_\d+d$",                   "NMF anchors",                "#ff7f0e", "h"),
    (r"^U_kmeans_\d+$",                 "k-means anchors",            "#7f7f7f", "p"),
]

# --- supervised reference ceilings --------------------------------------
LINEAR_REF = ("L_lin_rgb", "L_lin_hsv", "L_lin_lab", "L_lin_oklab", "L_lin_lch")
JOINT_REF = ("L_joint_rgb", "L_joint_hsv", "L_joint_lab", "L_joint_oklab",
             "L_joint_rgb_with_hue", "L_joint_oklab_with_h")
MANIFOLD_REF = ("M_hsv_bicone", "M_chroma_disk", "M_chroma_disk_plus_L",
                "M_cyl_hue_val", "M_torus_hue_sat", "M_sphere_hueval",
                "M_sphere_plus_chroma", "M_hsv_cone", "M_rgb_finer_grid")
NONPAR_REF = ("N_knn_rgb_k30", "N_knn_lab_k30", "N_knn_rgb_k10",
              "N_knn_lab_k10", "N_knn_oklab_k10")


def best(specs: dict, names: tuple[str, ...]) -> tuple[str, float] | None:
    cand = [(n, specs[n]["r2_macro_mean"]) for n in names
            if n in specs and specs[n].get("r2_macro_mean") is not None]
    if not cand:
        return None
    return max(cand, key=lambda t: t[1])


def main() -> int:
    d = json.loads(RES.read_text())
    specs = d["per_layer"]["L40"]["specs"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # plot each unsupervised family
    legend_handles = []
    for pat, label, color, marker in GROUPS:
        rx = re.compile(pat)
        pts = []
        for name, s in specs.items():
            if not rx.match(name):
                continue
            r2 = s.get("r2_macro_mean")
            if r2 is None:
                continue
            dim = parse_dim(name)
            if dim is None:
                continue
            std = s.get("r2_macro_std", 0.0) or 0.0
            pts.append((dim, float(r2), float(std), name))
        if not pts:
            continue
        pts.sort()
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        es = np.array([p[2] for p in pts])
        line, = ax.plot(xs, ys, marker=marker, color=color, label=f"{label} (n={len(pts)})",
                        lw=1.6, ms=8, alpha=0.95)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.12, lw=0)
        legend_handles.append(line)

    # ceiling lines
    refs = [
        ("best linear (supervised)",    LINEAR_REF,   "#444444", "--"),
        ("best joint color (supervised)", JOINT_REF,  "#222222", "-."),
        ("best parametric manifold",     MANIFOLD_REF, "#990000", ":"),
        ("best k-NN (nonparametric)",    NONPAR_REF,   "#005599", (0, (1, 1))),
    ]
    for lab, names, c, ls in refs:
        b = best(specs, names)
        if b is None:
            continue
        nm, val = b
        ax.axhline(val, color=c, ls=ls, lw=1.4, alpha=0.85,
                   label=f"{lab}: {nm} = {val:.3f}")

    # cosmetic
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128])
    ax.set_xticklabels([1, 2, 3, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128])
    ax.set_xlabel("latent dimension (or anchor count for k-means)", fontsize=11)
    ax.set_ylabel("held-out R²_macro  (5-fold CV by color, top-64 PCs)", fontsize=11)
    ax.set_title("cogito L40 · unsupervised manifold capacity scaling vs supervised ceilings",
                 fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_ylim(0.0, 1.02)

    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
