#!/usr/bin/env python3
"""Plot atom-compactness comparison from llm_probe results.json.

For each (concept, layer) pair, counts atoms above the moderate
correlation threshold for vanilla and curve SAEs and renders a
horizontal bar chart highlighting the architectural-localization
difference.

  python tools/plot_atom_compactness.py runs_cluster/llm_probe/results.json runs/atom_compactness.png
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", type=Path)
    ap.add_argument("output_png", type=Path, nargs="?", default=Path("atom_compactness.png"))
    args = ap.parse_args()

    d = json.loads(args.results_json.read_text())
    p2 = d.get("phase2", {}) or {}

    concepts_layers, van_n, crv_n = [], [], []
    for key, r in p2.items():
        if not isinstance(r, dict): continue
        v = r.get("vanilla") or {}
        cp = r.get("curve_position") or {}
        if not v or not cp: continue
        concepts_layers.append(key)
        van_n.append(v.get("n_atoms_above_moderate", 0))
        crv_n.append(cp.get("n_atoms_above_moderate", 0))

    if not concepts_layers:
        print("no phase2 data", file=sys.stderr); return 1

    order = np.argsort(crv_n)
    concepts_layers = [concepts_layers[i] for i in order]
    van_n = [van_n[i] for i in order]
    crv_n = [crv_n[i] for i in order]

    F = d.get("config", {}).get("sae_F_to_probe", 128)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(concepts_layers))))
    y = np.arange(len(concepts_layers)); w = 0.4
    ax.barh(y + w/2, van_n, w, label="vanilla TopK SAE", color="#bf616a")
    ax.barh(y - w/2, crv_n, w, label="Manifold-SAE curve atoms", color="#5e81ac")
    ax.axvline(F, color="gray", linestyle=":", label=f"F = {F} (total atoms)")
    ax.set_yticks(y); ax.set_yticklabels(concepts_layers)
    ax.set_xlabel("atoms with |Spearman(atom_signal, concept_rank)| > 0.5")
    ax.set_xlim(0, F + 5)
    for i, (vn, cn) in enumerate(zip(van_n, crv_n)):
        ax.text(vn + 1, i + w/2, str(vn), va="center", fontsize=9)
        ax.text(cn + 1, i - w/2, str(cn), va="center", fontsize=9, fontweight="bold")
    ax.set_title("How many SAE atoms correlate with each planted concept?\n"
                 "Lower = more localized representation")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"saved {args.output_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
