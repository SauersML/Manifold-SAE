"""UMAP of the color manifold at a few checkpoints (early->mid->final), each
color drawn in its true xkcd RGB -> visual of the color manifold emerging /
organizing over training. Frame-demeaned, L44. Read-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def color_vecs(extra: Path, layer: int):
    X = np.load(extra / "activations.npy"); recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    by, fr, hexc = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i); hexc[r["color"]] = r["hex"]
    Hd = H.copy()
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)
    cols = list(by); V = np.stack([Hd[by[c]].mean(0) for c in cols])
    return V, cols, hexc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpts", nargs="+", help="checkpoint dirs (labelled by name) early->final")
    ap.add_argument("--out", default="runs/ANALYSIS"); ap.add_argument("--layer", type=int, default=44)
    args = ap.parse_args()
    import umap, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n = len(args.ckpts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, ck in zip(axes, args.ckpts):
        V, cols, hexc = color_vecs(Path(ck) / "extra", args.layer)
        emb = umap.UMAP(n_neighbors=8, min_dist=0.4, metric="cosine", random_state=0).fit_transform(V)
        for i, c in enumerate(cols):
            ax.scatter(emb[i, 0], emb[i, 1], s=360, c=hexc[c], edgecolors="0.3", linewidths=0.7, zorder=3)
            ax.text(emb[i, 0], emb[i, 1], c, fontsize=5, ha="center", va="center",
                    color="white" if c in ("black", "navy", "indigo", "maroon", "purple") else "black", zorder=4)
        ax.set_title(Path(ck).name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Color manifold (UMAP of frame-demeaned L{args.layer} reps, dots in true RGB)")
    fig.savefig(out / "umap_color.png", dpi=150); plt.close(fig)
    print(f"[umap] wrote {out/'umap_color.png'}")


if __name__ == "__main__":
    main()
