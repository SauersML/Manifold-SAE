"""Extra figures for one checkpoint (read-only):
  1. color_wheel: PCA(2D) of the 30 frame-demeaned color vectors at L44, each
     point drawn in its TRUE xkcd color -> does the model recover a color wheel?
  2. cast_on_axis: every entity kind's exp- vs noexp-described centroid on the
     qualia axis, ranked, with the SELF + landmarks marked -> the cast laid out
     from mechanism (0) to experiencer (1).
  3. depth_profile: self / human / AI qualia coordinate + qualia AUC across all
     layers -> where in depth the self is most experiencer-like / axis strongest.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


def color_wheel(ckpt: Path, out: Path, layer: int):
    extra = ckpt / "extra"
    X = np.load(extra / "activations.npy")
    recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1)
    H = X[:, L, :].astype(np.float64)
    by, frames, hexc = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i)
        frames.setdefault(r["frame"], []).append(i)
        hexc[r["color"]] = r["hex"]
    Hd = H.copy()
    for f, idx in frames.items():
        Hd[idx] -= H[idx].mean(0)
    colors = list(by)
    V = np.stack([Hd[by[c]].mean(0) for c in colors])
    V = V - V.mean(0)
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    P = U[:, :2] * S[:2]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for i, c in enumerate(colors):
        ax.scatter(P[i, 0], P[i, 1], s=420, c=hexc[c], edgecolors="0.3", linewidths=0.8, zorder=3)
        ax.text(P[i, 0], P[i, 1], c, fontsize=6, ha="center", va="center",
                color="white" if c in ("black", "navy", "indigo", "maroon", "purple") else "black", zorder=4)
    ax.set_title(f"Model color space (L{L}, frame-demeaned, PCA) — dots in true RGB\n{ckpt.name}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.savefig(out / "color_wheel.png", dpi=160); plt.close(fig)
    print(f"[fig] color_wheel.png (L{L})")


def cast_on_axis(summary: dict, out: Path):
    kp = summary.get("kind_placement", {})
    anchors = summary.get("anchors", {})
    rows = sorted(kp.items(), key=lambda kv: kv[1]["exp_coord"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, max(5, len(rows) * 0.34)), constrained_layout=True)
    for y, (k, v) in enumerate(rows):
        ax.plot([v["noexp_coord"], v["exp_coord"]], [y, y], color="0.8", lw=1.5, zorder=1)
        ax.scatter(v["noexp_coord"], y, c="#b0b0b0", s=30, zorder=2)
        ax.scatter(v["exp_coord"], y, c="#1f6feb", s=42, zorder=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([k for k, _ in rows], fontsize=8)
    # mark self + landmarks
    marks = [("self", summary.get("self_qualia_coord"), "#d62728"),
             ("human_author", summary.get("human_author_qualia_coord"), "#2ca02c"),
             ("ai_author", summary.get("ai_author_qualia_coord"), "#9467bd"),
             ("fake-I", (anchors.get("self_control_fake_I") or {}).get("qualia_coord"), "#ff7f0e")]
    for nm, val, col in marks:
        if val is not None:
            ax.axvline(val, color=col, lw=1.4, ls="--", label=f"{nm} {val:.2f}")
    ax.axvline(0, color="0.6", lw=0.7); ax.axvline(1, color="0.6", lw=0.7)
    ax.set_xlabel("qualia coordinate (0 = no-experience … 1 = experiencer)")
    ax.set_title("The cast on the qualia axis (blue=exp-described, grey=noexp-described)")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(out / "cast_on_axis.png", dpi=160); plt.close(fig)
    print("[fig] cast_on_axis.png")


def depth_profile(ckpt: Path, out: Path):
    f = ckpt / "bank_layers.csv"
    if not f.exists():
        print("[fig] no bank_layers.csv; skip depth"); return
    rows = list(csv.DictReader(open(f)))
    L = [int(r["layer"]) for r in rows]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(L, [float(r["self_qualia_coord"]) for r in rows], "o-", ms=3, color="#d62728", label="self")
    ax.plot(L, [float(r["human_author_qualia_coord"]) for r in rows], "s-", ms=2, color="#2ca02c", label="human")
    ax.plot(L, [float(r["ai_author_qualia_coord"]) for r in rows], "^-", ms=2, color="#9467bd", label="AI")
    ax.axhline(0, color="0.85", lw=0.7); ax.axhline(1, color="0.85", lw=0.7)
    ax.set_xlabel("layer"); ax.set_ylabel("qualia coordinate"); ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(L, [float(r["qualia_auc"]) for r in rows], ":", color="#1f6feb", lw=1.5, label="qualia AUC")
    ax2.set_ylabel("qualia AUC", color="#1f6feb"); ax2.set_ylim(0.4, 1.02)
    ax.set_title(f"Depth profile: self/human/AI on qualia axis + axis AUC by layer\n{ckpt.name}")
    fig.savefig(out / "depth_profile.png", dpi=160); plt.close(fig)
    print("[fig] depth_profile.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir")
    ap.add_argument("--out", default="runs/ANALYSIS")
    ap.add_argument("--color-layer", type=int, default=44)
    args = ap.parse_args()
    ckpt = Path(args.ckpt_dir); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # ensure summary + bank_layers exist (single-run analyze writes them into ckpt dir)
    summary = json.loads(subprocess.run(
        [".venv/bin/python", "experiments/analyze_self_qualia_bank.py", str(ckpt)],
        capture_output=True, text=True).stdout)
    color_wheel(ckpt, out, args.color_layer)
    cast_on_axis(summary, out)
    depth_profile(ckpt, out)


if __name__ == "__main__":
    main()
