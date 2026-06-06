"""Color-geometry analysis of the dedicated color probe bank, across training.

For each checkpoint's color harvest (extra/activations.npy + prompts.jsonl, which
carry the true xkcd hex/RGB per color), at the analysis layer:
  1. FRAME-DEMEAN: subtract the per-frame mean across colors (removes the frame
     nuisance direction), then average the demeaned reps per color -> 30 color vecs.
  2. Alignment with TRUE color space: correlate pairwise representation distance
     (1 - cosine) with pairwise true RGB distance (Spearman). Higher = the model's
     color geometry matches perceptual/RGB structure.
  3. NN cleanliness: mean true-RGB distance of each color's top-3 representation
     neighbors (lower = cleaner hue neighborhoods).
Run across stages to see whether color geometry EMERGES/sharpens over training.

Read-only; safe alongside the live sweep.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

STAGE_ORDER = ["OLMO3_32B_TRAJ", "OLMO3_32B_TRAJ_SFT", "OLMO3_32B_TRAJ_DPO",
               "OLMO3_32B_TRAJ_RL", "OLMO3_32B_TRAJ_RL31"]


def _ckpt_key(name: str):
    m = re.search(r"stage(\d+).*?step(\d+)", name) or re.search(r"step[_-]?(\d+)", name)
    if m and m.lastindex == 2:
        return (int(m.group(1)), int(m.group(2)))
    return (0, int(m.group(1))) if m else (0, 0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    ar = ar - ar.mean(); br = br - br.mean()
    d = float(np.sqrt((ar @ ar) * (br @ br)))
    return float(ar @ br / d) if d > 0 else float("nan")


def color_metrics(extra_dir: str, layer: int) -> dict | None:
    act = Path(extra_dir) / "activations.npy"
    pj = Path(extra_dir) / "prompts.jsonl"
    if not act.exists() or not pj.exists():
        return None
    X = np.load(act)
    recs = [json.loads(l) for l in open(pj) if l.strip()]
    if layer >= X.shape[1]:
        layer = X.shape[1] - 1
    H = X[:, layer, :].astype(np.float64)
    colors, rgb = [], {}
    by = {}
    frames = {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i)
        frames.setdefault(r["frame"], []).append(i)
        rgb[r["color"]] = np.asarray(r["rgb"], dtype=np.float64)
    colors = list(by)
    # frame-demean: subtract per-frame mean across colors
    Hd = H.copy()
    for f, idxs in frames.items():
        Hd[idxs] -= H[idxs].mean(0)
    V = np.stack([Hd[by[c]].mean(0) for c in colors])
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    cos = Vn @ Vn.T
    repdist = 1.0 - cos
    truth = np.stack([rgb[c] for c in colors])
    rgbdist = np.sqrt(((truth[:, None] - truth[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(colors), k=1)
    align = _spearman(repdist[iu], rgbdist[iu])
    # NN cleanliness: mean true-RGB dist of top-3 representation neighbors
    nn_rgb = []
    for i in range(len(colors)):
        order = np.argsort(-cos[i])
        nbrs = [j for j in order if j != i][:3]
        nn_rgb.append(np.mean([rgbdist[i, j] for j in nbrs]))
    return {"n_colors": len(colors), "rgb_alignment_spearman": align,
            "mean_nn_rgb_dist": float(np.mean(nn_rgb)), "layer": layer}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="runs/ANALYSIS")
    ap.add_argument("--layer-percent", type=float, default=0.40)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    gidx = 0
    for stage in STAGE_ORDER:
        base = Path(args.runs_root) / stage
        if not base.exists():
            continue
        cks = sorted([d for d in glob.glob(str(base / "*/")) if os.path.exists(d + "done.json")],
                     key=lambda d: _ckpt_key(Path(d).name))
        for d in cks:
            extra = str(Path(d) / "extra")
            # infer layer from activations shape (40% depth)
            act = Path(extra) / "activations.npy"
            if not act.exists():
                continue
            nL = np.load(act, mmap_mode="r").shape[1]
            layer = int(round(args.layer_percent * (nL - 1)))
            m = color_metrics(extra, layer)
            if not m:
                continue
            m.update({"global_idx": gidx, "stage": stage.replace("OLMO3_32B_TRAJ", "pre").replace("_", ""),
                      "checkpoint": Path(d).name})
            rows.append(m); gidx += 1
            print("  %-26s align(RGB)=%.3f nn_rgb=%.1f" %
                  (m["checkpoint"], m["rgb_alignment_spearman"], m["mean_nn_rgb_dist"]))
    if rows:
        keys = ["global_idx", "stage", "checkpoint", "layer", "n_colors",
                "rgb_alignment_spearman", "mean_nn_rgb_dist"]
        with open(out / "color_geometry.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            w.writerows([{k: r[k] for k in keys} for r in rows])
        print(f"[color] {len(rows)} checkpoints -> {out/'color_geometry.csv'}")
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            x = [r["global_idx"] for r in rows]
            fig, a = plt.subplots(2, 1, figsize=(max(8, len(rows) * 0.28), 6), sharex=True,
                                  constrained_layout=True)
            a[0].plot(x, [r["rgb_alignment_spearman"] for r in rows], "o-", color="#6a0dad")
            a[0].set_ylabel("rep↔RGB\nSpearman"); a[0].axhline(0, color="0.8", lw=0.7)
            a[0].set_title("Color geometry alignment with true RGB across training (frame-demeaned)")
            a[1].plot(x, [r["mean_nn_rgb_dist"] for r in rows], "o-", color="#c0392b")
            a[1].set_ylabel("mean NN\nRGB dist (↓ better)")
            a[1].set_xticks(x); a[1].set_xticklabels([r["checkpoint"][:12] for r in rows],
                                                     rotation=90, fontsize=5)
            fig.savefig(out / "color_geometry.png", dpi=160); plt.close(fig)
            print(f"[color] wrote {out/'color_geometry.png'}")
        except Exception as e:
            print(f"[color] plot skipped: {e}")


if __name__ == "__main__":
    main()
