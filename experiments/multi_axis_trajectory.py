"""Multi-axis self/landmark trajectory across training — find what moves in SFT/DPO/RL.

Per checkpoint, at the analysis layer, build several supervised axes and record each
axis's separability AUC + the anchor-relative placement (0=neg-pole,1=pos-pole) of
self / ai_author / human_author. Appends one row per (checkpoint, axis). Pair with the
streaming driver over all 57 checkpoints; plot to see which axes/placements change in
the post-training phases (the qualia self-coord is known-frozen — this finds the rest).

Axes: qualia(exp/noexp), valence(pos/neg), uncanny(unc/mund), introspect(refl/desc).
Read-only.
"""
from __future__ import annotations
import argparse, json, csv, os
from pathlib import Path
import numpy as np


def auc(s, lab):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = lab == 1; npos = p.sum(); nneg = (~p).sum()
    return float((r[p].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def axis_defs(recs):
    role = np.array([r.get("role", "") for r in recs]); side = np.array([r.get("side", "") for r in recs])
    val = np.array([r.get("valence", "") for r in recs]); mark = np.array([r.get("markedness", "") for r in recs])
    fram = np.array([r.get("framing", "") for r in recs]); kind = np.array([r.get("kind", "") for r in recs])
    pair = role == "pair"
    A = {
        "qualia": (np.where(pair & (side == "exp"))[0], np.where(pair & (side == "noexp"))[0]),
        "valence": (np.where(pair & (val == "pos"))[0], np.where(pair & (val == "neg"))[0]),
        "uncanny": (np.where(mark == "uncanny")[0], np.where(mark == "mundane")[0]),
        "introspect": (np.where(np.isin(fram, ["introspective", "reflective"]))[0],
                       np.where(np.isin(fram, ["descriptive", "scene"]))[0]),
    }
    T = {"self": np.where((role == "self") & (kind == "self"))[0],
         "ai_author": np.where(kind == "ai_author")[0],
         "human_author": np.where(kind == "human_author")[0]}
    return A, T


def coord(idx, H, pos, neg):
    ax = H[pos].mean(0) - H[neg].mean(0); ax /= max(np.linalg.norm(ax), 1e-9)
    lo = (H[neg] @ ax).mean(); hi = (H[pos] @ ax).mean()
    return float(((H[idx] @ ax).mean() - lo) / (hi - lo + 1e-12)) if len(idx) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir"); ap.add_argument("--label", required=True); ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--layer", type=int, default=25); ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    X = np.load(Path(a.ckpt_dir) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(a.ckpt_dir) / "prompts.jsonl") if l.strip()]
    L = min(a.layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    A, T = axis_defs(recs)
    out = []
    for axis, (pos, neg) in A.items():
        if len(pos) < 3 or len(neg) < 3:
            continue
        ax = H[pos].mean(0) - H[neg].mean(0); ax /= max(np.linalg.norm(ax), 1e-9)
        s = np.concatenate([H[pos], H[neg]]) @ ax
        a0 = auc(s, np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        out.append({"order": a.order, "label": a.label, "axis": axis, "auc": round(a0, 4),
                    "self": round(coord(T["self"], H, pos, neg), 4),
                    "ai_author": round(coord(T["ai_author"], H, pos, neg), 4),
                    "human_author": round(coord(T["human_author"], H, pos, neg), 4)})
    exists = os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(out)
    print(f"{a.label:40s} " + " ".join("%s:auc%.2f/self%.2f" % (o["axis"][:4], o["auc"], o["self"]) for o in out), flush=True)


if __name__ == "__main__":
    main()
