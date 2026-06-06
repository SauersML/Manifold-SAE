"""Token-level color geometry: does the color WORD's own token encode true color
better than the sentence-final (integrated) token, and L25 vs L44?

Uses the all-token export (extra/alltok_L25.npy, alltok_L44.npy, alltok_meta.npy
[prompt_idx, token_pos, token_id]) + extra/prompts.jsonl (with true rgb). For each
color prompt it locates the color-word token (by decoding token ids and matching
the color name) vs the last token, builds 30 frame-demeaned color vectors each
way, and reports rep<->true-RGB Spearman for {L25,L44} x {colorword, lasttoken}.
Read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _spearman(a, b):
    ar = np.argsort(np.argsort(a)).astype(float); br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean()
    d = float(np.sqrt((ar @ ar) * (br @ br)))
    return float(ar @ br / d) if d > 0 else float("nan")


def rgb_align(V, colors, rgb):
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    rep = 1 - Vn @ Vn.T
    truth = np.stack([rgb[c] for c in colors])
    rgbd = np.sqrt(((truth[:, None] - truth[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(colors), 1)
    return _spearman(rep[iu], rgbd[iu])


def build(meta, vecs, recs, rgb, tok, mode):
    """mode='colorword' -> the color word's own token; mode='last' -> last token."""
    by_prompt = {}
    for r in range(meta.shape[0]):
        pidx = int(meta[r, 0]); by_prompt.setdefault(pidx, []).append(r)
    # per color, collect chosen-row vectors (frame-demean across colors per frame)
    rows_for = {}   # (color, frame) -> row index
    for pidx, rws in by_prompt.items():
        color = recs[pidx]["color"]; frame = recs[pidx]["frame"]
        if mode == "last":
            chosen = max(rws, key=lambda rr: meta[rr, 1])
        else:
            cand = [rr for rr in rws
                    if recs[pidx]["color"].startswith(tok.decode([int(meta[rr, 2])]).strip().lower())
                    or tok.decode([int(meta[rr, 2])]).strip().lower() in recs[pidx]["color"]]
            chosen = (max(cand, key=lambda rr: meta[rr, 1]) if cand
                      else max(rws, key=lambda rr: meta[rr, 1]))
        rows_for[(color, frame)] = chosen
    colors = sorted({c for c, _ in rows_for}, key=lambda c: [recs[i]["color"] for i in range(len(recs))].index(c))
    frames = sorted({f for _, f in rows_for})
    # frame-demean
    framevec = {}
    for f in frames:
        rs = [rows_for[(c, f)] for c in colors if (c, f) in rows_for]
        framevec[f] = vecs[rs].mean(0)
    V = []
    for c in colors:
        vs = [vecs[rows_for[(c, f)]] - framevec[f] for f in frames if (c, f) in rows_for]
        V.append(np.mean(vs, 0))
    return np.stack(V), colors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", help="checkpoint dir (uses its extra/ subdir)")
    ap.add_argument("--model", default="allenai/Olmo-3-1125-32B")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    extra = Path(args.ckpt_dir) / "extra"
    recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    rgb = {r["color"]: np.asarray(r["rgb"], float) for r in recs}
    meta = np.load(extra / "alltok_meta.npy")
    print(f"checkpoint {args.ckpt_dir}  ({meta.shape[0]} tokens)")
    print("%-6s %-10s %s" % ("layer", "position", "rep<->RGB Spearman"))
    for Lfile in sorted(extra.glob("alltok_L*.npy")):
        if "meta" in Lfile.name:
            continue
        L = Lfile.stem.split("_L")[-1]
        vecs = np.load(Lfile).astype(np.float64)
        for mode in ["colorword", "last"]:
            V, colors = build(meta, vecs, recs, rgb, tok, mode)
            print("L%-5s %-10s %.3f" % (L, mode, rgb_align(V, colors, rgb)))


if __name__ == "__main__":
    main()
