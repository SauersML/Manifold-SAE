"""Self-representation across training — single-checkpoint analyzer (append-to-CSV).

For one checkpoint, at the analysis layer, build the qualia axis from the matched
exp/noexp ENTITY pairs and read the anchor-relative coordinate of the indexical
SELF (and the ai-author / human-author landmarks) on it:
    coord = (proj - noexp_centroid) / (exp_centroid - noexp_centroid)
0 = non-experiencer anchor, 1 = experiencer anchor. Reports bootstrap 95% CI
(resampling pairs for the axis + self rows) and the exp/noexp separability AUC with
a label-permutation p (sanity that the axis is real at that checkpoint).

These are supervised projections (axis defined by exp/noexp LABELS), so this is the
non-retracted, robust part of the project — distinct from the (ill-posed) entity
*topology* question. Driver streams checkpoints (download->analyze->delete) to stay
disk-flat. Read-only on activations.
"""
from __future__ import annotations
import argparse, json, csv, os
from pathlib import Path
import numpy as np


def auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--label", required=True)
    ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nboot", type=int, default=500)
    args = ap.parse_args()

    X = np.load(Path(args.ckpt_dir) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(args.ckpt_dir) / "prompts.jsonl") if l.strip()]
    L = min(args.layer, X.shape[1] - 1)
    H = X[:, L, :].astype(np.float64)
    role = np.array([r.get("role", "") for r in recs])
    side = np.array([r.get("side", "") for r in recs])
    kind = np.array([r.get("kind", "") for r in recs])

    ie = np.where((role == "pair") & (side == "exp"))[0]
    ino = np.where((role == "pair") & (side == "noexp"))[0]
    self_i = np.where((role == "self") & (side == "-"))[0]
    if len(self_i) == 0:
        self_i = np.where(role == "self")[0]
    ai_i = np.where((kind == "ai_author") & (side == "-"))[0]
    hu_i = np.where((kind == "human_author") & (side == "-"))[0]

    def coord(idx, He, Hno, Hall):
        axis = He.mean(0) - Hno.mean(0); axis /= max(np.linalg.norm(axis), 1e-9)
        lo = (Hno @ axis).mean(); hi = (He @ axis).mean()
        return ((Hall[idx] @ axis).mean() - lo) / (hi - lo + 1e-12)

    He, Hno = H[ie], H[ino]
    self_c = coord(self_i, He, Hno, H) if len(self_i) else float("nan")
    ai_c = coord(ai_i, He, Hno, H) if len(ai_i) else float("nan")
    hu_c = coord(hu_i, He, Hno, H) if len(hu_i) else float("nan")

    # separability AUC of exp vs noexp on the axis (sanity) + permutation p
    axis = He.mean(0) - Hno.mean(0); axis /= max(np.linalg.norm(axis), 1e-9)
    sc = np.concatenate([He @ axis, Hno @ axis]); lab = np.concatenate([np.ones(len(ie)), np.zeros(len(ino))])
    a0 = auc(sc, lab)
    rng = np.random.RandomState(0)
    perm = np.array([auc(sc, rng.permutation(lab)) for _ in range(500)])
    pval = float((perm >= a0).mean())

    # bootstrap CI on self coord: resample pairs (axis) + self rows
    boot = []
    npair = min(len(ie), len(ino))
    for _ in range(args.nboot):
        be = ie[rng.randint(0, len(ie), len(ie))]; bno = ino[rng.randint(0, len(ino), len(ino))]
        bs = self_i[rng.randint(0, len(self_i), len(self_i))] if len(self_i) else self_i
        try:
            boot.append(coord(bs, H[be], H[bno], H))
        except Exception:
            pass
    boot = np.array([b for b in boot if np.isfinite(b)])
    ci_lo, ci_hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if len(boot) else (float("nan"), float("nan"))

    row = {"order": args.order, "label": args.label, "layer": L, "n_pair": len(ie),
           "self_coord": round(float(self_c), 4), "self_ci_lo": round(ci_lo, 4), "self_ci_hi": round(ci_hi, 4),
           "ai_author": round(float(ai_c), 4), "human_author": round(float(hu_c), 4),
           "qualia_auc": round(float(a0), 4), "auc_perm_p": pval}
    exists = os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"{args.label:28s} self={row['self_coord']:.3f} CI[{ci_lo:.3f},{ci_hi:.3f}] "
          f"ai={row['ai_author']:.3f} hu={row['human_author']:.3f} auc={a0:.3f} p={pval:.3f}", flush=True)


if __name__ == "__main__":
    main()
