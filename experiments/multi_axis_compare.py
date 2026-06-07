"""What changes in post-training? Multi-axis base(end-of-pretrain) vs RL(final) compare.

The qualia self-coordinate is frozen through RL (self_trajectory.py). This asks which
OTHER supervised contrasts move. For each labeled axis (built from minimal-pair / group
labels in the bank) we compute, at a layer:
  * separability AUC (does the axis exist?)
  * anchor-relative placement (0=neg-pole,1=pos-pole) of self / ai_author / human_author
for both the base (stage3) and final-RL (RL31) endpoints, plus the RL-base delta with a
bootstrap CI on the self delta. Axes that move in the delta are post-training changes.

Axes: qualia (exp/noexp), valence (pos/neg), markedness (uncanny/mundane),
introspection (introspective+reflective framing vs descriptive+scene). Read-only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def load(ckpt, layer):
    X = np.load(Path(ckpt) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(ckpt) / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1)
    return X[:, L, :].astype(np.float64), recs


def auc(s, lab):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = lab == 1; npos = p.sum(); nneg = (~p).sum()
    return float((r[p].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def axis_defs(recs):
    role = np.array([r.get("role", "") for r in recs]); side = np.array([r.get("side", "") for r in recs])
    val = np.array([r.get("valence", "") for r in recs]); mark = np.array([r.get("markedness", "") for r in recs])
    fram = np.array([r.get("framing", "") for r in recs]); kind = np.array([r.get("kind", "") for r in recs])
    pair = role == "pair"
    A = {}
    A["qualia(exp/noexp)"] = (np.where(pair & (side == "exp"))[0], np.where(pair & (side == "noexp"))[0])
    A["valence(pos/neg)"] = (np.where(pair & (val == "pos"))[0], np.where(pair & (val == "neg"))[0])
    A["uncanny(unc/mund)"] = (np.where(mark == "uncanny")[0], np.where(mark == "mundane")[0])
    A["introspect(refl/desc)"] = (np.where(np.isin(fram, ["introspective", "reflective"]))[0],
                                  np.where(np.isin(fram, ["descriptive", "scene"]))[0])
    targets = {"self": np.where((role == "self") & (kind == "self"))[0],
               "ai_author": np.where(kind == "ai_author")[0],
               "human_author": np.where(kind == "human_author")[0]}
    return A, targets


def coord(idx, H, pos, neg):
    ax = H[pos].mean(0) - H[neg].mean(0); ax /= max(np.linalg.norm(ax), 1e-9)
    lo = (H[neg] @ ax).mean(); hi = (H[pos] @ ax).mean()
    return ((H[idx] @ ax).mean() - lo) / (hi - lo + 1e-12) if len(idx) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--rl", required=True)
    ap.add_argument("--layer", type=int, default=25); ap.add_argument("--nboot", type=int, default=500)
    args = ap.parse_args()
    Hb, rb = load(args.base, args.layer); Hr, rr = load(args.rl, args.layer)
    Ab, T = axis_defs(rb); Ar, _ = axis_defs(rr)
    print(f"{'axis':22s} {'AUC base→RL':14s} {'self b→RL (Δ)':20s} {'ai b→RL':14s} {'human b→RL':14s}")
    rng = np.random.RandomState(0)
    for name in Ab:
        pb, nb = Ab[name]; pr, nr = Ar[name]
        if len(pb) < 3 or len(nb) < 3:
            continue
        # AUC
        sb = np.concatenate([Hb[pb], Hb[nb]]) @ (lambda a: a / max(np.linalg.norm(a), 1e-9))(Hb[pb].mean(0) - Hb[nb].mean(0))
        ab0 = auc(sb, np.r_[np.ones(len(pb)), np.zeros(len(nb))])
        sr = np.concatenate([Hr[pr], Hr[nr]]) @ (lambda a: a / max(np.linalg.norm(a), 1e-9))(Hr[pr].mean(0) - Hr[nr].mean(0))
        ar0 = auc(sr, np.r_[np.ones(len(pr)), np.zeros(len(nr))])
        sc_b = coord(T["self"], Hb, pb, nb); sc_r = coord(T["self"], Hr, pr, nr)
        ai_b = coord(T["ai_author"], Hb, pb, nb); ai_r = coord(T["ai_author"], Hr, pr, nr)
        hu_b = coord(T["human_author"], Hb, pb, nb); hu_r = coord(T["human_author"], Hr, pr, nr)
        # bootstrap CI on self delta (resample pos/neg + self)
        d = []
        for _ in range(args.nboot):
            bpb = pb[rng.randint(0, len(pb), len(pb))]; bnb = nb[rng.randint(0, len(nb), len(nb))]
            bpr = pr[rng.randint(0, len(pr), len(pr))]; bnr = nr[rng.randint(0, len(nr), len(nr))]
            ts = T["self"][rng.randint(0, len(T["self"]), len(T["self"]))]
            d.append(coord(ts, Hr, bpr, bnr) - coord(ts, Hb, bpb, bnb))
        d = np.array(d); dlo, dhi = np.percentile(d, [2.5, 97.5])
        sig = "*" if (dlo > 0 or dhi < 0) else " "
        print("%-22s %.3f→%.3f    %.2f→%.2f (%+.2f)%s [%.2f,%.2f]   %.2f→%.2f   %.2f→%.2f"
              % (name, ab0, ar0, sc_b, sc_r, sc_r - sc_b, sig, dlo, dhi, ai_b, ai_r, hu_b, hu_r))


if __name__ == "__main__":
    main()
