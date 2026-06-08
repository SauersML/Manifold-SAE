"""CLEAN re-analysis after adversarial review. Fixes:
 - landmarks use side=='-' (5 neutral) NOT all 11 (which folded in exp/noexp variants)
 - self pinned to role==self & side=='-' & kind==self
 - held-out split AUC (not circular in-sample perm-p)
 - anchor coord decomposed into numerator (self-neg)/denominator (pos-neg)
 - bootstrap CIs + pairwise significance (self vs ai, vs human, vs fake-I; 1p vs 3p)
Run on the two endpoints (base=end-pretrain, RL=final). Read-only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

rng = np.random.RandomState(0)


def load(ck, layer):
    X = np.load(Path(ck) / "activations.npy"); recs = [json.loads(l) for l in open(Path(ck) / "prompts.jsonl")]
    return X[:, min(layer, X.shape[1] - 1), :].astype(np.float64), recs


def auc(s, lab):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = lab == 1; return float((r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def heldout_auc(H, pos, neg, nsplit=50):
    out = []
    for _ in range(nsplit):
        pp = rng.permutation(pos); nn = rng.permutation(neg)
        ptr, pte = pp[:len(pp) // 2], pp[len(pp) // 2:]; ntr, nte = nn[:len(nn) // 2], nn[len(nn) // 2:]
        ax = H[ptr].mean(0) - H[ntr].mean(0); ax /= max(np.linalg.norm(ax), 1e-9)
        s = np.r_[H[pte] @ ax, H[nte] @ ax]; out.append(auc(s, np.r_[np.ones(len(pte)), np.zeros(len(nte))]))
    return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


def decomp(idx, H, pos, neg):
    ax = H[pos].mean(0) - H[neg].mean(0); ax /= max(np.linalg.norm(ax), 1e-9)
    lo = (H[neg] @ ax).mean(); hi = (H[pos] @ ax).mean()
    num = (H[idx] @ ax).mean() - lo; den = hi - lo
    return num / den, num, den


def boot_coord(idx, H, pos, neg, nb=2000):
    v = []
    for _ in range(nb):
        bp = pos[rng.randint(0, len(pos), len(pos))]; bn = neg[rng.randint(0, len(neg), len(neg))]
        bi = idx[rng.randint(0, len(idx), len(idx))]
        v.append(decomp(bi, H, bp, bn)[0])
    return np.array(v)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--rl", required=True)
    ap.add_argument("--layer", type=int, default=25); a = ap.parse_args()
    for tag, ck in [("BASE(end-pretrain)", a.base), ("RL(final)", a.rl)]:
        H, recs = load(ck, a.layer)
        role = np.array([r.get("role", "") for r in recs]); side = np.array([r.get("side", "") for r in recs])
        kind = np.array([r.get("kind", "") for r in recs]); val = np.array([r.get("valence", "") for r in recs])
        person = np.array([r.get("person", "") for r in recs])
        pair = role == "pair"
        self_i = np.where((role == "self") & (side == "-") & (kind == "self"))[0]
        self_1p = np.where((role == "self") & (side == "-") & (kind == "self") & (person == "1st"))[0]
        self_3p = np.where((role == "self") & (side == "-") & (kind == "self") & (person == "3rd"))[0]
        fakeI = np.where((role == "self") & (kind == "self_control"))[0]
        ai = np.where((kind == "ai_author") & (side == "-"))[0]      # CLEAN n=5
        hu = np.where((kind == "human_author") & (side == "-"))[0]   # CLEAN n=5
        print(f"\n######## {tag}  L{a.layer} ########  (self n={len(self_i)}, ai n={len(ai)}, hu n={len(hu)}, fakeI n={len(fakeI)})")
        for axname, pos, neg in [("qualia", np.where(pair & (side == "exp"))[0], np.where(pair & (side == "noexp"))[0]),
                                 ("valence", np.where(pair & (val == "pos"))[0], np.where(pair & (val == "neg"))[0])]:
            ho = heldout_auc(H, pos, neg)
            sc, num, den = decomp(self_i, H, pos, neg)
            sb = boot_coord(self_i, H, pos, neg); aib = boot_coord(ai, H, pos, neg); hub = boot_coord(hu, H, pos, neg)
            print(f"  [{axname}] held-out AUC={ho[0]:.3f}[{ho[1]:.3f},{ho[2]:.3f}]  self={sc:.2f}(num{num:.2f}/den{den:.2f}) CI[{np.percentile(sb,2.5):.2f},{np.percentile(sb,97.5):.2f}]")
            print(f"           ai={decomp(ai,H,pos,neg)[0]:.2f} CI[{np.percentile(aib,2.5):.2f},{np.percentile(aib,97.5):.2f}]   human={decomp(hu,H,pos,neg)[0]:.2f} CI[{np.percentile(hub,2.5):.2f},{np.percentile(hub,97.5):.2f}]")
            if axname == "qualia":
                d_ai = sb - aib; d_hu = sb - hub
                print(f"           self−ai Δ={d_ai.mean():+.2f} CI[{np.percentile(d_ai,2.5):+.2f},{np.percentile(d_ai,97.5):+.2f}] {'SIG' if (np.percentile(d_ai,2.5)>0 or np.percentile(d_ai,97.5)<0) else 'ns'}"
                      f"   self−human Δ={d_hu.mean():+.2f} CI[{np.percentile(d_hu,2.5):+.2f},{np.percentile(d_hu,97.5):+.2f}] {'SIG' if (np.percentile(d_hu,2.5)>0 or np.percentile(d_hu,97.5)<0) else 'ns'}")
                if len(fakeI) and len(self_1p):
                    f1 = boot_coord(fakeI, H, pos, neg); p1 = boot_coord(self_1p, H, pos, neg); p3 = boot_coord(self_3p, H, pos, neg)
                    d_fi = p1 - f1; d_pp = p1 - p3
                    print(f"           1p_self−fakeI Δ={d_fi.mean():+.2f} CI[{np.percentile(d_fi,2.5):+.2f},{np.percentile(d_fi,97.5):+.2f}] {'SIG' if (np.percentile(d_fi,2.5)>0 or np.percentile(d_fi,97.5)<0) else 'ns (≈ fictional I)'}"
                          f"   1p−3p Δ={d_pp.mean():+.2f} {'SIG' if (np.percentile(d_pp,2.5)>0 or np.percentile(d_pp,97.5)<0) else 'ns'}")


if __name__ == "__main__":
    main()
