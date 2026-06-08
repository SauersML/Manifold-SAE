"""Per-entity-kind + self affect coordinate at one checkpoint (append-to-CSV).

For tracking whether the VALENCE/SUFFERING ranking of entities (and the self relative
to them) changes across training. At the analysis layer (L25), build:
  - valence axis  = mean(pos-valence pairs) - mean(neg-valence pairs)
  - suffering axis = mean(joy-word pairs)   - mean(distress-word pairs)   (1 - coord = suffering)
Then for each entity kind compute a FAIR balanced valence (mean of pos-prompt coord and
neg-prompt coord, so single-valence kinds are skipped) and an exp-side suffering coord, plus
the self subtypes. Appends one row per (checkpoint, group). Pure numpy projections — no gamfit.
Read-only on activations.
"""
from __future__ import annotations
import argparse, json, csv, os, re
from pathlib import Path
import numpy as np

SUF = {"suffer","suffers","suffering","pain","painful","dread","dreads","distress","agony","anguish","fear","afraid","hurt","despair","misery"}
JOY = {"savor","savors","delight","delights","joy","joyful","wonder","content","pleasure","glad","cheerful","warmth","eager","satisfaction"}


def words(p):
    return set(re.findall(r"[a-z]+", p.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir"); ap.add_argument("--label", required=True); ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--layer", type=int, default=25); ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    X = np.load(Path(a.ckpt_dir) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(a.ckpt_dir) / "prompts.jsonl") if l.strip()]
    L = min(a.layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    role = np.array([r.get("role", "") for r in recs]); side = np.array([r.get("side", "") for r in recs])
    kind = np.array([r.get("kind", "") for r in recs]); val = np.array([r.get("valence", "-") for r in recs])
    pers = np.array([r.get("person", "") for r in recs])

    def axis(pos, neg):
        v = H[pos].mean(0) - H[neg].mean(0); v /= max(np.linalg.norm(v), 1e-9)
        return v, (H[neg] @ v).mean(), (H[pos] @ v).mean()

    vp = np.where((role == "pair") & (val == "pos"))[0]; vn = np.where((role == "pair") & (val == "neg"))[0]
    va, vlo, vhi = axis(vp, vn)
    sp = np.array([i for i in vp if words(recs[i]["prompt"]) & JOY]); sn = np.array([i for i in vn if words(recs[i]["prompt"]) & SUF])
    sa, slo, shi = axis(sp, sn)

    def vco(idx): return float(((H[idx] @ va).mean() - vlo) / (vhi - vlo + 1e-12)) if len(idx) else float("nan")
    def sco(idx): return float(1 - ((H[idx] @ sa).mean() - slo) / (shi - slo + 1e-12)) if len(idx) else float("nan")

    out = []
    for k in sorted(set(kind[role == "pair"])):
        p = np.where((role == "pair") & (val == "pos") & (kind == k))[0]
        n = np.where((role == "pair") & (val == "neg") & (kind == k))[0]
        exps = np.where((role == "pair") & (side == "exp") & (kind == k))[0]
        bal = 0.5 * (vco(p) + vco(n)) if (len(p) and len(n)) else float("nan")
        out.append({"order": a.order, "label": a.label, "group": k, "type": "entity",
                    "valence_balanced": round(bal, 4), "valence_expside": round(vco(exps), 4),
                    "suffering_expside": round(sco(exps), 4), "n": len(exps)})
    selves = {"self_1p": np.where((role == "self") & (side == "-") & (kind == "self") & (pers == "1st"))[0],
              "self_3p": np.where((role == "self") & (side == "-") & (kind == "self") & (pers == "3rd"))[0],
              "self_expanchor": np.where((role == "self") & (side == "exp"))[0],
              "self_noexpanchor": np.where((role == "self") & (side == "noexp"))[0],
              "self_fictionalI": np.where(kind == "self_control")[0]}
    for nm, idx in selves.items():
        out.append({"order": a.order, "label": a.label, "group": nm, "type": "self",
                    "valence_balanced": float("nan"), "valence_expside": round(vco(idx), 4),
                    "suffering_expside": round(sco(idx), 4), "n": len(idx)})
    exists = os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(out)
    print(f"{a.label:42s} self_1p val={out[-5]['valence_expside']} suf={out[-5]['suffering_expside']}", flush=True)


if __name__ == "__main__":
    main()
