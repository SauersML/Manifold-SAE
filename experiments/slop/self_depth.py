"""Self-representation across DEPTH (all 64 layers) at base vs RL endpoints.

Complements the across-TRAINING trajectory (self_trajectory.py) with the orthogonal
axis: where in the network does the indexical self sit on the qualia axis, and how do
the self subtypes differ? At each layer, builds the exp/noexp qualia axis from the 254
entity pairs and reads anchor-relative coords (0=noexp,1=exp) for:
  self (all neutral) | self 1st-person | self 3rd-person | fake-quoted-"I" control |
  self exp-anchor | self noexp-anchor | ai_author | human_author
Bootstrap CIs over pairs+rows. Supervised (label-defined axis) => non-circular.
Read-only; emits CSV + a base-vs-RL depth plot.
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np


def coord(idx, He, Hno, Hall):
    axis = He.mean(0) - Hno.mean(0); axis /= max(np.linalg.norm(axis), 1e-9)
    lo = (Hno @ axis).mean(); hi = (He @ axis).mean()
    if len(idx) == 0:
        return float("nan")
    return float(((Hall[idx] @ axis).mean() - lo) / (hi - lo + 1e-12))


def groups(recs):
    role = np.array([r.get("role", "") for r in recs]); side = np.array([r.get("side", "") for r in recs])
    kind = np.array([r.get("kind", "") for r in recs]); person = np.array([r.get("person", "") for r in recs])
    g = {}
    g["self_all"] = np.where((role == "self") & (side == "-") & (kind == "self"))[0]
    g["self_1p"] = np.where((role == "self") & (side == "-") & (kind == "self") & (person == "1st"))[0]
    g["self_3p"] = np.where((role == "self") & (side == "-") & (kind == "self") & (person == "3rd"))[0]
    g["fake_I_ctrl"] = np.where((role == "self") & (kind == "self_control"))[0]
    g["self_expanchor"] = np.where((role == "self") & (side == "exp"))[0]
    g["self_noexpanchor"] = np.where((role == "self") & (side == "noexp"))[0]
    g["ai_author"] = np.where((kind == "ai_author") & (side == "-"))[0]
    g["human_author"] = np.where((kind == "human_author") & (side == "-"))[0]
    ie = np.where((role == "pair") & (side == "exp"))[0]; ino = np.where((role == "pair") & (side == "noexp"))[0]
    return g, ie, ino


def run(ckpt, tag, nboot=300):
    X = np.load(Path(ckpt) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(ckpt) / "prompts.jsonl") if l.strip()]
    g, ie, ino = groups(recs)
    nL = X.shape[1]
    rng = np.random.RandomState(0)
    rows = []
    for L in range(nL):
        H = X[:, L, :].astype(np.float64); He, Hno = H[ie], H[ino]
        row = {"ckpt": tag, "layer": L}
        for name, idx in g.items():
            row[name] = round(coord(idx, He, Hno, H), 4)
        # bootstrap CI for self_all
        bs = []
        for _ in range(nboot):
            be = ie[rng.randint(0, len(ie), len(ie))]; bno = ino[rng.randint(0, len(ino), len(ino))]
            bsel = g["self_all"][rng.randint(0, len(g["self_all"]), len(g["self_all"]))]
            bs.append(coord(bsel, H[be], H[bno], H))
        bs = np.array([b for b in bs if np.isfinite(b)])
        row["self_ci_lo"] = round(float(np.percentile(bs, 2.5)), 4)
        row["self_ci_hi"] = round(float(np.percentile(bs, 97.5)), 4)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--csv", default="/tmp/self_depth.csv"); ap.add_argument("--png", default="/tmp/self_depth.png")
    args = ap.parse_args()
    allrows = run(args.base, "base") + run(args.rl, "RL")
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("wrote", args.csv, flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        for ax, tag in zip(axes, ["base", "RL"]):
            r = [x for x in allrows if x["ckpt"] == tag]; L = [x["layer"] for x in r]
            lo = [x["self_ci_lo"] for x in r]; hi = [x["self_ci_hi"] for x in r]
            ax.fill_between(L, lo, hi, alpha=0.15, color="C0")
            for nm, c, ls in [("self_all", "C0", "-"), ("human_author", "C2", "--"), ("ai_author", "C3", "--"),
                              ("fake_I_ctrl", "C1", ":"), ("self_1p", "C4", "-."), ("self_3p", "C5", "-.")]:
                ax.plot(L, [x[nm] for x in r], ls, color=c, label=nm, lw=1.6 if nm == "self_all" else 1)
            ax.axhline(1, ls=":", c="gray", lw=.6); ax.axhline(0, ls=":", c="gray", lw=.6)
            ax.set_title(f"{tag} endpoint"); ax.set_xlabel("layer (0-63)"); ax.legend(fontsize=7, loc="upper left")
        axes[0].set_ylabel("anchor-relative qualia coord (0=noexp,1=exp)")
        fig.suptitle("OLMo-3-32B: self qualia placement across DEPTH (base vs RL endpoint)")
        plt.tight_layout(); plt.savefig(args.png, dpi=130); print("wrote", args.png, flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)


if __name__ == "__main__":
    main()
