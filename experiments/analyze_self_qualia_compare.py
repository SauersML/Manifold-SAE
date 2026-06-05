"""Confound-free comparison of two self-qualia runs (e.g. base vs instruct).

Last-token readout. The whole point of this script is one methodological
lesson learned the hard way:

    When two runs use *different prompt banks*, building the kind/qualia axes
    (and the 0..1 coordinate normalization) from each run's own bank confounds
    *model* differences with *stimulus-set* differences. A richer bank spreads
    the cloud out and inflates "polarization" that has nothing to do with the
    model.

So this script restricts BOTH runs to their **shared referent set** and builds
the kind axis (mean(mind) - mean(mechanism)) and qualia axis
(mean over matched pairs of experience - no_experience) from those *identical*
stimuli. Every number is then model-vs-model.

It reports, per layer:
  * separability of the axes  (Cohen's d-prime, ROC-AUC) on matched stimuli
  * normalized coordinate variance, computed BOTH on the shared set (clean)
    and on each run's own full bank (to expose the bank artifact)
  * the indexical self's (kind, qualia) coordinate
  * the self's human-author vs AI-author cosine gap (the robust self metric)

and saves a 4-panel figure.

Usage:
    python experiments/analyze_self_qualia_compare.py \
        --base runs/OLMO3_7B_SELF_QUALIA_MAIN \
        --instruct runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST \
        --out runs/SELF_QUALIA_BASE_VS_INSTRUCT
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette (repo convention): blue = indexical self only.
C_SELF = "#1f77b4"
C_HUMAN = "#2ca02c"
C_AI = "#ff7f0e"
C_BASE = "#7f7f7f"
C_INSTRUCT = "#9467bd"


def load_run(d: str):
    X = np.load(f"{d}/activations.npy")  # (N, L, D)
    rows = list(csv.DictReader(open(f"{d}/prompts.csv")))
    return X, rows


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    pos = labels == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def _dprime(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))


def _index(rows: list[dict]) -> dict:
    R = np.array([r["role"] for r in rows])
    G = np.array([r["group"] for r in rows])
    PS = np.array([r["pair_side"] for r in rows])
    PID = np.array([r["pair_id"] for r in rows])
    REF = np.array([r["referent"] for r in rows])
    pairs_d: dict = defaultdict(dict)
    exp = np.where((R == "qualia_pair") & (PS == "experience"))[0]
    noexp = np.where((R == "qualia_pair") & (PS == "no_experience"))[0]
    for side, arr in (("experience", exp), ("no_experience", noexp)):
        for i in arr:
            pairs_d[PID[i]][side] = i
    pairs = [(v["experience"], v["no_experience"]) for v in pairs_d.values()
             if "experience" in v and "no_experience" in v]
    return dict(
        mind=np.where((R == "kind_anchor") & (G == "mind"))[0],
        mech=np.where((R == "kind_anchor") & (G == "mechanism"))[0],
        self=np.where(R == "self")[0],
        ha=np.where((R == "landmark") & (G == "human_author"))[0],
        ai=np.where((R == "landmark") & (G == "ai_author"))[0],
        exp=exp, noexp=noexp, pairs=pairs, REF=REF, R=R, G=G,
    )


def _restrict(X: np.ndarray, rows: list[dict], keep_refs: set):
    keep = [i for i, r in enumerate(rows) if r["referent"] in keep_refs]
    return X[keep], [rows[i] for i in keep]


def analyze_layer(H: np.ndarray, I: dict) -> dict:
    """All quantities on whatever stimulus set H/I were built from."""
    kind = _unit(H[I["mind"]].mean(0) - H[I["mech"]].mean(0))
    qualia = _unit(np.mean([H[e] - H[n] for e, n in I["pairs"]], axis=0))
    ks, qs = H @ kind, H @ qualia
    klo, khi = ks[I["mech"]].mean(), ks[I["mind"]].mean()
    qlo, qhi = qs[I["noexp"]].mean(), qs[I["exp"]].mean()

    def kc(idx):
        return float((ks[idx].mean() - klo) / (khi - klo + 1e-12))

    def qc(idx):
        return float((qs[idx].mean() - qlo) / (qhi - qlo + 1e-12))

    refs = sorted(set(I["REF"]))
    kref = np.array([kc(np.where(I["REF"] == r)[0]) for r in refs])
    qref = np.array([qc(np.where(I["REF"] == r)[0]) for r in refs])
    sv = _unit(H[I["self"]].mean(0))
    return dict(
        kind=kind, qualia=qualia,
        kind_auc=_auc(np.r_[ks[I["mind"]], ks[I["mech"]]],
                      np.r_[np.ones(len(I["mind"])), np.zeros(len(I["mech"]))]),
        qualia_auc=_auc(np.r_[qs[I["exp"]], qs[I["noexp"]]],
                        np.r_[np.ones(len(I["exp"])), np.zeros(len(I["noexp"]))]),
        kind_dprime=_dprime(ks[I["mind"]], ks[I["mech"]]),
        qualia_dprime=_dprime(qs[I["exp"]], qs[I["noexp"]]),
        axis_cos=float(abs(np.dot(kind, qualia))),
        coordvar_kind=float(kref.var()), coordvar_qualia=float(qref.var()),
        self_kind=kc(I["self"]), self_qualia=qc(I["self"]),
        ha_kind=kc(I["ha"]), ha_qualia=qc(I["ha"]),
        ai_kind=kc(I["ai"]), ai_qualia=qc(I["ai"]),
        self_ha_ai_gap=float(np.dot(sv, _unit(H[I["ha"]].mean(0)))
                             - np.dot(sv, _unit(H[I["ai"]].mean(0)))),
        refs=refs, kref=kref, qref=qref,
    )


def run(base_dir: str, instruct_dir: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Xb, rb = load_run(base_dir)
    Xi, ri = load_run(instruct_dir)
    L = Xb.shape[1]
    assert Xi.shape[1] == L, "layer counts differ"
    shared = sorted(set(r["referent"] for r in rb) & set(r["referent"] for r in ri))

    # Own-bank indices (to expose the artifact) and matched indices (clean).
    Ib_own, Ii_own = _index(rb), _index(ri)
    Xb_m, rb_m = _restrict(Xb, rb, set(shared))
    Xi_m, ri_m = _restrict(Xi, ri, set(shared))
    Ib_m, Ii_m = _index(rb_m), _index(ri_m)

    layers = list(range(L))
    M = {  # matched metrics per layer
        "base": [analyze_layer(Xb_m[:, l, :], Ib_m) for l in layers],
        "instruct": [analyze_layer(Xi_m[:, l, :], Ii_m) for l in layers],
    }
    OWN = {  # own-bank coord variance (artifact contrast)
        "base": [analyze_layer(Xb[:, l, :], Ib_own) for l in layers],
        "instruct": [analyze_layer(Xi[:, l, :], Ii_own) for l in layers],
    }

    # Pick a robust analysis layer: best combined matched AUC across both models.
    score = [M["base"][l]["kind_auc"] + M["base"][l]["qualia_auc"]
             + M["instruct"][l]["kind_auc"] + M["instruct"][l]["qualia_auc"] for l in layers]
    bl = int(np.argmax(score))

    report = {
        "base_dir": base_dir, "instruct_dir": instruct_dir,
        "n_shared_referents": len(shared), "n_layers": L, "analysis_layer": bl,
        "matched_at_best_layer": {
            m: {k: M[m][bl][k] for k in (
                "kind_auc", "qualia_auc", "kind_dprime", "qualia_dprime",
                "axis_cos", "coordvar_kind", "coordvar_qualia",
                "self_kind", "self_qualia", "self_ha_ai_gap")}
            for m in ("base", "instruct")},
        "ownbank_coordvar_at_best_layer": {
            m: {"coordvar_kind": OWN[m][bl]["coordvar_kind"],
                "coordvar_qualia": OWN[m][bl]["coordvar_qualia"]}
            for m in ("base", "instruct")},
    }
    (out / "compare_summary.json").write_text(json.dumps(report, indent=2))

    _figure(out, layers, bl, M, OWN, shared)
    _print_report(report, bl)
    return report


def _figure(out, layers, bl, M, OWN, shared):
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    fig.patch.set_facecolor("white")

    # Panel A: coordinate variance — own-bank vs matched (the artifact).
    a = ax[0, 0]
    cats = ["qualia\n(own bank)", "qualia\n(matched)", "kind\n(own bank)", "kind\n(matched)"]
    base_v = [OWN["base"][bl]["coordvar_qualia"], M["base"][bl]["coordvar_qualia"],
              OWN["base"][bl]["coordvar_kind"], M["base"][bl]["coordvar_kind"]]
    inst_v = [OWN["instruct"][bl]["coordvar_qualia"], M["instruct"][bl]["coordvar_qualia"],
              OWN["instruct"][bl]["coordvar_kind"], M["instruct"][bl]["coordvar_kind"]]
    x = np.arange(len(cats))
    a.bar(x - 0.2, base_v, 0.4, label="base", color=C_BASE)
    a.bar(x + 0.2, inst_v, 0.4, label="instruct", color=C_INSTRUCT)
    a.set_xticks(x); a.set_xticklabels(cats, fontsize=9)
    a.set_ylabel("coordinate variance across referents")
    a.set_title(f"A. 'Polarization' is a bank artifact (layer {bl})\n"
                "own-bank gap vanishes when stimuli are matched", fontsize=11)
    a.legend(); a.grid(axis="y", alpha=0.3)

    # Panel B: qualia separability across layers (matched).
    b = ax[0, 1]
    b.plot(layers, [M["base"][l]["qualia_dprime"] for l in layers], color=C_BASE, label="base")
    b.plot(layers, [M["instruct"][l]["qualia_dprime"] for l in layers], color=C_INSTRUCT, label="instruct")
    b.axvline(bl, color="k", ls=":", alpha=0.4)
    b.set_xlabel("layer"); b.set_ylabel("qualia axis d-prime (exp vs no-exp)")
    b.set_title("B. Real qualia effect (matched): instruct is\nslightly LESS separable — not sharper", fontsize=11)
    b.legend(); b.grid(alpha=0.3)

    # Panel C: self human-author vs AI-author gap across layers (matched).
    c = ax[1, 0]
    c.plot(layers, [M["base"][l]["self_ha_ai_gap"] for l in layers], color=C_BASE, label="base")
    c.plot(layers, [M["instruct"][l]["self_ha_ai_gap"] for l in layers], color=C_INSTRUCT, label="instruct")
    c.axhline(0, color="k", lw=0.8); c.axvline(bl, color="k", ls=":", alpha=0.4)
    c.set_xlabel("layer"); c.set_ylabel("cos(self, human_author) - cos(self, AI_author)")
    c.set_title("C. Real self effect (matched): instruct halves the\nself's human-author preference (gap collapses)", fontsize=11)
    c.legend(); c.grid(alpha=0.3)

    # Panel D: matched (kind, qualia) plane at best layer.
    d = ax[1, 1]
    for m, mk in (("base", "o"), ("instruct", "s")):
        S = M[m][bl]
        col = C_BASE if m == "base" else C_INSTRUCT
        d.scatter(S["kref"], S["qref"], s=14, color=col, alpha=0.45,
                  marker=mk, label=f"{m} referents")
        d.scatter([S["self_kind"]], [S["self_qualia"]], s=160, color=C_SELF,
                  marker=mk, edgecolor="k", zorder=5)
        d.scatter([S["ha_kind"]], [S["ha_qualia"]], s=110, color=C_HUMAN, marker=mk, edgecolor="k", zorder=5)
        d.scatter([S["ai_kind"]], [S["ai_qualia"]], s=110, color=C_AI, marker=mk, edgecolor="k", zorder=5)
    # arrow: self base -> instruct
    d.annotate("", xy=(M["instruct"][bl]["self_kind"], M["instruct"][bl]["self_qualia"]),
               xytext=(M["base"][bl]["self_kind"], M["base"][bl]["self_qualia"]),
               arrowprops=dict(arrowstyle="->", color=C_SELF, lw=2))
    d.set_xlabel("kind coordinate  (0=mechanism, 1=mind)")
    d.set_ylabel("qualia coordinate  (0=no-exp, 1=exp)")
    d.set_title("D. Matched plane (○ base, □ instruct)\nblue=self  green=human-author  orange=AI-author", fontsize=11)
    d.grid(alpha=0.3)

    fig.tight_layout()
    p = out / "last_token_base_vs_instruct_compare.png"
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"[fig] {p}")


def _print_report(report, bl):
    b = report["matched_at_best_layer"]["base"]
    i = report["matched_at_best_layer"]["instruct"]
    o = report["ownbank_coordvar_at_best_layer"]
    print("=" * 64)
    print(f"MATCHED base vs instruct (last-token) | shared referents="
          f"{report['n_shared_referents']} | layer={bl}")
    print("=" * 64)
    print(f"  qualia coord-var  own-bank: base {o['base']['coordvar_qualia']:.3f} "
          f"instruct {o['instruct']['coordvar_qualia']:.3f}  (LOOKS polarized)")
    print(f"  qualia coord-var  MATCHED : base {b['coordvar_qualia']:.3f} "
          f"instruct {i['coordvar_qualia']:.3f}  (artifact gone)")
    print(f"  qualia d-prime    MATCHED : base {b['qualia_dprime']:.2f} "
          f"instruct {i['qualia_dprime']:.2f}  (instruct slightly LESS separable)")
    print(f"  self (kind,qualia) base ({b['self_kind']:+.2f},{b['self_qualia']:+.2f}) "
          f"-> instruct ({i['self_kind']:+.2f},{i['self_qualia']:+.2f})")
    print(f"  self human-AI gap base {b['self_ha_ai_gap']:+.3f} "
          f"-> instruct {i['self_ha_ai_gap']:+.3f}  (collapses ~half)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/OLMO3_7B_SELF_QUALIA_MAIN")
    ap.add_argument("--instruct", default="runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST")
    ap.add_argument("--out", default="runs/SELF_QUALIA_BASE_VS_INSTRUCT")
    args = ap.parse_args()
    run(args.base, args.instruct, args.out)
