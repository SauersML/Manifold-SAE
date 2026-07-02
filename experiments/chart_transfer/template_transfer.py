"""Chart-transfer invariance: is a circle chart a property of the FEATURE or of the
PROMPT DISTRIBUTION?

This is the LLM analogue of the paper's Blender ground-truth validation -- and a claim
the paper CANNOT test, because a linear SAE has no intrinsic coordinate to transfer. We
fit the weekday / month circle chart on a SUBSET of the harvest's template sentences
(a TEMPLATE split, not a row split), then evaluate on the HELD-OUT templates:

  (a) COORDINATE CONSISTENCY -- does the same token (e.g. "Monday") receive the same
      recovered angle whether it appears in a fit-template or an eval-template sentence?
      (circular correlation of the per-token angle read on the two template groups.)
  (b) ADJACENCY PRESERVATION on unseen templates -- forwarding eval-template activations
      through the FROZEN chart, do the tokens still order correctly around the circle?
  (c) EV TRANSFER vs the LINEAR 2-PC plane transferred the SAME way -- the fair fight:
      does the 1-coordinate CHART generalize to new prompts better than, or as well as,
      the 2-coordinate PLANE? (chart-1 vs linear-1 vs linear-2, held-out-template EV.)

Everything is held out at the TEMPLATE level: the PCA reduction, the chart, and the
linear baseline are all fit on fit-templates only and evaluated on eval-templates.
Reuses the cached probe_out/*.npz harvest (no model load) and block_nursery's isolated
torch chart fit + circular stats. Leave-one-template-out CV for robust averages, plus a
fixed 3/2 split for the per-token consistency read.

Success = "the chart is a property of the FEATURE, not the prompt distribution", with
numbers. A null (chart no more transferable than the plane) is an equally-publishable
scoping result.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                                   # experiments/
# harvest source: default the original 5-template probe cache; CHART_TRANSFER_HARVEST
# overrides with a richer cache (e.g. the 14-template harvest_more/).
PROBE_OUT = Path(os.environ.get("CHART_TRANSFER_HARVEST", EXP / "probe_out"))
OUT_DIR = Path(os.environ.get("CHART_TRANSFER_OUT", HERE / "template_out"))

# import block_nursery for the isolated chart fit + circular stats (no reimplementation)
_spec = importlib.util.spec_from_file_location("block_nursery", EXP / "block_nursery.py")
bn = importlib.util.module_from_spec(_spec)
sys.modules["block_nursery"] = bn
_spec.loader.exec_module(bn)

circular_mean = bn.circular_mean
circular_corr = bn.circular_corr
recovered_angle = bn.recovered_angle
ev = bn.ev


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_demeaned(name: str):
    """Per-template-demeaned activations at every layer + rank/template_idx/n_labels.

    Per-template demeaning subtracts each template's OWN mean (within-template), so no
    statistic crosses the fit/eval template boundary."""
    z = np.load(PROBE_OUT / f"harvest_{name}.npz", allow_pickle=False)
    layers = [int(x) for x in z["layers"]]
    tidx = z["template_idx"].astype(int)
    rank = z["rank"].astype(int)
    demeaned = {}
    for L in layers:
        X = z[f"L{L}"].astype(np.float64)
        Xd = X.copy()
        for t in np.unique(tidx):
            mt = tidx == t
            Xd[mt] = X[mt] - X[mt].mean(0, keepdims=True)
        demeaned[L] = Xd
    return demeaned, layers, tidx, rank, int(z["n_labels"])


def best_layer(demeaned, layers, tidx, rank):
    """Pick the strongest-LINEAR layer (conservative for the baseline), same criterion
    as curved_feature_probes / block_nursery."""
    best_L, best = layers[0], -1.0
    for L in layers:
        Xc = demeaned[L] - demeaned[L].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:8].T
        rr = rank - rank.mean()
        sc = 0.0
        for k in range(proj.shape[1]):
            pk = proj[:, k] - proj[:, k].mean()
            d = np.sqrt((pk ** 2).sum() * (rr ** 2).sum())
            if d > 0:
                sc = max(sc, abs(float((pk * rr).sum() / d)))
        if sc > best:
            best_L, best = L, sc
    return best_L


# --------------------------------------------------------------------------- #
# Transferred reductions
# --------------------------------------------------------------------------- #
def pca_fit_project(X, fit_rows, r):
    """PCA fit on fit_rows (train-template), project ALL rows. Returns (Z_all, mu)."""
    mu = X[fit_rows].mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc[fit_rows], full_matrices=False)
    r = min(r, Vt.shape[0])
    return Xc @ Vt[:r].T, mu


def linear_transfer_ev(Z, fit_rows, eval_rows, L):
    """Held-out-template EV of the optimal linear L-dim reconstruction (PCA on fit_rows)."""
    mu = Z[fit_rows].mean(0)
    Ztr = Z[fit_rows] - mu
    _, _, Vt = np.linalg.svd(Ztr, full_matrices=False)
    Vt = Vt[:L]
    te = Z[eval_rows] - mu
    return round(ev(Z[eval_rows], te @ Vt.T @ Vt + mu), 4)


def cyclic_adjacency(tok_ang, n_tok):
    uniq = list(range(len(tok_ang)))
    seq = list(np.argsort(np.asarray(tok_ang) % (2 * np.pi)))
    true_adj = {frozenset((i, (i + 1) % len(uniq))) for i in uniq}
    rec_adj = {frozenset((seq[i], seq[(i + 1) % len(seq)])) for i in range(len(seq))}
    return round(len(true_adj & rec_adj) / len(uniq), 3)


def per_token_angle(angle, rank, tokens):
    """Circular mean recovered angle per token id (over the given rows)."""
    return np.array([circular_mean(angle[rank == u]) for u in tokens])


# --------------------------------------------------------------------------- #
# One template split
# --------------------------------------------------------------------------- #
def run_split(name, X, tidx, rank, n_tok, fit_templates, eval_templates, r, tag):
    fit_rows = np.where(np.isin(tidx, fit_templates))[0]
    eval_rows = np.where(np.isin(tidx, eval_templates))[0]
    Z, _ = pca_fit_project(X, fit_rows, r)
    # frozen chart: fit on fit-template rows, forward ALL rows
    fit = bn.fit_curved_isolated(Z, n_atoms=1, tag=tag, train_idx=fit_rows,
                                 test_idx=eval_rows, target_k=1)
    out = {"fit_templates": [int(t) for t in fit_templates],
           "eval_templates": [int(t) for t in eval_templates],
           "chart_status": fit["status"]}
    if fit["status"] != "CONVERGED":
        return out
    Zhat, _ = bn._load_fit(fit["out_path"])
    angle = recovered_angle(Zhat)                    # atan2 of reconstruction, all rows
    tokens = list(range(n_tok))
    # (a) coordinate consistency: same token's angle on fit vs eval template groups
    ang_fit = per_token_angle(angle[fit_rows], rank[fit_rows], tokens)
    ang_eval = per_token_angle(angle[eval_rows], rank[eval_rows], tokens)
    out["coord_consistency_circ_corr"] = round(abs(circular_corr(ang_fit, ang_eval)), 3)
    # per-token angular offset (median absolute wrapped difference), degrees
    diff = np.angle(np.exp(1j * (ang_fit - ang_eval)))
    out["coord_median_abs_offset_deg"] = round(float(np.median(np.abs(diff)) * 180 / np.pi), 1)
    # (b) adjacency on UNSEEN templates
    out["adjacency_eval_templates"] = cyclic_adjacency(ang_eval, n_tok)
    out["adjacency_fit_templates"] = cyclic_adjacency(ang_fit, n_tok)
    # (c) EV transfer -- chart(1 coord) vs linear plane transferred the same way
    out["chart_ev_eval"] = round(float(fit["ev"]), 4)     # held-out-template chart EV
    out["chart_ev_fit"] = round(float(fit["ev_train"]), 4)
    out["linear1_ev_eval"] = linear_transfer_ev(Z, fit_rows, eval_rows, 1)
    out["linear2_ev_eval"] = linear_transfer_ev(Z, fit_rows, eval_rows, 2)
    out["reduce_dim"] = int(r)
    return out


def run_set(name, r=8):
    demeaned, layers, tidx, rank, n_tok = load_demeaned(name)
    L = best_layer(demeaned, layers, tidx, rank)
    X = demeaned[L]
    templates = sorted(np.unique(tidx).tolist())
    T = len(templates)
    print(f"[{name}] layer L{L}, {T} templates, {n_tok} tokens", flush=True)

    result = {"set": name, "layer": int(L), "n_templates": T, "n_tokens": n_tok,
              "reduce_dim": r}

    # ---- fixed split: first ceil(T/2)+... use 3 fit / rest eval (the consistency read)
    n_fit = max(2, T - 2)
    split = run_split(name, X, tidx, rank, n_tok, templates[:n_fit], templates[n_fit:],
                      r, tag=f"tt_{name}_split")
    result["fixed_split"] = split
    print(f"[{name}] fixed split fit={split['fit_templates']} eval={split['eval_templates']}: "
          f"coord_consistency={split.get('coord_consistency_circ_corr')} "
          f"adj_eval={split.get('adjacency_eval_templates')} "
          f"chart_ev_eval={split.get('chart_ev_eval')} lin2_ev_eval={split.get('linear2_ev_eval')}",
          flush=True)

    # ---- leave-one-template-out CV (robust averages)
    loto = []
    for held in templates:
        fitT = [t for t in templates if t != held]
        s = run_split(name, X, tidx, rank, n_tok, fitT, [held], r,
                      tag=f"tt_{name}_loto{held}")
        loto.append(s)
        print(f"[{name}] LOTO held={held}: coord_consistency={s.get('coord_consistency_circ_corr')} "
              f"adj_eval={s.get('adjacency_eval_templates')} chart_ev_eval={s.get('chart_ev_eval')} "
              f"lin1={s.get('linear1_ev_eval')} lin2={s.get('linear2_ev_eval')}", flush=True)
    ok = [s for s in loto if s.get("chart_status") == "CONVERGED"]

    def agg(key, fn=np.mean):
        vals = [s[key] for s in ok if key in s]
        return round(float(fn(vals)), 3) if vals else None

    cc = [s["coord_consistency_circ_corr"] for s in ok if "coord_consistency_circ_corr" in s]
    result["loto"] = loto
    result["loto_summary"] = {
        "n_folds": len(ok),
        "coord_consistency_circ_corr": agg("coord_consistency_circ_corr"),
        "coord_consistency_median": agg("coord_consistency_circ_corr", np.median),
        "coord_consistency_frac_over_0.8": (round(float(np.mean(np.array(cc) > 0.8)), 2)
                                            if cc else None),
        "coord_median_abs_offset_deg": agg("coord_median_abs_offset_deg", np.median),
        "adjacency_eval_mean": agg("adjacency_eval_templates"),
        "chart_ev_eval_mean": agg("chart_ev_eval"),
        "linear1_ev_eval_mean": agg("linear1_ev_eval"),
        "linear2_ev_eval_mean": agg("linear2_ev_eval"),
    }
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = [s for s in ("weekday", "month") if (PROBE_OUT / f"harvest_{s}.npz").exists()]
    r = int(os.environ.get("CHART_TRANSFER_RDIM", "8"))
    allres = {"reduce_dim": r, "sets": {}}
    for name in sets:
        allres["sets"][name] = run_set(name, r=r)
        (OUT_DIR / "template_transfer.json").write_text(json.dumps(allres, indent=2, default=float))
        print(f"[saved] {OUT_DIR / 'template_transfer.json'}", flush=True)

    # verdict summary
    verdict = {}
    for name, res in allres["sets"].items():
        ls = res["loto_summary"]
        c, l1, l2 = (ls["chart_ev_eval_mean"], ls["linear1_ev_eval_mean"],
                     ls["linear2_ev_eval_mean"])
        verdict[name] = {
            "n_templates": res["n_templates"], "n_tokens": res["n_tokens"],
            "coord_consistency_circ_corr": ls["coord_consistency_circ_corr"],
            "adjacency_eval_mean": ls["adjacency_eval_mean"],
            "chart_ev_eval_mean": c,
            "linear1_ev_eval_mean": l1,
            "linear2_ev_eval_mean": l2,
            # the fair fights: chart uses 1 coord (compare to lin1); a circle needs 2
            # linear dims (compare to lin2). chart1>=lin1 and chart1~=lin2 = chart wins.
            "chart1_minus_linear1": None if c is None or l1 is None else round(c - l1, 3),
            "chart1_minus_linear2": None if c is None or l2 is None else round(c - l2, 3),
        }
    allres["verdict"] = verdict
    (OUT_DIR / "template_transfer.json").write_text(json.dumps(allres, indent=2, default=float))
    print(f"\n[VERDICT] {json.dumps(verdict, indent=2)}", flush=True)
    return allres


if __name__ == "__main__":
    main()
