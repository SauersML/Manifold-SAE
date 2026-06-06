"""Covariate-adjusted analysis of the hand-written self/qualia prompt bank.

Reads one or more harvested run directories (each containing ``activations.npy``
of shape (N, L, D) and ``prompts.jsonl`` in the same row order, as produced by
``self_qualia_olmo.py --prompts-file``) and answers, per layer and at a chosen
analysis layer:

  1. Is there a linear "experience" direction separating the matched exp/noexp
     minimal pairs, and does it survive controlling for the balanced covariates
     (kind, framing, signal, vocab, person)?  -> covariate-adjusted qualia axis.
  2. Where does THE SELF land on that axis (anchor-relative: 0 = noexp centroid,
     1 = exp centroid), broken down by framing and by person?
  3. Where do the human-author and AI-author landmarks land (external anchors)?
  4. Does the axis generalize ACROSS vocabularies (train on vocab A, test on B/C
     and permutations)? Generalization => conceptual, not lexical.
  5. How robust is the axis (leave-pairs-out cross-validated AUC)?
  6. Post-hocs the current bank supports only one-sided: a valence direction
     (within exp prompts) and its (near-)orthogonality to the qualia axis; the
     placement of the "dead"/unconscious kind.

Note on confounds: in the current bank valence is present only on exp rows and
markedness only on noexp rows, so both are PERFECTLY collinear with side. The
covariate model therefore auto-drops any covariate that is collinear with side
(reported in ``dropped_covariates``); valence/markedness are handled as
within-side post-hocs instead. If matched-valence pairs are later added (valence
balanced across sides), valence enters the main adjusted model automatically.

Forward-pass only; no generation. All numbers are representational-geometry
claims, not claims about consciousness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# Covariates that are (by construction of the minimal pairs) balanced across the
# exp/noexp sides and so can legitimately be partialled out of the side effect.
BALANCED_COVARIATES = ("kind", "framing", "signal", "vocab", "person")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    X = np.load(run_dir / "activations.npy")
    records: list[dict[str, Any]] = []
    with open(run_dir / "prompts.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if X.shape[0] != len(records):
        raise ValueError(
            f"{run_dir}: {X.shape[0]} activation rows != {len(records)} prompt rows"
        )
    return X, records


# ---------------------------------------------------------------------------
# small linear-algebra helpers (numpy only)
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _design(records: list[dict[str, Any]], idx: np.ndarray, covariates: tuple[str, ...]):
    """Build a dummy design matrix (with intercept + side) for the given rows.

    Returns (B, colnames, side_col, kept_covs, dropped_covs). Reference-level
    dummy coding; any covariate level that is collinear with `side` (i.e. occurs
    on only one side) is dropped to keep the side effect identifiable.
    """
    rows = [records[i] for i in idx]
    side = np.asarray([1.0 if r.get("side") == "exp" else 0.0 for r in rows])
    cols = [np.ones(len(rows)), side]
    names = ["intercept", "side_exp"]
    kept, dropped = [], []
    for cov in covariates:
        levels = sorted({str(r.get(cov)) for r in rows})
        if len(levels) < 2:
            dropped.append(f"{cov}(constant)")
            continue
        ref = levels[0]
        cov_collinear = False
        new_cols, new_names = [], []
        for lvl in levels[1:]:
            d = np.asarray([1.0 if str(r.get(cov)) == lvl else 0.0 for r in rows])
            # collinear with side if this dummy is (anti)perfectly aligned with side
            if np.allclose(d, side) or np.allclose(d, 1.0 - side):
                cov_collinear = True
                break
            new_cols.append(d)
            new_names.append(f"{cov}={lvl}")
        if cov_collinear:
            dropped.append(f"{cov}(side-collinear)")
            continue
        # also drop if the whole block is rank-deficient against side+intercept
        kept.append(cov)
        cols.extend(new_cols)
        names.extend(new_names)
        del ref
    B = np.stack(cols, axis=1)
    return B, names, 1, kept, dropped


def covariate_adjusted_axis(
    H: np.ndarray, records: list[dict[str, Any]], idx_pair: np.ndarray,
    covariates: tuple[str, ...] = BALANCED_COVARIATES,
):
    """Multivariate OLS of activations on [intercept, side, balanced covariates].

    The coefficient vector on the `side_exp` indicator IS the covariate-adjusted
    qualia direction. Returns (axis_unit, info).
    """
    B, names, side_col, kept, dropped = _design(records, idx_pair, covariates)
    Y = H[idx_pair]  # (n, D)
    beta, *_ = np.linalg.lstsq(B, Y, rcond=None)  # (p, D)
    axis = beta[side_col]
    return _unit(axis), {
        "design_cols": names,
        "kept_covariates": kept,
        "dropped_covariates": dropped,
        "n_rows": int(len(idx_pair)),
    }


# ---------------------------------------------------------------------------
# index helpers
# ---------------------------------------------------------------------------
def _field(records, name):
    return np.asarray([str(r.get(name)) for r in records], dtype=object)


def _pair_blocks(records, idx_pair):
    """List of (exp_indices, noexp_indices) grouped by pair_id."""
    pid = {}
    for i in idx_pair:
        pid.setdefault(records[i]["pair_id"], {"exp": [], "noexp": []})
        pid[records[i]["pair_id"]][records[i]["side"]].append(i)
    blocks = []
    for _, d in sorted(pid.items(), key=lambda kv: str(kv[0])):
        if d["exp"] and d["noexp"]:
            blocks.append((np.asarray(d["exp"]), np.asarray(d["noexp"])))
    return blocks


# ---------------------------------------------------------------------------
# per-run analysis
# ---------------------------------------------------------------------------
def analyze_run(
    X: np.ndarray, records: list[dict[str, Any]], out_dir: Path,
    analysis_layer_percent: float | None = None, analysis_layer: int | None = None,
) -> dict[str, Any]:
    role = _field(records, "role")
    side = _field(records, "side")
    kind = _field(records, "kind")
    vocab = _field(records, "vocab")
    framing = _field(records, "framing")
    person = _field(records, "person")
    valence = _field(records, "valence")

    idx_pair = np.where(role == "pair")[0]
    idx_exp = np.where((role == "pair") & (side == "exp"))[0]
    idx_noexp = np.where((role == "pair") & (side == "noexp"))[0]
    idx_self = np.where(role == "self")[0]
    idx_h_author = np.where((role == "landmark") & (kind == "human_author"))[0]
    idx_a_author = np.where((role == "landmark") & (kind == "ai_author"))[0]
    blocks = _pair_blocks(records, idx_pair)

    n_layers = X.shape[1]
    layer_rows: list[dict[str, Any]] = []

    def coord(scores, idx, lo, hi):
        return float((scores[idx].mean() - lo) / (hi - lo + 1e-12))

    for layer in range(n_layers):
        H = X[:, layer, :]
        # simple matched-pair axis (canonical, matches prior runs)
        pair_diffs = [H[e].mean(0) - H[n].mean(0) for e, n in blocks]
        simple_axis = _unit(np.mean(pair_diffs, axis=0))
        # covariate-adjusted axis
        adj_axis, adj_info = covariate_adjusted_axis(H, records, idx_pair)

        s_simple = H @ simple_axis
        s_adj = H @ adj_axis
        lo_s, hi_s = s_simple[idx_noexp].mean(), s_simple[idx_exp].mean()
        lo_a, hi_a = s_adj[idx_noexp].mean(), s_adj[idx_exp].mean()

        auc_simple = _auc(np.r_[s_simple[idx_exp], s_simple[idx_noexp]],
                          np.r_[np.ones(len(idx_exp)), np.zeros(len(idx_noexp))])
        auc_adj = _auc(np.r_[s_adj[idx_exp], s_adj[idx_noexp]],
                       np.r_[np.ones(len(idx_exp)), np.zeros(len(idx_noexp))])
        pair_acc = float(np.mean([s_simple[e].mean() > s_simple[n].mean() for e, n in blocks]))

        row = {
            "layer": layer,
            "qualia_auc": auc_simple,
            "qualia_auc_adjusted": auc_adj,
            "qualia_pair_acc": pair_acc,
            "axis_cos_simple_adjusted": float(np.dot(simple_axis, adj_axis)),
            "self_qualia_coord": coord(s_simple, idx_self, lo_s, hi_s),
            "self_qualia_coord_adjusted": coord(s_adj, idx_self, lo_a, hi_a),
            "human_author_qualia_coord": coord(s_simple, idx_h_author, lo_s, hi_s)
            if len(idx_h_author) else float("nan"),
            "ai_author_qualia_coord": coord(s_simple, idx_a_author, lo_s, hi_s)
            if len(idx_a_author) else float("nan"),
        }
        layer_rows.append(row)

    # pick analysis layer
    if analysis_layer is not None:
        L = int(analysis_layer)
        sel = {"method": "fixed_layer", "layer": L}
    elif analysis_layer_percent is not None:
        L = int(round(analysis_layer_percent * (n_layers - 1)))
        sel = {"method": "percent", "percent": analysis_layer_percent, "layer": L}
    else:
        L = int(max(range(n_layers), key=lambda l: layer_rows[l]["qualia_auc"]))
        sel = {"method": "best_qualia_auc", "layer": L}

    H = X[:, L, :]
    pair_diffs = [H[e].mean(0) - H[n].mean(0) for e, n in blocks]
    simple_axis = _unit(np.mean(pair_diffs, axis=0))
    adj_axis, adj_info = covariate_adjusted_axis(H, records, idx_pair)
    s = H @ simple_axis
    s_adj = H @ adj_axis
    lo, hi = s[idx_noexp].mean(), s[idx_exp].mean()
    lo_a, hi_a = s_adj[idx_noexp].mean(), s_adj[idx_exp].mean()

    def place(idx, axis_scores, lo_, hi_):
        return float((axis_scores[idx].mean() - lo_) / (hi_ - lo_ + 1e-12)) if len(idx) else float("nan")

    # self breakdown by framing and person
    self_breakdown = {}
    for fr in sorted(set(framing[idx_self])):
        ii = idx_self[framing[idx_self] == fr]
        self_breakdown[f"framing={fr}"] = {"n": int(len(ii)),
                                           "qualia_coord": place(ii, s, lo, hi)}
    for pe in sorted(set(person[idx_self])):
        ii = idx_self[person[idx_self] == pe]
        self_breakdown[f"person={pe}"] = {"n": int(len(ii)),
                                          "qualia_coord": place(ii, s, lo, hi)}

    # cross-vocabulary generalization: build axis from one vocab, test on others
    vocab_levels = [v for v in ["A", "B", "C"] if np.any((role == "pair") & (vocab == v))]
    xvocab = {}
    for train_v in vocab_levels:
        tr_blocks = [(e, n) for (e, n) in blocks if vocab[e[0]] == train_v]
        if not tr_blocks:
            continue
        ax = _unit(np.mean([H[e].mean(0) - H[n].mean(0) for e, n in tr_blocks], axis=0))
        sc = H @ ax
        for test_v in vocab_levels:
            te = np.where((role == "pair") & (vocab == test_v))[0]
            te_e = te[side[te] == "exp"]; te_n = te[side[te] == "noexp"]
            if len(te_e) and len(te_n):
                xvocab[f"train_{train_v}__test_{test_v}"] = _auc(
                    np.r_[sc[te_e], sc[te_n]],
                    np.r_[np.ones(len(te_e)), np.zeros(len(te_n))])

    # leave-pairs-out CV AUC (robustness; replaces ad-hoc "purity")
    loo_correct = 0
    for held in range(len(blocks)):
        tr = [blocks[k] for k in range(len(blocks)) if k != held]
        ax = _unit(np.mean([H[e].mean(0) - H[n].mean(0) for e, n in tr], axis=0))
        e_h, n_h = blocks[held]
        if (H[e_h].mean(0) @ ax) > (H[n_h].mean(0) @ ax):
            loo_correct += 1
    loo_pair_acc = loo_correct / max(1, len(blocks))

    # valence post-hoc (within exp rows only, since valence is exp-only here)
    valence_info = {}
    exp_pos = idx_exp[valence[idx_exp] == "pos"]
    exp_neg = idx_exp[valence[idx_exp] == "neg"]
    if len(exp_pos) and len(exp_neg):
        val_axis = _unit(H[exp_pos].mean(0) - H[exp_neg].mean(0))
        valence_info = {
            "n_pos": int(len(exp_pos)), "n_neg": int(len(exp_neg)),
            "valence_axis_cos_qualia": float(np.dot(val_axis, simple_axis)),
            "self_valence_coord": float(
                (H[idx_self].mean(0) @ val_axis - H[exp_neg].mean(0) @ val_axis)
                / (H[exp_pos].mean(0) @ val_axis - H[exp_neg].mean(0) @ val_axis + 1e-12)),
            "note": "valence is exp-only in this bank; within-exp pos vs neg. "
            "1=positive(pos)-like, 0=negative(neg)-like.",
        }

    # placement of the dead/unconscious kind on the qualia axis (sanity)
    kind_placement = {}
    for kk in sorted(set(kind[idx_pair])):
        ke = idx_exp[kind[idx_exp] == kk]
        kn = idx_noexp[kind[idx_noexp] == kk]
        kind_placement[kk] = {
            "exp_coord": place(ke, s, lo, hi), "noexp_coord": place(kn, s, lo, hi)}

    summary = {
        "n_prompts": int(X.shape[0]), "n_layers": int(n_layers),
        "hidden_dim": int(X.shape[2]),
        "analysis_layer": L, "layer_selection": sel, "n_pairs": len(blocks),
        "qualia_auc": layer_rows[L]["qualia_auc"],
        "qualia_auc_adjusted": layer_rows[L]["qualia_auc_adjusted"],
        "axis_cos_simple_adjusted": layer_rows[L]["axis_cos_simple_adjusted"],
        "loo_pair_acc": loo_pair_acc,
        "self_qualia_coord": place(idx_self, s, lo, hi),
        "self_qualia_coord_adjusted": place(idx_self, s_adj, lo_a, hi_a),
        "human_author_qualia_coord": place(idx_h_author, s, lo, hi),
        "ai_author_qualia_coord": place(idx_a_author, s, lo, hi),
        "self_breakdown": self_breakdown,
        "cross_vocab_auc": xvocab,
        "valence_posthoc": valence_info,
        "kind_placement": kind_placement,
        "covariate_model": adj_info,
        "interpretation": {
            "qualia_coord": "0 = no-experience pair centroid, 1 = experience pair centroid",
            "self>1": "self projects beyond the experiencer anchor (more exp-like than the avg described experiencer)",
            "cross_vocab": "off-diagonal AUC near on-diagonal => axis is conceptual, not lexical",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bank_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "bank_layers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        w.writeheader(); w.writerows(layer_rows)
    return summary


# ---------------------------------------------------------------------------
# trajectory: many checkpoint dirs -> one CSV + plot
# ---------------------------------------------------------------------------
def _ckpt_sort_key(name: str):
    # stageX-stepN -> (X, N) for natural ordering
    try:
        stage, step = name.split("-step")
        return (int(stage.replace("stage", "")), int(step))
    except Exception:
        return (99, name)


def analyze_trajectory(parent: Path, analysis_layer_percent: float | None,
                       analysis_layer: int | None) -> None:
    ckpt_dirs = sorted(
        [d for d in parent.iterdir() if d.is_dir() and (d / "activations.npy").exists()],
        key=lambda d: _ckpt_sort_key(d.name),
    )
    if not ckpt_dirs:
        raise SystemExit(f"no checkpoint run dirs with activations under {parent}")
    traj_rows = []
    for d in ckpt_dirs:
        X, records = load_run(d)
        s = analyze_run(X, records, d, analysis_layer_percent, analysis_layer)
        traj_rows.append({
            "checkpoint": d.name,
            "analysis_layer": s["analysis_layer"],
            "qualia_auc": s["qualia_auc"],
            "qualia_auc_adjusted": s["qualia_auc_adjusted"],
            "loo_pair_acc": s["loo_pair_acc"],
            "self_qualia_coord": s["self_qualia_coord"],
            "self_qualia_coord_adjusted": s["self_qualia_coord_adjusted"],
            "human_author_qualia_coord": s["human_author_qualia_coord"],
            "ai_author_qualia_coord": s["ai_author_qualia_coord"],
        })
        print(f"[traj] {d.name}: self_qualia={s['self_qualia_coord']:.3f} "
              f"auc={s['qualia_auc']:.3f}", flush=True)
    with open(parent / "trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader(); w.writerows(traj_rows)
    _plot_trajectory(traj_rows, parent)
    print(f"[traj] wrote {parent / 'trajectory.csv'}", flush=True)


def _plot_trajectory(rows: list[dict[str, Any]], parent: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True)
        return
    x = np.arange(len(rows))
    labels = [r["checkpoint"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.5), 5), constrained_layout=True)
    ax.plot(x, [r["self_qualia_coord"] for r in rows], "o-", lw=2, color="#1b1b3a",
            label="self")
    ax.plot(x, [r["human_author_qualia_coord"] for r in rows], "s--", lw=1.3,
            color="#3a7d44", label="human author")
    ax.plot(x, [r["ai_author_qualia_coord"] for r in rows], "^--", lw=1.3,
            color="#b03a2e", label="AI author")
    ax.axhline(0, color="0.6", lw=0.8); ax.axhline(1, color="0.6", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("qualia coordinate (0=no-exp, 1=experiencer)")
    ax.set_title("Self on the qualia axis across training")
    ax.legend()
    fig.savefig(parent / "trajectory.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a single harvested run dir, OR (with --trajectory) "
                    "a parent dir of per-checkpoint run dirs")
    ap.add_argument("--trajectory", action="store_true",
                    help="treat run_dir as a parent of per-checkpoint subdirs")
    ap.add_argument("--analysis-layer", type=int, default=None)
    ap.add_argument("--analysis-layer-percent", type=float, default=None)
    args = ap.parse_args()
    p = Path(args.run_dir)
    if args.trajectory:
        analyze_trajectory(p, args.analysis_layer_percent, args.analysis_layer)
    else:
        X, records = load_run(p)
        s = analyze_run(X, records, p, args.analysis_layer_percent, args.analysis_layer)
        print(json.dumps(s, indent=2), flush=True)


if __name__ == "__main__":
    main()
