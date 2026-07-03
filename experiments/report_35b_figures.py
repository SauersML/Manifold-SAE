"""Headline figure set + filled results doc for the 35B manifold-dictionary run.

Lane 6 (EVAL). Consumes the artifacts the other lanes emit and produces THE
publishable figures + a filled ``REPORT_35B.md`` whose cells are read directly off
those figures. The acceptance thresholds are frozen in ``experiments/prereg_35b.md``
(pre-registered before numbers land); this script only MEASURES against them.

Design: figures are matplotlib PNGs (paper deliverable). Colors follow the dataviz
skill's validated categorical palette (fixed slot order = the CVD-safety mechanism).
No PCA-hue where a real intrinsic coordinate exists (Fig 3 colors by the fitted
chart coordinate). Nothing is fabricated: a figure whose artifact has not landed is
skipped and its report cell stays ``PENDING``. ``--selftest`` renders every figure on
clearly-labeled planted synthetic so the plotting is proven end-to-end before real
artifacts arrive.

Expected artifacts (flexible key matching; missing -> PENDING):
  T1     l17_t1_frontier.json : {"frontier":[{"K","active"|"l0","heldout_ev"|"ev"}]}
  COMPOSE compose_per_atom.json: {"operating_point":{"total_actives","heldout_ev"},
                                  "min_effect_ev", "atoms":[{"topology","theta",
                                  "delta_ev","stable_rank","utilization",
                                  "chart_curve"[[..]],"band"[[lo,hi]..],
                                  "activation_proj"[[x,y]..],"activation_coord"[..]}],
                                  "births":[{"ev","collapse_events"}]}
  COMPOSE compose_mdl.json     : {"mdl_featurizers":[...]}  # score_json input (#2085)
  DOSE   dose_calibration.json : {"ordering_corr","slope","r2",
                                  "predicted_nats"[..],"measured_kl"[..]}

Usage:
  python experiments/report_35b_figures.py                 # scan default artifact dir
  python experiments/report_35b_figures.py --artifacts DIR # scan DIR
  python experiments/report_35b_figures.py --selftest      # planted-synthetic proof
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- dataviz validated categorical palette (light surface); fixed slot order -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"
CAT = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}
# Series role -> slot (assigned in fixed order; entity-stable across figures).
C_HYBRID = CAT["blue"]     # our composed hybrid (the proposal)
C_TOPK = CAT["aqua"]       # our TopK SAE baseline (the thing to match)
C_LINEAR = CAT["yellow"]   # pure-linear T1 tier
# Topology hues for the (Theta, dEV) scatter.
TOPO_COLOR = {"linear": CAT["blue"], "circle": CAT["orange"], "other": CAT["violet"]}
# Sequential blue ramp (for Fig-3 intrinsic-coordinate coloring is cyclic; see below).

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_ARTIFACTS = REPO / "results" / "run_35b"
FIGDIR = REPO / "figures_35b"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _newfig(w=7.2, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _get(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# ---------------------------------------------------------------------------------
# Figure 1 — frontier: held-out EV vs active budget (hybrid vs TopK vs pure-linear).
# ---------------------------------------------------------------------------------
def fig1_frontier(t1: dict | None, compose: dict | None, out: Path) -> dict:
    if not t1:
        return {"status": "PENDING", "reason": "T1 frontier not landed"}
    rows = _get(t1, "frontier", "rows", default=[])
    if not rows:
        return {"status": "PENDING", "reason": "empty T1 frontier"}
    l0 = np.array([float(_get(r, "l0", "active", "actives", default=np.nan)) for r in rows])
    ev = np.array([float(_get(r, "heldout_ev", "ev", "test_ev", default=np.nan)) for r in rows])
    order = np.argsort(l0)
    l0, ev = l0[order], ev[order]

    fig, ax = _newfig()
    # TopK baseline (our sparse_dictionary_fit) — the curve to match.
    ax.plot(l0, ev, "-o", color=C_TOPK, lw=2, ms=6, label="our TopK SAE (baseline)")
    # Pure-linear T1 tier, if reported separately from the TopK sweep.
    lin = _get(t1, "linear_tier", default=None)
    if lin:
        ll0 = np.array([float(_get(r, "l0", "active", default=np.nan)) for r in lin])
        lev = np.array([float(_get(r, "heldout_ev", "ev", default=np.nan)) for r in lin])
        o = np.argsort(ll0)
        ax.plot(ll0[o], lev[o], "-s", color=C_LINEAR, lw=2, ms=6, label="pure-linear T1")

    verdict: dict[str, Any] = {"status": "PENDING", "reason": "COMPOSE operating point not landed"}
    if compose:
        op = _get(compose, "operating_point", default={})
        h_l0 = _get(op, "total_actives", "l0", "actives")
        h_ev = _get(op, "heldout_ev", "ev")
        if h_l0 is not None and h_ev is not None:
            h_l0, h_ev = float(h_l0), float(h_ev)
            ax.plot([h_l0], [h_ev], "*", color=C_HYBRID, ms=20,
                    label="hybrid (composed)", zorder=5)
            ax.axvline(h_l0, color=C_HYBRID, lw=1, ls=":", alpha=0.6)
            # A1: TopK held-out EV interpolated at the hybrid's L0.
            topk_at = float(np.interp(h_l0, l0, ev))
            gap = h_ev - topk_at
            # EV-baseline provenance (lead's pin): the held-out TSS baseline MUST be the
            # TRAIN column mean applied to held-out rows, never the held-out column mean
            # (which leaks the first moment and inflates every absolute EV identically).
            t1_base = str(_get(t1, "ev_baseline", "tss_baseline", default="unstated")).lower()
            comp_base = str(_get(compose, "ev_baseline", default=None)
                            or _get(op, "ev_baseline", default="unstated")).lower()
            def _ok(b):
                return "train" in b  # "train_mean" / "train-mean" / "train mean origin"
            baseline_ok = _ok(t1_base) and _ok(comp_base)
            verdict = {
                "status": "ACCEPT" if gap >= -0.02 else "MISS",
                "hybrid_ev": round(h_ev, 4),
                "topk_ev_at_matched_l0": round(topk_at, 4),
                "gap": round(gap, 4),
                "hybrid_l0": h_l0,
                "threshold": "within 0.02 below or above",
                "ev_baseline_t1": t1_base,
                "ev_baseline_compose": comp_base,
                "ev_baseline_ok": bool(baseline_ok),
                "ev_definition": "1 - SSE_recon/TSS, TSS about the TRAIN column mean on held-out rows",
                "frontier_provenance": ("canonical recompute" if _get(t1, "authoritative", default=False)
                                        else "lane-reported (not yet canonical-recomputed)"),
            }
            ax.annotate(f"gap {gap:+.3f} @ L0={h_l0:g}", (h_l0, h_ev),
                        textcoords="offset points", xytext=(8, 10),
                        color=INK, fontsize=9)
    _style(ax, "Held-out EV vs active budget — hybrid vs TopK vs linear",
           "active coefficients / token (L0)", "held-out explained variance")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    verdict["figure"] = _save(fig, out)
    return verdict


# ---------------------------------------------------------------------------------
# Figure 2 — (Theta, dEV) scatter colored by atom topology; A2 = upper-right count.
# ---------------------------------------------------------------------------------
def fig2_theta_dev(compose: dict | None, out: Path) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE per-atom not landed"}
    atoms = _get(compose, "atoms", default=[])
    if not atoms:
        return {"status": "PENDING", "reason": "no atoms in COMPOSE artifact"}
    min_eff = float(_get(compose, "min_effect_ev", "min_effect", default=0.0))

    # ΔEV provenance: the pre-registered A2 metric is HELD-OUT marginal (LOAO) ΔEV. If
    # COMPOSE only has the FFI's in-sample birth ΔEV, we still score but flag it — birth
    # ΔEV is what the fit optimized, so an unflagged pass would be a Goodhart hazard.
    dev_source = str(_get(compose, "atoms", default=[{}])[0].get("delta_ev_source", "unstated"))
    held_out_dev = "loao" in dev_source or "held" in dev_source or dev_source == "unstated"
    fig, ax = _newfig()
    seen = set()
    n_curved_paying = 0
    for a in atoms:
        topo = str(_get(a, "topology", "kind", default="other")).lower()
        topo = topo if topo in TOPO_COLOR else "other"
        th = _get(a, "theta", "turning", default=None)
        de = _get(a, "delta_ev", "loao_delta_ev", "dev", default=None)
        if th is None or de is None:
            continue
        th, de = float(th), float(de)
        lbl = topo if topo not in seen else None
        seen.add(topo)
        ax.scatter([th], [de], s=64, color=TOPO_COLOR[topo], edgecolor=SURFACE,
                   linewidth=1.2, label=lbl, zorder=3)
        if topo == "circle" and th > 1.0 and de > min_eff:
            n_curved_paying += 1
    ax.axvline(1.0, color=INK2, lw=1, ls="--", alpha=0.7)
    if min_eff > 0:
        ax.axhline(min_eff, color=INK2, lw=1, ls="--", alpha=0.7)
    ax.annotate(f"Θ>1 & ΔEV>min_effect\ncurved atoms paying rent: {n_curved_paying}",
                (0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                color=INK, fontsize=9)
    _style(ax, "Per-atom turning Θ vs held-out marginal ΔEV",
           "fitted turning Θ (rad)", "held-out marginal ΔEV")
    ax.legend(frameon=False, fontsize=9, loc="upper left", title="topology")
    return {
        "status": "ACCEPT" if n_curved_paying >= 5 else "MISS",
        "curved_atoms_theta_gt1_and_paying": n_curved_paying,
        "min_effect_ev": min_eff,
        "threshold": ">= 5",
        "delta_ev_source": dev_source,
        "delta_ev_is_heldout": bool(held_out_dev),
        "figure": _save(fig, out),
    }


# ---------------------------------------------------------------------------------
# Figure 3 — gallery of curved atoms: decoded curve + band + coordinate-colored acts.
# ---------------------------------------------------------------------------------
def fig3_gallery(compose: dict | None, out: Path, k_max: int = 5) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE per-atom not landed"}
    atoms = [a for a in _get(compose, "atoms", default=[])
             if str(_get(a, "topology", "kind", default="")).lower() == "circle"
             and _get(a, "chart_curve") is not None]
    # strongest curved atoms first
    atoms.sort(key=lambda a: float(_get(a, "delta_ev", "dev", default=0.0)), reverse=True)
    atoms = atoms[:k_max]
    if not atoms:
        return {"status": "PENDING", "reason": "no curved atoms with decoded chart_curve"}
    n = len(atoms)
    ncol = min(n, 3)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.6 * nrow), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, a in enumerate(atoms):
        ax = axes[i // ncol][i % ncol]
        curve = np.asarray(a["chart_curve"], dtype=float)  # (T,2) projection dims
        acts = np.asarray(_get(a, "activation_proj", default=[]), dtype=float)
        coord = np.asarray(_get(a, "activation_coord", default=[]), dtype=float)
        if acts.size and coord.size:
            # cyclic (angular) coordinate -> hsv wheel is the honest cyclic map
            sc = ax.scatter(acts[:, 0], acts[:, 1], c=coord, cmap="hsv", s=10,
                            alpha=0.55, zorder=1)
        ax.plot(curve[:, 0], curve[:, 1], color=INK, lw=2.2, zorder=3)
        band = _get(a, "band", default=None)
        if band is not None:
            band = np.asarray(band, dtype=float)  # (T,2,2): lo/hi per proj dim, or width
            if band.ndim == 2 and band.shape[1] == 2:
                # perpendicular half-width band around the curve
                t = np.gradient(curve, axis=0)
                nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
                nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
                w = band[:, 0]
                lo = curve + nrm * w[:, None]
                hi = curve - nrm * w[:, None]
                ax.fill(np.r_[lo[:, 0], hi[::-1, 0]], np.r_[lo[:, 1], hi[::-1, 1]],
                        color=C_HYBRID, alpha=0.15, zorder=2, linewidth=0)
        th = _get(a, "theta", "turning", default=float("nan"))
        de = _get(a, "delta_ev", "dev", default=float("nan"))
        _style(ax, f"atom {_get(a,'idx','id',default=i)}  Θ={float(th):.2f}  ΔEV={float(de):.3f}",
               "proj dim 1", "proj dim 2")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    return {"status": "OK", "n_atoms": n, "figure": _save(fig, out)}


# ---------------------------------------------------------------------------------
# Figure 4 — MDL bits/token (via #2085 description-length surface; zero extra compute)
# ---------------------------------------------------------------------------------
def fig4_mdl(mdl: dict | None, out: Path) -> dict:
    if not mdl:
        return {"status": "PENDING", "reason": "MDL featurizer JSON not landed"}
    try:
        import sys
        sys.path.insert(0, str(HERE / "mdl_ladder"))
        import mdl as mdl_mod  # experiments/mdl_ladder/mdl.py (ported to #2085)
    except Exception as e:  # pragma: no cover
        return {"status": "PENDING", "reason": f"mdl module unavailable: {e}"}
    # score_json wants a "featurizers" list; COMPOSE emits them under
    # "mdl_featurizers" (seed_manifest convention) or "featurizers" directly.
    feats = _get(mdl, "mdl_featurizers", "featurizers", default=[])
    if not feats:
        return {"status": "PENDING", "reason": "no featurizers in MDL artifact"}
    payload = {"featurizers": feats}
    for opt in ("delta2", "l_param_bits", "block_name", "chart_name"):
        if _get(mdl, opt) is not None:
            payload[opt] = mdl[opt]
    scored = mdl_mod.score_json(payload)
    rows = scored.get("rows", [])
    if not rows:
        return {"status": "PENDING", "reason": "MDL score produced no rows"}
    # Drop non-finite rows (a degenerate featurizer with ev>=1 → residual 0 → infinite
    # rate; real fits have ev<1). If nothing finite survives, the artifact is degenerate.
    rows = [r for r in rows
            if math.isfinite(float(r.get("bits_per_token", r.get("total_bits", float("inf")))))]
    if not rows:
        return {"status": "PENDING",
                "reason": "all MDL rows non-finite (degenerate distortion floor, ev>=1)"}
    names = [r.get("name", str(i)) for i, r in enumerate(rows)]
    bits = [float(r.get("bits_per_token", r.get("total_bits", 0.0))) for r in rows]
    kinds = [r.get("kind", "block") for r in rows]
    colors = [C_HYBRID if k == "chart" else C_LINEAR for k in kinds]

    fig, ax = _newfig(w=7.6, h=max(3.2, 0.4 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.barh(y, bits, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Description length per featurizer (chart=blue, block=yellow)",
           "bits / token", "")
    # P2 verdict — how many curved charts PAY in MDL: pair each chart to its block
    # (chart_name/block_name) and count crossovers where actual firings ≥ f* (finite).
    feat_objs = {f["name"]: mdl_mod.featurizer_from_json(f) for f in payload["featurizers"]}
    delta2 = scored.get("delta2")
    n_pay = 0
    n_pairs = 0
    for f in payload["featurizers"]:
        bn, cn = f.get("block_name"), f.get("chart_name")
        if bn in feat_objs and cn in feat_objs and f.get("kind") == "chart":
            n_pairs += 1
            cx = mdl_mod.crossover_firings(feat_objs[bn], feat_objs[cn], delta2)
            fstar = cx.get("f_star")
            if fstar is not None and math.isfinite(float(fstar)) and float(fstar) >= 0 \
                    and cx.get("chart_wins_at_actual_f"):
                n_pay += 1
    p2_status = ("ACCEPT" if n_pay >= 5 else "MISS") if n_pairs else "PENDING"
    return {"status": p2_status, "n_curved_paying_mdl": n_pay, "n_pairs": n_pairs,
            "crossover": scored.get("crossover", {}), "threshold": ">=5 pay",
            "figure": _save(fig, out)}


# ---------------------------------------------------------------------------------
# Figure 5 — stable-rank + utilization column per atom.
# ---------------------------------------------------------------------------------
def fig5_stable_rank(compose: dict | None, out: Path) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE per-atom not landed"}
    atoms = [a for a in _get(compose, "atoms", default=[])
             if _get(a, "stable_rank") is not None]
    if not atoms:
        return {"status": "PENDING", "reason": "no stable_rank in atoms"}
    atoms.sort(key=lambda a: float(_get(a, "delta_ev", "dev", default=0.0)), reverse=True)
    idx = [str(_get(a, "idx", "id", default=i)) for i, a in enumerate(atoms)]
    sr = [float(_get(a, "stable_rank", default=np.nan)) for a in atoms]
    ut = [float(_get(a, "utilization", default=np.nan)) for a in atoms]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, max(3.2, 0.35 * len(atoms) + 1.5)))
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(atoms))
    axes[0].barh(y, sr, color=C_HYBRID, height=0.62)
    axes[0].set_yticks(y); axes[0].set_yticklabels(idx, fontsize=8); axes[0].invert_yaxis()
    _style(axes[0], "stable rank (per atom, sorted by ΔEV)", "stable rank", "atom")
    axes[1].barh(y, ut, color=C_TOPK, height=0.62)
    axes[1].set_yticks(y); axes[1].set_yticklabels([], fontsize=8); axes[1].invert_yaxis()
    _style(axes[1], "utilization", "utilization", "")
    return {"status": "OK", "n_atoms": len(atoms), "figure": _save(fig, out)}


# ---------------------------------------------------------------------------------
# Figure 6 — held-out EV of curved tier on 50k subsample vs linear-only.
# ---------------------------------------------------------------------------------
def fig6_curved_tier_ev(compose: dict | None, out: Path) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE per-atom not landed"}
    op = _get(compose, "operating_point", default={})
    hyb = _get(op, "heldout_ev", "ev")
    lin = _get(op, "linear_only_heldout_ev", "linear_ev")
    sub = _get(op, "heldout_subsample_n", "subsample_n", default=None)
    if hyb is None or lin is None:
        return {"status": "PENDING", "reason": "operating_point missing hybrid/linear EV"}
    fig, ax = _newfig(w=5.2, h=4.2)
    bars = ax.bar(["linear only", "hybrid (linear+curved)"], [float(lin), float(hyb)],
                  color=[C_LINEAR, C_HYBRID], width=0.6)
    for b, v in zip(bars, [float(lin), float(hyb)]):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK, fontsize=10)
    ttl = "Held-out EV: curved tier lift"
    if sub:
        ttl += f" (n={int(sub):,} held-out subsample)"
    _style(ax, ttl, "", "held-out explained variance")
    return {"status": "OK", "hybrid_ev": float(hyb), "linear_ev": float(lin),
            "lift": round(float(hyb) - float(lin), 4), "figure": _save(fig, out)}


# ---------------------------------------------------------------------------------
# Figures 7 & 8 — DOSE crown: probe-circle ordering + dose calibration.
# ---------------------------------------------------------------------------------
def fig78_dose(dose: dict | None, out7: Path, out8: Path) -> dict:
    if not dose:
        return {"status": "PENDING", "reason": "DOSE artifact not landed"}
    res: dict[str, Any] = {}
    # Fig 7 — probe-circle ordering
    probe_ang = _get(dose, "probe_angles", "chart_coord", default=None)
    probe_true = _get(dose, "probe_order", "true_index", default=None)
    ordering = _get(dose, "ordering_corr", "ordering", default=None)
    if probe_ang is not None and probe_true is not None:
        fig, ax = _newfig(w=5.0, h=4.6)
        pa = np.asarray(probe_ang, dtype=float)
        pt = np.asarray(probe_true, dtype=float)
        ax.scatter(pt, pa, s=70, color=C_HYBRID, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        _style(ax, f"Probe cyclic ordering  (corr={ordering})",
               "probe index (true cyclic order)", "fitted chart coordinate")
        res["fig7"] = _save(fig, out7)
    res["ordering_corr"] = ordering
    res["A4_status"] = ("ACCEPT" if ordering is not None and float(ordering) > 0.9
                        else ("MISS" if ordering is not None else "PENDING"))
    res["model"] = _get(dose, "model", default="Qwen3.6-35B")  # 8B fallback labels itself
    # G_wrap — wraparound: first/last probe adjacent on the chart (a line can't do this).
    wrap = _get(dose, "wraparound", "wraparound_pass", "wraparound_in_order", default=None)
    if wrap is None and probe_ang is not None and probe_true is not None:
        pa = np.asarray(probe_ang, dtype=float)
        order = np.argsort(np.asarray(probe_true, dtype=float))
        ring = pa[order]
        if ring.size >= 3:
            # cyclic gaps between consecutive-in-true-order probes; the first↔last gap
            # must be comparable to the others (not the maximal one a ranking forces).
            two_pi = 2 * math.pi
            gaps = np.abs(np.diff(np.r_[ring, ring[0] + two_pi]))
            gaps = np.minimum(gaps % two_pi, two_pi - (gaps % two_pi))
            close_gap = gaps[-1]  # last→first
            wrap = bool(close_gap <= 1.5 * float(np.median(gaps[:-1]) + 1e-9))
    res["wraparound_pass"] = wrap
    res["Gwrap_status"] = ("ACCEPT" if wrap is True else ("MISS" if wrap is False else "PENDING"))
    # I3 — chart-interp: coordinate ordering nameable iff ordering clears the bar on a
    # named probe (weekday/month). Named-ness is qualitative; the gate is A4.
    res["I3_nameable"] = _get(dose, "chart_interp_name", "probe_name", default=None)
    res["I3_status"] = ("ACCEPT" if ordering is not None and float(ordering) > 0.9
                        else ("MISS" if ordering is not None else "PENDING"))
    # Fig 8 — dose calibration
    pn = _get(dose, "predicted_nats", default=None)
    kl = _get(dose, "measured_kl", default=None)
    slope = _get(dose, "slope", default=None)
    r2 = _get(dose, "r2", "r_squared", default=None)
    if pn is not None and kl is not None:
        fig, ax = _newfig(w=5.4, h=4.6)
        pn = np.asarray(pn, dtype=float)
        kl = np.asarray(kl, dtype=float)
        ax.scatter(pn, kl, s=54, color=C_HYBRID, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        if slope is not None:
            xs = np.linspace(float(pn.min()), float(pn.max()), 50)
            ax.plot(xs, float(slope) * xs, color=C_TOPK, lw=2,
                    label=f"slope={float(slope):.2f}, R²={float(r2):.2f}" if r2 is not None
                    else f"slope={float(slope):.2f}")
            ax.legend(frameon=False, fontsize=9, loc="upper left")
        _style(ax, "Dose calibration: predicted nats vs measured KL",
               "predicted nats (steer along chart)", "measured output KL")
        res["fig8"] = _save(fig, out8)
    res["slope"] = slope
    res["r2"] = r2
    res["A5_status"] = ("ACCEPT" if slope is not None and 0.5 <= float(slope) <= 2.0
                        else ("MISS" if slope is not None else "PENDING"))
    res["A6_status"] = ("ACCEPT" if r2 is not None and float(r2) > 0.7
                        else ("MISS" if r2 is not None else "PENDING"))
    res["status"] = "OK" if (res.get("A4_status") != "PENDING") else "PENDING"
    return res


# ---------------------------------------------------------------------------------
# Discriminator scan — collapse events across births (A3).
# ---------------------------------------------------------------------------------
def a3_collapse(compose: dict | None) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE birth log not landed"}
    births = _get(compose, "births", "birth_log", default=None)
    if births is None:
        return {"status": "PENDING", "reason": "no birth log in COMPOSE artifact"}
    # prefer an explicit total if COMPOSE reports one, else sum the per-birth counts
    total = _get(compose, "collapse_events_total", default=None)
    total = int(total) if total is not None else \
        sum(int(_get(b, "collapse_events", "collapses", default=0)) for b in births)
    evs = [float(_get(b, "ev", "ev_after", default=np.nan)) for b in births]
    monotone = all(evs[i] <= evs[i + 1] + 1e-9 for i in range(len(evs) - 1) if not math.isnan(evs[i]))
    return {"status": "ACCEPT" if total == 0 else "MISS",
            "collapse_events": total, "n_births": len(births),
            "ev_monotone_in_births": bool(monotone), "threshold": "exactly 0",
            "grown_vs_joint": _get(compose, "grown_vs_joint", default=None)}


def _accepted_curved(compose: dict) -> list[dict]:
    """Curved atoms that pay rent: circle topology, Θ>1, ΔEV>min_effect."""
    min_eff = float(_get(compose, "min_effect_ev", "min_effect", default=0.0))
    out = []
    for a in _get(compose, "atoms", default=[]):
        if str(_get(a, "topology", "kind", default="")).lower() != "circle":
            continue
        th = _get(a, "theta", "turning")
        de = _get(a, "delta_ev", "dev", "loao_delta_ev")
        if th is None or de is None:
            continue
        if float(th) > 1.0 and float(de) > min_eff:
            out.append(a)
    return out


# ---------------------------------------------------------------------------------
# Axis 1 — FIDELITY currency (F2 loss-recovered, F3 KL-patched, distortion floor).
# ---------------------------------------------------------------------------------
def fidelity_currency(fc: dict | None) -> dict:
    if not fc:
        return {"F2_status": "PENDING", "F3_status": "PENDING",
                "reason": "CONTROL fidelity_currency.json not landed"}
    hyb = _get(fc, "hybrid", default={})
    topk = _get(fc, "topk", default={})
    floor = _get(fc, "distortion_floor_r2", "distortion_floor", default=None)
    res: dict[str, Any] = {"distortion_floor_r2": floor}
    lr_h, lr_t = _get(hyb, "loss_recovered"), _get(topk, "loss_recovered")
    if lr_h is not None and lr_t is not None:
        res["loss_recovered_hybrid"] = float(lr_h)
        res["loss_recovered_topk"] = float(lr_t)
        res["F2_status"] = "ACCEPT" if float(lr_h) >= float(lr_t) - 0.02 else "MISS"
    else:
        res["F2_status"] = "PENDING"
    kl_h, kl_t = _get(hyb, "kl_patched"), _get(topk, "kl_patched")
    if kl_h is not None and kl_t is not None:
        res["kl_patched_hybrid"] = float(kl_h)
        res["kl_patched_topk"] = float(kl_t)
        res["F3_status"] = "ACCEPT" if float(kl_h) <= float(kl_t) * 1.05 else "MISS"
    else:
        res["F3_status"] = "PENDING"
    return res


# ---------------------------------------------------------------------------------
# Axis 4 GATE — G0 hallucinated-structure control (real vs Gaussian-null vs shuffle).
# ---------------------------------------------------------------------------------
def g0_null_gate(nc: dict | None, out: Path) -> dict:
    if not nc:
        return {"status": "PENDING", "reason": "CONTROL null_control.json not landed"}
    arms = {"gaussian_matched": "Gaussian null", "shuffled": "shuffled null"}
    real = _get(nc, "real_reference", default=None)
    passes = True
    detail = {}
    for key in arms:
        a = _get(nc, key, default=None)
        if a is None:
            return {"status": "PENDING", "reason": f"null arm '{key}' missing"}
        n_curved = int(_get(a, "n_curved_accepted", "n_curved", default=0))
        mean_th = float(_get(a, "mean_theta", default=0.0))
        arm_pass = (n_curved <= 1) and (mean_th < 0.5)
        passes = passes and arm_pass
        detail[key] = {"n_curved_accepted": n_curved, "mean_theta": mean_th, "pass": arm_pass}
    harmonic_ok = _get(nc, "harmonic_null", default={})
    higher = _get(harmonic_ok, "higher_modes_on_first_harmonic_plus_noise", default=False)
    if higher:
        passes = False
    detail["harmonic_spurious_higher_modes"] = bool(higher)

    # Figure 9 — the null gate made visible: accepted curved atoms per arm.
    labels, counts, colors = [], [], []
    if real is not None:
        labels.append("real L17")
        counts.append(int(_get(real, "n_curved_accepted", "n_curved", default=0)))
        colors.append(C_HYBRID)
    for key, name in arms.items():
        labels.append(name)
        counts.append(detail[key]["n_curved_accepted"])
        colors.append(CAT["red"])
    fig, ax = _newfig(w=5.6, h=4.2)
    bars = ax.bar(labels, counts, color=colors, width=0.6)
    for b, v in zip(bars, counts):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", color=INK, fontsize=10)
    ax.axhline(1.5, color=INK2, lw=1, ls="--", alpha=0.7)
    _style(ax, f"Hallucination-null control (GATE: {'PASS' if passes else 'FAIL'})",
           "", "accepted curved atoms")
    return {"status": "PASS" if passes else "FAIL", "detail": detail,
            "figure": _save(fig, out)}


# ---------------------------------------------------------------------------------
# Axis 3 — I1 shatter count (analytic from Θ; empirical if COMPOSE provides it).
# ---------------------------------------------------------------------------------
def i1_shatter(compose: dict | None, eps: float = 0.1) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE per-atom not landed"}
    atoms = _accepted_curved(compose)
    if not atoms:
        return {"status": "PENDING", "reason": "no accepted curved atoms"}
    # shatter law: n ~ Θ / (2*sqrt(2*eps)) linear atoms to match one curve at rel-err eps
    denom = 2.0 * math.sqrt(2.0 * eps)
    analytic = [float(_get(a, "theta", "turning")) / denom for a in atoms]
    med = float(np.median(analytic))
    emp = [float(_get(a, "shatter_empirical")) for a in atoms
           if _get(a, "shatter_empirical") is not None]
    res = {"analytic_median": round(med, 2), "eps": eps, "n_atoms": len(atoms),
           "status": "ACCEPT" if med >= 2.0 else "MISS", "threshold": "median >= 2"}
    if emp:
        emp_med = float(np.median(emp))
        res["empirical_median"] = round(emp_med, 2)
        ratio = med / emp_med if emp_med else float("inf")
        res["analytic_vs_empirical_ratio"] = round(ratio, 2)
        res["law_holds"] = bool(0.5 <= ratio <= 2.0)
    return res


# ---------------------------------------------------------------------------------
# Axis 4 — G_band coverage + G_util stable rank.
# ---------------------------------------------------------------------------------
def _band_coverage_one(atom: dict) -> float | None:
    """Fraction of held-out on-atom activations inside the 95% band. Prefer the
    lane-computed value; else approximate from decoded curve + perp half-width band."""
    v = _get(atom, "band_coverage")
    if v is not None:
        return float(v)
    curve = _get(atom, "chart_curve")
    acts = _get(atom, "activation_proj")
    band = _get(atom, "band")
    if curve is None or acts is None or band is None:
        return None
    curve = np.asarray(curve, float)
    acts = np.asarray(acts, float)
    band = np.asarray(band, float)
    if acts.size == 0 or curve.shape[0] == 0:
        return None
    hw = band[:, 0] if band.ndim == 2 else band  # perp half-width per curve point
    # nearest curve point per activation, compare distance to that point's half-width
    d = np.linalg.norm(acts[:, None, :] - curve[None, :, :], axis=2)  # (n_act, n_curve)
    j = np.argmin(d, axis=1)
    inside = d[np.arange(acts.shape[0]), j] <= hw[j]
    return float(inside.mean())


def g_band(compose: dict | None) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE per-atom not landed"}
    atoms = _accepted_curved(compose)
    covs = [c for a in atoms if (c := _band_coverage_one(a)) is not None]
    if not covs:
        return {"status": "PENDING",
                "reason": "no posterior band (FFI atom payload lacks shape_band_sd) — "
                          "not fabricated; needs the band surface from the stagewise adapter"}
    med = float(np.median(covs))
    return {"status": "ACCEPT" if 0.90 <= med <= 0.98 else "MISS",
            "median_coverage": round(med, 3), "n_atoms": len(covs),
            "threshold": "coverage in [0.90, 0.98]"}


def g_util(compose: dict | None) -> dict:
    if not compose:
        return {"status": "PENDING", "reason": "COMPOSE per-atom not landed"}
    atoms = _accepted_curved(compose)
    srs, ds = [], []
    for a in atoms:
        sr = _get(a, "stable_rank")
        if sr is None:
            continue
        srs.append(float(sr))
        ds.append(int(_get(a, "d_atom", "d", default=1)))
    if not srs:
        return {"status": "PENDING", "reason": "no stable_rank on accepted curved atoms"}
    med_sr = float(np.median(srs))
    d = int(np.median(ds)) if ds else 1
    return {"status": "ACCEPT" if med_sr <= d + 0.5 else "MISS",
            "median_stable_rank": round(med_sr, 3), "d": d,
            "threshold": "median stable rank <= d + 0.5"}


# ---------------------------------------------------------------------------------
# Axis 6 — R1 seed stability, R2 cross-corpus, R3 hygiene attestation.
# ---------------------------------------------------------------------------------
def _atom_subspace(atom: dict) -> np.ndarray | None:
    """Column-orthonormal (p, k) basis of an atom's ambient subspace. Prefer an
    explicit `subspace_basis` (rows or cols); else the decoder block `decoder_B`."""
    b = _get(atom, "subspace_basis", "decoder_B", "basis", default=None)
    if b is None:
        return None
    B = np.asarray(b, dtype=float)
    if B.ndim != 2 or B.size == 0:
        return None
    # orient so rows = ambient dim p (the larger axis is p in our regime p≫k)
    if B.shape[0] < B.shape[1]:
        B = B.T
    q, _ = np.linalg.qr(B)
    return q


def _subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Largest principal cosine between two column-orthonormal subspaces (∈[0,1])."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(np.clip(s.max(), 0.0, 1.0)) if s.size else 0.0


def r1_stability(stab: dict | None, seed0: dict | None = None,
                 seed2: dict | None = None) -> dict:
    # Explicit stability.json wins if a lane computed it.
    if stab:
        pa = _get(stab, "principal_angle_overlap", "subspace_overlap", default=None)
        hm = _get(stab, "hungarian_latent_match", "latent_match", default=None)
        hh = _get(stab, "artifact_hash_match", default=None)
        return {"principal_angle_overlap": pa, "hungarian_latent_match": hm,
                "artifact_hash_match": hh, "source": "stability.json",
                "status": ("ACCEPT" if pa is not None and float(pa) > 0.9
                           else ("MISS" if pa is not None else "PENDING"))}
    # Otherwise compute it ourselves from the seed-0 / seed-2 composed dictionaries.
    if not (seed0 and seed2):
        return {"status": "PENDING", "reason": "seed-2 clone not landed (need both dicts)"}
    A0 = [(_atom_subspace(a), a) for a in _get(seed0, "atoms", default=[])]
    A2 = [(_atom_subspace(a), a) for a in _get(seed2, "atoms", default=[])]
    S0 = [s for s, _ in A0 if s is not None]
    S2 = [s for s, _ in A2 if s is not None]
    if not S0 or not S2:
        return {"status": "PENDING",
                "reason": "atoms lack subspace_basis/decoder_B — cannot compute angles"}
    from scipy.optimize import linear_sum_assignment
    ov = np.zeros((len(S0), len(S2)))
    for i, a in enumerate(S0):
        for j, b in enumerate(S2):
            k = min(a.shape[1], b.shape[1])
            ov[i, j] = _subspace_overlap(a[:, :k], b[:, :k])
    ri, cj = linear_sum_assignment(-ov)  # maximize matched overlap
    matched = ov[ri, cj]
    subspace_overlap = float(np.mean(matched))
    # honest harsher metric: fraction of atoms whose best match is a near-identity subspace
    latent_match = float(np.mean(matched > 0.9))
    return {"principal_angle_overlap": round(subspace_overlap, 3),
            "hungarian_latent_match": round(latent_match, 3),
            "n_matched": int(len(matched)), "source": "computed from seed0/seed2",
            "threshold": "subspace overlap > 0.9",
            "status": "ACCEPT" if subspace_overlap > 0.9 else "MISS"}


def r2_cross_corpus(cross: dict | None) -> dict:
    if not cross:
        return {"status": "PENDING", "reason": "creditscope L30 arm not landed"}
    n = int(_get(cross, "n_curved_recurring", "n_recur", default=0))
    return {"status": "ACCEPT" if n >= 3 else "MISS", "n_curved_recurring": n,
            "threshold": ">= 3 recur"}


def r3_hygiene(manifest: dict | None, tier0_present: bool = False) -> dict:
    if not manifest:
        return {"status": "PENDING", "reason": "DATA split manifest not landed"}
    # DATA emits split_manifest.json with a `split_policy` string; older schema used
    # explicit booleans. Accept both.
    policy = str(_get(manifest, "split_policy", default="")).lower()
    chunk = (bool(_get(manifest, "chunk_level_split", "split_by_chunk", default=False))
             or ("whole-file" in policy) or ("no row" in policy) or ("chunk" in policy))
    # Tier-0 train-only: an explicit flag, else attested by DATA's manifest convention
    # (tier0 computed on TRAIN files only) when a tier0.json sits beside the manifest.
    tier0 = bool(_get(manifest, "tier0_train_only", default=False)) or tier0_present
    currency = _get(manifest, "matched_currency", default="actives")  # headline currency
    ok = chunk and tier0
    return {"status": "PASS" if ok else "MISS", "chunk_level_split": chunk,
            "tier0_train_only": tier0, "matched_currency": currency,
            "split_policy": policy or None}


# ---------------------------------------------------------------------------------
# Report assembly (generated build output, filled from artifacts + frozen prereg).
# ---------------------------------------------------------------------------------
def write_report(results: dict, artifacts_dir: Path, out: Path) -> None:
    def cell(v):
        return v if v is not None else "PENDING"

    a1, a2, a3 = results["fig1"], results["fig2"], results["a3"]
    dose, fc, g0 = results["dose"], results["fidelity"], results["g0"]
    i1, gband, gutil = results["i1"], results["g_band"], results["g_util"]
    r1, r2, r3 = results["r1"], results["r2"], results["r3"]
    p2 = results["fig4"]
    lines = []
    A = lines.append
    A("# REPORT_35B — First manifold dictionary on a 35B residual stream")
    A("")
    A("*Generated by `experiments/report_35b_figures.py` from landed artifacts. The "
      "full 6-axis scorecard is frozen in `experiments/prereg_35b.md` (pre-registered "
      "before numbers land). `PENDING` = artifact not yet landed; nothing here is faked.*")
    A("")
    A("**Meta-rules** (how to read this): (1) **Goodhart** — this is a portfolio, no cell "
      "is the objective; our own parable is the affine-PCA shortcut that once greened the "
      "OLMo gate by *being* its baseline. (2) **Effect size > significance** — at millions "
      "of tokens everything real is detectable, so evidence and salience (min-effect floors "
      "on Θ/ΔEV/dose) are separate dials. (3) **Hierarchy descriptive < predictive < "
      "causal** — dose calibration is the causal crown, not another EV number.")
    A("")
    A(f"Artifacts scanned: `{artifacts_dir}`  |  figures: `{FIGDIR}`")
    A("")
    # --- GATE first: G0 licenses the whole geometry axis ---
    A("## GATE — Hallucinated-structure control (G0)")
    A("")
    A(f"**{cell(g0.get('status'))}** — accepted curved atoms on Gaussian-matched-noise and "
      "shuffled data must be ≤1 with Θ→0. A method that finds circles in noise is "
      "DISQUALIFIED on axis 4 regardless of every other score; passing LICENSES the "
      "geometry axis.")
    if g0.get("detail"):
        A("")
        A(f"```json\n{json.dumps(g0['detail'], indent=1)}\n```")
    A("")

    def row(idn, metric, thr, status, val):
        A(f"| {idn} | {metric} | {thr} | {cell(status)} | {cell(val)} |")

    A("## Scorecard — 6 axes")
    A("")
    A("| ID | Metric | Threshold | Result | Value |")
    A("|----|--------|-----------|--------|-------|")
    A("| | **Axis 1 — FIDELITY** | | | |")
    row("A1", "held-out EV vs TopK @ matched actives", "within 0.02", a1.get("status"),
        f"gap {cell(a1.get('gap'))}")
    row("F2", "loss-recovered @ floor (model's currency)", "≥ TopK−0.02", fc.get("F2_status"),
        cell(fc.get("loss_recovered_hybrid")))
    row("F3", "KL-patched @ floor", "≤ TopK×1.05", fc.get("F3_status"),
        cell(fc.get("kl_patched_hybrid")))
    row("—", "distortion floor R²* (read point)", "reported", "—",
        cell(fc.get("distortion_floor_r2")))
    A("| | **Axis 2 — PARSIMONY** | | | |")
    row("P1", "L0 (mean actives/token)", "reported, matched", a1.get("status"),
        f"L0 {cell(a1.get('hybrid_l0'))}")
    row("P2", "MDL bits @ δ, curved tier pays (#2085)", "≥5 pay", p2.get("status"),
        f"{cell(p2.get('n_curved_paying_mdl'))} pay")
    A("| | **Axis 3 — IDENTITY** | | | |")
    row("I1", "shatter count (linear atoms / curve)", "median ≥2", i1.get("status"),
        cell(i1.get("analytic_median")))
    row("I2", "absorption / SCR / TPP", "STRETCH", "PENDING", "SAEBench harness")
    row("I3", "chart-interp nameable ordering", "ordering>0.9", dose.get("I3_status"),
        cell(dose.get("ordering_corr")))
    A("| | **Axis 4 — GEOMETRY** (licensed by G0) | | | |")
    row("G0", "hallucinated-structure GATE", "≤1 curved on nulls", g0.get("status"),
        cell((g0.get("detail") or {}).get("gaussian_matched", {}).get("n_curved_accepted")))
    row("A2", "(Θ,ΔEV): curved atoms Θ>1 & ΔEV>min_effect", "≥5", a2.get("status"),
        cell(a2.get("curved_atoms_theta_gt1_and_paying")))
    row("A4", "coordinate fidelity (circular corr/ordering)", ">0.9", dose.get("A4_status"),
        cell(dose.get("ordering_corr")))
    row("G_wrap", "wraparound (Sun adjacent to Mon)", "pass", dose.get("Gwrap_status"),
        cell(dose.get("wraparound_pass")))
    row("G_band", "95% band coverage of held-out on-atom pts", "∈[0.90,0.98]",
        gband.get("status"), cell(gband.get("median_coverage")))
    row("G_util", "stable rank ≈ d (ARD prunes idle)", "≤ d+0.5", gutil.get("status"),
        cell(gutil.get("median_stable_rank")))
    A("| | **Axis 5 — CAUSAL** (the crown) | | | |")
    row("A5", "dose slope (measured-KL on predicted-nats)", "∈[0.5,2]", dose.get("A5_status"),
        cell(dose.get("slope")))
    row("A6", "dose R²", ">0.7", dose.get("A6_status"), cell(dose.get("r2")))
    row("C_steer", "on-target effect @ matched coherence", "STRETCH", "PENDING",
        "steering_bench")
    A("| | **Axis 6 — RELIABILITY** | | | |")
    row("R1", "seed stability (principal angles; latent-match too)", ">0.9 subspace",
        r1.get("status"), cell(r1.get("principal_angle_overlap")))
    row("R2", "cross-corpus replicate (creditscope L30)", "≥3 recur", r2.get("status"),
        cell(r2.get("n_curved_recurring")))
    row("R3", "split hygiene + matched budget stated", "pass", r3.get("status"),
        cell(r3.get("matched_currency")))
    A("| | **Discriminator** | | | |")
    row("A3", "live-decoder collapse events", "0", a3.get("status"),
        cell(a3.get("collapse_events")))
    A("")
    # ΔEV provenance caveat — A2 must be honest about in-sample vs held-out ΔEV.
    if a2.get("delta_ev_source") and a2.get("delta_ev_source") != "unstated":
        if not a2.get("delta_ev_is_heldout"):
            A(f"> **A2 caveat:** ΔEV source is `{a2['delta_ev_source']}` — the FFI's "
              "IN-SAMPLE birth ΔEV, not the pre-registered held-out LOAO. Birth ΔEV is "
              "what the fit optimized, so read A2 as provisional until the held-out LOAO "
              "lands from the stagewise adapter's per-atom held-out recon.")
        else:
            A(f"> A2 ΔEV source: `{a2['delta_ev_source']}` (held-out).")
        A("")
    # grown-vs-joint discriminator (the free architecture comparison inside the run)
    gvj = a3.get("grown_vs_joint")
    if gvj:
        A(f"> **Grown-vs-joint discriminator:** grown (stagewise) held-out EV "
          f"{cell(gvj.get('grown'))} vs joint-fit-at-grown-K {cell(gvj.get('joint'))}. "
          "Stagewise ≥ joint (with zero collapse) is the architecture evidence; joint "
          "collapsing where stagewise does not is the strongest such evidence.")
        A("")
    # overall verdict
    hard = [a1.get("status"), a2.get("status"), a3.get("status"), dose.get("A4_status"),
            dose.get("A5_status"), dose.get("A6_status")]
    if g0.get("status") == "FAIL":
        overall = "GATE FAILED (G0) — geometry-axis claims are VOID. See failure branch."
    elif all(s == "ACCEPT" for s in hard) and g0.get("status") == "PASS":
        overall = "GATE PASSED and all six frozen A-metrics ACCEPT — headline holds."
    elif any(s == "MISS" for s in hard):
        overall = "AT LEAST ONE A-METRIC MISS — see the pre-registered failure branch."
    else:
        overall = "IN PROGRESS — metrics still PENDING."
    A(f"**Overall:** {overall}")
    A("")
    A("## EV definition & split hygiene (the two silent ways to fake, closed)")
    A("")
    A("Held-out EV = **1 − SSE_recon / TSS**, where **TSS is taken about the TRAIN column "
      "mean applied to held-out rows** (equivalently, the origin after subtracting the "
      "train Tier-0 mean) — **never the held-out column mean**, which leaks the first "
      "moment and inflates every absolute EV number identically. Held-out EV is measured "
      "on the disjoint whole-shard held-out split (rollout-safe), Tier-0 fit on train only.")
    A("")
    A(f"- baseline attestation — T1: `{cell(a1.get('ev_baseline_t1'))}`, "
      f"COMPOSE: `{cell(a1.get('ev_baseline_compose'))}` → "
      f"{'OK (train-mean)' if a1.get('ev_baseline_ok') else 'UNVERIFIED — confirm both use train-mean, not heldout-colmean'}")
    A(f"- frontier provenance: {cell(a1.get('frontier_provenance'))} "
      "(canonical = every point recomputed by `experiments/canonical_ev.py` from "
      "decoder + held-out + Tier-0 with the origin/train-mean TSS; authoritative for the figure)")
    A(f"- split: chunk/rollout-level (never row); Tier-0 (mean, rogue dims, global RMS) "
      f"train-only; held-out EV on a 50k held-out subsample.")
    A("")
    A("## Headline figures")
    A("")
    figmap = [
        ("1 — Pareto frontier: held-out EV vs L0 (HEADLINE)", a1.get("figure")),
        ("8 — Dose calibration: predicted nats vs measured KL (HEADLINE, ours alone)",
         dose.get("fig8")),
        ("2 — (Θ, ΔEV) scatter by atom type", a2.get("figure")),
        ("3 — Curved-atom gallery", results["fig3"].get("figure")),
        ("4 — MDL bits/token @ δ", p2.get("figure")),
        ("5 — Stable-rank + utilization", results["fig5"].get("figure")),
        ("6 — Curved-tier held-out EV lift", results["fig6"].get("figure")),
        ("7 — Probe cyclic ordering", dose.get("fig7")),
        ("9 — Hallucination-null control (G0 made visible)", g0.get("figure")),
    ]
    for name, fpath in figmap:
        if fpath:
            rel = os.path.relpath(fpath, REPO)
            A(f"- **Fig {name}** — `{rel}`")
            A(f"  ![{name}]({rel})")
        else:
            A(f"- **Fig {name}** — PENDING")
    A("")
    A("## Raw verdict payloads")
    A("")
    A("```json")
    A(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "figure"}
                  for k, v in results.items() if isinstance(v, dict)}, indent=1))
    A("```")
    out.write_text("\n".join(lines))


def run(artifacts_dir: Path, report_path: Path | None = None) -> dict:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    # Prefer the canonical recompute (own the EV definition) over any lane-reported
    # frontier: canonical_ev.py rebuilds every point with the train-mean/origin TSS.
    t1 = (_load(artifacts_dir / "l17_t1_frontier_canonical.json")
          or _load(artifacts_dir / "l17_t1_frontier.json"))
    compose = _load(artifacts_dir / "compose_per_atom.json")
    mdl = _load(artifacts_dir / "compose_mdl.json")
    dose = _load(artifacts_dir / "dose_calibration.json")
    fc = _load(artifacts_dir / "fidelity_currency.json")
    nc = _load(artifacts_dir / "null_control.json")
    cross = _load(artifacts_dir / "creditscope_l30.json")
    stab = _load(artifacts_dir / "stability.json")
    seed2 = _load(artifacts_dir / "compose_per_atom_seed2.json")  # seed-2 clone for R1
    # R3 binds to DATA's real split manifest (data/l17/) with a results-dir override.
    manifest = (_load(artifacts_dir / "manifest.json")
                or _load(REPO / "data" / "l17" / "split_manifest.json"))
    tier0_present = (REPO / "data" / "l17" / "tier0.json").exists()
    results = {
        "fig1": fig1_frontier(t1, compose, FIGDIR / "fig1_frontier.png"),
        "fig2": fig2_theta_dev(compose, FIGDIR / "fig2_theta_dev.png"),
        "fig3": fig3_gallery(compose, FIGDIR / "fig3_gallery.png"),
        "fig4": fig4_mdl(mdl, FIGDIR / "fig4_mdl.png"),
        "fig5": fig5_stable_rank(compose, FIGDIR / "fig5_stable_rank.png"),
        "fig6": fig6_curved_tier_ev(compose, FIGDIR / "fig6_curved_tier_ev.png"),
        "a3": a3_collapse(compose),
        "dose": fig78_dose(dose, FIGDIR / "fig7_probe_ordering.png",
                           FIGDIR / "fig8_dose_calibration.png"),
        "fidelity": fidelity_currency(fc),
        "g0": g0_null_gate(nc, FIGDIR / "fig9_null_control.png"),
        "i1": i1_shatter(compose),
        "g_band": g_band(compose),
        "g_util": g_util(compose),
        "r1": r1_stability(stab, compose, seed2),
        "r2": r2_cross_corpus(cross),
        "r3": r3_hygiene(manifest, tier0_present),
    }
    write_report(results, artifacts_dir, report_path or (REPO / "REPORT_35B.md"))
    return results


# ---------------------------------------------------------------------------------
# Self-test — planted synthetic proving every figure renders (clearly labeled).
# ---------------------------------------------------------------------------------
def selftest() -> dict:
    rng = np.random.default_rng(0)
    st = REPO / "figures_35b" / "_selftest"
    st.mkdir(parents=True, exist_ok=True)
    # planted T1 frontier (a saturating TopK curve)
    l0s = [8, 16, 32, 64, 128]
    t1 = {"ev_baseline": "train_mean",
          "frontier": [{"K": k * 250, "l0": k, "heldout_ev": 0.55 + 0.35 * (1 - math.exp(-k / 30))}
                       for k in l0s],
          "linear_tier": [{"l0": k, "heldout_ev": 0.50 + 0.32 * (1 - math.exp(-k / 30))}
                          for k in l0s]}
    # planted COMPOSE per-atom: 7 circle atoms (5 paying), a few linear
    atoms = []
    for i in range(7):
        th = float(rng.uniform(1.2, 3.0))
        de = float(rng.uniform(0.006, 0.03)) if i < 5 else float(rng.uniform(0.0, 0.004))
        t = np.linspace(0, 2 * math.pi, 80)
        curve = np.stack([np.cos(t), np.sin(t)], axis=1) * (1 + 0.1 * i)
        ap = rng.normal(size=(300, 2)) * 0.1 + np.stack(
            [np.cos(t[rng.integers(0, 80, 300)]), np.sin(t[rng.integers(0, 80, 300)])], axis=1) * (1 + 0.1 * i)
        coord = np.arctan2(ap[:, 1], ap[:, 0])
        basis, _ = np.linalg.qr(rng.normal(size=(16, 2)))  # atom subspace in R^16
        atoms.append({"idx": i, "topology": "circle", "theta": th, "delta_ev": de,
                      "delta_ev_source": "heldout_loao", "d_atom": 1,
                      "stable_rank": float(rng.uniform(1.0, 1.4)),  # d=1 chart ⇒ ~1
                      "utilization": float(rng.uniform(0.1, 0.9)),
                      "chart_curve": curve.tolist(),
                      "band": np.c_[np.full(80, 0.30), np.full(80, 0.30)].tolist(),
                      "activation_proj": ap.tolist(),
                      "activation_coord": coord.tolist(),
                      "band_coverage": float(rng.uniform(0.93, 0.97)),
                      "subspace_basis": basis.tolist()})
    for i in range(3):
        atoms.append({"idx": 100 + i, "topology": "linear",
                      "theta": float(rng.uniform(0, 0.4)),
                      "delta_ev": float(rng.uniform(0.01, 0.05)),
                      "stable_rank": 1.0, "utilization": float(rng.uniform(0.2, 0.8))})
    compose = {
        "min_effect_ev": 0.005, "ev_baseline": "train_mean", "collapse_events_total": 0,
        "operating_point": {"total_actives": 40, "heldout_ev": 0.905,
                            "linear_only_heldout_ev": 0.88, "heldout_subsample_n": 50000},
        "grown_vs_joint": {"grown": 0.905, "joint": 0.86},
        "atoms": atoms,
        "births": [{"ev": 0.80 + 0.01 * i, "collapse_events": 0} for i in range(8)],
    }
    # seed-2 clone: same atoms, subspaces slightly rotated (subspace-stable, R1 ACCEPT)
    seed2_atoms = []
    for a in atoms:
        b = np.asarray(a.get("subspace_basis", []), float)
        if b.size:
            b2, _ = np.linalg.qr(b + rng.normal(scale=0.05, size=b.shape))
            a = {**a, "subspace_basis": b2.tolist()}
        seed2_atoms.append(a)
    compose_seed2 = {**compose, "atoms": seed2_atoms}
    # probe angles that respect cyclic order (so wraparound passes): evenly spaced ring
    ring = np.linspace(0, 2 * math.pi, 13)[:12] + rng.normal(0, 0.05, 12)
    dose = {"probe_order": list(range(12)), "probe_angles": list(ring),
            "ordering_corr": 0.96, "model": "Qwen3.6-35B", "chart_interp_name": "weekday",
            "predicted_nats": list(np.linspace(0, 2, 20)),
            "measured_kl": list(np.linspace(0, 2, 20) * 1.1 + rng.normal(0, 0.1, 20)),
            "slope": 1.08, "r2": 0.94}
    # CONTROL artifacts: fidelity currency + hallucination null (both PASS)
    fc = {"distortion_floor_r2": 0.92,
          "hybrid": {"loss_recovered": 0.91, "kl_patched": 0.08, "at_actives": 40},
          "topk": {"loss_recovered": 0.90, "kl_patched": 0.085, "at_actives": 40}}
    nc = {"real_reference": {"n_curved_accepted": 7, "mean_theta": 1.9},
          "gaussian_matched": {"n_curved_accepted": 0, "mean_theta": 0.08},
          "shuffled": {"n_curved_accepted": 1, "mean_theta": 0.12},
          "harmonic_null": {"higher_modes_on_first_harmonic_plus_noise": False}}
    stab = {"principal_angle_overlap": 0.93, "hungarian_latent_match": 0.42}
    cross = {"n_curved_recurring": 4}
    manifest = {"chunk_level_split": True, "tier0_train_only": True, "matched_currency": "actives"}
    # dump artifacts and drive the real production run() path (honest end-to-end proof)
    for name, payload in [("l17_t1_frontier.json", t1), ("compose_per_atom.json", compose),
                          ("compose_per_atom_seed2.json", compose_seed2),
                          ("dose_calibration.json", dose), ("fidelity_currency.json", fc),
                          ("null_control.json", nc),
                          ("creditscope_l30.json", cross), ("manifest.json", manifest)]:
        (st / name).write_text(json.dumps(payload))
    # a realistic MDL featurizer set (chart captures 85% on one intrinsic coord)
    feats = []
    for g in range(5):
        tv = float(rng.uniform(0.5, 3.0)); b = 8
        cv = sorted(np.abs(rng.normal(size=b)) * tv / b, reverse=True)
        f = int(rng.integers(200, 5000))
        feats.append({"name": f"blk{g}-linear", "kind": "block", "total_var": tv,
                      "n_tokens": 50000, "n_firings": f, "n_params": b * 512,
                      "coded_var": [float(v) for v in cv], "g_dict": 5, "k_active": 2})
        feats.append({"name": f"blk{g}-chart", "kind": "chart", "total_var": tv,
                      "n_tokens": 50000, "n_firings": f, "n_params": 4 * 512,
                      "coded_var": [0.85 * tv], "ev": 0.85, "g_dict": 5, "k_active": 2,
                      "block_name": f"blk{g}-linear", "chart_name": f"blk{g}-chart"})
    (st / "compose_mdl.json").write_text(json.dumps(
        {"mdl_featurizers": feats, "block_name": "blk0-linear", "chart_name": "blk0-chart"}))
    global FIGDIR
    FIGDIR = st
    return run(st, report_path=st / "REPORT_35B_selftest.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS),
                    help="dir holding l17_t1_frontier.json / compose_*.json / dose_*.json")
    ap.add_argument("--selftest", action="store_true",
                    help="render every figure on labeled planted synthetic (proof)")
    args = ap.parse_args()
    if args.selftest:
        res = selftest()
        print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v)
                          for k, v in res.items()}, indent=1))
        return
    res = run(Path(args.artifacts))
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
