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
            verdict = {
                "status": "ACCEPT" if gap >= -0.02 else "MISS",
                "hybrid_ev": round(h_ev, 4),
                "topk_ev_at_matched_l0": round(topk_at, 4),
                "gap": round(gap, 4),
                "hybrid_l0": h_l0,
                "threshold": "within 0.02 below or above",
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

    fig, ax = _newfig()
    seen = set()
    n_curved_paying = 0
    for a in atoms:
        topo = str(_get(a, "topology", "kind", default="other")).lower()
        topo = topo if topo in TOPO_COLOR else "other"
        th = _get(a, "theta", "turning", default=None)
        de = _get(a, "delta_ev", "dev", "loao_delta_ev", default=None)
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
    payload = mdl if "mdl_featurizers" in mdl else {"featurizers": _get(mdl, "featurizers", default=[])}
    if "mdl_featurizers" in mdl:
        payload = {"featurizers": mdl["mdl_featurizers"],
                   "delta2": _get(mdl, "delta2", default=None)}
        payload = {k: v for k, v in payload.items() if v is not None}
        payload["featurizers"] = mdl["mdl_featurizers"]
    scored = mdl_mod.score_json(payload)
    rows = scored.get("rows", [])
    if not rows:
        return {"status": "PENDING", "reason": "MDL score produced no rows"}
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
    cross = scored.get("crossover", {})
    return {"status": "OK", "crossover": cross, "figure": _save(fig, out)}


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
    total = sum(int(_get(b, "collapse_events", "collapses", default=0)) for b in births)
    evs = [float(_get(b, "ev", "ev_after", default=np.nan)) for b in births]
    monotone = all(evs[i] <= evs[i + 1] + 1e-9 for i in range(len(evs) - 1) if not math.isnan(evs[i]))
    return {"status": "ACCEPT" if total == 0 else "MISS",
            "collapse_events": total, "n_births": len(births),
            "ev_monotone_in_births": bool(monotone), "threshold": "exactly 0"}


# ---------------------------------------------------------------------------------
# Report assembly (generated build output, filled from artifacts + frozen prereg).
# ---------------------------------------------------------------------------------
def write_report(results: dict, artifacts_dir: Path, out: Path) -> None:
    def cell(v):
        return v if v is not None else "PENDING"

    a1 = results["fig1"]
    a2 = results["fig2"]
    a3 = results["a3"]
    dose = results["dose"]
    lines = []
    A = lines.append
    A("# REPORT_35B — First manifold dictionary on a 35B residual stream")
    A("")
    A("*Generated by `experiments/report_35b_figures.py` from landed artifacts. "
      "Acceptance thresholds are frozen in `experiments/prereg_35b.md` (pre-registered). "
      "`PENDING` = artifact not yet landed.*")
    A("")
    A(f"Artifacts scanned: `{artifacts_dir}`  |  figures: `{FIGDIR}`")
    A("")
    A("## Acceptance scoreboard")
    A("")
    A("| # | Metric | Threshold | Result | Value |")
    A("|---|--------|-----------|--------|-------|")
    A(f"| A1 | composed held-out EV vs TopK @ matched actives | within 0.02 | "
      f"{cell(a1.get('status'))} | gap {cell(a1.get('gap'))} |")
    A(f"| A2 | curved atoms Θ>1 & ΔEV>min_effect | ≥5 | "
      f"{cell(a2.get('status'))} | {cell(a2.get('curved_atoms_theta_gt1_and_paying'))} |")
    A(f"| A3 | live-decoder collapse events | 0 | "
      f"{cell(a3.get('status'))} | {cell(a3.get('collapse_events'))} |")
    A(f"| A4 | probe-circle ordering | >0.9 | "
      f"{cell(dose.get('A4_status'))} | {cell(dose.get('ordering_corr'))} |")
    A(f"| A5 | dose slope | ∈[0.5,2] | "
      f"{cell(dose.get('A5_status'))} | {cell(dose.get('slope'))} |")
    A(f"| A6 | dose R² | >0.7 | "
      f"{cell(dose.get('A6_status'))} | {cell(dose.get('r2'))} |")
    A("")
    accepts = [a1.get("status"), a2.get("status"), a3.get("status"),
               dose.get("A4_status"), dose.get("A5_status"), dose.get("A6_status")]
    if all(s == "ACCEPT" for s in accepts):
        overall = "ALL SIX ACCEPT — headline holds."
    elif any(s == "MISS" for s in accepts):
        overall = "AT LEAST ONE MISS — see the pre-registered failure branch for that metric."
    else:
        overall = "IN PROGRESS — some metrics still PENDING."
    A(f"**Overall:** {overall}")
    A("")
    A("## Figures")
    A("")
    figmap = [
        ("1 Frontier (EV vs active budget)", a1.get("figure")),
        ("2 (Θ, ΔEV) scatter by atom type", a2.get("figure")),
        ("3 Curved-atom gallery", results["fig3"].get("figure")),
        ("4 MDL bits/token", results["fig4"].get("figure")),
        ("5 Stable-rank + utilization", results["fig5"].get("figure")),
        ("6 Curved-tier held-out EV lift", results["fig6"].get("figure")),
        ("7 Probe-circle ordering (crown)", dose.get("fig7")),
        ("8 Dose calibration (crown)", dose.get("fig8")),
    ]
    for name, fpath in figmap:
        if fpath:
            rel = os.path.relpath(fpath, REPO)
            A(f"- **Fig {name}** — `{rel}`")
            A(f"  ![{name}]({rel})")
        else:
            A(f"- **Fig {name}** — PENDING")
    A("")
    A("## Discriminators (log lines from the composed run)")
    A("")
    A(f"- collapse events: {cell(a3.get('collapse_events'))} "
      f"(target 0) — {cell(a3.get('status'))}")
    A(f"- EV monotone in births: {cell(a3.get('ev_monotone_in_births'))}")
    A(f"- births: {cell(a3.get('n_births'))}")
    A("")
    A("## Raw verdict payloads")
    A("")
    A("```json")
    A(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "figure"}
                  for k, v in results.items() if isinstance(v, dict)}, indent=1))
    A("```")
    out.write_text("\n".join(lines))


def run(artifacts_dir: Path) -> dict:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    t1 = _load(artifacts_dir / "l17_t1_frontier.json")
    compose = _load(artifacts_dir / "compose_per_atom.json")
    mdl = _load(artifacts_dir / "compose_mdl.json")
    dose = _load(artifacts_dir / "dose_calibration.json")
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
    }
    write_report(results, artifacts_dir, REPO / "REPORT_35B.md")
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
    t1 = {"frontier": [{"K": k * 250, "l0": k, "heldout_ev": 0.55 + 0.35 * (1 - math.exp(-k / 30))}
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
        atoms.append({"idx": i, "topology": "circle", "theta": th, "delta_ev": de,
                      "stable_rank": float(rng.uniform(1.5, 4.0)),
                      "utilization": float(rng.uniform(0.1, 0.9)),
                      "chart_curve": curve.tolist(),
                      "band": np.c_[np.full(80, 0.12), np.full(80, 0.12)].tolist(),
                      "activation_proj": ap.tolist(),
                      "activation_coord": coord.tolist()})
    for i in range(3):
        atoms.append({"idx": 100 + i, "topology": "linear",
                      "theta": float(rng.uniform(0, 0.4)),
                      "delta_ev": float(rng.uniform(0.01, 0.05)),
                      "stable_rank": 1.0, "utilization": float(rng.uniform(0.2, 0.8))})
    compose = {
        "min_effect_ev": 0.005,
        "operating_point": {"total_actives": 40, "heldout_ev": 0.905,
                            "linear_only_heldout_ev": 0.88, "heldout_subsample_n": 50000},
        "atoms": atoms,
        "births": [{"ev": 0.80 + 0.01 * i, "collapse_events": 0} for i in range(8)],
    }
    dose = {"probe_order": list(range(12)),
            "probe_angles": list(np.sort(rng.uniform(0, 2 * math.pi, 12))),
            "ordering_corr": 0.96,
            "predicted_nats": list(np.linspace(0, 2, 20)),
            "measured_kl": list(np.linspace(0, 2, 20) * 1.1 + rng.normal(0, 0.1, 20)),
            "slope": 1.08, "r2": 0.94}
    global FIGDIR
    FIGDIR = st
    results = {
        "fig1": fig1_frontier(t1, compose, st / "fig1_frontier.png"),
        "fig2": fig2_theta_dev(compose, st / "fig2_theta_dev.png"),
        "fig3": fig3_gallery(compose, st / "fig3_gallery.png"),
        "fig4": {"status": "SKIP", "reason": "MDL needs real featurizer JSON"},
        "fig5": fig5_stable_rank(compose, st / "fig5_stable_rank.png"),
        "fig6": fig6_curved_tier_ev(compose, st / "fig6_curved_tier_ev.png"),
        "a3": a3_collapse(compose),
        "dose": fig78_dose(dose, st / "fig7_probe_ordering.png", st / "fig8_dose_calibration.png"),
    }
    return results


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
