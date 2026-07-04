"""Frontier plots for frontier_bench.py output: EV vs FLOPs and bits/token vs FLOPs.

Palette is the dataviz-skill default (light + dark aware via a single light render here,
matching the rest of the suite): BLUE #2a78d6 (curved chart, ours), AQUA #1baf7a
(same-lane linear control), YELLOW #eda100 (block/TopK linear). Surface #fcfcfb.

    python -m experiments.frontier_plots --in results/.../synth_p1024_curved.json \
        --out-dir results/suite_2026-07-03/frontiers/ --tag p1024_curved
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
LANE_STYLE = {
    "curved": (BLUE, "o", "Curved chart (ours)"),
    "manifold_linear": (AQUA, "s", "Same-lane linear (control)"),
    "linear": (YELLOW, "^", "Block/TopK linear"),
    "block": (YELLOW, "D", "Block-sparse (real 35B)"),
}


def _series(results, lane, xkey_path, ykey_path):
    xs, ys = [], []
    for r in results:
        if not r.get("ok") or r.get("lane") != lane:
            continue
        x = r
        for k in xkey_path:
            x = x.get(k) if isinstance(x, dict) else None
        y = r
        for k in ykey_path:
            y = y.get(k) if isinstance(y, dict) else None
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def plot_ev_vs_flops(payload, ax, which="infer"):
    key = "infer_macs_per_token" if which == "infer" else "train_macs_total"
    for lane, (c, mk, lab) in LANE_STYLE.items():
        xs, ys = _series(payload["results"], lane, ["flops", key], ["heldout_ev"])
        if len(xs):
            ax.plot(xs, ys, marker=mk, color=c, label=lab, lw=2, ms=7)
    ax.set_xscale("log")
    ax.set_xlabel(f"{which} MACs " + ("/ token" if which == "infer" else "(training, total)"))
    ax.set_ylabel("held-out explained variance")
    ax.set_title(f"EV vs compute ({which})")
    ax.legend(frameon=False, fontsize=9)


def plot_bits_vs_flops(payload, ax, currency="support_entropy"):
    ykey = f"bits_per_token_{currency}"
    for lane, (c, mk, lab) in LANE_STYLE.items():
        xs, ys = _series(payload["results"], lane, ["flops", "infer_macs_per_token"], ["mdl", ykey])
        if len(xs):
            ax.plot(xs, ys, marker=mk, color=c, label=lab, lw=2, ms=7)
    ax.set_xscale("log")
    ax.set_xlabel("infer MACs / token")
    ax.set_ylabel(f"bits / token @ matched distortion\n(selection: {currency.replace('_',' ')})")
    d2 = payload.get("matched_distortion_delta2")
    ax.set_title(f"Description length vs compute (delta2={d2:.3g})" if d2 else "Description length vs compute")
    ax.legend(frameon=False, fontsize=9)


def plot_ev_vs_k(payload, ax):
    for lane, (c, mk, lab) in LANE_STYLE.items():
        xs, ys = _series(payload["results"], lane, ["K"], ["heldout_ev"])
        if len(xs):
            ax.plot(xs, ys, marker=mk, color=c, label=lab, lw=2, ms=7)
    ax.set_xlabel("K (dictionary width)")
    ax.set_ylabel("held-out explained variance")
    ax.set_title("EV vs K (atom count)")
    ax.legend(frameon=False, fontsize=9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    payload = json.loads(Path(args.inp).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    plot_ev_vs_flops(payload, axes[0, 0], which="infer")
    plot_ev_vs_flops(payload, axes[0, 1], which="train")
    plot_bits_vs_flops(payload, axes[1, 0], currency="support_entropy")
    plot_ev_vs_k(payload, axes[1, 1])
    cfg = payload.get("config", {})
    fig.suptitle(
        f"Compute-matched frontier  |  DGP={cfg.get('dgp')} p={cfg.get('p')} "
        f"N={cfg.get('n')} concepts={cfg.get('concepts')} firing={cfg.get('firing_tail')}"
        f"(s={cfg.get('zipf_s')})",
        fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out_dir / f"frontier_{args.tag}.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
