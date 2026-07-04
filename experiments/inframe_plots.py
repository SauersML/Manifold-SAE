"""Plots for the in-frame curved frontier (experiments/inframe_frontier.py output).

Palette: BLUE #2a78d6 (curved, ours), YELLOW #eda100 (linear Tier-1), surface #fcfcfb.

    python -m experiments.inframe_plots --in results/suite_2026-07-03/frontiers/inframe_frontier.json \
        --out-dir results/suite_2026-07-03/frontiers/
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
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _by_p(rows, lane, key):
    ps = sorted({r["p"] for r in rows})
    mean = [np.mean([r[lane][key] for r in rows if r["p"] == p]) for p in ps]
    return np.array(ps), np.array(mean)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    d = json.loads(Path(args.inp).read_text())
    rows = d["rows"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # EV vs p, both lanes
    ps, ev_l = _by_p(rows, "linear", "ev"); _, ev_c = _by_p(rows, "curved", "ev")
    ax[0, 0].plot(ps, ev_c, "o-", color=BLUE, lw=2.4, ms=8, label="SAE + in-frame curved (ours)")
    ax[0, 0].plot(ps, ev_l, "^-", color=YELLOW, lw=2.4, ms=8, label="Tier-1 SAE (linear)")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("ambient p"); ax[0, 0].set_ylabel("explained variance")
    ax[0, 0].set_ylim(0, 1.02); ax[0, 0].set_title("EV vs ambient p (matched L0)")
    ax[0, 0].legend(frameon=False, fontsize=9)

    # ΔEV vs p (never below 0)
    dps = sorted({r["p"] for r in rows})
    dev_mean = [np.mean([r["delta_ev"] for r in rows if r["p"] == p]) for p in dps]
    dev_min = [np.min([r["delta_ev"] for r in rows if r["p"] == p]) for p in dps]
    ax[0, 1].plot(dps, dev_mean, "o-", color=BLUE, lw=2.4, ms=8, label="mean ΔEV")
    ax[0, 1].plot(dps, dev_min, "o--", color=AQUA, lw=1.6, ms=6, label="min ΔEV (worst seed)")
    ax[0, 1].axhline(0, color=INK2, lw=1.2, ls=":")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("ambient p")
    ax[0, 1].set_ylabel("ΔEV (curved − linear)"); ax[0, 1].set_title("Curved never loses EV")
    ax[0, 1].legend(frameon=False, fontsize=9)

    # bits vs p, both lanes
    _, b_l = _by_p(rows, "linear", "bits"); _, b_c = _by_p(rows, "curved", "bits")
    ax[1, 0].plot(ps, b_c, "o-", color=BLUE, lw=2.4, ms=8, label="in-frame curved")
    ax[1, 0].plot(ps, b_l, "^-", color=YELLOW, lw=2.4, ms=8, label="Tier-1 linear")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_xlabel("ambient p")
    ax[1, 0].set_ylabel("description length (bits, ½·border·log n + residual)")
    ax[1, 0].set_title("Curved wins bits at matched L0"); ax[1, 0].legend(frameon=False, fontsize=9)

    # border + inference-FLOP shrink vs p
    _, bord_l = _by_p(rows, "linear", "border"); _, bord_c = _by_p(rows, "curved", "border")
    _, fl_l = _by_p(rows, "linear", "infer_macs_per_token"); _, fl_c = _by_p(rows, "curved", "infer_macs_per_token")
    ax[1, 1].plot(ps, bord_l / bord_c, "o-", color=BLUE, lw=2.4, ms=8, label="border shrink M·p / M·r")
    ax[1, 1].plot(ps, fl_l / fl_c, "s--", color=AQUA, lw=1.8, ms=6, label="inference-MAC shrink")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("ambient p"); ax[1, 1].set_ylabel("× cheaper (curved vs linear)")
    ax[1, 1].set_title("In-frame curved is also cheaper (r ≪ p)")
    ax[1, 1].legend(frameon=False, fontsize=9)

    v = d.get("verdict", {})
    fig.suptitle(f"In-frame curved cascade frontier — never loses EV ({v.get('curved_never_loses_ev')}), "
                 f"wins bits ({v.get('curved_wins_bits')})  |  r_true={d['config']['r_true']} "
                 f"M={d['config']['m']} L0={d['config']['l0']}, {d['config']['seeds']} seeds",
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = out / "inframe_frontier.png"
    fig.savefig(p, dpi=150)
    print(f"[plot] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
