"""Verdicts from frontier_bench.py output: EV-at-matched-FLOP, bits crossover, overhead.

Turns the raw per-lane/per-K sweeps into the numbers the writeup asserts, honestly:

  * EV at matched inference-FLOP: interpolate each lane's held-out EV onto a shared
    FLOP grid; report min over the grid of (curved_EV - linear_EV). The claim
    "curved never loses EV at matched compute" is TRUE iff that min >= -tol.
  * bits at matched distortion: at a target held-out EV, interpolate each lane's
    bits/token (both selection currencies); report curved - linear (negative = curved
    wins bits).
  * pure-linear overhead: on a linear-DGP run, the EV and bits the curved lane gives up
    at matched FLOP (the honest cost line).

    python -m experiments.frontier_analyze --curved synth_curved.json [--linear synth_linear.json] \
        --out-md frontier_verdicts.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ok(payload, lane):
    rows = [r for r in payload["results"] if r.get("ok") and r.get("lane") == lane]
    return sorted(rows, key=lambda r: r["flops"]["infer_macs_per_token"])


def _interp(x_src, y_src, x_q):
    """Monotone-x linear interpolation of y(x) at x_q, clipped to the observed range."""
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    order = np.argsort(x_src)
    x_src, y_src = x_src[order], y_src[order]
    x_q = np.clip(x_q, x_src.min(), x_src.max())
    return np.interp(x_q, x_src, y_src)


def ev_at_matched_flop(payload, lane_a="curved", lane_b="linear", n_grid=40, tol=0.005):
    A, B = _ok(payload, lane_a), _ok(payload, lane_b)
    if not A or not B:
        return {"available": False, "reason": f"missing lane rows ({lane_a}:{len(A)} {lane_b}:{len(B)})"}
    fa = [r["flops"]["infer_macs_per_token"] for r in A]
    fb = [r["flops"]["infer_macs_per_token"] for r in B]
    lo, hi = max(min(fa), min(fb)), min(max(fa), max(fb))
    if hi <= lo:
        return {"available": False, "reason": "no overlapping FLOP range"}
    grid = np.geomspace(lo, hi, n_grid)
    ea = _interp(fa, [r["heldout_ev"] for r in A], grid)
    eb = _interp(fb, [r["heldout_ev"] for r in B], grid)
    delta = ea - eb
    i = int(np.argmin(delta))
    return {"available": True, "lane_a": lane_a, "lane_b": lane_b,
            "min_delta_ev": float(delta.min()), "at_infer_macs": float(grid[i]),
            "mean_delta_ev": float(delta.mean()),
            "curved_never_loses": bool(delta.min() >= -tol), "tol": tol,
            "flop_lo": float(lo), "flop_hi": float(hi)}


def bits_at_matched_ev(payload, target_ev, lane_a="curved", lane_b="linear",
                       currency="support_entropy"):
    key = f"bits_per_token_{currency}"
    A, B = _ok(payload, lane_a), _ok(payload, lane_b)
    if not A or not B:
        return {"available": False}
    def at(rows):
        evs = [r["heldout_ev"] for r in rows]
        bits = [r["mdl"][key] for r in rows if "mdl" in r]
        flps = [r["flops"]["infer_macs_per_token"] for r in rows]
        if len(bits) != len(evs) or max(evs) < target_ev:
            return None, None
        b = float(_interp(evs, bits, target_ev))
        f = float(_interp(evs, flps, target_ev))
        return b, f
    ba, fa = at(A)
    bb, fb = at(B)
    if ba is None or bb is None:
        return {"available": False, "reason": f"a lane cannot reach EV={target_ev}"}
    return {"available": True, "target_ev": target_ev, "currency": currency,
            "bits_a": ba, "bits_b": bb, "delta_bits": ba - bb,
            "curved_wins_bits": bool(ba < bb),
            "flop_a": fa, "flop_b": fb}


def summarize(curved_payload, linear_payload=None) -> dict:
    out = {"config": curved_payload.get("config"),
           "matched_distortion_delta2": curved_payload.get("matched_distortion_delta2")}
    # EV at matched FLOP, curved vs both linear references
    out["ev_matched_flop_vs_block"] = ev_at_matched_flop(curved_payload, "curved", "linear")
    out["ev_matched_flop_vs_samelane"] = ev_at_matched_flop(curved_payload, "curved", "manifold_linear")
    # bits at a few EV targets
    best_ev = min([max([r["heldout_ev"] for r in _ok(curved_payload, l)] or [0])
                   for l in ("curved", "linear") if _ok(curved_payload, l)] or [0])
    targets = [round(x, 3) for x in np.linspace(0.5 * best_ev, 0.95 * best_ev, 3)] if best_ev > 0 else []
    out["bits_matched_ev"] = []
    for t in targets:
        for cur in ("support_entropy", "combinatorial"):
            r = bits_at_matched_ev(curved_payload, t, "curved", "linear", cur)
            if r.get("available"):
                out["bits_matched_ev"].append(r)
    # pure-linear overhead
    if linear_payload is not None:
        out["pure_linear_overhead"] = {
            "ev_matched_flop": ev_at_matched_flop(linear_payload, "curved", "linear"),
            "note": "on a linear DGP, min_delta_ev is the EV the curved lane gives up; "
                    "positive delta_bits at matched EV is the bit overhead.",
        }
    return out


def to_markdown(summ: dict) -> str:
    L = ["# Frontier verdicts", ""]
    cfg = summ.get("config", {})
    L.append(f"**DGP** {cfg.get('dgp')} | p={cfg.get('p')} N={cfg.get('n')} "
             f"concepts={cfg.get('concepts')} firing={cfg.get('firing_tail')}(s={cfg.get('zipf_s')}) "
             f"| matched-distortion delta2={summ.get('matched_distortion_delta2')}")
    L.append("")
    for tag, key in [("vs block/TopK linear", "ev_matched_flop_vs_block"),
                     ("vs same-lane linear", "ev_matched_flop_vs_samelane")]:
        r = summ.get(key, {})
        if r.get("available"):
            verdict = "curved NEVER loses EV at matched FLOP" if r["curved_never_loses"] \
                else "curved LOSES EV somewhere (plotted)"
            L.append(f"- **EV at matched inference-FLOP ({tag})**: min Δ(curved−linear)EV = "
                     f"{r['min_delta_ev']:+.4f} at {r['at_infer_macs']:.3g} MACs/token; "
                     f"mean Δ = {r['mean_delta_ev']:+.4f}. → {verdict}.")
        else:
            L.append(f"- EV at matched FLOP ({tag}): n/a ({r.get('reason')}).")
    L.append("")
    if summ.get("bits_matched_ev"):
        L.append("**Bits/token at matched distortion (negative Δ = curved wins bits):**")
        L.append("")
        L.append("| target EV | currency | curved bits | linear bits | Δ bits | curved wins |")
        L.append("|---:|---|---:|---:|---:|:--:|")
        for r in summ["bits_matched_ev"]:
            L.append(f"| {r['target_ev']:.3f} | {r['currency']} | {r['bits_a']:.3f} | "
                     f"{r['bits_b']:.3f} | {r['delta_bits']:+.3f} | {'yes' if r['curved_wins_bits'] else 'no'} |")
        L.append("")
    ov = summ.get("pure_linear_overhead")
    if ov and ov["ev_matched_flop"].get("available"):
        r = ov["ev_matched_flop"]
        L.append(f"**Pure-linear DGP overhead (honest cost line):** min Δ(curved−linear)EV = "
                 f"{r['min_delta_ev']:+.4f} at matched FLOP; mean Δ = {r['mean_delta_ev']:+.4f}. "
                 f"This is the selection/refinement overhead paid when there is no curvature to find.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curved", required=True, help="curved-DGP frontier JSON")
    ap.add_argument("--linear", help="linear-DGP frontier JSON (pure-linear overhead)")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json")
    args = ap.parse_args(argv)
    cp = json.loads(Path(args.curved).read_text())
    lp = json.loads(Path(args.linear).read_text()) if args.linear else None
    summ = summarize(cp, lp)
    Path(args.out_md).write_text(to_markdown(summ) + "\n")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summ, indent=2))
    print(to_markdown(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
