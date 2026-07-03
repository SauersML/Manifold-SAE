#!/usr/bin/env python
"""Adapt a dose_calibration_real.json run into the schema Lane-6 EVAL consumes.

EVAL's figs 7/8 want, at results/run_<tag>/dose_calibration.json:
  model, probe_order, probe_angles, ordering_corr, predicted_nats, measured_kl, slope, r2

Usage: python dose_to_eval_schema.py <run_dir_or_json> <out_json> [--probe-feature month]
The probe arrays come from the chosen periodic feature's cyclic-ordering block; the
predicted/measured scatter and slope/R2 are the aggregate manifold-method calibration.
"""
from __future__ import annotations

import json
import os
import sys


def load_run(path):
    if os.path.isdir(path):
        path = os.path.join(path, "dose_calibration_real.json")
    with open(path) as fh:
        return json.load(fh), path


def pick_feature(per_atom, want):
    cand = [a for a in per_atom if a.get("cyclic_ordering")]
    if not cand:
        return None
    if want:
        for a in cand:
            if a["atom"] == want:
                return a
    # else the periodic feature with the most probes, then best correlation
    return sorted(cand, key=lambda a: (len(a["cyclic_ordering"]["words_present"]),
                                       a["cyclic_ordering"]["circ_corr"]))[-1]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want = None
    if "--probe-feature" in sys.argv:
        want = sys.argv[sys.argv.index("--probe-feature") + 1]
    run_path, out_path = args[0], args[1]
    run, src = load_run(run_path)

    per_atom = run["fit"]["per_atom"]
    feat = pick_feature(per_atom, want)
    if feat is None:
        raise SystemExit("no atom carries a cyclic_ordering block")
    co = feat["cyclic_ordering"]
    n_probe = len(co["words_present"])

    # Spacing-robust ordering correlation is the authoritative A4 score. Newer runs store it
    # as order_corr; for older runs recompute it from the stored angles (rank each word around
    # the loop, circularly-correlate the ranks with calendar order — ~1.0 for a correct cycle
    # regardless of non-uniform angular spacing, which raw-angle circ_corr would understate).
    def _circ_mean(a):
        return float(np.arctan2(np.sin(a).sum(), np.cos(a).sum()))

    def _circ_corr(a, b):
        a0, b0 = a - _circ_mean(a), b - _circ_mean(b)
        den = float(np.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2)))
        return float(np.sum(np.sin(a0) * np.sin(b0))) / den if den > 0 else 0.0

    ordering_corr = co.get("order_corr")
    if ordering_corr is None:
        ang = np.asarray(co["angles_rad"], dtype=float)
        ideal = 2.0 * np.pi * np.arange(len(ang)) / len(ang)
        ranks = np.argsort(np.argsort(ang)).astype(float)
        rank_ang = 2.0 * np.pi * ranks / len(ang)
        ordering_corr = max(_circ_corr(rank_ang, ideal), _circ_corr(rank_ang, -ideal))

    man = run["stats"]["manifold"]
    rows = [r for r in run["rows"] if r["method"] == "manifold"]

    payload = dict(
        model=run["model"],
        probe_feature=feat["atom"],
        probe_words=co["words_present"],
        probe_order=list(range(n_probe)),
        probe_angles=co["angles_rad"],
        ordering_corr=float(ordering_corr),
        ordering_corr_raw_angle=co["circ_corr"],
        wraparound_in_order=co["wraparound_in_order"],
        predicted_nats=[r["predicted_nats"] for r in rows],
        measured_kl=[r["measured_kl"] for r in rows],
        slope=man["log_slope"],
        r2=man["log_r2"],
        ratio_median=man["ratio_median"],
        source=os.path.basename(src),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[eval-schema] wrote {out_path}: model={payload['model'][:40]!r} "
          f"probe={feat['atom']} n_probe={n_probe} ordering_corr={ordering_corr:.3f} "
          f"slope={man['log_slope']:.3f} r2={man['log_r2']:.3f} "
          f"n_scatter={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
