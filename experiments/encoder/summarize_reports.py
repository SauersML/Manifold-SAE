"""Consolidate WS-E EncoderReport JSONs into one scoreboard.

    python summarize_reports.py /dev/shm/sauers_gpu/encoder/*.json

Prints one row per report: dictionary, K, agreement (coord/gate Lâˆž), fallback
rate, throughput (rows/s + PASS/FAIL vs 1e5), and cyclic-order recovery when a
probe run recorded it. A compact end-of-run readout for the lead.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def _fmt(v: object, spec: str = "") -> str:
    if v is None:
        return "-"
    try:
        return format(float(v), spec) if spec else str(v)
    except (TypeError, ValueError):
        return str(v)


def main(argv: list[str]) -> int:
    paths: list[str] = []
    for a in argv or ["/dev/shm/sauers_gpu/encoder/*.json"]:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print("no report JSONs found")
        return 1

    print(f"{'report':28} {'K':>2} {'coordLinf':>10} {'gateLinf':>9} "
          f"{'fallback':>8} {'rows/s':>12} {'gate':>5} {'cyc':>5}")
    print("-" * 92)
    for p in paths:
        try:
            r = json.loads(Path(p).read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"{Path(p).name:28} <unreadable: {exc}>")
            continue
        cyc = r.get("cyclic_order_recovery") or {}
        cyc_frac = cyc.get("cyclic_order_fraction")
        print(
            f"{Path(p).stem:28.28} "
            f"{_fmt(r.get('k_atoms')):>2} "
            f"{_fmt(r.get('coord_linf_max'), '.2e'):>10} "
            f"{_fmt(r.get('assignment_linf_max'), '.2e'):>9} "
            f"{_fmt(r.get('fallback_rate_overall'), '.3f'):>8} "
            f"{_fmt(r.get('throughput_rows_per_s'), ',.0f'):>12} "
            f"{'PASS' if r.get('throughput_passes_gate') else 'FAIL':>5} "
            f"{_fmt(cyc_frac, '.2f') if cyc_frac is not None else '-':>5}"
        )
        dec = r.get("fallback_by_freq_decile")
        if dec:
            rates = " ".join(f"{d['fallback_rate']:.2f}" for d in dec)
            print(f"    decile fallback (rare->freq): {rates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
