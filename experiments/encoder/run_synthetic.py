"""End-to-end WS-E validation on a synthetic composed dictionary.

Builds a planted composed dictionary (circles + linear atoms), fits it, distils
the amortized encoder, and reports agreement + certificate-fallback (overall and
by token-frequency decile) + throughput. Runs at tiny scale locally and at real
scale on node2 (via Heimdall).

    python run_synthetic.py --scale local
    python run_synthetic.py --scale node --circles 3 --linear 1 --ambient 256 \
        --n-train 200000 --n-eval 200000 --out /dev/shm/sauers_gpu/encoder/synth_node.json

No env vars, no wall-clock budgets (SPEC.md). CLI flags only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_harness as dh          # noqa: E402
import synth_dictionary as sd         # noqa: E402


def _log(*a: object) -> None:
    print(*a)
    sys.stdout.flush()


def build_config(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WS-E synthetic encoder validation")
    ap.add_argument("--scale", choices=("local", "node"), default="local",
                    help="tiny local smoke (K=1) vs real-scale node run")
    ap.add_argument("--circles", type=int, default=None)
    ap.add_argument("--linear", type=int, default=None)
    ap.add_argument("--ambient", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--fit-iter", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, nargs="+", default=None)
    ap.add_argument("--throughput-rows", type=int, default=None)
    ap.add_argument("--gate-rows-per-s", type=float, default=1.0e5)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="write JSON report here")
    args = ap.parse_args(argv)

    # Scale presets (overridable per flag). Local stays within the 8GB / tiny-fit
    # doctrine (n<=2000, p<=64); node uses real corpus-sweep sizes.
    if args.scale == "local":
        d = dict(circles=1, linear=0, ambient=16, n_train=1200, n_eval=800,
                 epochs=200, hidden=[64, 64], throughput_rows=200_000)
    else:
        d = dict(circles=2, linear=1, ambient=128, n_train=50_000, n_eval=50_000,
                 epochs=400, hidden=[128, 128], throughput_rows=1_000_000)
    for key, val in d.items():
        if getattr(args, key.replace("-", "_")) is None:
            setattr(args, key.replace("-", "_"), val)
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_config(argv)
    _log(f"[WS-E] scale={args.scale} circles={args.circles} linear={args.linear} "
         f"ambient={args.ambient} n_train={args.n_train} n_eval={args.n_eval}")

    t0 = time.perf_counter()
    corpus = sd.planted_circles(
        n_train=args.n_train, n_eval=args.n_eval, ambient_dim=args.ambient,
        n_circles=args.circles, n_linear=args.linear, noise=args.noise,
        random_state=args.random_state,
    )
    _log(f"[WS-E] corpus: {corpus.description} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    model = sd.fit_dictionary(corpus, n_iter=args.fit_iter, random_state=args.random_state)
    _log(f"[WS-E] dictionary fit: K={len(model.atoms)} "
         f"R2={getattr(model,'reconstruction_r2', float('nan')):.4f} "
         f"({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    _enc, report = dh.distill_and_gate(
        model, corpus.X_train, corpus.X_eval,
        dictionary_source=f"synthetic: {corpus.description}",
        token_freq=corpus.token_freq_eval,
        hidden=tuple(args.hidden), epochs=args.epochs,
        random_state=args.random_state,
        throughput_target_rows=args.throughput_rows,
        throughput_gate_rows_per_s=args.gate_rows_per_s,
        notes=[f"distill+gate+throughput in {time.perf_counter()-t0:.1f}s (measured below)"],
    )
    _log("\n===== WS-E ENCODER REPORT =====")
    _log(report.summary())
    _log("================================")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dh.write_report(report, args.out)
        _log(f"[WS-E] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
