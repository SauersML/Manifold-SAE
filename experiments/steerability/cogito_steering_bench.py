"""Orchestrator: run cogito_intervene.py across all 11 axes and the full
31-prompt color elicitation set.

For each axis in {hue, sat, val, pc1, pc2, pc3, red, blue, green,
achromatic, random}:
  for alpha in {-3,-2,-1,0,+1,+2,+3} (in per-axis-std units):
    for prompt in color_prompts.json:
      POST /v1/completions with extra_body.interventions=[...] and
      capture top-K logprobs + KL vs baseline.

Each (prompt x axis x alpha) is appended as one JSONL line to
`cogito_intervention_results.jsonl`. Tolerant of server hiccups: per-call
retry + progress logged to stderr.

Total: 11 axes x 31 prompts x 7 alphas = 2387 calls.
At ~2-4 s/call on B200 TP=8 -> ~60-80 min.

Endpoint comes from COGITO_API_BASE (NEVER hardcoded).

Usage:
  python cogito_steering_bench.py --dry-run
  COGITO_API_BASE=http://<host>:8000 python cogito_steering_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERVENE = HERE / "cogito_intervene.py"
DEFAULT_OUT = HERE / "cogito_intervention_results.jsonl"
PROMPTS = HERE / "color_prompts.json"

ALL_AXES = [
    "hue", "sat", "val",
    "pc1", "pc2", "pc3",
    "red", "blue", "green", "achromatic",
    "random",
]
ALPHAS = "-3,-2,-1,0,1,2,3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base",
                    default=os.environ.get("COGITO_API_BASE", ""))
    ap.add_argument("--model",
                    default=os.environ.get("COGITO_MODEL", "cogito"))
    ap.add_argument("--layer", type=int,
                    default=int(os.environ.get("COGITO_LAYER", "40")))
    ap.add_argument("--axes", default=",".join(ALL_AXES),
                    help="comma-separated axis names")
    ap.add_argument("--alphas", default=ALPHAS)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-retries-per-axis", type=int, default=3)
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    n_prompts = len(json.loads(Path(args.prompts).read_text())["prompts"])
    n_alphas = len(args.alphas.split(","))
    total = len(axes) * n_prompts * n_alphas
    print(f"[bench] axes={len(axes)} prompts={n_prompts} alphas={n_alphas} "
          f"-> {total} calls (est ~{total * 3 / 60:.0f} min @3s/call)",
          flush=True)

    if not args.dry_run and not args.api_base:
        raise SystemExit(
            "Set COGITO_API_BASE or pass --api-base, or use --dry-run.")

    overall_t0 = time.time()
    failures: list[tuple[str, str]] = []

    for ai, axis in enumerate(axes):
        print(f"\n[bench] ===== axis {ai+1}/{len(axes)}: {axis} =====",
              file=sys.stderr, flush=True)
        cmd = [
            sys.executable, str(INTERVENE),
            "--axis", axis,
            "--prompts", args.prompts,
            "--out", args.out,
            "--alphas", args.alphas,
            "--top-k", str(args.top_k),
            "--layer", str(args.layer),
            "--model", args.model,
        ]
        if args.api_base:
            cmd += ["--api-base", args.api_base]
        if args.dry_run:
            cmd += ["--dry-run", "--limit-prompts", "2"]

        ok = False
        for attempt in range(args.max_retries_per_axis):
            t0 = time.time()
            print(f"[bench] $ {' '.join(cmd)}", file=sys.stderr, flush=True)
            rc = subprocess.call(cmd)
            dt = time.time() - t0
            if rc == 0:
                print(f"[bench] axis={axis} ok in {dt:.1f}s",
                      file=sys.stderr, flush=True)
                ok = True
                break
            print(f"[bench] axis={axis} attempt {attempt+1} FAILED rc={rc} "
                  f"after {dt:.1f}s; sleeping 10s before retry",
                  file=sys.stderr, flush=True)
            time.sleep(10)
        if not ok:
            failures.append((axis, f"rc!=0 after {args.max_retries_per_axis} tries"))

    total_dt = time.time() - overall_t0
    print(f"\n[bench] done in {total_dt/60:.1f} min", flush=True)
    if failures:
        print(f"[bench] {len(failures)} axis failures:", flush=True)
        for a, msg in failures:
            print(f"  - {a}: {msg}", flush=True)
        return 1
    print(f"[bench] all axes succeeded; results -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
