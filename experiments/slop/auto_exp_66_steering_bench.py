"""auto_exp_66 — head-to-head SAE steering benchmark.

Runs the 4-protocol steering benchmark from
``manifold_sae/eval/steering_bench.py`` over every checkpoint in
``runs/sae_comparison/`` and prints a leaderboard.

Hypothesis (from auto_exp_44 + the gauge-fix story):
  * Protocol 1 (linear push): a vanilla L1 SAE with dense, near-linear
    decoders should be competitive — pushing an atom moves the decoded
    vector almost rigidly along its dictionary direction.
  * Protocol 2 (anchor swap): the **Manifold-SAE** is supposed to win,
    because it has explicit (theta, amp) factorization — swapping the
    amp block while keeping theta fixed is exactly an anchor swap by
    construction. auto_exp_44 already saw anchor-offset > tangent-push
    qualitatively; this protocol pins it down quantitatively.
  * Protocol 3 (magnitude scaling): Manifold > L1 (Fourier basis is
    designed for smooth amplitude response); TopK may saturate.
  * Protocol 4 (compositional): Manifold > L1 if the (theta, amp)
    factorization actually disentangles axes; L1 will leak hue into
    value and vice-versa.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_steering_bench import run, expand_globs  # noqa: E402


def main() -> None:
    models = expand_globs([str(ROOT / "runs/sae_comparison/model_*.pt")])
    if not models:
        print("[auto_exp_66] no models found under runs/sae_comparison/")
        return
    out_dir = ROOT / "runs/auto_exp_66_steering_bench"
    lb = run(models, str(ROOT / "runs/COLOR_COGITO_L40/X_L40.npy"),
             str(out_dir), device="cpu", max_val_rows=1200)
    print("\n=== HEAD-TO-HEAD STEERING BENCH ===")
    for p, rows in lb["ranking"].items():
        print(f"\n[{p}]")
        for rank, (name, score) in enumerate(rows, 1):
            print(f"  {rank}. {name:30s}  {score:+.3f}")

    summary_path = out_dir / "head_to_head.json"
    summary_path.write_text(json.dumps(lb["ranking"], indent=2, default=float))
    print(f"\n[wrote] {summary_path}")
    print(f"[wrote] {out_dir/'radar.png'}")


if __name__ == "__main__":
    main()
