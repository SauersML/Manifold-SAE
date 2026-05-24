"""CLI: regenerate the full SAE-family leaderboard.

Usage:
    python -m scripts.run_full_leaderboard \
        --output runs/full_leaderboard/ \
        --data runs/COLOR_COGITO_L40/X_L40.npy
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from manifold_sae.eval.leaderboard_v2 import run_leaderboard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/full_leaderboard/")
    ap.add_argument("--data", default="runs/COLOR_COGITO_L40/X_L40.npy")
    ap.add_argument("--root", default=".")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-val-rows", type=int, default=1024)
    ap.add_argument("--ablation-subset", type=int, default=16)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    payload = run_leaderboard(
        root=Path(args.root),
        data_path=Path(args.data),
        output_dir=Path(args.output),
        device=args.device,
        max_val_rows=args.max_val_rows,
        ablation_subset=args.ablation_subset,
    )
    print(f"[run_full_leaderboard] scored {payload['n_scored']}"
          f" of {payload['n_checkpoints_found']} checkpoints")
    rows = payload["rows"]
    if rows:
        winner = rows[0]
        print(f"[run_full_leaderboard] winner: {winner['variant']}"
              f" ({Path(winner['file']).stem})"
              f"  composite={winner['composite_score']:.3f}"
              f"  R²={winner['val_r2']:.3f}"
              f"  steer={winner['steering_composite']:.3f}")
    for s in payload["skipped"]:
        print(f"  skipped {s['path']}: {s['reason']}")


if __name__ == "__main__":
    main()
