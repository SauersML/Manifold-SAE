"""auto_exp_63: AdaptiveK SAE vs TopK-32 baseline on cogito-L40.

Thin wrapper over scripts/train_adaptive_k.py that also prints the
side-by-side comparison and writes runs/auto_exp_63_adaptive_k.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_adaptive_k import train  # noqa: E402


def main():
    summary = train(epochs=15, sparsity_weight=0.01, k_min=8, k_max=80)
    base = summary["baseline_topk32"]
    print("=" * 60)
    print("AdaptiveK SAE vs TopK-32 baseline on cogito-L40 (val):")
    print(f"  AdaptiveK : R2={summary['final_val_r2']:.4f}  mean_k={summary['final_mean_k']:.1f}  "
          f"k_range=[{summary['k_per_row_min']:.0f}, {summary['k_per_row_max']:.0f}]  "
          f"active={summary['n_active']}  dead={summary['dead_rate']:.3f}")
    print(f"  TopK-32   : R2={base['val_r2']:.4f}  mean_k={base['mean_k']}  active={base['n_active']}")
    print("=" * 60)
    out_path = ROOT / "runs" / "auto_exp_63_adaptive_k.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp63] {out_path}")
    return summary


if __name__ == "__main__":
    main()
