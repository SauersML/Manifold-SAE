"""auto_exp_64: Complete Replacement Model (CRM) on 3-layer synthetic stack."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_crm import train  # noqa: E402


def main():
    summary = train(epochs=5)
    print("=" * 60)
    print("CRM per-stage R2 (L0 input recon, L1/L2 activation match):")
    for l, r in enumerate(summary["final_per_stage_r2"]):
        print(f"  stage {l}: R2={r:.4f}")
    print("=" * 60)
    out_path = ROOT / "runs" / "auto_exp_64_crm.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp64] {out_path}")
    return summary


if __name__ == "__main__":
    main()
