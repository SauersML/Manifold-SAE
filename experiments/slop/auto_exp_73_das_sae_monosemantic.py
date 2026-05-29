"""auto_exp_73: DAS-SAE monosemantic-hue sweep on cogito-L40.

Tests the falsifier from project_das_sae_cogito.md: does ANY regime in
(F, λ_intv, λ_l1) space produce a SINGLE feature whose swap reproduces the
target hue change with R² ≥ 0.5?

  - max_r2 ≥ 0.5  → monosemantic hue atom EXISTS; "hue is distributed" claim
                    under TopK was a regularization artifact, not a property
                    of cogito-L40.
  - max_r2 <  0.5 → hue is genuinely distributed in cogito-L40; no single
                    atom carries it even when L1 is allowed to grow K freely
                    and λ_intv dominates the loss.

Calls scripts/train_das_sae_l1_monosemantic.main() directly and writes a
verdict json + a console report.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/Users/user/Manifold-SAE")
sys.path.insert(0, str(ROOT))

from scripts import train_das_sae_l1_monosemantic as sweep_mod

OUT = ROOT / "runs" / "auto_exp_73_das_sae_monosemantic"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Defer to the sweep script's arg parser via sys.argv override.
    saved_argv = sys.argv
    sys.argv = ["train_das_sae_l1_monosemantic.py", "--epochs", "5"]
    try:
        grid = sweep_mod.main()
    finally:
        sys.argv = saved_argv

    winner = grid["winner"]
    max_r2 = winner["max_single_feature_r2"]
    threshold = 0.5
    if max_r2 >= threshold:
        verdict = "MONOSEMANTIC_HUE_ATOM_EXISTS"
        narrative = (
            f"Single-feature swap R² = {max_r2:.3f} ≥ {threshold} at "
            f"F={winner['F']}, λ_intv={winner['lambda_intv']}, "
            f"λ_l1={winner['lambda_l1']}. The 'hue is distributed' result "
            f"from auto_exp_65 was a TopK artifact; under L1 with sufficient "
            f"interchange pressure, a single SAE atom carries hue causally."
        )
    else:
        verdict = "HUE_GENUINELY_DISTRIBUTED"
        narrative = (
            f"Best single-feature swap R² across the grid = {max_r2:.3f} < "
            f"{threshold} even at F={winner['F']}, λ_intv={winner['lambda_intv']}, "
            f"λ_l1={winner['lambda_l1']}. Confirms hue in cogito-L40 is "
            f"genuinely a distributed direction; no single SAE atom carries "
            f"it under either TopK (auto_exp_65) or L1-only (this sweep)."
        )

    report = {
        "experiment": "auto_exp_73_das_sae_monosemantic",
        "grid_summary": {
            "Fs": grid["Fs"],
            "intvs": grid["intvs"],
            "l1s": grid["l1s"],
            "n_cfgs": len(grid["grid"]),
        },
        "winner": winner,
        "threshold": threshold,
        "verdict": verdict,
        "narrative": narrative,
        "runtime_seconds": time.time() - t0,
        "grid_json_path": str(ROOT / "runs" / "das_sae_l1_sweep" / "grid.json"),
        "heatmap_path": str(ROOT / "runs" / "das_sae_l1_sweep" / "grid_heatmap.png"),
        "winner_model_path": str(ROOT / "runs" / "das_sae_l1_sweep" / "winner.pt"),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 70)
    print(f"[auto_exp_73] VERDICT: {verdict}")
    print(f"[auto_exp_73] max single-feature swap R² = {max_r2:.3f}")
    print(f"[auto_exp_73] winner: F={winner['F']} λ_intv={winner['lambda_intv']} "
          f"λ_l1={winner['lambda_l1']}")
    print(f"[auto_exp_73] {narrative}")
    print(f"[auto_exp_73] report -> {OUT / 'report.json'}")
    return report


if __name__ == "__main__":
    main()
