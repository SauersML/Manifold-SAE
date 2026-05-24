"""auto_exp_76: Run full SAE-family leaderboard, identify winner.

Prints the final ranked table, the across-the-board winner, the per-
protocol winners (linear_push / anchor_swap / magnitude / compositional),
and the 3 most-improved variants vs the Manifold-SAE baseline
(R²=0.913).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from manifold_sae.eval.leaderboard_v2 import run_leaderboard


MANIFOLD_BASELINE_R2 = 0.913


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "full_leaderboard"
    data = root / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"

    payload = run_leaderboard(
        root=root, data_path=data, output_dir=out,
        device="cpu", max_val_rows=1024, ablation_subset=16,
    )
    rows = payload["rows"]
    print("\n========== FULL LEADERBOARD ==========")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r['variant']:<14} composite={r['composite_score']:.3f}"
              f"  R²={r['val_r2']:.3f}  steer={r['steering_composite']:.3f}"
              f"  HSV={r['hsv_coherence']:.3f}  dead={r['dead_rate']:.3f}"
              f"  ({Path(r['file']).stem})")
    if rows:
        w = rows[0]
        print(f"\n>>> across-the-board winner: {w['variant']} ({Path(w['file']).stem})"
              f" composite={w['composite_score']:.3f}")
    # Per-protocol winners.
    protos = ["steering_linear_push", "steering_anchor_swap",
              "steering_magnitude", "steering_compositional"]
    print("\nPer-protocol winners:")
    for p in protos:
        best = max(rows, key=lambda r: r.get(p, float("-inf")))
        print(f"  {p:<28} -> {best['variant']:<14} ({best[p]:.3f})")
    print(f"  hsv_coherence                -> "
          f"{max(rows, key=lambda r: r['hsv_coherence'])['variant']}")
    print(f"  val_r2                       -> "
          f"{max(rows, key=lambda r: r['val_r2'])['variant']}")

    # 3 most-improved vs Manifold baseline R²=0.913.
    deltas = sorted(rows, key=lambda r: -(r["val_r2"] - MANIFOLD_BASELINE_R2))
    print("\nTop-3 most-improved vs Manifold-SAE baseline R²=0.913:")
    for r in deltas[:3]:
        d = r["val_r2"] - MANIFOLD_BASELINE_R2
        print(f"  {r['variant']:<14} ΔR²={d:+.3f}  (now {r['val_r2']:.3f})")

    # Write insights markdown.
    _write_insights(rows, out / "insights.md")
    print(f"\nWrote {out / 'leaderboard.json'}")
    print(f"Wrote {out / 'leaderboard.md'}")
    print(f"Wrote {out / 'insights.md'}")


def _write_insights(rows, path: Path) -> None:
    if not rows:
        path.write_text("# Insights\n\nNo rows scored.\n")
        return
    # Architectural-family grouping.
    families = {
        "topk_like":      ["topk", "adaptive_k", "das_sae"],
        "l1_like":        ["l1"],
        "manifold_s1":    ["manifold"],
        "ot_wasserstein": ["wasserstein"],
        "multi_layer":    ["sheaf"],
    }
    by_fam = {f: [] for f in families}
    for r in rows:
        for fam, vs in families.items():
            if r["variant"] in vs:
                by_fam[fam].append(r["composite_score"])
                break
    lines = ["# Full Leaderboard — Cross-cut Analysis\n\n"]
    lines.append("## Architectural family vs composite score\n\n")
    lines.append("| Family | n | mean composite |\n|---|---|---|\n")
    for fam, vals in by_fam.items():
        if vals:
            lines.append(f"| {fam} | {len(vals)} | {sum(vals)/len(vals):.3f} |\n")
    # Pareto: R² vs sparsity (mean_K).
    lines.append("\n## Pareto frontier: sparsity vs R²\n\n")
    lines.append("| Variant | mean_K (active per row) | R² |\n|---|---|---|\n")
    for r in sorted(rows, key=lambda x: x["mean_K"]):
        lines.append(f"| {r['variant']} | {r['mean_K']:.1f} | {r['val_r2']:.3f} |\n")
    # Pareto pruning: a variant is Pareto-optimal if no other has both
    # lower K AND higher R².
    def is_pareto(r):
        return not any(
            (o["mean_K"] < r["mean_K"]) and (o["val_r2"] > r["val_r2"])
            for o in rows
        )
    front = [r for r in rows if is_pareto(r)]
    lines.append("\n### Pareto-optimal variants\n\n")
    for r in front:
        lines.append(f"- {r['variant']} (K={r['mean_K']:.1f}, R²={r['val_r2']:.3f})\n")
    # Recommendations: top-3 by composite.
    lines.append("\n## Recommendations for next round\n\n")
    for i, r in enumerate(rows[:3], 1):
        lines.append(f"{i}. **{r['variant']}** — composite {r['composite_score']:.3f},"
                     f" R²={r['val_r2']:.3f}, HSV={r['hsv_coherence']:.3f}\n")
    path.write_text("".join(lines))


if __name__ == "__main__":
    main()
