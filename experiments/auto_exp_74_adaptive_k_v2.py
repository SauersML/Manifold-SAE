"""auto_exp_74: AdaptiveKv2 (target-K loss) vs TopK-32 vs AdaptiveK v1.

Runs both v2 loss variants (clipped, squared) and plots R² vs mean-K against
the TopK-32 baseline (R²=0.874 at K=32) and the v1 AdaptiveK best
(R²=0.839 at K≈8.2). Writes JSON + PNG to runs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.train_adaptive_k_v2 import main as train_v2_main  # noqa: E402


def main():
    summary = train_v2_main()
    results = summary["results"]

    # Reference points.
    base_topk = {"label": "TopK-32 baseline", "mean_k": 32.0, "r2": 0.874}
    base_v1 = {"label": "AdaptiveK v1", "mean_k": 8.2, "r2": 0.839}
    v2_clip = {
        "label": "AdaptiveKv2 (clipped)",
        "mean_k": results["clipped"]["final_mean_k"],
        "r2": results["clipped"]["final_val_r2"],
    }
    v2_sq = {
        "label": "AdaptiveKv2 (squared)",
        "mean_k": results["squared"]["final_mean_k"],
        "r2": results["squared"]["final_val_r2"],
    }
    points = [base_topk, base_v1, v2_clip, v2_sq]

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        colors = {"TopK-32 baseline": "C0", "AdaptiveK v1": "C1",
                  "AdaptiveKv2 (clipped)": "C2", "AdaptiveKv2 (squared)": "C3"}
        for p in points:
            ax.scatter(p["mean_k"], p["r2"], s=130, c=colors[p["label"]],
                       label=p["label"], edgecolors="black", zorder=3)
        ax.axhline(0.880, ls="--", color="grey", alpha=0.6, label="goal R²=0.880")
        ax.axvline(32, ls=":", color="grey", alpha=0.4)
        ax.set_xlabel("mean K (per row)")
        ax.set_ylabel("val R²")
        ax.set_title("AdaptiveKv2 vs TopK-32 vs v1 on cogito-L40")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        out_png = ROOT / "runs" / "auto_exp_74_adaptive_k_v2.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        print(f"[plot] {out_png}", flush=True)
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)

    payload = {
        "v2_summary": summary,
        "comparison_points": points,
        "winner": summary["winner"],
        "winner_r2": summary["winner_r2"],
        "winner_mean_k": summary["winner_mean_k"],
        "goal_met": summary["goal_met"],
        "baseline_topk32_r2": 0.874,
        "v1_adaptive_r2": 0.839,
        "v1_adaptive_meank": 8.2,
    }
    out_path = ROOT / "runs" / "auto_exp_74_adaptive_k_v2.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print("=" * 60)
    print("AdaptiveKv2 vs baselines on cogito-L40 (val):")
    for p in points:
        marker = " <-- WIN" if p["label"].endswith(f"({summary['winner']})") else ""
        print(f"  {p['label']:28s}: R2={p['r2']:.4f}  mean_k={p['mean_k']:.1f}{marker}")
    print(f"  goal (R2>=0.880 at K<=32) met: {summary['goal_met']}")
    print("=" * 60)
    print(f"[exp74] {out_path}", flush=True)
    return payload


if __name__ == "__main__":
    main()
