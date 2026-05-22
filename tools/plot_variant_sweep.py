#!/usr/bin/env python3
"""Cross-variant sweep plot.

Aggregates all `runs_cluster/llm_sweep*/results.json` files and renders
a single grid plot comparing architectural variants side by side:

  rows: layer / model
  cols: vanilla EV / curve EV / locked EV / alive atoms
  x-axis per cell: F

So you can see at a glance how each variant performs across the F sweep.

  python tools/plot_variant_sweep.py [runs_dir] [output_png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_variant_results(runs_dir: Path) -> list[dict]:
    """Look for results.json in each llm_sweep* subdir."""
    out = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith("llm_sweep"):
            continue
        rj = d / "results.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        # Format A: dict with config, var, results
        if isinstance(data, dict) and "results" in data:
            cfg = data.get("config", {})
            rows = data.get("results", [])
        elif isinstance(data, list):
            cfg = {}
            rows = data
        else:
            continue
        out.append({
            "run_name": d.name,
            "config": cfg,
            "rows": rows,
        })
    return out


def main():
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs_cluster")
    out_png = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/variant_sweep_comparison.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    variants = load_variant_results(runs_dir)
    if not variants:
        print(f"no llm_sweep* results found under {runs_dir}", file=sys.stderr)
        return 1
    print(f"found {len(variants)} variant runs:")
    for v in variants:
        cfg = v["config"]
        meta = (
            f"  {v['run_name']:<30} "
            f"layer={cfg.get('layer','?')} "
            f"model={cfg.get('model_name','?').split('/')[-1][:14]:<14} "
            f"R={cfg.get('sae_intrinsic_rank','?')} "
            f"K={cfg.get('sae_n_basis','?')} "
            f"rows={len(v['rows'])}"
        )
        print(meta)

    # 2x2 grid: vanilla_expl, curve_expl, locked_expl, alive_atoms
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    metrics = [
        ("vanilla_explained", "Vanilla TopK SAE expl. variance", axes[0, 0]),
        ("curve_explained",    "Manifold-SAE (per-batch B) expl. variance", axes[0, 1]),
        ("curve_locked_explained", "Manifold-SAE (locked B) expl. variance", axes[1, 0]),
        ("curve_alive", "Manifold-SAE alive atoms (curve)", axes[1, 1]),
    ]
    cmap = plt.cm.tab10
    for idx, v in enumerate(variants):
        rows = sorted(v["rows"], key=lambda r: r.get("F", 0))
        Fs = [r["F"] for r in rows]
        color = cmap(idx % 10)
        label = v["run_name"].replace("llm_sweep_", "")
        for key, _, ax in metrics:
            ys = [r.get(key) for r in rows if r.get(key) is not None]
            xs = [r["F"] for r in rows if r.get(key) is not None]
            if not ys:
                continue
            ax.plot(xs, ys, "o-", color=color, label=label, linewidth=1.5, markersize=7)

    for key, title, ax in metrics:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("dictionary size F")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if key.endswith("_alive"):
            ax.set_ylabel("alive atoms")
        else:
            ax.set_ylabel("explained variance")
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Manifold-SAE sweep variants — cross-comparison",
                  fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
