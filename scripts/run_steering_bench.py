"""Run the SAE Steering Benchmark across one or more checkpoints.

Usage:

  python scripts/run_steering_bench.py \
      --models runs/sae_comparison/model_topk.pt \
               runs/sae_comparison/model_l1.pt \
               runs/sae_comparison/model_manifold.pt \
      --data   runs/COLOR_COGITO_L40/X_L40.npy \
      --output runs/steering_bench/

Outputs:
  <output>/<model_name>.json       per-model raw scores
  <output>/leaderboard.json        combined comparison
  <output>/leaderboard.md          ranked table
  <output>/radar.png               radar plot
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from manifold_sae.eval.registry import loader_for
from manifold_sae.eval.run import prepare
from manifold_sae.eval.steering_bench import SteeringBench, BenchResult


def expand_globs(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        matched = glob.glob(p)
        if matched:
            out.extend(matched)
        else:
            out.append(p)
    return sorted(set(out))


def run(models: list[str], data: str, output: str,
        device: str = "cpu", max_val_rows: int = 1500) -> dict:
    out_dir = Path(output); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    X_train_np, X_val, labels, D = prepare(
        root, Path(data), device=device, max_val_rows=max_val_rows
    )
    if labels.color_hsv is None or labels.row_color_idx is None:
        raise RuntimeError("prepare() didn't return color labels; needed for bench.")
    hsv = labels.color_hsv[labels.row_color_idx]
    name_labels = labels.row_color_idx

    all_results: dict[str, dict] = {}
    for path in models:
        try:
            loader = loader_for(path)
        except KeyError as e:
            print(f"[bench] skipping {path}: {e}", flush=True)
            continue
        print(f"[bench] loading {path}", flush=True)
        try:
            wrapper = loader(path, d_in=D, device=device)
        except Exception as e:
            print(f"[bench] load failed for {path}: {e}", flush=True)
            continue
        bench = SteeringBench(wrapper, X_val, hsv_labels=hsv, name_labels=name_labels)
        print(f"[bench] scoring {wrapper.name}", flush=True)
        res = bench.run()
        d = res.to_dict()
        all_results[wrapper.name] = d
        with open(out_dir / f"{wrapper.name}.json", "w") as f:
            json.dump(d, f, indent=2, default=float)

    leaderboard = {"per_model": all_results,
                   "ranking": _rank(all_results)}
    with open(out_dir / "leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2, default=float)
    _render_md(leaderboard, out_dir / "leaderboard.md")
    try:
        _render_radar(all_results, out_dir / "radar.png")
    except Exception as e:
        print(f"[bench] radar failed: {e}", flush=True)
    print(f"[bench] wrote {out_dir}/", flush=True)
    return leaderboard


def _rank(per_model: dict) -> dict:
    protos = ["linear_push", "anchor_swap", "magnitude_scaling", "compositional"]
    rank: dict[str, list] = {}
    for p in protos:
        scored = []
        for name, d in per_model.items():
            r = d["protocols"].get(p, {}).get("steering_r2", float("-inf"))
            scored.append((name, r))
        scored.sort(key=lambda kv: -kv[1])
        rank[p] = scored
    # composite
    comp = [(name, d["summary"]["composite"]) for name, d in per_model.items()]
    comp.sort(key=lambda kv: -kv[1])
    rank["composite"] = comp
    return rank


def _render_md(lb: dict, path: Path) -> None:
    protos = ["linear_push", "anchor_swap", "magnitude_scaling", "compositional"]
    lines = ["# SAE Steering Benchmark Leaderboard\n"]
    lines.append("Higher steering_r2 = better. Lower side_effect = better.\n")
    lines.append("\n## Per-protocol R² ranking\n")
    for p in protos:
        lines.append(f"\n### {p}\n")
        lines.append("| Rank | Model | steering_r² | side_effect | monotonicity |\n")
        lines.append("|---|---|---|---|---|\n")
        scored = lb["ranking"][p]
        for i, (name, _) in enumerate(scored, 1):
            d = lb["per_model"][name]["protocols"][p]
            lines.append(f"| {i} | {name} | {d['steering_r2']:.3f}"
                         f" | {d['side_effect_norm']:.3f}"
                         f" | {d['monotonicity']:.3f} |\n")
    lines.append("\n## Composite\n")
    lines.append("| Rank | Model | composite |\n|---|---|---|\n")
    for i, (name, v) in enumerate(lb["ranking"]["composite"], 1):
        lines.append(f"| {i} | {name} | {v:.3f} |\n")
    path.write_text("".join(lines))


def _render_radar(per_model: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protos = ["linear_push", "anchor_swap", "magnitude_scaling", "compositional"]
    angles = np.linspace(0, 2 * np.pi, len(protos), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    for name, d in per_model.items():
        vals = [max(0.0, min(1.0, d["protocols"][p]["steering_r2"])) for p in protos]
        vals += vals[:1]
        ax.plot(angles, vals, label=name, lw=2)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(protos)
    ax.set_ylim(0, 1)
    ax.set_title("Steering Bench (radar; higher = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="runs/COLOR_COGITO_L40/X_L40.npy")
    ap.add_argument("--output", default="runs/steering_bench/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-val-rows", type=int, default=1500)
    args = ap.parse_args()
    models = expand_globs(args.models)
    run(models, args.data, args.output, device=args.device,
        max_val_rows=args.max_val_rows)


if __name__ == "__main__":
    main()
