"""Compare two self/qualia runs, usually last-token vs mean-pool readouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


WHITE = "#ffffff"
INK = "#111827"
GRID = "#d9dee7"
COL_LAST = "#2563eb"
COL_MEAN = "#d97706"
COL_HUMAN = "#059669"
COL_AI = "#64748b"
PREFIX = "last_token_vs_mean_pool"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titleweight": "bold",
            "legend.frameon": True,
            "legend.facecolor": WHITE,
            "legend.edgecolor": "#e5e7eb",
        }
    )


def load_run(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text())
    meta = json.loads((path / "run_meta.json").read_text())
    rows = list(csv.DictReader((path / "layers.csv").open()))
    b = summary["best_layer_metrics"]
    return {
        "path": str(path),
        "pooling": meta.get("pooling", path.name),
        "model": meta.get("model"),
        "revision": meta.get("revision"),
        "n_prompts": summary["n_prompts"],
        "n_layers": summary["n_layers"],
        "hidden_dim": summary["hidden_dim"],
        "layer": summary["best_layer"],
        "layer_selection": summary.get("layer_selection", {}),
        "metrics": b,
        "rows": rows,
    }


def write_outputs(last: dict[str, Any], mean: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = [
        "kind_auc",
        "qualia_auc",
        "qualia_pair_acc",
        "axis_cosine_kind_qualia",
        "self_kind_coord",
        "self_qualia_coord",
        "human_author_kind_coord",
        "human_author_qualia_coord",
        "ai_author_kind_coord",
        "ai_author_qualia_coord",
        "self_cos_mind",
        "self_cos_mechanism",
        "self_cos_human_author",
        "self_cos_ai_author",
    ]
    rows = []
    for run in [last, mean]:
        row = {
            "pooling": run["pooling"],
            "model": run["model"],
            "revision": run["revision"],
            "layer": run["layer"],
            "n_prompts": run["n_prompts"],
            "n_layers": run["n_layers"],
            "hidden_dim": run["hidden_dim"],
        }
        for key in keys:
            row[key] = run["metrics"][key]
        rows.append(row)

    for path in [
        out_dir / "pooling_comparison.csv",
        out_dir / "pooling_comparison.json",
        out_dir / "pooling_comparison_plane.png",
        out_dir / "pooling_comparison_bars.png",
        out_dir / "pooling_comparison_depth.png",
        out_dir / f"{PREFIX}_pooling_comparison.csv",
        out_dir / f"{PREFIX}_pooling_comparison.json",
        out_dir / f"{PREFIX}_pooling_comparison_plane.png",
        out_dir / f"{PREFIX}_pooling_comparison_bars.png",
        out_dir / f"{PREFIX}_pooling_comparison_depth.png",
    ]:
        if path.exists():
            path.unlink()

    with (out_dir / f"{PREFIX}_pooling_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    comparison = {
        "last": {k: v for k, v in last.items() if k != "rows"},
        "mean": {k: v for k, v in mean.items() if k != "rows"},
        "delta_last_minus_mean": {
            key: float(last["metrics"][key]) - float(mean["metrics"][key]) for key in keys
        },
    }
    (out_dir / f"{PREFIX}_pooling_comparison.json").write_text(json.dumps(comparison, indent=2))

    configure_style()
    _plot_plane(last, mean, out_dir)
    _plot_bars(last, mean, out_dir)
    _plot_depth(last, mean, out_dir)


def _plot_plane(last: dict[str, Any], mean: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for val in [0, 0.5, 1]:
        ax.axhline(val, color=GRID, lw=1)
        ax.axvline(val, color=GRID, lw=1)
    points = [
        (last, COL_LAST, "last-token self", "self_kind_coord", "self_qualia_coord", "o", 180),
        (mean, COL_MEAN, "mean-pool self", "self_kind_coord", "self_qualia_coord", "o", 180),
        (last, COL_HUMAN, "human author", "human_author_kind_coord", "human_author_qualia_coord", "s", 120),
        (last, COL_AI, "AI author", "ai_author_kind_coord", "ai_author_qualia_coord", "^", 130),
    ]
    for run, color, label, kx, ky, marker, size in points:
        m = run["metrics"]
        ax.scatter([m[kx]], [m[ky]], color=color, marker=marker, s=size, alpha=0.88,
                   edgecolor="white", linewidth=1.4, label=label)
    ax.plot(
        [last["metrics"]["self_kind_coord"], mean["metrics"]["self_kind_coord"]],
        [last["metrics"]["self_qualia_coord"], mean["metrics"]["self_qualia_coord"]],
        color="#9ca3af",
        lw=1.4,
        alpha=0.8,
    )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.18)
    ax.set_xlabel("kind coordinate")
    ax.set_ylabel("qualia coordinate")
    ax.set_title(f"Pooling comparison at fixed layer {last['layer']}")
    ax.legend(loc="lower right")
    fig.savefig(out_dir / f"{PREFIX}_pooling_comparison_plane.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bars(last: dict[str, Any], mean: dict[str, Any], out_dir: Path) -> None:
    labels = [
        "self kind",
        "self qualia",
        "human qualia",
        "AI qualia",
        "kind AUC",
        "qualia AUC",
    ]
    keys = [
        "self_kind_coord",
        "self_qualia_coord",
        "human_author_qualia_coord",
        "ai_author_qualia_coord",
        "kind_auc",
        "qualia_auc",
    ]
    last_vals = np.asarray([float(last["metrics"][key]) for key in keys])
    mean_vals = np.asarray([float(mean["metrics"][key]) for key in keys])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.bar(x - width / 2, last_vals, width, color=COL_LAST, alpha=0.78, label="last token")
    ax.bar(x + width / 2, mean_vals, width, color=COL_MEAN, alpha=0.72, label="mean pool")
    ax.axhline(0, color=GRID, lw=1)
    ax.axhline(1, color=GRID, lw=1)
    ax.set_ylim(0, 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("coordinate / score")
    ax.set_title(f"Readout comparison at fixed layer {last['layer']}")
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.72)
    ax.legend(loc="upper right")
    fig.savefig(out_dir / f"{PREFIX}_pooling_comparison_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _layer_array(run: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in run["rows"]])


def _plot_depth(last: dict[str, Any], mean: dict[str, Any], out_dir: Path) -> None:
    layers = np.asarray([int(r["layer"]) for r in last["rows"]])
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.2), sharex=True)
    for ax, key, title in [
        (axes[0], "self_kind_coord", "Self kind coordinate"),
        (axes[1], "self_qualia_coord", "Self qualia coordinate"),
    ]:
        ax.plot(layers, _layer_array(last, key), color=COL_LAST, lw=2.4, alpha=0.86,
                label="last token")
        ax.plot(layers, _layer_array(mean, key), color=COL_MEAN, lw=2.4, alpha=0.86,
                label="mean pool")
        ax.axvline(last["layer"], color=INK, lw=1.1, ls="--", alpha=0.42)
        ax.axhline(0, color=GRID, lw=1)
        ax.axhline(1, color=GRID, lw=1)
        ax.grid(color=GRID, lw=0.8, alpha=0.72)
        ax.set_title(title, loc="left")
        ax.set_ylabel(key.replace("_", " "))
    axes[1].set_xlabel("layer")
    axes[0].legend(loc="lower right")
    fig.savefig(out_dir / f"{PREFIX}_pooling_comparison_depth.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--last-dir", required=True)
    ap.add_argument("--mean-dir", required=True)
    ap.add_argument("--out-dir", default="runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_COMPARISON")
    args = ap.parse_args()
    write_outputs(load_run(Path(args.last_dir)), load_run(Path(args.mean_dir)), Path(args.out_dir))
    print(f"wrote comparison to {args.out_dir}")


if __name__ == "__main__":
    main()
