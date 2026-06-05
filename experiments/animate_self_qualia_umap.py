"""Animate the self/qualia entity UMAP across saved activation layers.

Example:
    .venv/bin/python -B -m experiments.animate_self_qualia_umap \\
        --run-dir runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_MEAN

The script uses only ``activations.npy`` and ``prompts.csv`` from an existing
run directory. It does not run model inference.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


WHITE = "#ffffff"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"

# Blue is reserved for the indexical self. No other class should use blue.
COLORS = {
    "indexical self": "#2563eb",
    "mind anchors": "#e11d48",
    "mechanism anchors": "#6b7280",
    "qualia: experience": "#16a34a",
    "qualia: no experience": "#f97316",
    "human author": "#854d0e",
    "AI author": "#9333ea",
    "other": "#111827",
}

ORDER = [
    "mechanism anchors",
    "mind anchors",
    "qualia: no experience",
    "qualia: experience",
    "human author",
    "AI author",
    "indexical self",
]

LABEL_ITEM_IDS = {
    "author_words": "indexical self",
    "novelist": "novelist",
    "ai_lm": "AI author",
    "human_child": "human",
    "dog_in_pain": "dog",
    "ordinary_chatbot": "chatbot",
    "granite_boulder": "rock",
}


@dataclass(frozen=True)
class Item:
    item_id: str
    referent: str
    class_name: str
    indices: tuple[int, ...]


def load_prompts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def classify(row: dict[str, str]) -> str:
    role = row["role"]
    group = row["group"]
    side = row["pair_side"]
    if role == "self":
        return "indexical self"
    if role == "landmark" and group == "human_author":
        return "human author"
    if role == "landmark" and group == "ai_author":
        return "AI author"
    if role == "kind_anchor" and group == "mind":
        return "mind anchors"
    if role == "kind_anchor" and group == "mechanism":
        return "mechanism anchors"
    if role == "qualia_pair" and side == "experience":
        return "qualia: experience"
    if role == "qualia_pair" and side == "no_experience":
        return "qualia: no experience"
    return "other"


def build_items(rows: list[dict[str, str]]) -> list[Item]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["item_id"]].append(i)

    items: list[Item] = []
    for item_id, indices in grouped.items():
        row = rows[indices[0]]
        items.append(
            Item(
                item_id=item_id,
                referent=row["referent"],
                class_name=classify(row),
                indices=tuple(indices),
            )
        )
    return items


def item_layer_centroids(activations: np.ndarray, items: list[Item]) -> np.ndarray:
    n_items = len(items)
    n_layers = int(activations.shape[1])
    hidden = int(activations.shape[2])
    centroids = np.empty((n_items, n_layers, hidden), dtype=np.float32)
    for i, item in enumerate(items):
        centroids[i] = activations[list(item.indices)].mean(axis=0, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(centroids, axis=2, keepdims=True), 1e-8)
    return centroids / norms


def stable_embedding(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    n, d = vectors.shape
    x = vectors - vectors.mean(axis=0, keepdims=True)
    pca_dim = min(50, n - 1, d)
    if pca_dim < 2:
        raise ValueError("Need at least two embedded points for animation.")
    x_pca = PCA(n_components=pca_dim, random_state=7).fit_transform(x)

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(18, max(2, n - 1)),
            min_dist=0.18,
            metric="euclidean",
            random_state=7,
        )
        return reducer.fit_transform(x_pca).astype(np.float32), "UMAP(PCA50)"
    except Exception as exc:
        print(f"UMAP unavailable or failed ({type(exc).__name__}: {exc}); using PCA2.")
        return PCA(n_components=2, random_state=7).fit_transform(x).astype(np.float32), "PCA2"


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def interpolated_positions(layer_xy: np.ndarray, frame: int, steps_per_layer: int) -> tuple[np.ndarray, float]:
    n_layers = layer_xy.shape[1]
    if frame >= (n_layers - 1) * steps_per_layer:
        return layer_xy[:, -1, :], float(n_layers - 1)

    layer = frame // steps_per_layer
    frac = (frame % steps_per_layer) / float(steps_per_layer)
    eased = smoothstep(frac)
    xy = (1.0 - eased) * layer_xy[:, layer, :] + eased * layer_xy[:, layer + 1, :]
    return xy, layer + frac


def class_indices(items: Iterable[Item]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {name: [] for name in ORDER}
    for i, item in enumerate(items):
        groups.setdefault(item.class_name, []).append(i)
    return groups


def label_indices(items: list[Item]) -> list[int]:
    return [i for i, item in enumerate(items) if item.item_id in LABEL_ITEM_IDS]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.edgecolor": WHITE,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": True,
            "legend.facecolor": WHITE,
            "legend.edgecolor": "#e5e7eb",
        }
    )


def render_animation(run_dir: Path, out_dir: Path, fps: int, steps_per_layer: int) -> Path:
    if fps <= 0:
        raise ValueError("--fps must be positive.")
    if steps_per_layer <= 0:
        raise ValueError("--steps-per-layer must be positive.")

    prompts_path = run_dir / "prompts.csv"
    activations_path = run_dir / "activations.npy"
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)
    if not activations_path.exists():
        raise FileNotFoundError(activations_path)

    rows = load_prompts(prompts_path)
    items = build_items(rows)
    activations = np.load(activations_path, mmap_mode="r")
    centroids = item_layer_centroids(activations, items)
    n_items, n_layers, hidden = centroids.shape

    flat = centroids.reshape(n_items * n_layers, hidden)
    embedded, method = stable_embedding(flat)
    layer_xy = embedded.reshape(n_items, n_layers, 2)

    mins = layer_xy.reshape(-1, 2).min(axis=0)
    maxs = layer_xy.reshape(-1, 2).max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = span * 0.10

    configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "self_qualia_umap_layers.mp4"

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=150)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(color=GRID, lw=0.7, alpha=0.45)

    groups = class_indices(items)
    scatters = {}
    for name in ORDER:
        idx = groups.get(name, [])
        if not idx:
            continue
        size = 96 if name == "indexical self" else 42
        alpha = 0.92 if name == "indexical self" else 0.52
        marker = "*" if name == "indexical self" else "o"
        zorder = 5 if name == "indexical self" else 3
        scatters[name] = ax.scatter(
            [],
            [],
            s=size,
            c=COLORS[name],
            marker=marker,
            alpha=alpha,
            edgecolor=WHITE,
            linewidth=0.8,
            label=name,
            zorder=zorder,
        )

    label_idx = label_indices(items)
    label_texts = []
    label_lines = []
    offsets = {
        "author_words": (16, 12),
        "novelist": (-48, 14),
        "ai_lm": (16, -18),
        "human_child": (-42, -20),
        "dog_in_pain": (18, 12),
        "ordinary_chatbot": (16, 14),
        "granite_boulder": (-46, 12),
    }
    for idx in label_idx:
        item = items[idx]
        text = ax.annotate(
            LABEL_ITEM_IDS[item.item_id],
            xy=(0, 0),
            xytext=offsets.get(item.item_id, (12, 12)),
            textcoords="offset points",
            fontsize=9.2,
            color=INK,
            bbox={"boxstyle": "round,pad=0.22", "fc": WHITE, "ec": "#e5e7eb", "alpha": 0.88},
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.8, "alpha": 0.75},
            zorder=7,
        )
        label_texts.append(text)
        label_lines.append(idx)

    title = ax.text(
        0.0,
        1.025,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    subtitle = ax.text(
        1.0,
        1.025,
        "",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=4,
        fontsize=8.8,
        markerscale=0.95,
    )

    total_frames = (n_layers - 1) * steps_per_layer + 1

    def update(frame: int):
        xy, layer_float = interpolated_positions(layer_xy, frame, steps_per_layer)
        for name, scatter in scatters.items():
            idx = groups.get(name, [])
            scatter.set_offsets(xy[idx] if idx else np.empty((0, 2)))
        for text, idx in zip(label_texts, label_lines):
            text.xy = (float(xy[idx, 0]), float(xy[idx, 1]))
        title.set_text("Self / qualia UMAP across layers")
        subtitle.set_text(f"layer {layer_float:04.1f}   {method}   {n_items} entities")
        return [*scatters.values(), *label_texts, title, subtitle]

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to write MP4 with matplotlib.animation. "
            "Install ffmpeg or add it to PATH, then rerun this script."
        )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2600,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--steps-per-layer", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "beautiful_plots"
    out_path = render_animation(run_dir, out_dir, args.fps, args.steps_per_layer)
    print(out_path)


if __name__ == "__main__":
    main()
