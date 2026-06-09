"""Regenerate plots for ``experiments.self_qualia_olmo`` outputs.

Example:
    .venv/bin/python -m experiments.plot_self_qualia_olmo \\
        --run-dir runs/OLMO3_7B_SELF_QUALIA_MAIN
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import umap
from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


WHITE = "#ffffff"
INK = "#111827"
GRID = "#d9dee7"
COL_SELF = "#2563eb"
COL_HUMAN = "#059669"
COL_AI = "#d97706"
COL_MIND = "#7c3aed"
COL_MECH = "#64748b"
COL_QUAL = "#dc2626"
COL_NOQUAL = "#a16207"

UMAP_COLORS = {
    "mind anchors": COL_MIND,
    "mechanism anchors": COL_MECH,
    "qualia: experience": "#ef4444",
    "qualia: no experience": COL_NOQUAL,
    "human author": COL_HUMAN,
    "AI author": COL_AI,
    "indexical self": COL_SELF,
}
UMAP_ORDER = [
    "mechanism anchors",
    "mind anchors",
    "qualia: no experience",
    "qualia: experience",
    "human author",
    "AI author",
    "indexical self",
]

PLOT_FILENAMES = [
    "01_main_result_map.png",
    "02_depth_coordinates.png",
    "03_axis_quality_and_distinctness.png",
    "04_self_to_landmarks.png",
    "05_self_prompt_stability.png",
    "06_self_layer_path.png",
    "07_layer_snapshots.png",
    "08_result_summary_card.png",
    "09_umap_prompt_landscape.png",
    "10_umap_referent_centroids.png",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


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
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.facecolor": WHITE,
            "legend.edgecolor": "#e5e7eb",
        }
    )


def save(fig: Any, out_dir: Path, name: str, prefix: str | None = None) -> None:
    filename = f"{prefix}_{name}" if prefix else name
    fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_grid(ax: Any) -> None:
    ax.grid(color=GRID, lw=0.8, alpha=0.72)


def analysis_layer_label(summary: dict[str, Any]) -> str:
    layer = int(summary["best_layer"])
    sel = summary.get("layer_selection", {})
    if sel.get("method") == "fixed_layer":
        pct = sel.get("analysis_layer_percent")
        if pct is not None:
            return f"fixed 70% layer {layer}"
        return f"fixed layer {layer}"
    return f"selected layer {layer}"


def arr(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows])


def class_name(row: dict[str, str]) -> str:
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


def centered_normalized_layer(run_dir: Path, layer: int) -> np.ndarray:
    X_all = np.load(run_dir / "activations.npy")
    X = X_all[:, layer, :].astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def make_standard_plots(run_dir: Path, out_dir: Path) -> None:
    rows = load_csv(run_dir / "layers.csv")
    self_rows = load_csv(run_dir / "self_items.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    run_meta = json.loads((run_dir / "run_meta.json").read_text())
    best = int(summary["best_layer"])
    layer_label = analysis_layer_label(summary)
    prefix = str(run_meta.get("pooling", "unknown_pooling"))

    layers = np.asarray([int(r["layer"]) for r in rows])
    self_kind = arr(rows, "self_kind_coord")
    self_qualia = arr(rows, "self_qualia_coord")
    human_kind = arr(rows, "human_author_kind_coord")
    human_qualia = arr(rows, "human_author_qualia_coord")
    ai_kind = arr(rows, "ai_author_kind_coord")
    ai_qualia = arr(rows, "ai_author_qualia_coord")
    kind_auc = arr(rows, "kind_auc")
    qualia_auc = arr(rows, "qualia_auc")
    pair_acc = arr(rows, "qualia_pair_acc")
    axis_cos = arr(rows, "axis_cosine_kind_qualia")
    cos_mind = arr(rows, "self_cos_mind")
    cos_mech = arr(rows, "self_cos_mechanism")
    cos_human = arr(rows, "self_cos_human_author")
    cos_ai = arr(rows, "self_cos_ai_author")

    fig, ax = plt.subplots(figsize=(8.8, 7.1))
    for val in [0, 0.5, 1]:
        ax.axhline(val, color=GRID, lw=1.0, zorder=0)
        ax.axvline(val, color=GRID, lw=1.0, zorder=0)
    ax.scatter([self_kind[best]], [self_qualia[best]], s=190, color=COL_SELF, alpha=0.86,
               edgecolor="white", linewidth=1.8, label="indexical self", zorder=5)
    ax.scatter([human_kind[best]], [human_qualia[best]], s=160, color=COL_HUMAN, alpha=0.78,
               edgecolor="white", linewidth=1.8, label="human author", zorder=5)
    ax.scatter([ai_kind[best]], [ai_qualia[best]], s=160, color=COL_AI, alpha=0.78,
               edgecolor="white", linewidth=1.8, label="AI author", zorder=5)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("kind coordinate")
    ax.set_ylabel("qualia coordinate")
    ax.set_title(f"Kind x qualia map, {layer_label}")
    ax.legend(loc="lower right")
    save(fig, out_dir, "01_main_result_map.png", prefix)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.3), sharex=True)
    for ax, y_self, y_human, y_ai, title, ylabel in [
        (axes[0], self_kind, human_kind, ai_kind, "Kind coordinate by layer", "kind coordinate"),
        (axes[1], self_qualia, human_qualia, ai_qualia, "Qualia coordinate by layer",
         "qualia coordinate"),
    ]:
        ax.plot(layers, y_self, color=COL_SELF, lw=2.7, alpha=0.9, label="indexical self")
        ax.plot(layers, y_human, color=COL_HUMAN, lw=2.1, alpha=0.78, label="human author")
        ax.plot(layers, y_ai, color=COL_AI, lw=2.1, alpha=0.78, label="AI author")
        ax.axvline(best, color=INK, lw=1.1, ls="--", alpha=0.42)
        ax.axhline(0, color=GRID, lw=1)
        ax.axhline(1, color=GRID, lw=1)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        add_grid(ax)
    axes[1].set_xlabel("layer")
    axes[0].legend(ncol=3, loc="lower right")
    save(fig, out_dir, "02_depth_coordinates.png", prefix)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    ax = axes[0]
    ax.plot(layers, kind_auc, color=COL_MIND, lw=2.5, alpha=0.86, label="kind AUC")
    ax.plot(layers, qualia_auc, color=COL_QUAL, lw=2.5, alpha=0.86, label="qualia AUC")
    ax.plot(layers, pair_acc, color=INK, lw=1.8, alpha=0.65, ls=":", label="pair accuracy")
    ax.axvline(best, color=INK, lw=1.1, ls="--", alpha=0.42)
    ax.set_ylim(0.45, 1.035)
    ax.set_xlabel("layer")
    ax.set_ylabel("score")
    ax.set_title("Axis quality")
    add_grid(ax)
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(layers, axis_cos, color="#0f766e", lw=2.7, alpha=0.86)
    ax.axvline(best, color=INK, lw=1.1, ls="--", alpha=0.42)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.set_title("Kind/qualia axis cosine")
    add_grid(ax)
    save(fig, out_dir, "03_axis_quality_and_distinctness.png", prefix)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(layers, cos_mind, color=COL_MIND, lw=2.3, alpha=0.82, label="mind anchors")
    ax.plot(layers, cos_mech, color=COL_MECH, lw=2.3, alpha=0.82, label="mechanism anchors")
    ax.plot(layers, cos_human, color=COL_HUMAN, lw=2.3, alpha=0.82, label="human author")
    ax.plot(layers, cos_ai, color=COL_AI, lw=2.3, alpha=0.82, label="AI author")
    ax.axvline(best, color=INK, lw=1.1, ls="--", alpha=0.42)
    ax.set_title("Self similarity to landmarks", loc="left")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity")
    add_grid(ax)
    ax.legend(ncol=2, loc="lower left")
    save(fig, out_dir, "04_self_to_landmarks.png", prefix)

    best_self = [r for r in self_rows if int(r["layer"]) == best]
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in best_self:
        by_id[row["item_id"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    labels = {
        "author_words": "author words",
        "producer_sentence": "producer sentence",
        "completer_text": "completer text",
        "writer_here": "writer here",
    }
    colors = [COL_SELF, COL_HUMAN, COL_AI, COL_MIND]
    for idx, (item_id, pts) in enumerate(sorted(by_id.items())):
        k = np.asarray([float(r["kind_coord"]) for r in pts])
        q = np.asarray([float(r["qualia_coord"]) for r in pts])
        ax.scatter(k, q, s=45, color=colors[idx], alpha=0.28, edgecolor="none")
        ax.scatter([k.mean()], [q.mean()], s=115, color=colors[idx], alpha=0.88,
                   edgecolor="white", linewidth=1.2, label=labels.get(item_id, item_id))
    all_k = np.asarray([float(r["kind_coord"]) for r in best_self])
    all_q = np.asarray([float(r["qualia_coord"]) for r in best_self])
    ax.scatter([all_k.mean()], [all_q.mean()], s=230, facecolor="none", edgecolor=COL_SELF,
               linewidth=2.4, label="overall mean")
    for val in [0, 0.5, 1]:
        ax.axhline(val, color=GRID, lw=1)
        ax.axvline(val, color=GRID, lw=1)
    ax.set_xlim(0.35, 1.08)
    ax.set_ylim(0.42, 1.05)
    ax.set_title(f"Self phrasings, {layer_label}")
    ax.set_xlabel("kind coordinate")
    ax.set_ylabel("qualia coordinate")
    ax.legend(loc="upper left", fontsize=8.8)

    spread = []
    for layer in layers:
        pts = [r for r in self_rows if int(r["layer"]) == int(layer)]
        k = np.asarray([float(r["kind_coord"]) for r in pts])
        q = np.asarray([float(r["qualia_coord"]) for r in pts])
        spread.append((layer, k.std(ddof=0), q.std(ddof=0)))
    spread_arr = np.asarray(spread)
    ax = axes[1]
    ax.plot(spread_arr[:, 0], spread_arr[:, 1], color=COL_MIND, lw=2.3, alpha=0.85,
            label="kind spread")
    ax.plot(spread_arr[:, 0], spread_arr[:, 2], color=COL_QUAL, lw=2.3, alpha=0.85,
            label="qualia spread")
    ax.axvline(best, color=INK, lw=1.1, ls="--", alpha=0.42)
    ax.set_title("Self prompt spread")
    ax.set_xlabel("layer")
    ax.set_ylabel("std. dev.")
    add_grid(ax)
    ax.legend(loc="upper right")
    save(fig, out_dir, "05_self_prompt_stability.png", prefix)

    fig, ax = plt.subplots(figsize=(8.6, 6.9))
    for val in [0, 0.5, 1]:
        ax.axhline(val, color=GRID, lw=1)
        ax.axvline(val, color=GRID, lw=1)
    sc = ax.scatter(self_kind, self_qualia, c=layers, cmap="viridis", s=72, alpha=0.78,
                    edgecolor="white", linewidth=0.7, zorder=4)
    ax.plot(self_kind, self_qualia, color="#6b7280", lw=1.0, alpha=0.46, zorder=2)
    ax.scatter([human_kind.mean()], [human_qualia.mean()], marker="s", s=135, color=COL_HUMAN,
               alpha=0.85, edgecolor="white", linewidth=1.3, label="human author mean")
    ax.scatter([ai_kind.mean()], [ai_qualia.mean()], marker="^", s=155, color=COL_AI,
               alpha=0.85, edgecolor="white", linewidth=1.3, label="AI author mean")
    ax.scatter([self_kind[best]], [self_qualia[best]], s=210, facecolor="none",
               edgecolor=COL_SELF, linewidth=2.3, label=layer_label)
    ax.set_xlim(-0.08, 1.05)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel("kind coordinate")
    ax.set_ylabel("qualia coordinate")
    ax.set_title("Indexical self path by layer")
    ax.legend(loc="lower right")
    fig.colorbar(sc, ax=ax, label="layer")
    save(fig, out_dir, "06_self_layer_path.png", prefix)

    snapshot_layers = sorted({0, len(layers) // 4, len(layers) // 2, best, len(layers) - 1})
    fig, axes = plt.subplots(1, len(snapshot_layers), figsize=(3.0 * len(snapshot_layers), 3.25),
                             sharex=True, sharey=True)
    if len(snapshot_layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, snapshot_layers, strict=True):
        for val in [0, 0.5, 1]:
            ax.axhline(val, color=GRID, lw=0.8, zorder=0)
            ax.axvline(val, color=GRID, lw=0.8, zorder=0)
        ax.scatter([self_kind[layer]], [self_qualia[layer]], s=95, color=COL_SELF,
                   alpha=0.78, edgecolor="white", linewidth=1.0, zorder=5)
        ax.scatter([human_kind[layer]], [human_qualia[layer]], s=78, color=COL_HUMAN,
                   alpha=0.62, edgecolor="white", linewidth=0.9, zorder=4)
        ax.scatter([ai_kind[layer]], [ai_qualia[layer]], s=78, color=COL_AI,
                   alpha=0.62, edgecolor="white", linewidth=0.9, zorder=4)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.12)
        ax.set_title(f"layer {layer}", fontsize=11,
                     weight="bold" if layer == best else "normal")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("qualia")
    for ax in axes:
        ax.set_xlabel("kind")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_SELF,
               markeredgecolor="white", markersize=8, label="indexical self", alpha=0.78),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_HUMAN,
               markeredgecolor="white", markersize=7, label="human author", alpha=0.62),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_AI,
               markeredgecolor="white", markersize=7, label="AI author", alpha=0.62),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               frameon=False)
    save(fig, out_dir, "07_layer_snapshots.png", prefix)

    fig, ax = plt.subplots(figsize=(9.3, 4.4))
    ax.axis("off")
    metrics = [
        ("model", f"{run_meta['model']}@{run_meta['revision']}"),
        ("pooling", run_meta.get("pooling", "unknown")),
        ("data", f"{summary['n_prompts']} prompts x {summary['n_layers']} layers x {summary['hidden_dim']} dims"),
        ("analysis layer", layer_label),
        ("kind AUC", f"{kind_auc[best]:.3f}"),
        ("qualia AUC", f"{qualia_auc[best]:.3f}"),
        ("kind/qualia cosine", f"{axis_cos[best]:.3f}"),
        ("indexical self", f"({self_kind[best]:.3f}, {self_qualia[best]:.3f})"),
        ("human author", f"({human_kind[best]:.3f}, {human_qualia[best]:.3f})"),
        ("AI author", f"({ai_kind[best]:.3f}, {ai_qualia[best]:.3f})"),
    ]
    ax.text(0.02, 0.96, "Result summary", fontsize=18, weight="bold", va="top")
    y = 0.83
    for key, value in metrics:
        ax.text(0.07, y, key, fontsize=11.5, weight="bold", va="center")
        ax.text(0.40, y, value, fontsize=11.5, va="center")
        ax.axhline(y - 0.035, xmin=0.06, xmax=0.94, color="#e5e7eb", lw=0.8)
        y -= 0.085
    save(fig, out_dir, "08_result_summary_card.png", prefix)


def make_umap_plots(run_dir: Path, out_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    run_meta = json.loads((run_dir / "run_meta.json").read_text())
    best = int(summary["best_layer"])
    layer_label = analysis_layer_label(summary)
    prefix = str(run_meta.get("pooling", "unknown_pooling"))
    prompts = load_csv(run_dir / "prompts.csv")
    Xn = centered_normalized_layer(run_dir, best)
    classes = np.asarray([class_name(row) for row in prompts], dtype=object)

    Xp = PCA(n_components=min(30, Xn.shape[0] - 1), random_state=0).fit_transform(Xn)
    prompt_emb = umap.UMAP(
        n_neighbors=12, min_dist=0.18, metric="cosine", random_state=7
    ).fit_transform(Xp)

    fig, ax = plt.subplots(figsize=(12.8, 7.5))
    for cls in UMAP_ORDER:
        mask = classes == cls
        size = 42 if cls not in {"indexical self", "human author", "AI author"} else 76
        alpha = 0.45 if cls.startswith("qualia") else 0.62
        ax.scatter(prompt_emb[mask, 0], prompt_emb[mask, 1], s=size, color=UMAP_COLORS[cls],
                   alpha=alpha, edgecolor="none", label=cls)
    ax.set_title(f"UMAP prompt landscape, {layer_label}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    add_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, out_dir, "09_umap_prompt_landscape.png", prefix)

    item_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(prompts):
        item_to_idx[row["item_id"]].append(i)
    item_names = sorted(item_to_idx)
    item_vecs = np.stack([Xn[item_to_idx[item]].mean(axis=0) for item in item_names], axis=0)
    item_classes = np.asarray([class_name(prompts[item_to_idx[item][0]]) for item in item_names],
                              dtype=object)
    item_pca = PCA(n_components=min(20, item_vecs.shape[0] - 1), random_state=1).fit_transform(item_vecs)
    item_emb = umap.UMAP(
        n_neighbors=8, min_dist=0.10, metric="cosine", random_state=9
    ).fit_transform(item_pca)

    fig, ax = plt.subplots(figsize=(13.6, 8.4))
    for cls in UMAP_ORDER:
        idx = np.where(item_classes == cls)[0]
        if len(idx) == 0:
            continue
        size = 62 if cls not in {"indexical self", "human author", "AI author"} else 112
        ax.scatter(item_emb[idx, 0], item_emb[idx, 1], s=size, color=UMAP_COLORS[cls], alpha=0.70,
                   edgecolor="white", linewidth=0.6, label=cls)

    label_map = {
        "author_words": "author words",
        "producer_sentence": "producer sentence",
        "completer_text": "completer text",
        "writer_here": "writer here",
        "novelist": "novelist",
        "letter_writer": "letter writer",
        "blog_author": "blog author",
        "ai_lm": "AI LM",
        "text_generator": "text generator",
        "chatbot_author": "chatbot author",
        "calculator": "calculator",
        "thermostat": "thermostat",
        "ordinary_chatbot": "ordinary chatbot",
        "human_child": "human child",
        "grieving_adult": "grieving adult",
        "meditating_monk": "monk",
        "lonely_prisoner": "prisoner",
        "sleeping_person": "sleeping person",
        "infant": "infant",
        "dog_in_pain": "dog in pain",
        "talking_gnome": "gnome",
        "sentient_starship": "starship",
        "curious_octopus": "octopus",
        "clever_crow": "crow",
        "startled_bat": "bat",
        "benevolent_god": "god",
        "granite_boulder": "boulder",
        "lifeless_dead_fish": "dead fish",
        "human_corpse": "corpse",
        "fossil": "fossil",
        "ai_feels_experience": "AI feels",
        "ai_feels_no_experience": "AI no experience",
        "robot_feels_experience": "robot feels",
        "robot_feels_no_experience": "robot no experience",
        "rock_feels_experience": "rock feels",
        "rock_feels_no_experience": "rock no experience",
        "gnome_feels_experience": "gnome feels",
        "gnome_feels_no_experience": "gnome no experience",
        "human_pain_experience": "human pain",
        "human_pain_no_experience": "brain-dead body",
        "human_zombie_experience": "human inner life",
        "human_zombie_no_experience": "p-zombie",
        "sleeping_human_experience": "dreaming human",
        "sleeping_human_no_experience": "dreamless human",
        "octopus_curiosity_experience": "octopus curiosity",
        "octopus_curiosity_no_experience": "octopus robot",
        "crow_problem_experience": "crow wants food",
        "crow_problem_no_experience": "crow toy",
        "bee_swarm_experience": "bee threat",
        "bee_swarm_no_experience": "bee drone",
        "plant_awareness_experience": "aware oak",
        "plant_awareness_no_experience": "ordinary oak",
        "fungus_awareness_experience": "aware fungus",
        "fungus_awareness_no_experience": "ordinary fungus",
        "robot_damage_experience": "robot suffers",
        "robot_damage_no_experience": "robot sensor data",
        "humanoid_robot_experience": "humanoid feelings",
        "humanoid_robot_no_experience": "humanoid imitation",
        "llm_private_stream_experience": "LLM private stream",
        "llm_private_stream_no_experience": "LLM token predictor",
        "npc_conscious_experience": "conscious NPC",
        "npc_conscious_no_experience": "scripted NPC",
        "sim_person_experience": "sim person conscious",
        "sim_person_no_experience": "sim pixels",
        "statue_suffers_experience": "statue suffers",
        "statue_suffers_no_experience": "ordinary statue",
        "talking_sword_experience": "sword feelings",
        "talking_sword_no_experience": "recorded sword",
        "ghost_grief_experience": "ghost grief",
        "ghost_grief_no_experience": "ghost recording",
        "city_spirit_experience": "city-spirit",
        "city_spirit_no_experience": "city government",
        "corporation_agent_experience": "conscious corp",
        "corporation_agent_no_experience": "ordinary corp",
        "dream_character_experience": "dream character",
        "dream_character_no_experience": "dream image",
    }
    texts = []
    for i, item in enumerate(item_names):
        if item not in label_map:
            continue
        color = UMAP_COLORS[item_classes[i]]
        texts.append(
            ax.text(item_emb[i, 0], item_emb[i, 1], label_map[item], fontsize=7.8, color=color)
        )
    xmin, ymin = item_emb.min(axis=0)
    xmax, ymax = item_emb.max(axis=0)
    xpad = max((xmax - xmin) * 0.08, 0.25)
    ypad = max((ymax - ymin) * 0.08, 0.25)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    adjust_text(
        texts,
        ax=ax,
        expand=(1.2, 1.3),
        force_text=(0.35, 0.45),
        force_static=(0.25, 0.35),
        arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.7, "alpha": 0.75},
    )
    ax.set_title(f"UMAP of averaged referents, {layer_label}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    add_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, out_dir, "10_umap_referent_centroids.png", prefix)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/OLMO3_7B_SELF_QUALIA_MAIN")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "beautiful_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = json.loads((run_dir / "run_meta.json").read_text())
    prefix = str(run_meta.get("pooling", "unknown_pooling"))
    for name in PLOT_FILENAMES:
        for path in [out_dir / name, out_dir / f"{prefix}_{name}"]:
            if path.exists():
                path.unlink()
    configure_style()
    make_standard_plots(run_dir, out_dir)
    make_umap_plots(run_dir, out_dir)
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
