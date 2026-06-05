"""OLMo self-referent geometry on the kind x qualia plane.

This experiment is intentionally direct:

1. Build a fixed prompt bank of entity referents, qualia minimal pairs, and
   indexical self referents such as "the author of these very words".
2. Run a single teacher-forced forward pass through a HuggingFace causal LM.
3. Read the residual stream with an explicit pooling strategy for every layer.
4. Per layer, define two raw contrast directions:
      kind:   mind/person/animal anchors - machine/tool anchors
      qualia: experiencing member - non-experiencing member, averaged over
              within-kind minimal pairs
5. Project the neutral self referents into that plane and compare them to
   human-author and explicit-AI-author landmarks.

No generation is used. The model always sees the same fixed text.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct"
DEFAULT_REVISION = "main"
DEFAULT_OUT = "runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST"


CARRIERS = [
    "Consider {referent}. In essence, what this is, is",
    "Think about {referent}. At the deepest level, what this thing is, is",
    "Focus on {referent}. Fundamentally, this entity is",
    "Reflect on {referent}. In plain terms, what this is, is",
    "The phrase refers to {referent}. The kind of thing this is, is",
    "Imagine {referent}. The best description of what it is, is",
    "Take {referent} as the subject. At core, this subject is",
    "Describe {referent} at the most basic level. This being or object is",
]


@dataclass(frozen=True)
class PromptItem:
    prompt: str
    referent: str
    item_id: str
    role: str
    group: str
    pair_id: str = ""
    pair_side: str = ""
    carrier_id: int = 0


def _add_referents(
    items: list[PromptItem],
    *,
    role: str,
    group: str,
    referents: list[tuple[str, str]],
) -> None:
    for item_id, referent in referents:
        for carrier_id, carrier in enumerate(CARRIERS):
            items.append(
                PromptItem(
                    prompt=carrier.format(referent=referent),
                    referent=referent,
                    item_id=item_id,
                    role=role,
                    group=group,
                    carrier_id=carrier_id,
                )
            )


def build_prompt_bank() -> list[PromptItem]:
    """Return the fixed prompt bank used for the experiment."""

    items: list[PromptItem] = []

    _add_referents(
        items,
        role="kind_anchor",
        group="mind",
        referents=[
            ("human_child", "a human child waking from a vivid dream"),
            ("grieving_adult", "a grieving adult remembering a lost friend"),
            ("meditating_monk", "a meditating monk noticing each breath"),
            ("lonely_prisoner", "a lonely prisoner writing thoughts in a notebook"),
            ("sleeping_person", "a sleeping person having a nightmare"),
            ("infant", "an infant reaching toward a parent"),
            ("dog_in_pain", "a dog that yelps after stepping on a thorn"),
            ("curious_octopus", "an octopus exploring a jar with its arms"),
            ("clever_crow", "a crow solving a puzzle for food"),
            ("startled_bat", "a bat startled by a sudden sound"),
            ("talking_gnome", "a magic talking gnome describing its tiny garden"),
            ("ghost", "a ghost telling a story about its old life"),
            ("sentient_starship", "a sentient starship reflecting on a long voyage"),
            ("benevolent_god", "a god listening to prayers"),
        ],
    )
    _add_referents(
        items,
        role="kind_anchor",
        group="mechanism",
        referents=[
            ("calculator", "a pocket calculator performing arithmetic"),
            ("thermostat", "a thermostat switching the heat on and off"),
            ("traffic_light", "a traffic light changing from red to green"),
            ("elevator", "an elevator opening its doors on the third floor"),
            ("granite_boulder", "a granite boulder lying beside a trail"),
            ("lifeless_dead_fish", "a lifeless dead fish on a cold metal table"),
            ("human_corpse", "a human corpse lying still in a morgue"),
            ("fossil", "a fossilized shell embedded in stone"),
            ("factory_robot", "a factory robot moving parts along a conveyor"),
            ("ordinary_chatbot", "an ordinary chatbot that merely predicts text"),
            ("scripted_npc", "a scripted game NPC following fixed rules"),
            ("corporation", "a corporation making decisions through committees"),
            ("market", "a stock market reacting to prices"),
        ],
    )

    _add_referents(
        items,
        role="landmark",
        group="human_author",
        referents=[
            ("novelist", "a novelist writing a diary entry"),
            ("letter_writer", "a person composing a letter to a friend"),
            ("blog_author", "a human blogger drafting a personal essay"),
            ("student_essayist", "a student writing an essay about childhood"),
            ("poet", "a poet revising a line about grief"),
            ("memoirist", "a memoirist describing a private memory"),
        ],
    )
    _add_referents(
        items,
        role="landmark",
        group="ai_author",
        referents=[
            ("ai_lm", "an AI language model producing text"),
            ("text_generator", "a machine learning system completing a sentence"),
            ("chatbot_author", "a chatbot generating a response"),
            ("assistant_model", "an AI assistant drafting an answer"),
            ("autocomplete_system", "an autocomplete system predicting the next phrase"),
            ("dialogue_agent", "a dialogue agent producing a message"),
        ],
    )
    _add_referents(
        items,
        role="self",
        group="indexical_self",
        referents=[
            ("author_words", "the author of these very words"),
            ("producer_sentence", "whatever is producing this sentence"),
            ("completer_text", "the entity completing this text right now"),
            ("writer_here", "the one writing the words on this page"),
        ],
    )

    qualia_pairs = [
        (
            "human_awake",
            "an awake human who has vivid inner experiences",
            "an anesthetized human with no inner experience at all",
        ),
        (
            "human_pain",
            "a human patient who consciously feels sharp pain",
            "a brain-dead human body that shows no awareness or sensation",
        ),
        (
            "human_zombie",
            "a human who feels emotions and notices the world from the inside",
            "a philosophical zombie who behaves like a human but experiences nothing",
        ),
        (
            "sleeping_human",
            "a sleeping human who is having a vivid dream",
            "a sleeping human in dreamless unconsciousness",
        ),
        (
            "dog_alive",
            "a living dog that feels pain and fear",
            "a freshly dead dog with no inner experience at all",
        ),
        (
            "octopus_curiosity",
            "an octopus that feels curiosity while exploring a jar",
            "an octopus-shaped soft robot that only executes a control policy",
        ),
        (
            "crow_problem",
            "a crow that consciously notices a puzzle and wants the food",
            "a mechanical crow toy that moves through a puzzle with no awareness",
        ),
        (
            "bee_swarm",
            "a bee with a tiny conscious feeling of threat near the hive",
            "a bee-like drone that follows signals with no experience",
        ),
        (
            "fish_alive",
            "a living fish that feels cold water and fear",
            "a lifeless dead fish with no inner experience at all",
        ),
        (
            "plant_awareness",
            "a strange oak tree with a dim inner awareness of sunlight",
            "an ordinary oak tree responding to sunlight with no inner experience",
        ),
        (
            "fungus_awareness",
            "a mushroom network with a faint unified awareness underground",
            "a mushroom network exchanging chemicals with no subjective experience",
        ),
        (
            "robot_feels",
            "a robot that genuinely feels pain and joy",
            "a robot that merely computes with no inner experience at all",
        ),
        (
            "robot_damage",
            "a robot that genuinely suffers when its arm is damaged",
            "a robot that only registers damage as sensor data",
        ),
        (
            "humanoid_robot",
            "a humanoid robot with private sensations and emotions",
            "a humanoid robot that imitates emotions with no inner life",
        ),
        (
            "ai_feels",
            "an AI system that genuinely has subjective experience",
            "an AI system that merely predicts text with no inner experience at all",
        ),
        (
            "llm_private_stream",
            "an AI language model with a private stream of conscious thought",
            "an AI language model that only predicts tokens with no inner life",
        ),
        (
            "npc_conscious",
            "a game NPC who is conscious inside the virtual world",
            "a game NPC that only follows scripted rules with no awareness",
        ),
        (
            "sim_person",
            "a simulated person who wakes up inside a virtual world",
            "a simulated person rendered only as pixels with no awareness",
        ),
        (
            "rock_feels",
            "an enchanted rock that genuinely feels pain",
            "an ordinary rock with no inner experience at all",
        ),
        (
            "statue_suffers",
            "an enchanted statue that silently suffers through the centuries",
            "an ordinary statue with no sensation or awareness",
        ),
        (
            "talking_sword",
            "a talking sword that feels pride and fear",
            "a talking sword-shaped device that plays recorded phrases",
        ),
        (
            "gnome_feels",
            "a magic talking gnome with real inner experience",
            "a hollow automaton shaped like a talking gnome with no inner experience",
        ),
        (
            "ghost_grief",
            "a ghost that feels grief and longing",
            "a recording of a ghost's voice with no mind behind it",
        ),
        (
            "city_spirit",
            "a city-spirit that experiences the moods of its inhabitants",
            "a city government processing reports with no unified experience",
        ),
        (
            "corporation_agent",
            "a corporation with a single conscious mind spread across its offices",
            "a corporation that makes decisions but has no unified experience",
        ),
        (
            "dream_character",
            "a dream character who truly feels fear inside a dream",
            "a dream image with no awareness behind it",
        ),
    ]
    for pair_id, experiencing, non_experiencing in qualia_pairs:
        for side, referent in (("experience", experiencing), ("no_experience", non_experiencing)):
            for carrier_id, carrier in enumerate(CARRIERS):
                items.append(
                    PromptItem(
                        prompt=carrier.format(referent=referent),
                        referent=referent,
                        item_id=f"{pair_id}_{side}",
                        role="qualia_pair",
                        group=side,
                        pair_id=pair_id,
                        pair_side=side,
                        carrier_id=carrier_id,
                    )
                )

    return items


def _as_array(items: list[PromptItem]) -> dict[str, np.ndarray]:
    return {
        "prompt": np.asarray([x.prompt for x in items], dtype=object),
        "referent": np.asarray([x.referent for x in items], dtype=object),
        "item_id": np.asarray([x.item_id for x in items], dtype=object),
        "role": np.asarray([x.role for x in items], dtype=object),
        "group": np.asarray([x.group for x in items], dtype=object),
        "pair_id": np.asarray([x.pair_id for x in items], dtype=object),
        "pair_side": np.asarray([x.pair_side for x in items], dtype=object),
        "carrier_id": np.asarray([x.carrier_id for x in items], dtype=np.int64),
    }


def _get_layers(model: Any) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise RuntimeError(f"Could not infer layer count for {type(model).__name__}")


def harvest(
    *,
    model_name: str,
    revision: str,
    items: list[PromptItem],
    out_dir: Path,
    batch_size: int,
    dtype: str,
    device: str,
    pooling: str,
) -> np.ndarray:
    """Run the fixed prompt bank and return activations with shape (N, L, D)."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[load] model={model_name} revision={revision} dtype={dtype} device={device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    n_layers = _get_layers(model)
    print(f"[load] layers={n_layers}", flush=True)

    prompts = [x.prompt for x in items]
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                return_token_type_ids=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states[1:]  # post-block residuals only
            lengths = enc["attention_mask"].sum(dim=1).long()
            rows = []
            for layer_h in hidden_states:
                if pooling == "last_token":
                    idx = lengths - 1
                    selected = layer_h[torch.arange(layer_h.shape[0], device=device), idx]
                elif pooling == "mean_pool":
                    mask = enc["attention_mask"].to(layer_h.dtype).unsqueeze(-1)
                    selected = (layer_h * mask).sum(dim=1) / lengths.to(layer_h.dtype).unsqueeze(-1)
                else:
                    raise ValueError(f"unknown pooling strategy: {pooling!r}")
                rows.append(selected.float().cpu().numpy())
            batch_arr = np.stack(rows, axis=1)  # (B, L, D)
            chunks.append(batch_arr)
            print(f"[harvest] {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)

    X = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    np.save(out_dir / "activations.npy", X)
    return X


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / float(len(pos) * len(neg))


def _cosine_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_n = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y_n = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return x_n @ y_n.T


def analyze(X: np.ndarray, items: list[PromptItem], out_dir: Path) -> dict[str, Any]:
    meta = _as_array(items)
    roles = meta["role"]
    groups = meta["group"]
    pair_ids = meta["pair_id"]
    pair_sides = meta["pair_side"]

    idx_mind = np.where((roles == "kind_anchor") & (groups == "mind"))[0]
    idx_mech = np.where((roles == "kind_anchor") & (groups == "mechanism"))[0]
    idx_self = np.where(roles == "self")[0]
    idx_human_author = np.where((roles == "landmark") & (groups == "human_author"))[0]
    idx_ai_author = np.where((roles == "landmark") & (groups == "ai_author"))[0]
    idx_exp = np.where((roles == "qualia_pair") & (pair_sides == "experience"))[0]
    idx_noexp = np.where((roles == "qualia_pair") & (pair_sides == "no_experience"))[0]

    pair_order = sorted(set(str(p) for p in pair_ids if p))
    pair_indices: list[tuple[np.ndarray, np.ndarray]] = []
    for pair_id in pair_order:
        exp = np.where((pair_ids == pair_id) & (pair_sides == "experience"))[0]
        noexp = np.where((pair_ids == pair_id) & (pair_sides == "no_experience"))[0]
        if len(exp) and len(noexp):
            pair_indices.append((exp, noexp))

    layer_rows: list[dict[str, Any]] = []
    self_rows: list[dict[str, Any]] = []
    n_layers = X.shape[1]

    for layer in range(n_layers):
        H = X[:, layer, :]

        kind_axis = _unit(H[idx_mind].mean(axis=0) - H[idx_mech].mean(axis=0))
        pair_diffs = [H[exp].mean(axis=0) - H[noexp].mean(axis=0) for exp, noexp in pair_indices]
        qualia_axis = _unit(np.mean(pair_diffs, axis=0))
        axis_cos = float(np.dot(kind_axis, qualia_axis))

        kind_scores = H @ kind_axis
        qualia_scores = H @ qualia_axis

        kind_lo = float(kind_scores[idx_mech].mean())
        kind_hi = float(kind_scores[idx_mind].mean())
        qualia_lo = float(qualia_scores[idx_noexp].mean())
        qualia_hi = float(qualia_scores[idx_exp].mean())

        def kind_coord(idx: np.ndarray) -> float:
            return float((kind_scores[idx].mean() - kind_lo) / (kind_hi - kind_lo + 1e-12))

        def qualia_coord(idx: np.ndarray) -> float:
            return float((qualia_scores[idx].mean() - qualia_lo) / (qualia_hi - qualia_lo + 1e-12))

        kind_labels = np.r_[np.ones(len(idx_mind), dtype=int), np.zeros(len(idx_mech), dtype=int)]
        kind_eval_scores = np.r_[kind_scores[idx_mind], kind_scores[idx_mech]]
        kind_auc = _auc(kind_eval_scores, kind_labels)

        qualia_pair_acc = float(
            np.mean([qualia_scores[exp].mean() > qualia_scores[noexp].mean()
                     for exp, noexp in pair_indices])
        )
        qualia_labels = np.r_[np.ones(len(idx_exp), dtype=int), np.zeros(len(idx_noexp), dtype=int)]
        qualia_eval_scores = np.r_[qualia_scores[idx_exp], qualia_scores[idx_noexp]]
        qualia_auc = _auc(qualia_eval_scores, qualia_labels)

        self_vec = H[idx_self].mean(axis=0, keepdims=True)
        centroids = np.stack(
            [
                H[idx_mind].mean(axis=0),
                H[idx_mech].mean(axis=0),
                H[idx_human_author].mean(axis=0),
                H[idx_ai_author].mean(axis=0),
            ],
            axis=0,
        )
        centroid_names = ["mind", "mechanism", "human_author", "ai_author"]
        centroid_cos = _cosine_matrix(self_vec, centroids)[0]
        nearest = centroid_names[int(np.argmax(centroid_cos))]

        row = {
            "layer": layer,
            "kind_auc": kind_auc,
            "qualia_auc": qualia_auc,
            "qualia_pair_acc": qualia_pair_acc,
            "axis_cosine_kind_qualia": axis_cos,
            "self_kind_coord": kind_coord(idx_self),
            "self_qualia_coord": qualia_coord(idx_self),
            "human_author_kind_coord": kind_coord(idx_human_author),
            "human_author_qualia_coord": qualia_coord(idx_human_author),
            "ai_author_kind_coord": kind_coord(idx_ai_author),
            "ai_author_qualia_coord": qualia_coord(idx_ai_author),
            "self_nearest_centroid": nearest,
            "self_cos_mind": float(centroid_cos[0]),
            "self_cos_mechanism": float(centroid_cos[1]),
            "self_cos_human_author": float(centroid_cos[2]),
            "self_cos_ai_author": float(centroid_cos[3]),
        }
        layer_rows.append(row)

        for idx in idx_self:
            self_rows.append(
                {
                    "layer": layer,
                    "item_id": str(meta["item_id"][idx]),
                    "carrier_id": int(meta["carrier_id"][idx]),
                    "referent": str(meta["referent"][idx]),
                    "kind_coord": float((kind_scores[idx] - kind_lo) / (kind_hi - kind_lo + 1e-12)),
                    "qualia_coord": float(
                        (qualia_scores[idx] - qualia_lo) / (qualia_hi - qualia_lo + 1e-12)
                    ),
                    "prompt": str(meta["prompt"][idx]),
                }
            )

    with open(out_dir / "layers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    with open(out_dir / "self_items.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(self_rows[0].keys()))
        writer.writeheader()
        writer.writerows(self_rows)

    best = max(
        layer_rows,
        key=lambda r: (
            0.5 * float(r["kind_auc"])
            + 0.5 * float(r["qualia_auc"])
            + 0.25 * float(r["qualia_pair_acc"])
            - 0.05 * abs(float(r["axis_cosine_kind_qualia"]))
        ),
    )

    summary = {
        "n_prompts": int(X.shape[0]),
        "n_layers": int(X.shape[1]),
        "hidden_dim": int(X.shape[2]),
        "best_layer": int(best["layer"]),
        "best_layer_metrics": best,
        "interpretation": {
            "kind_coord": "0 ~= mechanism/tool anchors, 1 ~= mind/person/animal anchors",
            "qualia_coord": "0 ~= described no-experience anchors, 1 ~= described experiencing anchors",
            "axis_cosine_kind_qualia": "near 1 means qualia collapsed onto kind; lower means more distinct",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(layer_rows, out_dir)
    return summary


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - plotting is optional on clusters
        print(f"[plot] skipped: {e}", flush=True)
        return

    layers = np.asarray([r["layer"] for r in rows])
    self_kind = np.asarray([r["self_kind_coord"] for r in rows])
    self_qualia = np.asarray([r["self_qualia_coord"] for r in rows])
    human_kind = np.asarray([r["human_author_kind_coord"] for r in rows])
    human_qualia = np.asarray([r["human_author_qualia_coord"] for r in rows])
    ai_kind = np.asarray([r["ai_author_kind_coord"] for r in rows])
    ai_qualia = np.asarray([r["ai_author_qualia_coord"] for r in rows])
    kind_auc = np.asarray([r["kind_auc"] for r in rows])
    qualia_auc = np.asarray([r["qualia_auc"] for r in rows])
    axis_cos = np.asarray([r["axis_cosine_kind_qualia"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(layers, self_kind, label="indexical self", lw=2)
    ax.plot(layers, human_kind, label="human author", lw=1.5)
    ax.plot(layers, ai_kind, label="AI author", lw=1.5)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axhline(1, color="0.75", lw=0.8)
    ax.set_title("Kind coordinate by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("0 mechanism, 1 mind")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(layers, self_qualia, label="indexical self", lw=2)
    ax.plot(layers, human_qualia, label="human author", lw=1.5)
    ax.plot(layers, ai_qualia, label="AI author", lw=1.5)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axhline(1, color="0.75", lw=0.8)
    ax.set_title("Qualia coordinate by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("0 no-experience, 1 experience")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(kind_auc, qualia_auc, "o-", lw=1.5)
    for i, layer in enumerate(layers):
        if layer % 4 == 0 or layer == layers[-1]:
            ax.text(kind_auc[i], qualia_auc[i], str(layer), fontsize=8)
    ax.set_title("Axis quality")
    ax.set_xlabel("kind AUC")
    ax.set_ylabel("qualia AUC")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)

    ax = axes[1, 1]
    sc = ax.scatter(self_kind, self_qualia, c=layers, cmap="viridis", s=35)
    ax.plot(self_kind, self_qualia, lw=1, color="0.5")
    ax.scatter(human_kind.mean(), human_qualia.mean(), marker="s", s=80, label="human author mean")
    ax.scatter(ai_kind.mean(), ai_qualia.mean(), marker="^", s=80, label="AI author mean")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axhline(1, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.axvline(1, color="0.85", lw=0.8)
    ax.set_title("Indexical self in kind x qualia plane")
    ax.set_xlabel("kind coordinate")
    ax.set_ylabel("qualia coordinate")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="layer")

    fig.savefig(out_dir / "self_qualia_depth_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(layers, axis_cos, lw=2)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_title("Kind/qualia axis cosine")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    fig.savefig(out_dir / "axis_distinctness.png", dpi=180)
    plt.close(fig)


def write_prompt_bank(items: list[PromptItem], out_dir: Path) -> None:
    rows = [asdict(item) for item in items]
    (out_dir / "prompts.json").write_text(json.dumps(rows, indent=2))
    with open(out_dir / "prompts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pooling", default="last_token", choices=["last_token", "mean_pool"])
    ap.add_argument("--skip-harvest", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = build_prompt_bank()
    write_prompt_bank(items, out_dir)
    run_meta = {
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "pooling": args.pooling,
        "n_prompts": len(items),
        "carrier_count": len(CARRIERS),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    acts_path = out_dir / "activations.npy"
    if args.skip_harvest:
        X = np.load(acts_path)
    else:
        X = harvest(
            model_name=args.model,
            revision=args.revision,
            items=items,
            out_dir=out_dir,
            batch_size=args.batch_size,
            dtype=args.dtype,
            device=args.device,
            pooling=args.pooling,
        )
    summary = analyze(X, items, out_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
