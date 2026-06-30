"""Harvesting and loading residual-stream activations from a HuggingFace LLM.

The headline use-case: pull layer-28 residual-stream activations from Llama 3.1 8B
on a curated set of cyclic-concept prompts (days, months, letters, ...), to feed
the Manifold-SAE.

Design notes:
- Harvest writes a single ``.pt`` file (PyTorch's torch.save) with a dict
  ``{"activations": Tensor(N, D), "labels": list[str], "categories": list[str],
     "values": list[str], "prompts": list[str], "meta": dict}``. We deliberately
  don't reach for safetensors here because we want to bundle the metadata lists
  in the same file; safetensors stores only tensors.
- Resumable: if ``output_path`` exists we load it, skip already-processed prompts
  (matched by exact prompt string), and append. Saves after every batch so a
  killed job leaves a usable shard.
- We hook the *output* of ``model.model.layers[layer]`` which on Llama is the
  residual stream *after* the layer (so layer=28 means "what layer 28 produced").
  We don't use ``output_hidden_states`` because forward hooks are cheaper and let
  us select exactly one layer without materializing all 33.
- target_token_strategy resolves the BPE position whose residual we keep:
  ``"value_token"`` finds the token spelling the concept value;
  ``"last_token"`` uses the final non-pad token (often the most informative on
  causal LMs because attention has read everything); a callable lets the user do
  anything else with ``(prompt, value, tokenizer, input_ids)``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Iterable

import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# Token-strategy resolution
# ----------------------------------------------------------------------

TargetTokenStrategy = str | Callable[..., int]


def _find_value_token_index(
    prompt: str,
    value: str,
    tokenizer: Any,
    input_ids: torch.Tensor,
) -> int:
    """Return the position of the last token belonging to ``value`` inside ``prompt``.

    Strategy: tokenize value both with and without a leading space (Llama's BPE
    treats " Monday" and "Monday" as different first tokens). Try each variant
    as a contiguous subsequence inside ``input_ids``. We prefer the *last*
    occurrence — if a prompt repeats the value, the trailing one usually has
    more context.
    """
    ids = input_ids.tolist()

    candidates: list[list[int]] = []
    for variant in (value, " " + value):
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if toks:
            candidates.append(toks)

    best: int = -1
    for cand in candidates:
        L = len(cand)
        for start in range(len(ids) - L + 1):
            if ids[start : start + L] == cand:
                # take last token of the value span; that's where the model has
                # actually committed to the concept.
                best = max(best, start + L - 1)

    if best == -1:
        # Fall back to last token; loud-but-not-fatal so harvest doesn't die on
        # a single odd prompt.
        return len(ids) - 1
    return best


def _resolve_target_index(
    strategy: TargetTokenStrategy,
    prompt: str,
    value: str,
    tokenizer: Any,
    input_ids: torch.Tensor,
) -> int:
    if callable(strategy):
        return int(strategy(prompt=prompt, value=value, tokenizer=tokenizer, input_ids=input_ids))
    if strategy == "value_token":
        return _find_value_token_index(prompt, value, tokenizer, input_ids)
    if strategy == "last_token":
        return int(input_ids.shape[-1] - 1)
    raise ValueError(f"Unknown target_token_strategy: {strategy!r}")


# ----------------------------------------------------------------------
# Harvest
# ----------------------------------------------------------------------


@dataclass
class HarvestRecord:
    activation: torch.Tensor  # (D,)
    label: str
    category: str
    value: str
    prompt: str


def _load_existing(output_path: str) -> tuple[list[HarvestRecord], set[str]]:
    if not os.path.exists(output_path):
        return [], set()
    blob = torch.load(output_path, map_location="cpu")
    acts = blob["activations"]
    labels = blob.get("labels", [])
    cats = blob.get("categories", [])
    vals = blob.get("values", [])
    prompts = blob.get("prompts", [])
    records: list[HarvestRecord] = []
    for i in range(acts.shape[0]):
        records.append(
            HarvestRecord(
                activation=acts[i],
                label=labels[i] if i < len(labels) else "",
                category=cats[i] if i < len(cats) else "",
                value=vals[i] if i < len(vals) else "",
                prompt=prompts[i] if i < len(prompts) else "",
            )
        )
    return records, set(prompts)


def _save_records(
    records: list[HarvestRecord],
    output_path: str,
    meta: dict,
) -> None:
    if not records:
        return
    acts = torch.stack([r.activation for r in records], dim=0)
    blob = {
        "activations": acts.contiguous(),
        "labels": [r.label for r in records],
        "categories": [r.category for r in records],
        "values": [r.value for r in records],
        "prompts": [r.prompt for r in records],
        "meta": meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    torch.save(blob, tmp_path)
    os.replace(tmp_path, output_path)


def _select_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[name]


def harvest_activations(
    model_name: str,
    prompts: list[dict],
    layer: int,
    target_token_strategy: TargetTokenStrategy = "value_token",
    device: str = "cuda",
    batch_size: int = 4,
    output_path: str = "activations.pt",
    dtype: str = "bfloat16",
    max_prompts: int | None = None,
    print_sanity_samples: int = 3,
) -> dict:
    """Harvest residual-stream activations from a HF causal LM.

    Imports of ``transformers`` happen inside the function so that the module
    is usable (and importable) without the optional ``[llm]`` extra installed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    torch_dtype = _select_dtype(dtype)

    print(f"[harvest] loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[harvest] loading model {model_name} (dtype={dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    # Locate the residual-stream layer. Llama exposes ``model.model.layers``.
    # Other architectures (gpt-neox, etc.) differ; raise loudly so the user
    # knows to adapt.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "Expected a Llama-style architecture with model.model.layers; got "
            f"{type(model).__name__}. Adapt the hook target manually."
        )
    layer_module = model.model.layers[layer]

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: Any, _inputs: Any, output: Any) -> None:
        # Llama layer outputs a tuple (hidden_states, ...); take [0].
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden.detach()

    handle = layer_module.register_forward_hook(_hook)

    # Resume
    records, seen = _load_existing(output_path)
    print(f"[harvest] {len(records)} records already on disk; skipping those")

    to_process = [p for p in prompts if p["prompt"] not in seen]
    if max_prompts is not None:
        to_process = to_process[:max_prompts]

    if device == "cpu":
        print(
            "[harvest] WARNING: device=cpu. Llama 3.1 8B is enormous on CPU; "
            "this is only viable for very small models or sanity checks."
        )

    # Sanity preview: tokenize a few prompts and show the chosen target token.
    for p in to_process[:print_sanity_samples]:
        enc = tokenizer(p["prompt"], return_tensors="pt")
        ids = enc["input_ids"][0]
        idx = _resolve_target_index(
            target_token_strategy, p["prompt"], p["value"], tokenizer, ids
        )
        tok_str = tokenizer.decode([ids[idx].item()])
        print(
            f"[harvest][sanity] cat={p['category']:<10} value={p['value']:<10} "
            f"idx={idx:<3} tok={tok_str!r}"
        )

    meta = {
        "model_name": model_name,
        "layer": layer,
        "target_token_strategy": (
            target_token_strategy if isinstance(target_token_strategy, str) else "callable"
        ),
        "dtype": dtype,
    }

    n_done = 0
    try:
        with torch.no_grad():
            for start in range(0, len(to_process), batch_size):
                batch = to_process[start : start + batch_size]
                texts = [b["prompt"] for b in batch]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                _ = model(**enc)
                h = captured["h"]  # (B, T, D)

                # Per-example target index, respecting per-example unpadded length.
                attn = enc["attention_mask"]
                for i, item in enumerate(batch):
                    # Unpadded ids for this row, for index computation.
                    unpadded_len = int(attn[i].sum().item())
                    ids_i = enc["input_ids"][i, :unpadded_len].detach().cpu()
                    idx = _resolve_target_index(
                        target_token_strategy,
                        item["prompt"],
                        item["value"],
                        tokenizer,
                        ids_i,
                    )
                    idx = max(0, min(idx, unpadded_len - 1))
                    vec = h[i, idx].to(torch.float32).cpu().clone()
                    records.append(
                        HarvestRecord(
                            activation=vec,
                            label=f"{item['category']}:{item['value']}",
                            category=item["category"],
                            value=str(item["value"]),
                            prompt=item["prompt"],
                        )
                    )
                n_done += len(batch)
                _save_records(records, output_path, meta)
                print(
                    f"[harvest] processed {n_done}/{len(to_process)} "
                    f"(total on disk: {len(records)})"
                )
    finally:
        handle.remove()

    return {
        "n_records": len(records),
        "output_path": output_path,
        "meta": meta,
    }


# ----------------------------------------------------------------------
# Dataset wrapper
# ----------------------------------------------------------------------


class ActivationDataset(Dataset):
    """Wraps a harvested .pt file. ``__getitem__`` returns only the activation
    tensor so it drops straight into a DataLoader feeding the SAE. Labels and
    categories are stored as attributes for downstream evaluation.
    """

    def __init__(self, path: str, dtype: torch.dtype = torch.float32) -> None:
        blob = torch.load(path, map_location="cpu")
        self.activations: torch.Tensor = blob["activations"].to(dtype)
        self.labels: list[str] = list(blob.get("labels", []))
        self.categories: list[str] = list(blob.get("categories", []))
        self.values: list[str] = list(blob.get("values", []))
        self.prompts: list[str] = list(blob.get("prompts", []))
        self.meta: dict = dict(blob.get("meta", {}))

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]

    @property
    def dim(self) -> int:
        return int(self.activations.shape[1])


def partition_by_category(dataset: ActivationDataset) -> dict[str, CategorySubset]:
    """Group items by category; values are lightweight views (no copy)."""
    buckets: dict[str, list[int]] = {}
    for i, c in enumerate(dataset.categories):
        buckets.setdefault(c, []).append(i)
    return {c: CategorySubset(dataset, idxs) for c, idxs in buckets.items()}


@dataclass
class CategorySubset:
    parent: ActivationDataset
    indices: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.indices)

    def activations(self) -> torch.Tensor:
        return self.parent.activations[self.indices]

    def values(self) -> list[str]:
        return [self.parent.values[i] for i in self.indices]

    def labels(self) -> list[str]:
        return [self.parent.labels[i] for i in self.indices]

    def prompts(self) -> list[str]:
        return [self.parent.prompts[i] for i in self.indices]


# ----------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------


def load_prompts(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def iter_batches(
    activations: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterable[torch.Tensor]:
    """Tiny in-memory batcher — Manifold-SAE training fits easily in RAM once
    we have a few hundred prompts' activations, so we skip DataLoader overhead.
    """
    N = activations.shape[0]
    while True:
        if shuffle:
            perm = torch.randperm(N, generator=generator)
        else:
            perm = torch.arange(N)
        for start in range(0, N, batch_size):
            sel = perm[start : start + batch_size]
            yield activations[sel]
