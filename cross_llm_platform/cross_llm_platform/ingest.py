"""HuggingFace activation harvesting for decoder-only causal LMs.

The public entry point is :func:`harvest_activations`, which loads any
``AutoModelForCausalLM`` checkpoint, runs batched prompts with
``output_hidden_states=True``, optionally applies an activation steering hook
during the forward pass, and returns a typed :class:`HarvestResult`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Pool = Literal["last_token", "mean", "first_token"]
Probe = int | Sequence[int]


class SteeringHook(Protocol):
    """Callable that edits hidden states during activation harvest."""

    def __call__(self, layer: int, hidden_states: Any, attention_mask: Any) -> Any:
        """Return edited hidden states for a model layer."""


@dataclass(frozen=True, slots=True)
class HarvestResult:
    """Activation harvest output.

    Attributes:
        activations: ``(n_prompts, width)`` for one layer or
            ``(n_prompts, n_layers, width)`` for multiple layers.
        prompts: Harvested prompt strings.
        model_name: HuggingFace model id or path.
        layers: Captured hidden-state indices.
        pool: Sequence pooling strategy.
    """

    activations: NDArray[np.float32]
    prompts: tuple[str, ...]
    model_name: str
    layers: tuple[int, ...]
    pool: Pool

    def save(self, path: str | Path) -> None:
        """Save activations plus JSON metadata as ``.npz``."""
        np.savez_compressed(
            Path(path),
            activations=self.activations,
            prompts=np.asarray(self.prompts, dtype=object),
            model_name=np.asarray(self.model_name, dtype=object),
            layers=np.asarray(self.layers, dtype=np.int64),
            pool=np.asarray(self.pool, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HarvestResult":
        """Load a result written by :meth:`save`."""
        data = np.load(Path(path), allow_pickle=True)
        return cls(
            activations=np.asarray(data["activations"], dtype=np.float32),
            prompts=tuple(str(x) for x in data["prompts"].tolist()),
            model_name=str(data["model_name"].item()),
            layers=tuple(int(x) for x in data["layers"]),
            pool=str(data["pool"].item()),  # type: ignore[arg-type]
        )


def harvest_activations(
    model_name_or_path: str,
    prompts: Sequence[str],
    layer: Probe,
    *,
    batch_size: int = 4,
    max_length: int = 256,
    pool: Pool = "last_token",
    device: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    steering_hook: SteeringHook | None = None,
) -> HarvestResult:
    """Harvest decoder-only hidden states from a HuggingFace causal LM.

    Args:
        model_name_or_path: HuggingFace id or local checkpoint path.
        prompts: Prompt strings.
        layer: Hidden-state index or indices. ``0`` is embedding output;
            positive block indices follow HuggingFace ``hidden_states``.
        batch_size: Tokenized prompt batch size.
        max_length: Tokenizer truncation length.
        pool: Sequence pooling strategy.
        device: Torch device. Defaults to CUDA when available.
        dtype: ``auto``, ``float32``, ``float16``, or ``bfloat16``.
        trust_remote_code: Forwarded to HuggingFace loaders.
        steering_hook: Optional callable that edits block outputs while
            harvesting, enabling per-prompt steering-injection probes.

    Returns:
        :class:`HarvestResult` with float32 activations.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise NotImplementedError(
            "HuggingFace harvesting requires torch and transformers. "
            "Install cross-llm-platform[hf]."
        ) from exc

    layer_ids = (int(layer),) if isinstance(layer, int) else tuple(int(x) for x in layer)
    if not layer_ids:
        raise ValueError("at least one layer must be requested")
    torch_dtype = _torch_dtype(dtype, torch)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    if not bool(getattr(model.config, "is_decoder", True)):
        raise ValueError("harvest_activations supports decoder-only causal language models")
    model.eval()

    handles = _install_steering_hooks(model, steering_hook) if steering_hook is not None else []
    rows: list[NDArray[np.float32]] = []
    try:
        with torch.no_grad():
            for batch in _batches(tuple(prompts), batch_size):
                encoded = tokenizer(
                    list(batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                output = model(**encoded, output_hidden_states=True, use_cache=False)
                hidden_states = output.hidden_states
                by_layer = [
                    _pool_hidden(hidden_states[idx], encoded["attention_mask"], pool)
                    for idx in layer_ids
                ]
                stacked = torch.stack(by_layer, dim=1)
                rows.append(stacked.detach().to(torch.float32).cpu().numpy().astype(np.float32))
    finally:
        for handle in handles:
            handle.remove()

    arr = np.concatenate(rows, axis=0)
    if len(layer_ids) == 1:
        arr = arr[:, 0, :]
    return HarvestResult(arr, tuple(prompts), model_name_or_path, layer_ids, pool)


def load_prompts(path: str | Path) -> list[str]:
    """Load prompts from JSON, JSONL, or plain text."""
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return [str(x["prompt"] if isinstance(x, dict) and "prompt" in x else x) for x in data]
        raise ValueError("JSON prompt file must contain a list")
    if p.suffix == ".jsonl":
        return [
            str((obj := json.loads(line))["prompt"] if isinstance(obj, dict) and "prompt" in obj else obj)
            for line in p.read_text().splitlines()
            if line.strip()
        ]
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _torch_dtype(dtype: str, torch: Any) -> Any:
    if dtype == "auto":
        return "auto"
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in table:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return table[dtype]


def _batches(items: Sequence[str], batch_size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(items), max(1, int(batch_size))):
        yield tuple(items[start : start + batch_size])


def _pool_hidden(hidden: Any, attention_mask: Any, pool: Pool) -> Any:
    if pool == "first_token":
        return hidden[:, 0, :]
    lengths = attention_mask.sum(dim=1).clamp(min=1)
    if pool == "last_token":
        idx = lengths - 1
        return hidden[hidden.new_tensor(range(hidden.shape[0]), dtype=idx.dtype).long(), idx.long(), :]
    if pool == "mean":
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / lengths.to(hidden.dtype).unsqueeze(-1)
    raise ValueError(f"unknown pool {pool!r}")


def _install_steering_hooks(model: Any, hook: SteeringHook) -> list[Any]:
    layers = _decoder_layers(model)
    handles = []
    for layer_idx, module in enumerate(layers, start=1):
        def _hook(_module: Any, inputs: tuple[Any, ...], output: Any, idx: int = layer_idx) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            mask = inputs[1] if len(inputs) > 1 else None
            edited = hook(idx, hidden, mask)
            if isinstance(output, tuple):
                return (edited, *output[1:])
            return edited

        handles.append(module.register_forward_hook(_hook))
    return handles


def _decoder_layers(model: Any) -> Sequence[Any]:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise ValueError("could not locate decoder block list on this model")
