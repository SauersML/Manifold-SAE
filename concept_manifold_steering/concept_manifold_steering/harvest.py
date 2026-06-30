"""Generic HuggingFace activation-harvest interface.

Loads any decoder-only ``AutoModelForCausalLM`` checkpoint and returns
mean-pooled hidden states at a chosen layer for a list of prompts.  This
mirrors the cogito-L40 harvest (see Manifold-SAE
``experiments/slop/steerability/harvest_hex.py``) but is generic over model and
aggregation strategy.

For *remote* models exposed via an OpenAI-compatible vLLM server with a
``/v1/encode`` endpoint, see :func:`harvest_via_server` which avoids
loading model weights locally.

Typical use
-----------
>>> X = harvest_activations(
...     "meta-llama/Llama-3.1-8B",
...     prompts=["The color of a tomato is", "A pumpkin is"],
...     layer=20,
...     aggregate="last_token",
... )
>>> X.shape
(2, 4096)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable, Literal, Sequence

import numpy as np

Aggregate = Literal["mean", "last_token", "first_token", "max"]


def _pool(hidden: "np.ndarray", attention_mask: "np.ndarray", how: Aggregate) -> np.ndarray:
    """Pool (T, D) hidden states using an attention mask of length T."""
    mask = attention_mask.astype(np.float32)
    if how == "mean":
        return (hidden * mask[:, None]).sum(0) / max(mask.sum(), 1.0)
    if how == "last_token":
        # last *real* (non-pad) token
        idx = int(mask.sum()) - 1
        idx = max(idx, 0)
        return hidden[idx]
    if how == "first_token":
        return hidden[0]
    if how == "max":
        h = hidden.copy()
        h[mask == 0] = -np.inf
        return h.max(0)
    raise ValueError(f"unknown aggregate {how!r}")


def harvest_activations(
    model_name_or_path: str,
    prompts: Sequence[str],
    layer: int,
    *,
    aggregate: Aggregate = "mean",
    batch_size: int = 8,
    max_length: int = 128,
    device: str | None = None,
    dtype: str = "float16",
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Harvest hidden-state activations at one layer.

    Parameters
    ----------
    model_name_or_path
        HuggingFace hub id (e.g. ``"deepcogito/cogito-v1-preview-llama-8B"``)
        or local path accepted by ``AutoModelForCausalLM.from_pretrained``.
    prompts
        Iterable of prompt strings.
    layer
        0-indexed transformer block whose output hidden state we record
        (uses ``output_hidden_states=True``; ``layer=0`` is the embedding
        output, ``layer=L`` is after block ``L-1``).
    aggregate
        Pooling strategy over the sequence dimension. ``"mean"`` matches
        the cogito harvest; ``"last_token"`` matches typical steering
        intervention sites; ``"max"`` and ``"first_token"`` are exposed
        for ablations.
    batch_size, max_length
        Tokenisation knobs.
    device
        Override torch device. Defaults to ``"cuda"`` if available else
        ``"cpu"``.
    dtype
        ``"float16" | "bfloat16" | "float32"``.
    trust_remote_code
        Forwarded to ``from_pretrained`` (cogito, etc. need this).

    Returns
    -------
    X : ndarray of shape (len(prompts), d_model), dtype float32.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "harvest_activations requires the 'hf' extra: "
            "pip install concept_manifold_steering[hf]"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]

    if verbose:
        print(f"[harvest] loading {model_name_or_path!r} on {device} ({dtype}) ...",
              file=sys.stderr, flush=True)

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        output_hidden_states=True,
    ).to(device)
    model.eval()

    prompts = list(prompts)
    out: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s : s + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
            res = model(**enc, output_hidden_states=True, use_cache=False)
            hs = res.hidden_states[layer]  # (B, T, D)
            mask = enc["attention_mask"]
            hs_np = hs.to(torch.float32).cpu().numpy()
            mask_np = mask.cpu().numpy()
            for b in range(hs_np.shape[0]):
                out.append(_pool(hs_np[b], mask_np[b], aggregate))
            if verbose:
                done = len(out)
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  [harvest] {done}/{len(prompts)} ({rate:.1f}/s)",
                      file=sys.stderr, flush=True)

    return np.stack(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Remote-server harvest (vLLM with /v1/encode, matches Manifold-SAE setup)
# ---------------------------------------------------------------------------

def harvest_via_server(
    api_base: str,
    prompts: Sequence[str],
    layer: int,
    *,
    aggregate: Aggregate = "mean",
    batch_size: int = 32,
    max_length: int = 64,
    timeout: float = 120.0,
    verbose: bool = True,
) -> np.ndarray:
    """Harvest activations through a remote /v1/encode endpoint.

    This avoids loading 70B+ weights locally and is what the cogito
    Manifold-SAE pipeline uses in production.  The expected request
    shape is::

        POST {api_base}/v1/encode
        {"texts": [...], "layers": [layer], "aggregate": "mean",
         "max_length": 64}

    and the response is ``{"results": [{"layer_{L}": [...D...]}, ...]}``.
    """
    out: list[np.ndarray] = []
    key = f"layer_{layer}"
    t0 = time.time()
    for s in range(0, len(prompts), batch_size):
        batch = list(prompts[s : s + batch_size])
        payload = {
            "texts": batch,
            "layers": [int(layer)],
            "aggregate": aggregate,
            "max_length": int(max_length),
        }
        req = urllib.request.Request(
            f"{api_base.rstrip('/')}/v1/encode",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for r in data["results"]:
            out.append(np.asarray(r[key], dtype=np.float32))
        if verbose:
            done = len(out)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  [server-harvest] {done}/{len(prompts)} ({rate:.1f}/s)",
                  file=sys.stderr, flush=True)
    return np.stack(out, axis=0).astype(np.float32)
