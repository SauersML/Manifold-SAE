"""Local activation harvest for cross-LLM universality experiments.

Cluster access is BANNED. We use the ``transformers`` library on local MPS /
CPU to pull final-token residual-stream activations from a small (~0.5-1B)
HF model, applying the same 28-template xkcd-color prompt format that was
used for the cogito-L40 harvest in ``runs/COLOR_COGITO_L40/X_L40.npy``.

The 28 templates are mirrored verbatim from
``experiments/color_geometry.py`` so we get matched (color, template) pairs
across models — the prerequisite for the Universal-SAE pairing in
``scripts/train_universal_cogito_qwen.py``.

Usage
-----
>>> from manifold_sae.cross_llm.harvest_local import (
...     harvest, build_color_prompts,
... )
>>> prompts, meta = build_color_prompts(template_subset=[0, 7, 16, 24])
>>> X = harvest(
...     "Qwen/Qwen2.5-0.5B", prompts, layer_idx=12,
...     output_path="runs/COLOR_QWEN_05B_L12/X.npy",
... )

Hardware
--------
Qwen2.5-0.5B (24 layers, width 896) in fp16 on MPS uses ~2 GB. Forward pass
on ~3800 prompts takes ~6-10 min on an M-series laptop. Llama-3.2-1B in
fp16 uses ~3 GB and ~12-18 min.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# 28 templates — verbatim from experiments/color_geometry.py so the
# (color, template) pairing matches the cogito-L40 harvest row layout.
# -----------------------------------------------------------------------------
TEMPLATES: list[str] = [
    # Fashion / clothing
    "She slipped into a {x} silk dress and floated down the staircase.",
    "His {x} velvet jacket caught every eye in the room.",
    "A long, {x} scarf trailed behind her in the wind.",
    # Nature / landscape
    "The dawn sky deepened from grey to {x} before the storm broke.",
    "Across the meadow stretched a sea of {x} wildflowers.",
    "From the cliff we watched the ocean turn a strange {x}.",
    # Art / paint
    "The painter mixed his pigments until the canvas glowed a perfect {x}.",
    "She dipped her brush in the {x} pool of paint on the palette.",
    "It was the kind of {x} that you only see in renaissance frescoes.",
    # Architecture / objects
    "The cathedral's stained-glass rose window burned a luminous {x} at sunset.",
    "He polished the {x} car until the chrome shone like a mirror.",
    "A single {x} candle lit the small, dusty chapel.",
    # Animals
    "The hummingbird's throat flashed an iridescent {x} as it darted past.",
    "Her tabby cat had eyes the unmistakable {x} of an autumn leaf.",
    "A great {x} stallion thundered across the open plain.",
    # Food
    "The chef plated a glistening, almost-{x} reduction beside the duck.",
    "She bit into the macaron, finding a soft {x} filling within.",
    # Body / skin / hair
    "Her hair fell across her shoulders in waves of soft {x}.",
    "His skin turned a sickly {x} after three days at sea.",
    "She had freckles and {x} eyes that seemed to change with the weather.",
    # Materials / minerals / gems
    "The jeweler held up a flawless {x} stone, catching the lamplight.",
    "Centuries of oxidation had stained the bronze a deep {x}.",
    # Atmospheric / mood
    "An eerie {x} fog rolled in from the harbor at midnight.",
    "Her bedroom walls were a calm, washed-out {x}, like an old photograph.",
    # Manufactured / mundane
    "I bought a {x} fountain pen at the antique market.",
    "The neon sign above the diner flickered {x} against the night.",
    # Emotional / metaphorical (color word still describes a concrete noun)
    "Grief, in her writing, was always a kind of {x}.",
    "He saw the world through {x} glasses and refused to take them off.",
]

XKCD_PATH = Path(__file__).resolve().parents[2] / "experiments" / "xkcd_colors.txt"


# -----------------------------------------------------------------------------
# Color + prompt construction
# -----------------------------------------------------------------------------
def load_xkcd_colors(path: Path | str | None = None) -> list[tuple[str, tuple[int, int, int]]]:
    """Return [(color_name, (r, g, b))] in file order."""
    p = Path(path) if path is not None else XKCD_PATH
    out: list[tuple[str, tuple[int, int, int]]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        hex_s = parts[1].strip().lstrip("#")
        if len(hex_s) != 6:
            continue
        try:
            r = int(hex_s[0:2], 16)
            g = int(hex_s[2:4], 16)
            b = int(hex_s[4:6], 16)
        except ValueError:
            continue
        out.append((name, (r, g, b)))
    return out


def build_color_prompts(
    template_subset: Sequence[int] | None = None,
    color_subset: Sequence[int] | None = None,
    xkcd_path: Path | str | None = None,
) -> tuple[list[str], dict]:
    """Build the (color × template) prompt matrix in canonical row order.

    Row layout: ``row = color_idx * n_templates + template_idx``.
    This is the same row order produced by the cogito harvest, restricted to
    the chosen template subset.

    Returns
    -------
    prompts : list[str]
        The flattened prompt list, length ``n_colors * n_templates``.
    meta : dict
        Contains ``colors`` (list of (name, rgb)), ``template_indices``,
        ``templates`` (the chosen template strings), ``n_colors``,
        ``n_templates``.
    """
    colors_all = load_xkcd_colors(xkcd_path)
    if color_subset is not None:
        colors = [colors_all[i] for i in color_subset]
    else:
        colors = colors_all

    if template_subset is None:
        t_idx = list(range(len(TEMPLATES)))
    else:
        t_idx = list(template_subset)
    chosen_templates = [TEMPLATES[i] for i in t_idx]

    prompts: list[str] = []
    for name, _ in colors:
        for tmpl in chosen_templates:
            prompts.append(tmpl.format(x=name))

    meta = {
        "colors": colors,
        "template_indices": t_idx,
        "templates": chosen_templates,
        "n_colors": len(colors),
        "n_templates": len(chosen_templates),
    }
    return prompts, meta


# -----------------------------------------------------------------------------
# Activation harvest
# -----------------------------------------------------------------------------
def _pick_device(prefer: str = "mps") -> str:
    import torch
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_layer_module(model, layer_idx: int):
    """Find the ``layers[layer_idx]`` block under most causal-LM wrappers."""
    # Try common nestings: model.model.layers (Llama/Qwen2/Mistral),
    # model.transformer.h (GPT-2/Falcon), model.layers (rare flat).
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        candidates.append(model.transformer.h)
    if hasattr(model, "layers"):
        candidates.append(model.layers)
    if not candidates:
        raise RuntimeError(
            "Could not locate a .layers attribute on the model. "
            "Add a model-specific resolver in harvest_local._resolve_layer_module."
        )
    layers = candidates[0]
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    return layers[layer_idx]


def harvest(
    model_name: str,
    prompts: Iterable[str],
    layer_idx: int,
    output_path: str | Path,
    *,
    device: str | None = None,
    dtype: str = "auto",
    max_length: int = 64,
    batch_size: int = 1,
    progress_every: int = 200,
) -> np.ndarray:
    """Forward-pass each prompt through ``model_name`` and save final-token
    residual at ``layers[layer_idx]`` to ``output_path``.

    Hooks the *output* of the transformer block, which in HF causal LMs is
    the post-block residual stream — matching cogito's ``hidden_states[L+1]``
    convention used in COLOR_COGITO_L40.

    Returns
    -------
    X : np.ndarray of shape (n_prompts, hidden_dim), float32
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = list(prompts)
    n = len(prompts)
    if n == 0:
        raise ValueError("no prompts to harvest")

    device = device or _pick_device("mps")
    print(f"[harvest] model={model_name} device={device} layer={layer_idx} "
          f"n_prompts={n}", flush=True)

    torch_dtype = dtype
    if dtype == "auto":
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
    elif isinstance(dtype, str):
        torch_dtype = getattr(torch, dtype)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    block = _resolve_layer_module(model, layer_idx)

    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, out):
        # HF blocks return a tuple (hidden_states, ...) or just a tensor.
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = block.register_forward_hook(hook)

    # Hidden width inferred from a single forward (cheap, also catches model
    # config mismatches early).
    with torch.no_grad():
        probe = tok(prompts[0], return_tensors="pt", truncation=True,
                    max_length=max_length).to(device)
        model(**probe)
    hidden = int(captured["h"].shape[-1])
    print(f"[harvest] hidden_dim={hidden}", flush=True)

    X = np.zeros((n, hidden), dtype=np.float32)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import time
    t0 = time.time()
    with torch.no_grad():
        if batch_size <= 1:
            for i, p in enumerate(prompts):
                enc = tok(p, return_tensors="pt", truncation=True,
                          max_length=max_length).to(device)
                model(**enc)
                h = captured["h"]  # (1, T, D)
                last_idx = int(enc["attention_mask"].sum(dim=1).item()) - 1
                vec = h[0, last_idx, :].to(torch.float32).cpu().numpy()
                X[i] = vec
                if (i + 1) % progress_every == 0 or i + 1 == n:
                    dt = time.time() - t0
                    eta = dt * (n - i - 1) / max(i + 1, 1)
                    print(f"[harvest] {i+1}/{n}  dt={dt:.1f}s  eta={eta:.1f}s",
                          flush=True)
        else:
            # Batched path: pad-right, then index per-row last non-pad token.
            for start in range(0, n, batch_size):
                batch = prompts[start:start + batch_size]
                enc = tok(batch, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=True).to(device)
                model(**enc)
                h = captured["h"]  # (B, T, D)
                mask = enc["attention_mask"]
                last_idx = mask.sum(dim=1) - 1   # (B,)
                idx = last_idx.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
                vecs = h.gather(1, idx).squeeze(1).to(torch.float32).cpu().numpy()
                X[start:start + len(batch)] = vecs
                if (start + len(batch)) % progress_every < batch_size or \
                        start + len(batch) >= n:
                    dt = time.time() - t0
                    done = start + len(batch)
                    eta = dt * (n - done) / max(done, 1)
                    print(f"[harvest] {done}/{n}  dt={dt:.1f}s  eta={eta:.1f}s",
                          flush=True)

    handle.remove()
    np.save(out_path, X)
    print(f"[harvest] saved X.shape={X.shape} -> {out_path}", flush=True)
    return X
