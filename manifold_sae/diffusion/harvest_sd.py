"""Harvest SD-1.4 (or SDXL-Turbo) UNet cross-attention residual activations.

For each xkcd color × template prompt we run a few diffusion-denoising steps
and capture the residual activation at a hooked UNet block. We DO NOT generate
or save the final image — the residual itself is the analog of the cogito-L40
hidden state.

Row layout matches the cogito convention:
    row = color_idx * n_templates + template_idx_within_subset

Output (under ``runs/COLOR_SD_UNET_MID/`` by default):
    X.npy        (n_prompts, D) float32 — mean-pooled spatial residual
    meta.json    {colors, templates, template_indices, n_colors, n_templates,
                  model_name, hook_block, n_steps, height, width, D, dtype}

Wall-time estimate on local Apple-Silicon MPS:
    SD-1.5      ~6-10 s per prompt @ 5 steps, 256x256 → 949×4 = 3796 prompts
                ≈ 6-10 hours (use n_colors=64 for the fast smoke path first).
    SDXL-Turbo  ~1-2 s per prompt @ 2 steps, 512x512  → ~1-2 hours full.

CAVEAT BUDGET (matches the spec):
    - If SD-1.5 OOMs locally, the CLI in ``scripts/harvest_sd_unet.py`` falls
      back to ``stabilityai/sdxl-turbo``.
    - Use ``--n_colors 64`` for the fast smoke path first.
    - If still slow, document honestly and use ``n_colors=128`` subset.

This module avoids importing diffusers/torch at module load so the file is
importable even on machines without the heavy deps (e.g. CI). Both are
imported lazily inside ``harvest_sd``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from manifold_sae.cross_llm.harvest_local import (
    TEMPLATES,
    build_color_prompts,
    load_xkcd_colors,
)


# Default = mid-stack mid_block (deepest, smallest spatial extent → low-D pool).
# down_blocks[2] is also reasonable; we expose ``hook_block`` so the caller
# can choose.
DEFAULT_HOOK = "mid_block"


@dataclass
class HarvestConfig:
    model_name: str = "runwayml/stable-diffusion-v1-5"
    hook_block: str = DEFAULT_HOOK     # "mid_block" | "down_blocks.2" | "up_blocks.1"
    n_steps: int = 5                   # denoising steps to run (residual is captured each step then mean-pooled across steps)
    height: int = 256                  # downsize from default 512 for MPS speed
    width: int = 256
    device: str = "mps"                # "mps" | "cuda" | "cpu"
    dtype: str = "fp16"                # "fp16" | "fp32"
    pool: str = "mean_spatial_time"    # mean over (H', W') and over n_steps → (D,)
    seed: int = 0


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
def _pick_device(prefer: str) -> str:
    import torch
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_hook_module(unet, hook_block: str):
    """Return the nn.Module to hook for ``hook_block`` ('mid_block' / 'down_blocks.2' / ...)."""
    parts = hook_block.split(".")
    obj = unet
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def _load_pipe(model_name: str, device: str, dtype_str: str):
    """Lazy import of diffusers; build a StableDiffusionPipeline / AutoPipeline."""
    import torch
    from diffusers import AutoPipelineForText2Image

    torch_dtype = torch.float16 if dtype_str == "fp16" and device != "cpu" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # Disable any safety-checker / NSFW filter heads — we never save images.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe = pipe.to(device)
    # Reduce memory; MPS doesn't support attention_slicing on every version
    # but it's a no-op when unavailable.
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_sd_prompts(
    template_subset: Sequence[int] | None = (0, 7, 16, 24),
    color_subset: Sequence[int] | None = None,
) -> tuple[list[str], dict]:
    """Same row layout as the cogito harvest. Default = 4-template subset."""
    return build_color_prompts(
        template_subset=template_subset,
        color_subset=color_subset,
    )


def harvest_sd(
    prompts: Iterable[str],
    cfg: HarvestConfig,
    output_dir: Path | str = "runs/COLOR_SD_UNET_MID",
    meta_extra: dict | None = None,
    progress_every: int = 32,
) -> np.ndarray:
    """Run diffusion + capture residuals.

    Returns
    -------
    X : np.ndarray, shape (n_prompts, D), float32
    """
    import torch

    device = _pick_device(cfg.device)
    pipe = _load_pipe(cfg.model_name, device, cfg.dtype)
    unet = pipe.unet
    hook_mod = _resolve_hook_module(unet, cfg.hook_block)

    captured: list[torch.Tensor] = []

    def _hook(_mod, _inp, out):
        # UNet blocks return either a Tensor or a tuple; the hidden-states are
        # always the first element. Detach + cast to cpu/fp32 at the boundary.
        h = out[0] if isinstance(out, (tuple, list)) else out
        # Mean-pool over spatial dims → (B, C). We accumulate per-step.
        if h.dim() == 4:               # (B, C, H', W')
            pooled = h.float().mean(dim=(2, 3))
        elif h.dim() == 3:             # (B, N_tok, C) for transformer blocks
            pooled = h.float().mean(dim=1)
        else:
            pooled = h.float().reshape(h.shape[0], -1)
        captured.append(pooled.detach().to("cpu"))

    handle = hook_mod.register_forward_hook(_hook)

    rows: list[np.ndarray] = []
    prompts_list = list(prompts)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    try:
        with torch.inference_mode():
            for i, prompt in enumerate(prompts_list):
                captured.clear()
                # Run only the denoising loop — output image is discarded.
                _ = pipe(
                    prompt=prompt,
                    num_inference_steps=cfg.n_steps,
                    height=cfg.height,
                    width=cfg.width,
                    guidance_scale=0.0,        # turbo / fast: no CFG
                    generator=generator,
                    output_type="latent",      # skip VAE-decode time
                )
                if not captured:
                    raise RuntimeError(f"hook captured nothing on prompt {i}")
                # mean over denoising steps → (B, C); B==1 here (and CFG-doubling
                # is avoided because guidance_scale=0). Take first row.
                stacked = torch.stack(captured, dim=0).mean(dim=0)  # (B, C)
                rows.append(stacked[0].numpy().astype(np.float32))
                if (i + 1) % progress_every == 0 or i == len(prompts_list) - 1:
                    print(f"[harvest_sd] {i + 1}/{len(prompts_list)} "
                          f"D={rows[0].shape[-1]}", flush=True)
    finally:
        handle.remove()

    X = np.stack(rows, axis=0)        # (n_prompts, D)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)

    meta = {
        "model_name": cfg.model_name,
        "hook_block": cfg.hook_block,
        "n_steps": cfg.n_steps,
        "height": cfg.height,
        "width": cfg.width,
        "dtype": cfg.dtype,
        "pool": cfg.pool,
        "seed": cfg.seed,
        "D": int(X.shape[1]),
        "n_prompts": int(X.shape[0]),
    }
    if meta_extra is not None:
        meta.update(meta_extra)
    # Serialize colors / templates if we got them from build_sd_prompts.
    # Tuples-of-(name, rgb) need to be JSON-coerced.
    if "colors" in meta:
        meta["colors"] = [
            (n, [int(c) for c in rgb]) for (n, rgb) in meta["colors"]
        ]
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[harvest_sd] saved {out_dir/'X.npy'} shape={X.shape}", flush=True)
    return X


__all__ = [
    "HarvestConfig",
    "DEFAULT_HOOK",
    "build_sd_prompts",
    "harvest_sd",
    "TEMPLATES",
    "load_xkcd_colors",
]
