"""CLI: harvest UNet residuals for the SAE-on-Diffusion contrarian probe.

Walltime estimate (Apple Silicon MPS, no warm cache):

    Model              steps  H×W       per-prompt   3796 prompts
    sd-1.5             5      256×256   ~6-10 s      ~6-10 hours
    sdxl-turbo         2      512×512   ~1-2 s       ~1-2 hours
    sdxl-turbo         1      512×512   ~0.5-1 s     ~30-60 min

The default fast-smoke path uses ``--n_colors 64 --model sdxl-turbo`` so the
caller can verify the pipeline in ~5-10 min before committing to the full 949.

Usage
-----
    # Fast smoke (~5-10 min):
    python scripts/harvest_sd_unet.py --n_colors 64 --model sdxl-turbo

    # Full SD-1.5 (multi-hour):
    python scripts/harvest_sd_unet.py --model sd1.5 --out_dir runs/COLOR_SD_UNET_MID

    # SDXL-Turbo full (faster):
    python scripts/harvest_sd_unet.py --model sdxl-turbo --out_dir runs/COLOR_SD_UNET_MID
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from manifold_sae.diffusion.harvest_sd import (
    HarvestConfig,
    DEFAULT_HOOK,
    build_sd_prompts,
    harvest_sd,
)


MODEL_ALIASES = {
    "sd1.5":       "runwayml/stable-diffusion-v1-5",
    "sd1.4":       "CompVis/stable-diffusion-v1-4",
    "sdxl-turbo":  "stabilityai/sdxl-turbo",
    "sd-turbo":    "stabilityai/sd-turbo",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sdxl-turbo",
                    help="alias (sd1.5 / sd1.4 / sdxl-turbo / sd-turbo) or full HF id")
    ap.add_argument("--hook_block", default=DEFAULT_HOOK,
                    help="UNet sub-module to hook (e.g. mid_block, down_blocks.2)")
    ap.add_argument("--n_steps", type=int, default=2)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    ap.add_argument("--n_colors", type=int, default=64,
                    help="default 64 for the fast smoke path; use 949 for full")
    ap.add_argument("--templates", type=int, nargs="+", default=[0, 7, 16, 24])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/COLOR_SD_UNET_MID")
    args = ap.parse_args()

    model_id = MODEL_ALIASES.get(args.model, args.model)
    cfg = HarvestConfig(
        model_name=model_id,
        hook_block=args.hook_block,
        n_steps=args.n_steps,
        height=args.height,
        width=args.width,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    color_subset = list(range(args.n_colors)) if args.n_colors > 0 else None
    prompts, meta = build_sd_prompts(
        template_subset=tuple(args.templates),
        color_subset=color_subset,
    )

    print(f"[harvest_sd_unet] model={model_id}  hook={args.hook_block}  "
          f"steps={args.n_steps}  res={args.height}x{args.width}  "
          f"prompts={len(prompts)}  ({meta['n_colors']}×{meta['n_templates']})",
          flush=True)
    t0 = time.time()
    X = harvest_sd(
        prompts=prompts,
        cfg=cfg,
        output_dir=args.out_dir,
        meta_extra=meta,
        progress_every=16,
    )
    dt = time.time() - t0
    per = dt / max(1, len(prompts))
    full_estimate_h = per * 949 * len(args.templates) / 3600.0
    print(f"[harvest_sd_unet] DONE in {dt:.1f}s  ({per:.2f}s/prompt)  "
          f"X.shape={tuple(X.shape)}", flush=True)
    print(f"[harvest_sd_unet] EXTRAPOLATED full 949-color × "
          f"{len(args.templates)}-template: ~{full_estimate_h:.2f}h",
          flush=True)

    # Append timing info to meta.json
    meta_path = Path(args.out_dir) / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        m["wall_seconds"] = dt
        m["per_prompt_seconds"] = per
        m["extrapolated_full_949_hours"] = full_estimate_h
        meta_path.write_text(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
