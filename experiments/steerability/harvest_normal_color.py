"""Control harvest: 30 'normal' color names (orange, deep red, pale yellow,
...) under the SAME 5 templates as harvest_hex.py for direct hex-vs-name
comparison. Saves (150, 7168) -> X_L40_normal.npy + normal_prompt_index.json.

Usage:
  python harvest_normal_color.py --dry-run
  COGITO_API_BASE=http://<host>:8000 python harvest_normal_color.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the encode/templates helpers from harvest_hex (sibling file).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harvest_hex import TEMPLATES, post_encode, DEFAULT_API, DEFAULT_LAYER  # noqa: E402


COLOR_NAMES = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "gray", "cyan", "magenta", "navy", "teal",
    "deep red", "pale yellow", "bright green", "dark blue", "light pink",
    "burnt orange", "forest green", "sky blue", "royal purple",
    "charcoal gray", "off white", "lime green", "hot pink",
    "olive", "beige",
]


def build_prompts(names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        for ti, tpl in enumerate(TEMPLATES):
            # The hex templates use {hex} placeholder; map name into it.
            rows.append({"name": name, "template_idx": ti,
                          "prompt": tpl.format(hex=name)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default=DEFAULT_API)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-x", default=str(HERE / "X_L40_normal.npy"))
    ap.add_argument("--out-idx", default=str(HERE / "normal_prompt_index.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts = build_prompts(COLOR_NAMES)
    print(f"[normal] {len(COLOR_NAMES)} color names x {len(TEMPLATES)} "
          f"templates = {len(prompts)} prompts", flush=True)

    if args.dry_run:
        print("[dry-run] first 3 requests that WOULD be POSTed:")
        for p in prompts[:3]:
            payload = {"texts": [p["prompt"]], "layers": [args.layer],
                       "aggregate": "mean", "max_length": 64}
            print(f"  POST {args.api_base.rstrip('/')}/v1/encode")
            print(f"    {json.dumps(payload)}")
        return 0

    feats: list[np.ndarray] = []
    t0 = time.time()
    key = f"layer_{args.layer}"
    for s in range(0, len(prompts), args.batch_size):
        batch = prompts[s:s + args.batch_size]
        texts = [r["prompt"] for r in batch]
        data = post_encode(args.api_base, texts, layer=args.layer)
        for r in data["results"]:
            arr = np.asarray(r[key], dtype=np.float32)
            feats.append(arr)
        elapsed = time.time() - t0
        rate = len(feats) / max(elapsed, 1e-6)
        eta = (len(prompts) - len(feats)) / max(rate, 1e-6)
        print(f"  [encode] {len(feats)}/{len(prompts)} ({rate:.1f}/s, "
              f"ETA {eta:5.1f}s)", flush=True)

    X = np.stack(feats, axis=0)
    np.save(args.out_x, X)
    Path(args.out_idx).write_text(json.dumps(prompts, indent=2))
    print(f"[done] X={X.shape} -> {args.out_x}", flush=True)
    print(f"[done] index ({len(prompts)} rows) -> {args.out_idx}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
