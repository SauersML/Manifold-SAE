"""Harvest Qwen2.5-0.5B layer-12 activations on the xkcd-color prompt set.

Hardware
--------
- Apple Silicon (M-series), MPS backend, ~2 GB GPU memory required.
- ~3796 prompts (949 colors × 4 templates) at hidden_dim=896 (f32 on disk).
- Expected wall time: ~6-10 minutes on M2/M3.
- Disk: ~13 MB float32 (NOT 1.5 GB — the docstring in the original spec
  used Cogito's 7168-d width; Qwen2.5-0.5B is only 896-d).

Output
------
runs/COLOR_QWEN_05B_L12/
  X.npy               (n_prompts, 896) float32
  meta.json           {colors, templates, template_indices, n_colors, n_templates,
                       model_name, layer_idx, layer_total}

The row layout matches the cogito-L40 convention restricted to the chosen
template subset: row = color_idx * n_templates + template_idx_within_subset.
This makes the row pairing for the UniversalSAE explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifold_sae.cross_llm.harvest_local import (
    build_color_prompts,
    harvest,
)


# Templates chosen as a representative subset of the 28 cogito templates,
# covering distinct semantic / syntactic categories. Indices into
# experiments/color_geometry.py TEMPLATES list.
DEFAULT_TEMPLATE_SUBSET = [0, 7, 16, 24]
#   0: "She slipped into a {x} silk dress ..."          (fashion)
#   7: "She dipped her brush in the {x} pool of paint." (art)
#  16: "She bit into the macaron, finding a soft {x} filling within."  (food)
#  24: "I bought a {x} fountain pen at the antique market." (object)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_LAYER = 12  # mid-stack of 24-layer Qwen2.5-0.5B


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument(
        "--templates", type=int, nargs="+", default=DEFAULT_TEMPLATE_SUBSET,
        help="Indices into the 28-template list.",
    )
    ap.add_argument("--n-colors", type=int, default=None,
                    help="Limit to first N xkcd colors (default: all 949).")
    ap.add_argument("--out-dir", default="runs/COLOR_QWEN_05B_L12")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="MPS often slower with batching due to padding; default 1.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    color_subset = list(range(args.n_colors)) if args.n_colors else None
    prompts, meta = build_color_prompts(
        template_subset=args.templates, color_subset=color_subset,
    )
    print(f"[harvest_qwen_small] n_prompts={len(prompts)} "
          f"colors={meta['n_colors']} templates={meta['n_templates']}",
          flush=True)
    print(f"[harvest_qwen_small] template indices: {meta['template_indices']}",
          flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = harvest(
        args.model, prompts, layer_idx=args.layer,
        output_path=out_dir / "X.npy",
        device=args.device, batch_size=args.batch_size,
    )

    # Persist meta alongside X. Convert RGB tuples to lists for JSON.
    meta_out = {
        "model_name": args.model,
        "layer_idx": args.layer,
        "n_colors": meta["n_colors"],
        "n_templates": meta["n_templates"],
        "template_indices": meta["template_indices"],
        "templates": meta["templates"],
        "colors": [[name, list(rgb)] for name, rgb in meta["colors"]],
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "row_layout": "row = color_idx * n_templates + template_subset_idx",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"[harvest_qwen_small] meta -> {out_dir/'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
