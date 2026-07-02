#!/usr/bin/env python
"""WS-D probe harvest: calendar + color token-site activations on the frontier LM.

Small dedicated harvests (not a corpus job) of the residual-stream activation at
a target token embedded in natural template sentences, for four cyclic/ordered
concept sets:

    weekday  -- 7-point circle (cyclic)
    month    -- 12-point circle (cyclic)
    year     -- ordered non-cyclic curve
    color    -- hue circle (cyclic), continuous HSV label

For each (set, layer) we write a small ShardWriter directory whose manifest
carries the per-row labels, ground-truth order/continuous coordinate, and the
cyclic flag, so the downstream curved-feature probes (WS-F) can align rows to
concepts. Readout is the residual at the LAST sub-token of the target word.

Same model / same mid-stack layers as the big harvest, so charts are comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

for _cand in (
    "/models/sauers_build/gam_fable/examples",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gam", "examples"),
    "/Users/user/gam/examples",
):
    if os.path.isfile(os.path.join(_cand, "residual_shard_io.py")):
        sys.path.insert(0, _cand)
        break
from residual_shard_io import MANIFEST_NAME, ShardWriter, tokenizer_hash  # noqa: E402

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
YEARS = [str(y) for y in range(1950, 2021, 5)]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple",
          "pink", "brown", "black", "white", "gray", "cyan", "magenta"]


def _color_hue() -> list[float]:
    import matplotlib.colors as mcolors
    out = []
    for name in COLORS:
        rgb = mcolors.to_rgb(mcolors.CSS4_COLORS.get(name, "#808080"))
        out.append(float(mcolors.rgb_to_hsv(rgb)[0]))  # hue in [0,1)
    return out


def token_sets() -> dict[str, dict]:
    return {
        "weekday": {"labels": WEEKDAYS, "order": list(range(7)), "cyclic": True,
                    "templates": ["I will see you on {x}.",
                                  "The meeting is scheduled for {x}.",
                                  "She was born on a {x}.",
                                  "We always rest on {x}.",
                                  "By {x}, everything was ready."]},
        "month": {"labels": MONTHS, "order": list(range(12)), "cyclic": True,
                  "templates": ["It happened in {x}.",
                                "We got married in {x}.",
                                "The festival is held every {x}.",
                                "By {x} the snow had melted.",
                                "Her birthday is in {x}."]},
        "year": {"labels": YEARS, "order": [int(y) for y in YEARS], "cyclic": False,
                 "templates": ["It happened in {x}.",
                               "The book was published in {x}.",
                               "She was born in {x}.",
                               "By the year {x} everything had changed.",
                               "The war ended in {x}."]},
        "color": {"labels": COLORS, "order": _color_hue(), "cyclic": True,
                  "templates": ["The wall was painted {x}.",
                                "She wore a bright {x} dress.",
                                "The sky turned {x}.",
                                "He bought a {x} car.",
                                "The flowers were mostly {x}."]},
    }


def target_last_pos(tok, template: str, label: str) -> tuple[list[int], int]:
    """Token ids for template.format(x=label) and the index of the label's last
    sub-token. Computed by prefix-length differencing (no special tokens)."""
    prefix = template.split("{x}")[0]
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tok(template.format(x=label), add_special_tokens=False)["input_ids"]
    with_label = tok(prefix + label, add_special_tokens=False)["input_ids"]
    last = len(with_label) - 1
    if last >= len(full_ids):
        last = len(full_ids) - 1
    if last < len(pre_ids):
        last = len(full_ids) - 1  # fallback: last token overall
    return full_ids, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_gpu = torch.cuda.device_count()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map="auto" if n_gpu > 1 else 0, low_cpu_mem_usage=True,
    ).eval()
    in_device = next(model.parameters()).device
    dec = model.model.layers
    d_model = int(model.config.hidden_size)

    grabbed: dict[int, torch.Tensor] = {}

    def make_hook(L: int):
        def hook(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [dec[L].register_forward_hook(make_hook(L)) for L in layers]

    sets = token_sets()
    written = {}
    for sname, spec in sets.items():
        rows = {L: [] for L in layers}
        row_labels, row_order = [], []
        for li, label in enumerate(spec["labels"]):
            for template in spec["templates"]:
                ids, pos = target_last_pos(tok, template, label)
                t = torch.tensor([ids], device=in_device)
                with torch.no_grad():
                    model(t)
                for L in layers:
                    rows[L].append(grabbed[L][0, pos, :].float().cpu().numpy())
                grabbed.clear()
                row_labels.append(label)
                row_order.append(spec["order"][li])
        for L in layers:
            d = os.path.join(args.out_root, f"{args.tag}_probe_{sname}_l{L}")
            w = ShardWriter(d, d_model=d_model, rows_per_shard=10_000_000, meta={
                "model_name": args.model, "layer": L, "probe_set": sname,
                "cyclic": spec["cyclic"], "labels": row_labels,
                "order": row_order, "templates": spec["templates"],
                "tokenizer_hash": tokenizer_hash(tok), "workstream": "WS-D",
                "readout": "residual at last sub-token of target word",
            })
            w.append(np.stack(rows[L]).astype(np.float32))
            man = w.close()
            with open(os.path.join(d, MANIFEST_NAME), "w") as f:
                json.dump(man, f)
            written[f"{sname}_l{L}"] = {"dir": d, "rows": man["total_tokens"]}
        print(f"[probe] {sname}: {len(row_labels)} rows/layer x {len(layers)} layers", flush=True)

    for h in handles:
        h.remove()
    with open(os.path.join(args.out_root, f"{args.tag}_PROBES_SUMMARY.json"), "w") as f:
        json.dump(written, f, indent=2)
    print("[probe] DONE", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # avoid pyarrow/torch teardown abort; outputs already flushed


if __name__ == "__main__":
    main()
