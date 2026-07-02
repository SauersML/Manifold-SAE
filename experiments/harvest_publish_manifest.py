#!/usr/bin/env python
"""Assemble the top-level harvest MANIFEST.json from all shard-dir manifests.

Scans an --out-root for residual_shard harvest directories (big multi-layer sets
and small probe sets), reads each ``manifest.json``, and writes a single
``MANIFEST.json`` index that C-tier1 / WS-E / WS-F can consume to locate shard
paths, token counts, d_model, layer, and the T0 stats block.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(d: str) -> dict | None:
    p = os.path.join(d, "manifest.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    corpus, probes = [], []
    for d in sorted(glob.glob(os.path.join(args.out_root, "*"))):
        if not os.path.isdir(d):
            continue
        man = load(d)
        if man is None or man.get("format") != "residual_shard_bf16":
            continue
        entry = {
            "dir": d,
            "model_name": man.get("model_name"),
            "layer": man.get("layer"),
            "d_model": man.get("d_model"),
            "total_tokens": man.get("total_tokens"),
            "n_shards": len(man.get("shards", [])),
            "provisional": man.get("provisional", False),
        }
        if "probe_set" in man:
            entry["probe_set"] = man["probe_set"]
            entry["cyclic"] = man.get("cyclic")
            entry["n_rows"] = man.get("total_tokens")
            probes.append(entry)
        else:
            t0 = man.get("t0", {})
            entry["has_t0"] = bool(t0)
            entry["n_rogue_dims"] = len(t0.get("rogue_dims", {}).get("index", []))
            entry["scale_median_std"] = t0.get("scale_median_std")
            entry["token_cap"] = man.get("token_cap")
            corpus.append(entry)

    partial = any(e.get("provisional") or not e.get("has_t0") for e in corpus)
    out = {
        "harvest_root": args.out_root,
        "partial": partial,
        "reader": "gam/examples/residual_shard_io.py :: ShardReader(dir).batches(n)",
        "note": ("bf16 memmap shards; each manifest carries per-dim T0 "
                 "(mean/std/rms/rogue_dims/scale). partial=true means the big "
                 "harvest is still running — shards are append-only, re-open "
                 "load_shards() to pick up new rows; T0 finalizes at completion."),
        "corpus_sets": corpus,
        "probe_sets": probes,
    }
    outp = os.path.join(args.out_root, "MANIFEST.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[publish] wrote {outp}: {len(corpus)} corpus + {len(probes)} probe sets")
    for e in corpus:
        print(f"  corpus {e['dir']} L{e['layer']} tokens={e['total_tokens']} "
              f"d={e['d_model']} rogue={e['n_rogue_dims']}")
    for e in probes:
        print(f"  probe  {e['dir']} {e.get('probe_set')} L{e['layer']} rows={e['n_rows']}")


if __name__ == "__main__":
    main()
