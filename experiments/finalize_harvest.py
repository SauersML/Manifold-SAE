#!/usr/bin/env python
"""Finalize (or interim-stamp) a residual_shard harvest directory from its shards.

Streams the on-disk ``shard_*.bf16`` files, recomputes per-dim mean/RMS in
float64, and writes a complete ``manifest.json`` with the T0 block. Robust to a
harvest that was cancelled mid-run (or an actively-growing last shard: only the
complete rows on disk are counted). Provenance meta is carried over from an
existing (provisional) manifest when present.

Usage:
  finalize_harvest.py <harvest_dir> [--d-model N]   # N required if no manifest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

for _cand in (
    "/models/sauers_build/gam_fable/examples",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gam", "examples"),
    "/Users/user/gam/examples",
):
    if os.path.isfile(os.path.join(_cand, "residual_shard_io.py")):
        sys.path.insert(0, _cand)
        break
from residual_shard_io import (  # noqa: E402
    FORMAT_NAME, FORMAT_VERSION, MANIFEST_NAME, bf16_bits_to_float32,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_frontier_multilayer import compute_t0  # noqa: E402

_DTYPE = np.dtype("<u2")
_CARRY = ("model_name", "layer", "tokenizer_hash", "text_dataset", "text_subset",
          "seq_len", "workstream", "harvest_args", "token_cap", "probe_set",
          "cyclic", "labels", "order", "templates", "readout")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest_dir")
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--rows-per-shard", type=int, default=1_000_000)
    args = ap.parse_args()

    d = args.harvest_dir
    prev = {}
    mp = os.path.join(d, MANIFEST_NAME)
    if os.path.isfile(mp):
        with open(mp) as f:
            prev = json.load(f)
    d_model = args.d_model or prev.get("d_model")
    if not d_model:
        raise SystemExit("d_model unknown: pass --d-model")
    rowbytes = d_model * _DTYPE.itemsize

    shard_files = sorted(glob.glob(os.path.join(d, "shard_*.bf16")))
    if not shard_files:
        raise SystemExit(f"no shards in {d}")

    _sum = np.zeros(d_model, dtype=np.float64)
    _sumsq = np.zeros(d_model, dtype=np.float64)
    total = 0
    shards_meta = []
    for path in shard_files:
        nbytes = os.path.getsize(path)
        rows = nbytes // rowbytes  # floor: ignore a partially-written tail row
        if rows == 0:
            continue
        mm = np.memmap(path, dtype=_DTYPE, mode="r", shape=(rows, d_model))
        for i in range(0, rows, 200_000):
            blk = bf16_bits_to_float32(np.asarray(mm[i:i + 200_000])).astype(np.float64)
            _sum += blk.sum(axis=0)
            _sumsq += np.square(blk).sum(axis=0)
        total += rows
        shards_meta.append({"file": os.path.basename(path), "rows": int(rows),
                            "bytes": int(rows * rowbytes)})
        del mm

    mean = _sum / total
    rms = np.sqrt(_sumsq / total)
    manifest = {
        "format": FORMAT_NAME, "format_version": FORMAT_VERSION,
        "dtype": "bfloat16", "byte_order": "little", "d_model": int(d_model),
        "rows_per_shard": prev.get("rows_per_shard", args.rows_per_shard),
        "total_tokens": int(total), "shards": shards_meta,
        "stats": {"mean": mean.astype(np.float32).tolist(),
                  "norm": rms.astype(np.float32).tolist()},
        "finalized_from_shards": True,
    }
    for k in _CARRY:
        if k in prev:
            manifest[k] = prev[k]
    manifest["t0"] = compute_t0(mean, rms)
    with open(mp, "w") as f:
        json.dump(manifest, f)
    print(f"[finalize] {d}: total_tokens={total} shards={len(shards_meta)} "
          f"d_model={d_model} n_rogue={len(manifest['t0']['rogue_dims']['index'])} "
          f"scale_median_std={manifest['t0']['scale_median_std']:.4f}")


if __name__ == "__main__":
    main()
