#!/usr/bin/env python
"""End-to-end sanity check of a residual_shard harvest dir through load_shards.

Reads the first batch via the real reader path, and (if present) the T0 block,
asserting dtype/shape/finiteness so a format bug surfaces on a small slice rather
than after the full harvest. Exits non-zero on any failure.
"""
from __future__ import annotations

import os
import sys

import numpy as np

for _cand in (
    "/models/sauers_build/gam_fable/examples",
    "/Users/user/gam/examples",
):
    if os.path.isfile(os.path.join(_cand, "residual_shard_io.py")):
        sys.path.insert(0, _cand)
        break
from residual_shard_io import load_shards  # noqa: E402


def main() -> None:
    d = sys.argv[1]
    r = load_shards(d)
    assert r.d_model > 0, "d_model not positive"
    it = r.batches(4096)
    batch = next(it)
    assert batch.dtype == np.float32, f"batch dtype {batch.dtype} != float32"
    assert batch.ndim == 2 and batch.shape[1] == r.d_model, f"bad shape {batch.shape}"
    assert np.isfinite(batch).all(), "non-finite values in first batch"
    man = r.manifest
    t0 = man.get("t0")
    checks = {
        "d_model": r.d_model,
        "total_tokens": r.total_tokens,
        "n_shards": len(r.shards),
        "batch_shape": list(batch.shape),
        "batch_finite": True,
        "batch_abs_max": float(np.abs(batch).max()),
        "has_t0": bool(t0),
    }
    if t0:
        for k in ("mean", "std", "rms"):
            v = np.asarray(t0[k], dtype=np.float64)
            assert v.shape[0] == r.d_model, f"t0[{k}] wrong length"
            assert np.isfinite(v).all(), f"t0[{k}] has non-finite"
        checks["t0_scale_median_std"] = t0["scale_median_std"]
        checks["t0_n_rogue"] = len(t0["rogue_dims"]["index"])
        checks["t0_finite"] = True
    print("VALIDATE_OK", checks)


if __name__ == "__main__":
    main()
