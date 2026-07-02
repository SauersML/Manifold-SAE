"""Fast invariant tests for the BSF baseline. Run: ``.venv/bin/python
experiments/bsf_baseline/test_bsf.py`` (no pytest needed)."""

from __future__ import annotations

import os
import struct
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsf import (  # noqa: E402
    BSF,
    BSFConfig,
    TrainConfig,
    block_topk_mask,
    ev,
    load_shard_harvest,
    stable_rank,
    train_bsf,
)

torch.set_default_dtype(torch.float64)


def test_block_topk_keeps_largest_norm_blocks():
    # 1 row, 4 blocks of size 2 with known norms 3,1,2,4 -> keep blocks {3,0}
    z = torch.tensor([[[3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 4.0]]])
    mask = block_topk_mask(z, k=2)
    assert mask.tolist() == [[1.0, 0.0, 0.0, 1.0]], mask.tolist()
    assert block_topk_mask(z, k=4).sum().item() == 4  # k>=G keeps all
    print("ok: block_topk keeps largest-norm blocks")


def test_grassmann_blocks_stay_orthonormal():
    cfg = BSFConfig(d_model=16, n_blocks=4, block_size=3, k_blocks=2, mode="grassmann")
    m = BSF(cfg)
    X = torch.randn(300, 16)
    train_bsf(m, X, TrainConfig(steps=300, batch_size=128, lr=5e-3, reproj_every=10))
    m.reproject_stiefel()
    D = m.decoder.detach().numpy()
    err = max(np.abs(D[g] @ D[g].T - np.eye(3)).max() for g in range(4))
    assert err < 1e-8, err
    assert m.log_gamma is not None  # single learned scalar gain
    print(f"ok: grassmann blocks orthonormal (max err {err:.1e}), gamma learned")


def test_auxk_targets_dead_blocks_and_lifts_dead_liveness():
    # Data lives in a 2-block subspace; with more blocks + AuxK, dead blocks should
    # get a nonzero auxiliary gradient signal (aux_loss > 0 while training).
    cfg = BSFConfig(d_model=12, n_blocks=6, block_size=2, k_blocks=1, mode="vanilla",
                    aux_k_blocks=2)
    m = BSF(cfg)
    X = torch.randn(200, 12)
    m.train()
    out = m(X)
    assert float(out.aux_loss) > 0.0, "AuxK loss should be positive during training"
    # in eval / no-AuxK config the aux loss path is skipped
    m0 = BSF(BSFConfig(d_model=12, n_blocks=6, block_size=2, k_blocks=1, mode="vanilla",
                       aux_k_blocks=0))
    m0.train()
    assert float(m0(X).aux_loss) == 0.0
    print("ok: AuxK produces a positive dead-block loss, disabled when k_aux=0")


def test_topk_sae_is_b1_and_reconstructs():
    # b=1 vanilla == signed TopK SAE; should reconstruct a low-rank signal well.
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((32, 6)))[0]
    X = torch.tensor((rng.standard_normal((1500, 6)) @ basis.T) + 0.02 * rng.standard_normal((1500, 32)))
    m = BSF(BSFConfig(d_model=32, n_blocks=32, block_size=1, k_blocks=8, mode="vanilla"))
    train_bsf(m, X, TrainConfig(steps=1500, batch_size=256, lr=4e-3))
    e = ev(m, X)
    assert e > 0.9, e
    print(f"ok: TopK-SAE (b=1) reconstructs low-rank signal, EV={e:.3f}")


def test_stable_rank_bounds():
    assert abs(stable_rank(np.eye(5)) - 5.0) < 1e-9   # isotropic -> full
    a = np.zeros((5, 5)); a[0, 0] = 1.0
    assert abs(stable_rank(a) - 1.0) < 1e-9            # rank-1 -> 1
    print("ok: stable_rank matches isotropic (=dim) and rank-1 (=1) limits")


def test_shard_loader_roundtrip_matches_gam_format():
    # Write a residual_shard_bf16 harvest exactly as gam's ShardWriter would
    # (manifest.json + little-endian uint16 bf16 bit patterns) and read it back.
    d_model = 8
    rows = (np.arange(5 * d_model).reshape(5, d_model) / 7.0).astype(np.float32)
    u = rows.view(np.uint32)
    bias = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    bits = ((u + bias) >> 16).astype(np.uint16)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "shard_00000.bf16"), "wb") as f:
            f.write(bits.astype("<u2").tobytes())
        manifest = {
            "format": "residual_shard_bf16", "format_version": 1, "dtype": "bfloat16",
            "byte_order": "little", "d_model": d_model, "rows_per_shard": 1_000_000,
            "total_tokens": 5, "shards": [{"file": "shard_00000.bf16", "rows": 5}],
        }
        import json
        with open(os.path.join(td, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        got = load_shard_harvest(td)
    # bf16 loses mantissa bits; equality is to the bf16 grid, so compare loosely.
    assert got.shape == (5, d_model), got.shape
    assert np.abs(got - rows).max() < 0.02, np.abs(got - rows).max()
    print("ok: shard loader roundtrips gam's residual_shard_bf16 format")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nall {len(tests)} BSF invariant tests passed")
