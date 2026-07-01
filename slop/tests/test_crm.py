"""Tests for Complete Replacement Model skeleton."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.crm import CompleteReplacementModel, CRMConfig


def test_chain_shapes():
    cfg = CRMConfig(layer_dims=[16, 24, 20], n_features_per_sae=32,
                    transcoder_mid=64)
    model = CompleteReplacementModel(cfg)
    xs = [torch.randn(5, d) for d in [16, 24, 20]]
    out = model.forward(xs)
    assert len(out["recons"]) == 3
    for r, d in zip(out["recons"], [16, 24, 20]):
        assert r.shape == (5, d)
    assert len(out["latents_tc"]) == 2


def test_loss_and_backward():
    cfg = CRMConfig(layer_dims=[8, 12, 10], n_features_per_sae=16,
                    transcoder_mid=32)
    model = CompleteReplacementModel(cfg)
    xs = [torch.randn(4, d) for d in [8, 12, 10]]
    out = model.loss(xs)
    out["loss"].backward()
    assert out["loss"].item() > 0
    assert len(out["per_stage_mse"]) == 3


def test_per_stage_r2_runs():
    torch.manual_seed(0)
    cfg = CRMConfig(layer_dims=[8, 8, 8], n_features_per_sae=16,
                    transcoder_mid=32)
    model = CompleteReplacementModel(cfg)
    xs = [torch.randn(20, 8) for _ in range(3)]
    r2 = model.per_stage_r2(xs)
    assert len(r2) == 3
    # all finite
    for r in r2:
        assert r == r  # not nan
