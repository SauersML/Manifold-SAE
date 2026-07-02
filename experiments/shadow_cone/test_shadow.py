"""Fast invariant tests for shadow_cone. Run:
``.venv/bin/python experiments/shadow_cone/test_shadow.py``."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shadow_cone import GatedBSF, GatedConfig, roc_auc, train_gated  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_roc_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(500)
    labels = (scores + 0.5 * rng.standard_normal(500) > 0).astype(int)
    from sklearn.metrics import roc_auc_score
    ours = roc_auc(scores, labels)
    ref = float(roc_auc_score(labels, scores))
    assert abs(ours - ref) < 1e-9, (ours, ref)
    # perfect + inverted limits
    assert abs(roc_auc(np.array([0.0, 1.0, 2.0]), np.array([0, 0, 1])) - 1.0) < 1e-9
    assert abs(roc_auc(np.array([2.0, 1.0, 0.0]), np.array([0, 0, 1])) - 0.0) < 1e-9
    print(f"ok: roc_auc matches sklearn (ours={ours:.4f}, ref={ref:.4f})")


def test_gate_is_bounded_and_hard_thresholds():
    m = GatedBSF(GatedConfig(d_model=12, n_blocks=4, block_size=3, seed=0))
    x = torch.randn(50, 12)
    a = m.gate(x)
    assert (a >= 0).all() and (a < 1.0 + 1e-9).all(), "soft gate must be in [0,1)"
    ah = m.gate(x, hard=True)
    assert set(np.unique(ah.detach().numpy()).tolist()) <= {0.0, 1.0}, "hard gate must be 0/1"
    # gate is zero exactly where the presence logit is non-positive
    ell = m.presence_logit(x)
    assert bool((a[ell <= 0] == 0).all()), "gate must be dark when logit<=0"
    print("ok: presence gate bounded [0,1), hard gate is 0/1, dark below threshold")


def test_gated_blocks_reconstruct_and_stay_orthonormal():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((24, 4)))[0]
    X = torch.tensor((rng.standard_normal((800, 4)) @ basis.T) + 0.02 * rng.standard_normal((800, 24)))
    m = GatedBSF(GatedConfig(d_model=24, n_blocks=6, block_size=4, l0_coef=1e-3, seed=0))
    train_gated(m, X, steps=800, lr=4e-3)
    D = m.decoder.detach().numpy()
    err = max(np.abs(D[g] @ D[g].T - np.eye(4)).max() for g in range(6))
    assert err < 1e-8, err
    x_hat, _, _ = m(X)
    sst = float(((X - X.mean(0)) ** 2).sum())
    ev = 1.0 - float(((x_hat - X) ** 2).sum()) / sst
    assert ev > 0.7, ev
    print(f"ok: gated BSF reconstructs (EV={ev:.3f}) with orthonormal blocks (err {err:.1e})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nall {len(tests)} shadow_cone invariant tests passed")
