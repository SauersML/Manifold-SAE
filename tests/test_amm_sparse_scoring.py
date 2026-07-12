"""Exact compact-support scoring checks for the full AMM benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AMM_DIR = Path(__file__).resolve().parents[1] / "experiments" / "amm_zoo"
sys.path.insert(0, str(AMM_DIR))

from amm import generate_amm  # noqa: E402
from metrics import _oracle_recovered, score_arm  # noqa: E402


def test_compact_oracle_scoring_is_exact() -> None:
    dataset = generate_amm(
        seed=19,
        sigma_frac=0.05,
        n_train=500,
        n_test=240,
        d=32,
        k=3,
    )
    recovered = _oracle_recovered(dataset, "test")
    for factor in recovered:
        assert factor.contribution.shape == (factor.rows.size, dataset.d)
        assert factor.rows.size == int(factor.active.sum())
        assert np.all(factor.active[factor.rows])
        assert factor.rows.size < dataset.test.n

    report = score_arm(dataset, recovered, "test", seed=19, geodesic_sample=64)
    assert report["overall"]["mean_contribution_r2"] == 1.0
    assert report["overall"]["topology_id_accuracy"] == 1.0
    assert report["overall"]["intrinsic_dim_accuracy"] == 1.0
