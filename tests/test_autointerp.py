"""Smoke tests for manifold_sae.autointerp.{explain, score}.

Pure-numpy/torch — no model checkpoints loaded, no real activations needed.
Verifies the rule-based hypothesizer + simulation-accuracy regression + bootstrap
on a tiny synthetic atom designed to be trivially recoverable.
"""
from __future__ import annotations

import numpy as np
import pytest

from manifold_sae.autointerp.explain import (
    rgb_to_hsv,
    collect_top_activating,
    hypothesize_atom,
    AtomHypothesis,
)
from manifold_sae.autointerp.score import (
    simulation_features,
    score_hypothesis,
    aggregate_model_scores,
    bootstrap_ci,
)


N_COLORS = 40
N_TPL = 4


def _toy_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    color_names = [f"color_{i:02d}_{'red' if i < 10 else 'blue'}" for i in range(N_COLORS)]
    # First 10 colors red-ish, next 30 blue-ish
    rgb = np.zeros((N_COLORS, 3), dtype=np.float32)
    rgb[:10] = np.array([[0.9, 0.1, 0.1]]) + 0.05 * rng.standard_normal((10, 3))
    rgb[10:] = np.array([[0.1, 0.1, 0.9]]) + 0.05 * rng.standard_normal((30, 3))
    rgb = np.clip(rgb, 0.01, 0.99)
    color_hsv = rgb_to_hsv(rgb)
    N = N_COLORS * N_TPL
    row_color = np.arange(N) // N_TPL
    row_template = np.arange(N) % N_TPL
    # Atom fires strongly on red colors, template 0 dominant
    acts = np.zeros((N, 1), dtype=np.float32)
    for r in range(N):
        if row_color[r] < 10:
            base = 1.0 + 0.2 * rng.standard_normal()
            tpl_boost = 0.5 if row_template[r] == 0 else 0.0
            acts[r, 0] = max(0.0, base + tpl_boost)
    return color_names, color_hsv, row_color, row_template, acts


def test_rgb_to_hsv_shape():
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    hsv = rgb_to_hsv(rgb)
    assert hsv.shape == (3, 3)
    # red -> H≈0, green -> H≈1/3, blue -> H≈2/3
    assert hsv[0, 0] == pytest.approx(0.0, abs=0.02)
    assert hsv[1, 0] == pytest.approx(1.0 / 3.0, abs=0.02)
    assert hsv[2, 0] == pytest.approx(2.0 / 3.0, abs=0.02)


def test_collect_top_activating_returns_sorted_dicts():
    names, _, row_c, row_t, acts = _toy_data()
    top = collect_top_activating(acts, 0, row_c, row_t, names, n_top=10)
    assert len(top) == 10
    assert all(top[i]["act"] >= top[i + 1]["act"] for i in range(9))
    # all top examples should be from the red block
    assert all(t["color_idx"] < 10 for t in top)


def test_hypothesize_atom_produces_structured_fields():
    names, hsv, row_c, row_t, acts = _toy_data()
    top = collect_top_activating(acts, 0, row_c, row_t, names, n_top=20)
    h = hypothesize_atom(0, "ToyModel", top, hsv, names, n_templates=N_TPL)
    assert isinstance(h, AtomHypothesis)
    assert h.n_active == 20
    assert h.atom_id == 0
    assert h.model_name == "ToyModel"
    # Red sits near hue 0; hypothesizer should pick a tight range there.
    lo, hi = h.hue_range
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    # template_pattern should prefer template 0
    assert 0 in h.template_pattern
    # NL explanation is a non-empty string
    assert isinstance(h.explanation, str) and len(h.explanation) > 10


def test_dead_atom_handled_gracefully():
    names, hsv, *_ = _toy_data()
    h = hypothesize_atom(99, "ToyModel", [], hsv, names, n_templates=N_TPL)
    assert h.n_active == 0
    assert "dead" in h.explanation.lower()


def test_score_hypothesis_recovers_signal_for_planted_atom():
    names, hsv, row_c, row_t, acts = _toy_data()
    top = collect_top_activating(acts, 0, row_c, row_t, names, n_top=20)
    h = hypothesize_atom(0, "ToyModel", top, hsv, names, n_templates=N_TPL)
    sc = score_hypothesis(h, acts, hsv, names, row_c, row_t)
    # Planted atom (red + template 0) is essentially noiseless rule-defined.
    # R² should be high.
    assert sc["r2"] > 0.7, f"expected sim R² > 0.7 on planted atom, got {sc['r2']:.3f}"
    assert sc["n_active"] > 0


def test_simulation_features_shape():
    names, hsv, row_c, row_t, acts = _toy_data()
    top = collect_top_activating(acts, 0, row_c, row_t, names, n_top=20)
    h = hypothesize_atom(0, "ToyModel", top, hsv, names, n_templates=N_TPL)
    feats = simulation_features(h, hsv, names, row_c, row_t)
    assert feats.shape == (N_COLORS * N_TPL, 5)
    assert feats.min() >= 0.0 and feats.max() <= 1.0


def test_aggregate_filters_low_active_atoms():
    per_atom = [
        {"r2": 0.9, "n_active": 100, "mean_act": 1.0},
        {"r2": 0.8, "n_active": 50, "mean_act": 1.0},
        {"r2": -0.5, "n_active": 1, "mean_act": 0.0},  # should be filtered (< 5)
    ]
    agg = aggregate_model_scores(per_atom, min_active=5)
    assert agg["n_atoms_evaluated"] == 2
    assert agg["median_r2"] == pytest.approx(0.85)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=0.5, scale=0.2, size=50).tolist()
    pt, lo, hi = bootstrap_ci(vals, n_boot=500, statistic="mean", seed=0)
    assert lo < pt < hi
    assert pt == pytest.approx(float(np.mean(vals)))
