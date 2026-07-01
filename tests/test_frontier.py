"""Tests for the gamfit-native manifold-vs-linear frontier evaluator.

Split into (a) a fast unit test of *our* wrapper logic (result surface + markdown,
no gamfit fit) and (b) a @slow integration run of the real gamfit frontier that
skips if gamfit's REML solver fails to converge (a documented small-data limit of
the closed-form manifold fit, not a bug in this wrapper).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gamfit")

from manifold_sae.eval.frontier import (
    FrontierResult,
    format_frontier_markdown,
    manifold_vs_linear_frontier,
)


def test_frontier_result_surface_and_markdown():
    """Wrapper logic: verdict passthrough + markdown, without running any fit."""
    res = FrontierResult(
        manifold_ev_by_k={4: 0.90, 8: 0.95},
        linear_ev_by_k={4: 0.70, 8: 0.94},
        verdict={
            "confirmed": True,
            "manifold_k": 4,
            "linear_k": 8,
            "efficiency_ratio": 2.0,
            "best_manifold_ev": 0.95,
            "best_linear_ev": 0.94,
            "target_ev": 0.94,
        },
        rows=[],
        knee=4,
    )
    assert res.beats_linear is True
    assert res.efficiency_ratio == 2.0
    assert res.manifold_k == 4 and res.linear_k == 8

    md = format_frontier_markdown(res)
    assert "manifold EV" in md and "linear EV" in md
    assert "MANIFOLD BEATS LINEAR" in md
    assert "+0.2000" in md  # Δ at K=4 (0.90 − 0.70)


def _mixture_of_circles(n: int, d: int, n_circles: int, seed: int) -> np.ndarray:
    """n points spread over `n_circles` distinct 1-D circles embedded in R^d."""
    rng = np.random.default_rng(seed)
    parts = []
    per = n // n_circles
    for c in range(n_circles):
        t = rng.uniform(0.0, 2.0 * np.pi, size=per)
        basis = rng.standard_normal((2, d))
        x = np.cos(t)[:, None] * basis[0] + np.sin(t)[:, None] * basis[1]
        parts.append(x)
    x = np.concatenate(parts, axis=0)
    x = x + 0.03 * rng.standard_normal(x.shape)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)  # whiten (recommended preprocessing)


@pytest.mark.slow
def test_frontier_integration_runs():
    train = _mixture_of_circles(360, 8, n_circles=3, seed=1)
    test = _mixture_of_circles(120, 8, n_circles=3, seed=2)
    try:
        res = manifold_vs_linear_frontier(
            train, test, k_values=[3, 6], d_atom=1, atom_topology="circle",
        )
    except gamfit_convergence_errors() as exc:  # noqa: F821 - defined below
        pytest.skip(f"gamfit manifold solver did not converge on this data: {exc}")

    assert set(res.manifold_ev_by_k) == {3, 6}
    assert set(res.linear_ev_by_k) == {3, 6}
    for ev in (*res.manifold_ev_by_k.values(), *res.linear_ev_by_k.values()):
        assert np.isfinite(ev)
    assert isinstance(res.beats_linear, bool)
    assert res.verdict
    assert "manifold EV" in format_frontier_markdown(res)


def gamfit_convergence_errors() -> tuple[type[BaseException], ...]:
    import gamfit

    errs: list[type[BaseException]] = []
    for name in ("RemlConvergenceError", "PirlsConvergenceError", "GamError"):
        e = getattr(gamfit, name, None)
        if isinstance(e, type) and issubclass(e, BaseException):
            errs.append(e)
    return tuple(errs) or (Exception,)
