"""Shared pytest configuration for the Manifold-SAE test suite.

Sets the torch default dtype to f64 (the gamfit Rust backend is f64) and seeds
numpy and torch for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: heavier numerical checks (gradcheck) that take a few seconds each",
    )


@pytest.fixture(scope="session", autouse=True)
def _torch_default_dtype_f64() -> None:
    """All tests run with torch default dtype f64 (gamfit's backend is f64)."""
    torch.set_default_dtype(torch.float64)


@pytest.fixture(autouse=True)
def _seed_rngs() -> None:
    """Deterministic per-test seeds so failures are reproducible."""
    np.random.seed(0)
    torch.manual_seed(0)
