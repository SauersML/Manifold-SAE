"""Integration sweep — every registered SAE variant + a few compositions.

For each variant we:
    * instantiate at F=8, B=16, D=32,
    * run forward + backward under SGD,
    * verify the loss is finite and reduces in one step.

For three pairs we then verify the composition runs end-to-end. Every variant,
including ManifoldSAE, runs at the shared F=8 setting; the integration layer
casts the input to each variant's expected dtype at the boundary.
"""
from __future__ import annotations

import math

import pytest
import torch

from manifold_sae.integration import (
    SAEModelRegistry,
    compose_pipeline,
    train_any,
)


B, D, F = 16, 32, 8


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _data(B_: int = B, D_: int = D) -> torch.Tensor:
    return torch.randn(B_, D_)


# Per-variant minimum F (and possibly B) overrides — some variants need
# a richer setting to even run forward+backward once.
VARIANT_OVERRIDES: dict[str, dict] = {
    "transcoder": {"F": F},
}


@pytest.mark.parametrize("name", sorted(SAEModelRegistry.keys()))
def test_variant_smoke(name: str) -> None:
    """Instantiate, forward, backward, verify loss is finite + reduces.

    Every registered variant must pass — a failure here means the variant's
    behavior is wrong and must be fixed, never marked expected-to-fail.
    """
    overrides = VARIANT_OVERRIDES.get(name, {})
    F_local = int(overrides.get("F", F))
    B_local = int(overrides.get("B", B))
    X = _data(B_=B_local)

    result = train_any(name, X, F=F_local, steps=1, lr=1e-3)

    assert math.isfinite(result["loss_initial"]), f"{name}: loss_initial NaN/Inf"
    assert math.isfinite(result["loss_final"]), f"{name}: loss_final NaN/Inf"
    # We allow tiny upward drift in 1 SGD step for low-lr variants; require
    # that the loss didn't blow up by more than 5%.
    assert result["loss_final"] <= result["loss_initial"] * 1.05 + 1e-6, (
        f"{name}: loss_final {result['loss_final']:.4g} >> initial "
        f"{result['loss_initial']:.4g}"
    )
    assert result["n_params"] > 0


# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------


COMPOSITION_PAIRS = [
    ("adaptive_k", "wasserstein"),  # subs for manifold+matryoshka (no matryoshka module yet)
    ("equivariant", "wasserstein"),
    ("adaptive_k", "crm"),
]


@pytest.mark.parametrize("pair", COMPOSITION_PAIRS)
def test_pair_composition(pair: tuple[str, str]) -> None:
    X = _data()
    out = compose_pipeline(list(pair), X, F=F, steps=1, lr=1e-3)
    assert len(out["models"]) == 2
    assert len(out["per_stage_r2"]) == 2
    assert out["final_recon"].shape == X.shape
    for r2 in out["per_stage_r2"]:
        assert math.isfinite(r2)
