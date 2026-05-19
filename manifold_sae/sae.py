"""Manifold-SAE: encoder + gamfit-backed manifold decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import decode
from .encoder import ManifoldEncoder
from .gamfit_glue import BasisSpec


@dataclass
class ManifoldSAEConfig:
    input_dim: int
    n_features: int
    n_basis: int
    hidden_dim: int | None = None
    sparsity_weight: float = 1e-3
    reml_weight: float = 1e-2
    position_spread_weight: float = 1e-3


@dataclass
class ManifoldSAEOutput:
    reconstruction: torch.Tensor
    positions: torch.Tensor
    amplitudes: torch.Tensor
    reml_score: torch.Tensor
    lambdas: torch.Tensor
    edf: torch.Tensor
    coefficients: torch.Tensor


class ManifoldSAE(nn.Module):
    """Encode (B, D) -> positions, amplitudes; decode via gamfit REML inner solve."""

    def __init__(self, config: ManifoldSAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ManifoldEncoder(
            input_dim=config.input_dim,
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
        )
        self.basis_spec: BasisSpec = BasisSpec(n_basis=config.n_basis)

    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        positions, amplitudes = self.encoder(x)
        fit = decode(positions, amplitudes, x, self.basis_spec)
        return ManifoldSAEOutput(
            reconstruction=fit["reconstruction"],
            positions=positions,
            amplitudes=amplitudes,
            reml_score=fit["reml_score"],
            lambdas=fit["lambdas"],
            edf=fit["edf"],
            coefficients=fit["coefficients"],
        )
