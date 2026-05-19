"""Manifold-SAE: encoder + gamfit Duchon REML decoder.

Each feature is a 1D smooth curve in residual stream space, fit via gamfit's
Duchon-basis REML solve (gamfit picks the smoothing parameter automatically
per feature). Amplitudes are binary (straight-through TopK) so they can't
leak position information through a continuous gauge.

The architecture is the spec's minimal form: encoder + gamfit. Persistent
feature identity must come from training dynamics (curriculum) rather than
architectural anchors.
"""

from __future__ import annotations

from dataclasses import dataclass

import gamfit.torch as gt
import numpy as np
import torch
from torch import nn

from .encoder import ManifoldEncoder


@dataclass
class ManifoldSAEConfig:
    input_dim: int
    n_features: int
    n_basis: int
    top_k: int
    sparsity_weight: float = 1e-3


@dataclass
class ManifoldSAEOutput:
    reconstruction: torch.Tensor
    positions: torch.Tensor
    amplitudes: torch.Tensor
    coefficients: torch.Tensor
    reml_score: torch.Tensor


class ManifoldSAE(nn.Module):
    def __init__(self, config: ManifoldSAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ManifoldEncoder(
            input_dim=config.input_dim,
            n_features=config.n_features,
            top_k=config.top_k,
        )
        K = int(config.n_basis)
        centers = np.linspace(0.0, 1.0, K, dtype=np.float64)
        penalty = np.eye(K, dtype=np.float64)
        self.register_buffer("centers", torch.from_numpy(centers))
        self.register_buffer("penalty", torch.from_numpy(penalty))

    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        positions, amplitudes = self.encoder(x)
        B, F = positions.shape
        t_packed = positions.t().contiguous().view(-1).to(torch.float64)
        by_packed = amplitudes.t().contiguous().view(-1).to(torch.float64)
        y_packed = x.to(torch.float64).repeat(F, 1)
        offsets = (torch.arange(F + 1, device=positions.device) * B).to(torch.uint64)
        out = gt.gaussian_reml_fit_positions_batched(
            t_packed, y_packed, offsets,
            "duchon", self.centers, self.penalty,
            basis_order=2, periodic=False,
            by=by_packed,
        )
        per_feature = out.fitted.view(F, B, x.shape[1])
        reconstruction = per_feature.sum(dim=0).to(x.dtype)
        reml_per_feat = torch.nan_to_num(out.reml_score, nan=0.0, posinf=0.0, neginf=0.0)
        return ManifoldSAEOutput(
            reconstruction=reconstruction,
            positions=positions,
            amplitudes=amplitudes,
            coefficients=out.coefficients.to(x.dtype),
            reml_score=reml_per_feat.sum().to(x.dtype),
        )


@torch.no_grad()
def extract_feature_curves(
    sae: ManifoldSAE,
    activations: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Per-feature learned curves on ``t_grid``, restricted to actually-fired
    position range to avoid extrapolation."""
    device = next(sae.parameters()).device
    activations = activations.to(device)
    t_grid_f64 = t_grid.to(device=device, dtype=torch.float64)

    sae.eval()
    out = sae(activations)
    coefficients = out.coefficients.to(torch.float64)
    amp = out.amplitudes
    pos = out.positions
    firing = amp > 1e-3
    F = coefficients.shape[0]
    T = t_grid_f64.shape[0]
    D = coefficients.shape[-1]
    curves = torch.zeros(F, T, D, dtype=torch.float64, device=device)
    for k in range(F):
        m = firing[:, k]
        if m.sum() < 2:
            continue
        pos_k = pos[m, k].to(torch.float64)
        t_lo = float(pos_k.quantile(0.02).item())
        t_hi = float(pos_k.quantile(0.98).item())
        if t_hi - t_lo < 1e-3:
            continue
        t_k = t_lo + (t_hi - t_lo) * t_grid_f64
        phi_k = gt.duchon_basis_1d(t_k, sae.centers, m=2, periodic=False)
        mean_amp_k = amp[m, k].to(torch.float64).mean()
        curves[k] = mean_amp_k * (phi_k @ coefficients[k])
    return curves.to(activations.dtype)
