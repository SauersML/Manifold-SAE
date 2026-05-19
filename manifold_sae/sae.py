"""Manifold-SAE: factored architecture with persistent per-feature subspace.

Each feature k owns a learnable rank-R ambient subspace ``W_k ∈ R^(D, R)`` —
the "tube" the curve lives in. Gamfit's Duchon REML fits the 1D smooth path
inside that subspace. ``W_k`` is the manifold-analog of ``W_dec`` in a
standard SAE; it carries persistent feature identity across batches.

Position parameterization
-------------------------
The encoder produces UNBOUNDED scalar position logits. A per-batch soft
min-max rescaling maps them into ``[0, 1]`` so positions span the basis
domain by construction every batch — no init dependence, no rank-deficient
inner solves from clustered positions. This decouples the encoder's learned
representation from gamfit's required domain; the curve's reparameterization
invariance makes the rescaling lossless.

Amplitudes
----------
Binary via straight-through TopK. Amplitudes can only encode "feature fires
y/n", not magnitude — closing the amp/coef gauge so positions must carry
where-on-the-curve information.
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
    intrinsic_rank: int = 2
    sparsity_weight: float = 1e-3
    cumulant_weight: float = 1e-2   # 4th-moment ICA-style identifiability
    ortho_weight: float = 1e-2      # ||W_k^T W_k − I_R||² per feature
    periodic: bool = False          # use periodic Duchon basis (gamfit 0.1.69+)


@dataclass
class ManifoldSAEOutput:
    reconstruction: torch.Tensor
    positions: torch.Tensor
    amplitudes: torch.Tensor       # binary (TopK + straight-through)
    mask_soft: torch.Tensor        # pre-binarization sigmoid probabilities
    coefficients: torch.Tensor
    directions: torch.Tensor
    cumulant_loss: torch.Tensor
    ortho_loss: torch.Tensor
    monotonicity_loss: torch.Tensor


def _soft_rescale_positions(z_raw: torch.Tensor, beta: float = 10.0, eps: float = 1e-4) -> torch.Tensor:
    """Smooth per-batch min-max normalization of unbounded scalar logits to [0, 1].

    ``z_raw`` has shape (B, F). For each feature, compute a smooth min and
    soft max across the batch dimension, then rescale.
    ``-(1/β) logsumexp(-β z)`` is a smooth lower envelope; ``(1/β) logsumexp(β z)``
    is a smooth upper envelope. β controls sharpness; β=10 is tight enough
    that positions span [eps, 1-eps] reliably without being non-smooth.
    """
    soft_max = (1.0 / beta) * torch.logsumexp(beta * z_raw, dim=0)  # (F,)
    soft_min = -(1.0 / beta) * torch.logsumexp(-beta * z_raw, dim=0)  # (F,)
    span = (soft_max - soft_min).clamp(min=1e-6)
    t = (z_raw - soft_min.unsqueeze(0)) / span.unsqueeze(0)
    return t.clamp(eps, 1.0 - eps)


class ManifoldSAE(nn.Module):
    def __init__(self, config: ManifoldSAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ManifoldEncoder(
            intrinsic_rank=config.intrinsic_rank,
            n_features=config.n_features,
            input_dim=config.input_dim,
            top_k=config.top_k,
        )
        K = int(config.n_basis)
        D = int(config.input_dim)
        R = int(config.intrinsic_rank)
        F = int(config.n_features)

        centers = np.linspace(0.0, 1.0, K, dtype=np.float64)
        penalty = np.eye(K, dtype=np.float64)
        self.register_buffer("centers", torch.from_numpy(centers))
        self.register_buffer("penalty", torch.from_numpy(penalty))

        # Persistent per-feature ambient subspace W_k ∈ R^(D, R). Initialize
        # so the UNION of all features' columns is one orthonormal frame
        # when D ≥ F*R; otherwise fall back to per-feature independent QR.
        if D >= F * R:
            Q, _ = torch.linalg.qr(torch.randn(D, F * R))
            directions = Q.reshape(D, F, R).permute(1, 0, 2).contiguous()
        else:
            directions = torch.empty(F, D, R)
            for k in range(F):
                q, _ = torch.linalg.qr(torch.randn(D, R))
                directions[k] = q
        self.directions = nn.Parameter(directions.to(torch.float32))

    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        x_dtype = x.dtype
        dirs = self.directions.to(x_dtype)
        z_raw, mask_soft, mask_binary = self.encoder(x)
        positions = _soft_rescale_positions(z_raw)
        y_proj = torch.einsum("bd,fdr->bfr", x, dirs)  # (B, F, R)

        B, F = positions.shape
        R = dirs.shape[-1]

        # gamfit per-batch REML solve in R-dim subspace.
        t_packed = positions.t().contiguous().view(-1).to(torch.float64)
        by_packed = mask_binary.t().contiguous().view(-1).to(torch.float64)
        y_packed = y_proj.permute(1, 0, 2).contiguous().view(F * B, R).to(torch.float64)
        offsets = (torch.arange(F + 1, device=positions.device) * B).to(torch.uint64)
        out = gt.gaussian_reml_fit_positions_batched(
            t_packed, y_packed, offsets,
            "duchon", self.centers, self.penalty,
            basis_order=2,
            periodic=self.config.periodic,
            period=1.0 if self.config.periodic else None,
            by=by_packed,
            init_lambda=1e-4,  # bias REML toward low smoothing — preserves
                                # curvature of sharper features (cubic, tanh)
                                # that REML otherwise flattens at default init.
        )
        fitted_intrinsic = out.fitted.view(F, B, R).to(x_dtype)
        contribution = torch.einsum("fbr,fdr->bfd", fitted_intrinsic, dirs)
        recon = contribution.sum(dim=1)  # (B, D)

        # Identifiability: 4th-moment cumulant contrast on PRE-binarization
        # mask probabilities. Pushes features toward statistical independence
        # so the planted decomposition becomes the unique low-loss attractor.
        # Use mask_soft (gradient-friendly) rather than mask_binary.
        if self.config.cumulant_weight > 0:
            a = mask_soft
            a_c = a - a.mean(dim=0, keepdim=True)
            a_std = a_c.std(dim=0, keepdim=True).clamp(min=1e-6)
            a_n = a_c / a_std
            pair_4th = (a_n.unsqueeze(-1) * a_n.unsqueeze(-2)).pow(2).mean(dim=0)  # (F, F)
            n_off = max(F * F - F, 1)
            cumulant_loss = (pair_4th.sum() - pair_4th.diagonal().sum()) / n_off
        else:
            cumulant_loss = torch.zeros((), dtype=x_dtype, device=x.device)

        # Orthonormality of per-feature W_k.
        I_R = torch.eye(R, dtype=dirs.dtype, device=dirs.device).unsqueeze(0)
        WtW = torch.einsum("fdr,fds->frs", dirs, dirs)
        ortho_loss = ((WtW - I_R) ** 2).mean()

        # Monotonicity: position should be a monotone function of the
        # principal-axis projection x @ W_k[:, 0]. Penalty is
        # 1 - |Pearson corr(position, principal-projection)| per feature
        # over firing tokens. Anchored to principal axis only — angular
        # variant on top introduces twisting for genuinely monotone curves.
        principal = y_proj[..., 0]  # (B, F)
        mask_f = mask_binary.detach()
        eps = 1e-6
        terms = []
        for k in range(F):
            m = mask_f[:, k] > 0.5
            if m.sum() < 5:
                continue
            p = positions[m, k]
            q = principal[m, k]
            p_c = p - p.mean()
            q_c = q - q.mean()
            denom = (p_c.pow(2).sum() * q_c.pow(2).sum()).clamp(min=eps).sqrt()
            terms.append(1.0 - (p_c * q_c).sum().abs() / denom)
        monotonicity_loss = torch.stack(terms).mean() if terms else torch.zeros((), dtype=x_dtype, device=x.device)

        return ManifoldSAEOutput(
            reconstruction=recon,
            positions=positions,
            amplitudes=mask_binary,
            mask_soft=mask_soft,
            coefficients=out.coefficients.to(x_dtype),
            directions=self.directions,
            cumulant_loss=cumulant_loss,
            ortho_loss=ortho_loss,
            monotonicity_loss=monotonicity_loss,
        )


@torch.no_grad()
def extract_feature_curves(
    sae: ManifoldSAE,
    activations: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Per-feature learned curves on ``t_grid`` in ambient space.

    Evaluates the Duchon basis at t_grid mapped to each feature's actually-
    observed position range, contracts with the per-batch coefficients (in
    R-dim subspace), lifts to ambient via the persistent W_k.
    """
    device = next(sae.parameters()).device
    activations = activations.to(device)
    t_grid_f64 = t_grid.to(device=device, dtype=torch.float64)

    sae.eval()
    out = sae(activations)
    coefficients = out.coefficients.to(torch.float64)
    directions = sae.directions.to(torch.float64)
    amp = out.amplitudes
    pos = out.positions
    firing = amp > 1e-3
    F = coefficients.shape[0]
    T = t_grid_f64.shape[0]
    D = directions.shape[1]
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
        phi_k = gt.duchon_basis_1d(t_k, sae.centers, m=2, periodic=sae.config.periodic)
        intrinsic_curve = phi_k @ coefficients[k]
        ambient_curve = intrinsic_curve @ directions[k].T
        curves[k] = ambient_curve
    return curves.to(activations.dtype)
