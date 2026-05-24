"""CylinderSAESharedEnc — shared-encoder variant of CylinderSAE.

Motivation
----------
``cylinder_sae.CylinderEncoder`` allocates a per-feature MLP weight
``fc1_w`` of shape (F, D, H). At F=512 D=7168 H=256 that's already
940M parameters in encoder alone, dominating the whole model. The
hypothesis is the per-atom specialization is supposed to live in the
*decoder* (curve basis `B_dec` of shape (F, M, D)), and forcing the
encoder to also be per-feature is mostly memory waste.

This module replaces the per-feature MLP with two shared Linears:

    shared_h  = GELU(LayerNorm(x) @ W_in)             (B, H_shared)
    head_out  = shared_h @ W_head + b_head            (B, F*4)
    (theta, ell, amp) extracted from the F*4 dims.

Parameter count drops from O(F·D·H + F·H·4) to O(D·H + H·F·4),
i.e. F× smaller in the first layer (the dominant one).

Atoms / decoder / loss are identical to CylinderSAE so the comparison
is purely on the encoder factorization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .cylinder_sae import bspline_basis, fourier_basis


@dataclass
class CylinderSAESharedEncConfig:
    input_dim: int
    n_features: int = 512
    fourier_harm: int = 3
    lightness_basis_k: int = 4
    top_k: int = 32
    sparsity_weight: float = 1e-3
    ard_weight: float = 1e-3
    hidden_dim: int = 512  # shared hidden size


class _SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, n_features: int, hidden_dim: int = 512):
        super().__init__()
        self.F = int(n_features)
        self.H = int(hidden_dim)
        self.norm = nn.LayerNorm(input_dim) if input_dim >= 4 else nn.Identity()
        self.in_proj = nn.Linear(input_dim, self.H)
        # Single shared linear projects shared-hidden to F*4 outputs.
        self.head = nn.Linear(self.H, self.F * 4)
        self.act = nn.GELU()
        # Initialize head bias so amp_soft starts ≈ 0 (gate mostly off, matches
        # behaviour of per-feature CylinderEncoder which inherits b≈0).
        with torch.no_grad():
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_n = self.norm(x)
        h = self.act(self.in_proj(x_n))            # (B, H)
        out = self.head(h)                          # (B, F*4)
        out = out.view(x_n.shape[0], self.F, 4)
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        cos_l = out[..., 0]
        sin_l = out[..., 1]
        ell = out[..., 2]
        amp_logit = out[..., 3]
        theta = torch.atan2(sin_l, cos_l)
        amp_soft = torch.sigmoid(amp_logit)
        return theta, ell, amp_soft


class CylinderSAESharedEnc(nn.Module):
    """Cylinder SAE with shared-encoder factorization.

    Public API mirrors ``cylinder_sae.CylinderSAE`` so existing diagnostics
    (atom_norms, loss, forward) work unchanged.
    """

    def __init__(
        self,
        input_dim: int | CylinderSAESharedEncConfig,
        n_features: int = 512,
        fourier_harm: int = 3,
        lightness_basis_k: int = 4,
        top_k: int = 32,
        sparsity_weight: float = 1e-3,
        ard_weight: float = 1e-3,
        hidden_dim: int = 512,
    ):
        super().__init__()
        if isinstance(input_dim, CylinderSAESharedEncConfig):
            cfg = input_dim
        else:
            cfg = CylinderSAESharedEncConfig(
                input_dim=int(input_dim),
                n_features=int(n_features),
                fourier_harm=int(fourier_harm),
                lightness_basis_k=int(lightness_basis_k),
                top_k=int(top_k),
                sparsity_weight=float(sparsity_weight),
                ard_weight=float(ard_weight),
                hidden_dim=int(hidden_dim),
            )
        self.config = cfg

        D = cfg.input_dim
        F = cfg.n_features
        self.H = cfg.fourier_harm
        self.K_ell = cfg.lightness_basis_k
        self.M = (2 * self.H + 1) * self.K_ell

        self.encoder = _SharedEncoder(D, F, hidden_dim=cfg.hidden_dim)

        scale = 1.0 / math.sqrt(self.M)
        self.B_dec = nn.Parameter(torch.randn(F, self.M, D) * scale * 0.1)
        self.b_dec = nn.Parameter(torch.zeros(D))

        self.top_k = int(cfg.top_k)
        self.sparsity_weight = float(cfg.sparsity_weight)
        self.ard_weight = float(cfg.ard_weight)

    def encode(self, x: torch.Tensor):
        x_c = x - self.b_dec
        theta, ell, amp_soft = self.encoder(x_c)
        F = self.config.n_features
        if self.top_k is not None and self.top_k < F:
            _v, idx = torch.topk(amp_soft, self.top_k, dim=1)
            hard = torch.zeros_like(amp_soft)
            hard.scatter_(1, idx, 1.0)
            amp_binary = hard + (amp_soft - amp_soft.detach())
        else:
            amp_binary = torch.ones_like(amp_soft) + (amp_soft - amp_soft.detach())
        return theta, ell, amp_binary, amp_soft

    def basis(self, theta: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        phi_t = fourier_basis(theta, self.H)
        phi_l = bspline_basis(ell, self.K_ell)
        phi = phi_t.unsqueeze(-1) * phi_l.unsqueeze(-2)
        return phi.reshape(*theta.shape, self.M)

    def decode(self, theta, ell, amp):
        phi = self.basis(theta, ell)
        contrib = torch.einsum("bfm,fmd->bfd", phi, self.B_dec)
        contrib = contrib * amp.unsqueeze(-1)
        return contrib.sum(dim=1) + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        theta, ell, amp_b, amp_s = self.encode(x)
        recon = self.decode(theta, ell, amp_b)
        return {"x_hat": recon, "theta": theta, "ell": ell,
                "amp": amp_b, "amp_soft": amp_s}

    def atom_norms(self) -> torch.Tensor:
        return self.B_dec.reshape(self.config.n_features, -1).norm(dim=-1)

    def loss(self, x: torch.Tensor):
        out = self(x)
        recon = ((out["x_hat"] - x) ** 2).mean()
        sparsity = self.sparsity_weight * out["amp_soft"].mean()
        ard = self.ard_weight * self.atom_norms().mean()
        total = recon + sparsity + ard
        with torch.no_grad():
            active = (out["amp"] > 0.5).float()
            k_eff = active.sum(dim=1).mean()
            dead_rate = 1.0 - (active.sum(dim=0) > 0).float().mean()
        return total, {
            "recon": recon.detach(),
            "sparsity": sparsity.detach(),
            "ard": ard.detach(),
            "k_eff": k_eff,
            "dead_rate": dead_rate,
        }
