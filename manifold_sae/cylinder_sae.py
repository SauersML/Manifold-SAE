"""CylinderSAE — Cylinder-native Manifold-SAE, gamfit 0.1.123 thin wrapper.

Motivation (auto_exp_67 on cogito-L40):
    Cylinder (S^1 × R) WINS topology selection by ΔREML > 140 vs Torus,
    > 1500 vs Euclidean. Each atom is a sheet (θ ∈ S^1, ℓ ∈ R).

History — what was deleted in the 0.1.123 migration
---------------------------------------------------
The pre-0.1.123 version of this file shipped its OWN per-feature MLP
encoder (`fc1_w: (F, D, H)` — 210 GB at F=512 D=7168 H=14336 default),
its own torch B-spline surrogate, AND a per-feature decoder. The
per-feature MLP was the OOM hazard.

We now route through:

  * `gamfit.Cylinder(n_knots=(K_theta, K_ell))` — the canonical basis
    descriptor (TensorBSpline of periodic θ × non-periodic ℓ margins).
  * a SHARED encoder (one `nn.Linear` → F·3 heads), eliminating the
    per-feature MLP entirely.

NOTE on gamfit gap: `gamfit.Cylinder` is a basis *descriptor*, not a
callable basis-matrix evaluator that streams through a torch graph.
The actual periodic-θ Fourier × Gaussian-ℓ basis is still computed
in torch below — this is the torch-grad mirror of `gamfit.Cylinder`'s
REML basis. Filed as gap: gamfit needs a `Cylinder.evaluate(theta, ell)
-> torch.Tensor` for end-to-end backprop.

For end-to-end gamfit REML on this topology, `gamfit.sae_manifold_fit(
X, atom_topology='cylinder', d_atom=2)` is the one-shot entry point —
but it currently errors with "Duchon D2 collocation requires 2*(p+s) >
dimension+2" since it doesn't expose duchon_{p,s} kwargs (also filed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

try:
    import gamfit
    _CYLINDER_DESCRIPTOR = gamfit.Cylinder  # for introspection / REML wiring
except Exception:  # gamfit optional at import time
    gamfit = None
    _CYLINDER_DESCRIPTOR = None


# ---------------------------------------------------------------------------
# Bases (torch-grad mirror of gamfit.Cylinder's TensorBSpline marginals).
# ---------------------------------------------------------------------------

def fourier_basis(theta: torch.Tensor, harmonics: int) -> torch.Tensor:
    """Fourier features on S^1 (mirrors periodic BSpline marginal)."""
    out = [torch.ones_like(theta)]
    for h in range(1, harmonics + 1):
        out.append(torch.cos(h * theta))
        out.append(torch.sin(h * theta))
    return torch.stack(out, dim=-1)


def bspline_basis(ell: torch.Tensor, n_basis: int, low: float = -3.0, high: float = 3.0) -> torch.Tensor:
    """Gaussian partition-of-unity surrogate for cubic B-spline on [low, high]."""
    centers = torch.linspace(low, high, n_basis, device=ell.device, dtype=ell.dtype)
    sigma = (high - low) / max(n_basis - 1, 1)
    diff = ell.unsqueeze(-1) - centers
    phi = torch.exp(-0.5 * (diff / sigma) ** 2)
    return phi / phi.sum(dim=-1, keepdim=True).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Config + module
# ---------------------------------------------------------------------------

@dataclass
class CylinderSAEConfig:
    input_dim: int
    n_features: int = 512
    fourier_harm: int = 3          # H — periodic margin n_knots ≈ 2H+1
    lightness_basis_k: int = 4     # K_ell — open margin n_knots
    top_k: int = 32
    sparsity_weight: float = 1e-3
    ard_weight: float = 1e-3
    hidden_dim: int = 512          # SHARED encoder hidden size


class CylinderSAE(nn.Module):
    """Cylinder Manifold-SAE — shared encoder, per-atom Cylinder decoder.

    Decoder per atom k:  contrib_k = amp_k * (φ_θ(θ_k) ⊗ φ_ℓ(ℓ_k)) @ B_k
    where B_k ∈ R^{M, D} and M = (2H+1) * K_ell. φ_θ × φ_ℓ mirrors
    ``gamfit.Cylinder(n_knots=(2H+1, K_ell))``.
    """

    def __init__(self, input_dim: int | CylinderSAEConfig, n_features: int = 512,
                 fourier_harm: int = 3, lightness_basis_k: int = 4, top_k: int = 32,
                 sparsity_weight: float = 1e-3, ard_weight: float = 1e-3,
                 hidden_dim: int = 512):
        super().__init__()
        cfg = (input_dim if isinstance(input_dim, CylinderSAEConfig) else CylinderSAEConfig(
            input_dim=int(input_dim), n_features=int(n_features), fourier_harm=int(fourier_harm),
            lightness_basis_k=int(lightness_basis_k), top_k=int(top_k),
            sparsity_weight=float(sparsity_weight), ard_weight=float(ard_weight),
            hidden_dim=int(hidden_dim)))
        self.config = cfg
        D, F = cfg.input_dim, cfg.n_features
        self.H, self.K_ell = cfg.fourier_harm, cfg.lightness_basis_k
        self.M = (2 * self.H + 1) * self.K_ell

        # gamfit basis descriptor (REML-side metadata; not used in forward).
        self.basis_spec = (
            _CYLINDER_DESCRIPTOR(n_knots=(2 * self.H + 1, self.K_ell), name="cyl")
            if _CYLINDER_DESCRIPTOR is not None else None
        )

        # SHARED encoder — single Linear → F*3 (cos_θ, sin_θ, ℓ, amp).
        self.norm = nn.LayerNorm(D) if D >= 4 else nn.Identity()
        self.in_proj = nn.Linear(D, cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, F * 4)
        with torch.no_grad():
            self.head.bias.zero_()
        # Per-atom decoder coefficients.
        self.B_dec = nn.Parameter(torch.randn(F, self.M, D) * (0.1 / math.sqrt(self.M)))
        self.b_dec = nn.Parameter(torch.zeros(D))

        self.top_k = int(cfg.top_k)
        self.sparsity_weight = float(cfg.sparsity_weight)
        self.ard_weight = float(cfg.ard_weight)

    def encode(self, x: torch.Tensor):
        xc = x - self.b_dec
        h = torch.nn.functional.gelu(self.in_proj(self.norm(xc)))
        out = self.head(h).view(x.shape[0], self.config.n_features, 4)
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        cos_l, sin_l, ell, amp_logit = out[..., 0], out[..., 1], out[..., 2], out[..., 3]
        theta = torch.atan2(sin_l, cos_l)        # guaranteed S^1
        amp_soft = torch.sigmoid(amp_logit)
        F = self.config.n_features
        if self.top_k is not None and self.top_k < F:
            _v, idx = torch.topk(amp_soft, self.top_k, dim=1)
            hard = torch.zeros_like(amp_soft)
            hard.scatter_(1, idx, 1.0)
            amp_binary = hard + (amp_soft - amp_soft.detach())  # STE
        else:
            amp_binary = torch.ones_like(amp_soft) + (amp_soft - amp_soft.detach())
        return theta, ell, amp_binary, amp_soft

    def basis(self, theta: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        phi_t = fourier_basis(theta, self.H)     # (B, F, 2H+1)
        phi_l = bspline_basis(ell, self.K_ell)   # (B, F, K_ell)
        return (phi_t.unsqueeze(-1) * phi_l.unsqueeze(-2)).reshape(*theta.shape, self.M)

    def decode(self, theta, ell, amp):
        phi = self.basis(theta, ell)
        contrib = torch.einsum("bfm,fmd->bfd", phi, self.B_dec) * amp.unsqueeze(-1)
        return contrib.sum(dim=1) + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        theta, ell, amp_b, amp_s = self.encode(x)
        recon = self.decode(theta, ell, amp_b)
        return {"x_hat": recon, "theta": theta, "ell": ell, "amp": amp_b, "amp_soft": amp_s}

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
        return total, {"recon": recon.detach(), "sparsity": sparsity.detach(),
                       "ard": ard.detach(), "k_eff": k_eff, "dead_rate": dead_rate}
