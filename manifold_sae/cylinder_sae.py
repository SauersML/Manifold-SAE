"""CylinderSAE: Cylinder-native Manifold-SAE.

Motivation (auto_exp_67 topology selector on cogito-L40):
    Cylinder (S^1 x R) WINS on cogito-L40 — REML beats Torus by Δ=140.8,
    beats Euclidean by Δ>1500, and beats pure Circle by Δ=697.

    The existing ManifoldSAE (manifold_sae/sae.py) treats each atom as a
    1-D curve on [0,1] (possibly periodic via ``periodic=True``). That is
    pure S^1 per atom. Cogito-L40 demands a richer per-atom topology:
    each atom is a SHEET parameterized by (θ, ℓ), where θ ∈ S^1 (angle,
    e.g. hue) and ℓ ∈ R (lightness / "lift" axis).

Architecture
------------
    For each of F atoms:
        θ_i ∈ S^1   — per-token angle (periodic, encoded as (cos θ, sin θ))
        ℓ_i ∈ R     — per-token lightness/lift coordinate

    Decoder basis per atom:
        φ_θ(θ) = [1, cos θ, sin θ, cos 2θ, sin 2θ, ..., cos Hθ, sin Hθ]
                  shape (2H+1,)
        φ_ℓ(ℓ) = B-spline basis of order ``lightness_basis_k`` on a fixed
                  grid (knots in [-3, 3], cubic interior, natural ends)
                  shape (K_ell,)

        TENSOR product: φ_i = vec( outer(φ_θ, φ_ℓ) ) ∈ R^M, M = (2H+1)*K_ell

    Each atom k owns a coefficient matrix B_k ∈ R^(M, D); per-token
    contribution is amp_k * (φ_i @ B_k).

Encoder
-------
    Per-feature parallel MLP (mirrors ManifoldEncoder structure but emits
    THREE scalars per feature: (theta_logit, ell, amp_logit)). theta is
    wrapped to [-π, π] via atan2 of a (cos, sin) head — guaranteed S^1.

Gate
----
    IBP-Gumbel-style straight-through TopK on the soft mask (sigmoid of
    amp_logit), matching ManifoldEncoder's binary STE pattern. ARD on
    per-atom decoder Frobenius norm.

Single-token feedforward. No gamfit call. Everything pure torch / MPS.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Basis builders
# ---------------------------------------------------------------------------

def fourier_basis(theta: torch.Tensor, harmonics: int) -> torch.Tensor:
    """Fourier features on S^1.

    Args:
        theta: (...,) angles in radians (any range; cos/sin handle wrap).
        harmonics: H ≥ 0; basis has size 2H+1 (DC + H cos/sin pairs).
    Returns:
        (..., 2H+1) tensor; column 0 is the DC term.
    """
    shape = theta.shape
    out = [torch.ones_like(theta)]
    for h in range(1, harmonics + 1):
        out.append(torch.cos(h * theta))
        out.append(torch.sin(h * theta))
    return torch.stack(out, dim=-1)


def bspline_basis(
    ell: torch.Tensor,
    n_basis: int,
    low: float = -3.0,
    high: float = 3.0,
) -> torch.Tensor:
    """Cubic-ish radial-basis surrogate for a B-spline basis on R.

    Pure torch (no scipy at runtime) so gradients flow on MPS. Uses
    evenly spaced Gaussian bumps with bandwidth set by knot spacing —
    a faithful smooth-positive partition-of-unity surrogate that has
    the right "local support" intuition without needing recursive
    Cox-de Boor (which is hard on MPS).

    Args:
        ell: (...,) lightness coordinates, will be soft-clamped to
            [low, high] via tanh-scaling before basis evaluation.
        n_basis: number of basis functions K_ell.
    Returns:
        (..., n_basis) basis values; rows sum to ~1 (Gaussian PoU).
    """
    centers = torch.linspace(low, high, n_basis, device=ell.device, dtype=ell.dtype)
    # bandwidth = knot spacing
    sigma = (high - low) / max(n_basis - 1, 1)
    diff = ell.unsqueeze(-1) - centers  # (..., K)
    phi = torch.exp(-0.5 * (diff / sigma) ** 2)
    # Soft normalize so rows sum to ~1 (partition of unity), and gradient
    # is well-conditioned even when ell is far from all centers.
    phi = phi / phi.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return phi


# ---------------------------------------------------------------------------
# Config + module
# ---------------------------------------------------------------------------

@dataclass
class CylinderSAEConfig:
    input_dim: int
    n_features: int = 512
    fourier_harm: int = 3
    lightness_basis_k: int = 4
    top_k: int = 32
    sparsity_weight: float = 1e-3
    ard_weight: float = 1e-3
    hidden_dim: int | None = None


class CylinderEncoder(nn.Module):
    """Per-feature parallel MLP emitting (cos_θ, sin_θ, ℓ, amp_logit).

    Mirrors manifold_sae.encoder.ManifoldEncoder structure but with
    a 4-channel head: (cos_logit, sin_logit, lightness, amp_logit).
    """

    def __init__(self, input_dim: int, n_features: int, hidden_dim: int | None = None):
        super().__init__()
        D = input_dim
        F = n_features
        # Default H caps at 256: the per-feature `(F, D, H)` weight is F× bigger
        # than a shared encoder, so old default max(2*D, 4*F) OOMs at F·D ~ 3.6M.
        H = hidden_dim if hidden_dim is not None else min(256, max(64, 4 * F))
        self.norm = nn.LayerNorm(D) if D >= 4 else nn.Identity()
        self.fc1_w = nn.Parameter(torch.randn(F, D, H) / max(D, 1) ** 0.5)
        self.fc1_b = nn.Parameter(torch.zeros(F, H))
        # 4 outputs: cos_theta_logit, sin_theta_logit, ell, amp_logit
        self.fc2_w = nn.Parameter(torch.randn(F, H, 4) / max(H, 1) ** 0.5)
        self.fc2_b = nn.Parameter(torch.zeros(F, 4))
        self.act = nn.GELU()
        self.n_features = F

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (theta, ell, amp_soft) each of shape (B, F)."""
        x_n = self.norm(x)
        B = x_n.shape[0]
        # (B, F, D) x (F, D, H) -> (B, F, H)
        x_b = x_n.unsqueeze(1).expand(B, self.n_features, x_n.shape[-1])
        h = torch.einsum("bfd,fdh->bfh", x_b, self.fc1_w) + self.fc1_b.unsqueeze(0)
        h = self.act(h)
        out = torch.einsum("bfh,fho->bfo", h, self.fc2_w) + self.fc2_b.unsqueeze(0)
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        cos_l = out[..., 0]
        sin_l = out[..., 1]
        ell = out[..., 2]  # unbounded R
        amp_logit = out[..., 3]
        theta = torch.atan2(sin_l, cos_l)  # in [-π, π]; guaranteed wrap-safe
        amp_soft = torch.sigmoid(amp_logit)
        return theta, ell, amp_soft


class CylinderSAE(nn.Module):
    """Cylinder-native Manifold-SAE.

    Args:
        input_dim: D, ambient dim (e.g. 7168 for cogito L40).
        n_features: F (default 512).
        fourier_harm: H ≥ 1 (default 3 → 7 angular basis functions).
        lightness_basis_k: K_ell (default 4).
        top_k: TopK gate.
    """

    def __init__(
        self,
        input_dim: int | CylinderSAEConfig,
        n_features: int = 512,
        fourier_harm: int = 3,
        lightness_basis_k: int = 4,
        top_k: int = 32,
        sparsity_weight: float = 1e-3,
        ard_weight: float = 1e-3,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if isinstance(input_dim, CylinderSAEConfig):
            cfg = input_dim
        else:
            cfg = CylinderSAEConfig(
                input_dim=int(input_dim),
                n_features=int(n_features),
                fourier_harm=int(fourier_harm),
                lightness_basis_k=int(lightness_basis_k),
                top_k=int(top_k),
                sparsity_weight=float(sparsity_weight),
                ard_weight=float(ard_weight),
                hidden_dim=hidden_dim,
            )
        self.config = cfg

        D = cfg.input_dim
        F = cfg.n_features
        self.H = cfg.fourier_harm
        self.K_ell = cfg.lightness_basis_k
        self.M = (2 * self.H + 1) * self.K_ell

        self.encoder = CylinderEncoder(D, F, hidden_dim=cfg.hidden_dim)

        # Per-atom decoder coefficients: (F, M, D)
        # Small init so early-training reconstructions don't blow up.
        scale = 1.0 / math.sqrt(self.M)
        self.B_dec = nn.Parameter(torch.randn(F, self.M, D) * scale * 0.1)
        self.b_dec = nn.Parameter(torch.zeros(D))

        self.top_k = int(cfg.top_k)
        self.sparsity_weight = float(cfg.sparsity_weight)
        self.ard_weight = float(cfg.ard_weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (theta, ell, amp_binary, amp_soft).

        amp_binary is a straight-through TopK gate over amp_soft (IBP-Gumbel
        style); amp_soft is the raw sigmoid for diagnostics / sparsity term.
        """
        x_c = x - self.b_dec
        theta, ell, amp_soft = self.encoder(x_c)
        if self.top_k is not None and self.top_k < self.config.n_features:
            _v, idx = torch.topk(amp_soft, self.top_k, dim=1)
            hard = torch.zeros_like(amp_soft)
            hard.scatter_(1, idx, 1.0)
            amp_binary = hard + (amp_soft - amp_soft.detach())  # STE
        else:
            amp_binary = torch.ones_like(amp_soft) + (amp_soft - amp_soft.detach())
        return theta, ell, amp_binary, amp_soft

    def basis(self, theta: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        """Per-token, per-feature tensor-product basis.

        Args:
            theta: (B, F)
            ell:   (B, F)
        Returns:
            phi: (B, F, M)  M = (2H+1) * K_ell
        """
        # (B, F, 2H+1)
        phi_t = fourier_basis(theta, self.H)
        # (B, F, K_ell)
        phi_l = bspline_basis(ell, self.K_ell)
        # outer product per (b, f) -> (B, F, 2H+1, K_ell) -> flatten
        phi = phi_t.unsqueeze(-1) * phi_l.unsqueeze(-2)
        return phi.reshape(*theta.shape, self.M)

    def decode(
        self,
        theta: torch.Tensor,
        ell: torch.Tensor,
        amp: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction. theta/ell/amp all (B, F)."""
        phi = self.basis(theta, ell)  # (B, F, M)
        # contribution_k = amp_k * (phi_k @ B_k)
        # contrib = einsum("bfm,fmd->bfd"); recon = sum over f
        contrib = torch.einsum("bfm,fmd->bfd", phi, self.B_dec)
        contrib = contrib * amp.unsqueeze(-1)
        return contrib.sum(dim=1) + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        theta, ell, amp_b, amp_s = self.encode(x)
        recon = self.decode(theta, ell, amp_b)
        return {
            "x_hat": recon,
            "theta": theta,
            "ell": ell,
            "amp": amp_b,
            "amp_soft": amp_s,
        }

    def atom_norms(self) -> torch.Tensor:
        """Per-atom decoder Frobenius norm; used for ARD pruning diagnostics."""
        return self.B_dec.reshape(self.config.n_features, -1).norm(dim=-1)

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out = self(x)
        recon = ((out["x_hat"] - x) ** 2).mean()
        # IBP-Gumbel sparsity: encourage few atoms via soft mask
        sparsity = self.sparsity_weight * out["amp_soft"].mean()
        # ARD on per-atom decoder norms — push unused atoms to zero
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
