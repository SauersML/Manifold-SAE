"""Equivariant / Lie-group SAE (Mendel et al. arXiv:2511.09432).

Architecture
------------
A small set of *atoms* a ∈ {1..F}. Each atom carries:

  - a 2-frame  W_a ∈ R^{D x 2}        (ambient embedding of the rep plane)
  - a group rep ρ(g_a) ∈ R^{2x2}      (here g_a ∈ SO(2))
  - per-atom amplitude  z_a ∈ R       (continuous, softplus-gated by a TopK head)

The decoder is

    x̂  =  b_dec  +  Σ_a  z_a · W_a · ρ(g_a) · e_1

where e_1 = (1, 0)^T is a canonical "phase 0" vector in the rep plane.
Equivalently, since ρ(θ)·e_1 = (cosθ, sinθ), the per-atom contribution is the
column-2 frame W_a applied to a unit vector that *rotates with the atom's
learned phase*. This is the SO(2) special case of  ρ(g) · W · z  from the
proposal.

For TRIVIAL atoms (group = {e}), ρ ≡ I_d and the head produces just (z_a · w_a)
with w_a ∈ R^D — i.e. the standard linear-decoder SAE atom.

Composition with the existing pipeline
--------------------------------------
- IBP-Gumbel gating + TopK encoder reused (same headed encoder as the
  curve-decoder SAE in train_sae_comparison.py).
- ARD penalty on per-atom amplitude scale (composes with EquivariantPenalty).
- EquivariantPenalty: ½ ‖ [ρ(g), W] z ‖²  commutator residual + per-group
  bandwidth ARD — drives W_a to span an irrep of the group, so the atom
  represents a *manifold* (S^1 for SO(2)) and not a single point.

Hue is the prototype: 64 SO(2) atoms produce the cyclic hue manifold; the
remaining trivial atoms absorb everything else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


GroupName = Literal["SO2", "SO3", "R1", "Trivial"]


# ---------------------------------------------------------------------------
# Group representations (analytic)
# ---------------------------------------------------------------------------

def rho_so2(theta: torch.Tensor) -> torch.Tensor:
    """SO(2) rep in R^2.  theta: (...,) -> (..., 2, 2)."""
    c, s = torch.cos(theta), torch.sin(theta)
    # [[c, -s], [s, c]]
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def rho_so3(omega: torch.Tensor) -> torch.Tensor:
    """SO(3) rep via Rodrigues. omega: (..., 3) axis-angle -> (..., 3, 3)."""
    angle = omega.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # (..., 1)
    axis = omega / angle                                       # (..., 3)
    ax = axis[..., 0]; ay = axis[..., 1]; az = axis[..., 2]
    zero = torch.zeros_like(ax)
    K = torch.stack([
        torch.stack([zero, -az,  ay], dim=-1),
        torch.stack([az,   zero, -ax], dim=-1),
        torch.stack([-ay,  ax,  zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(K.shape)
    s = torch.sin(angle).unsqueeze(-1)
    c1 = (1.0 - torch.cos(angle)).unsqueeze(-1)
    return I + s * K + c1 * (K @ K)


def rho(group: GroupName, g: torch.Tensor) -> torch.Tensor:
    if group == "SO2":
        return rho_so2(g.squeeze(-1) if g.dim() and g.shape[-1] == 1 else g)
    if group == "SO3":
        return rho_so3(g)
    if group == "R1":
        # Translation in R; rep as 1x1 identity (additive). Used only for
        # API completeness; in the decoder this collapses to a trivial atom
        # with the parameter consumed by W.
        return torch.ones(g.shape + (1, 1), dtype=g.dtype, device=g.device)
    if group == "Trivial":
        return torch.ones(g.shape[:-1] + (1, 1), dtype=g.dtype, device=g.device) \
            if g.dim() > 0 else torch.ones((1, 1), dtype=g.dtype, device=g.device)
    raise ValueError(f"unknown group {group!r}")


GROUP_DIM = {"SO2": 1, "SO3": 3, "R1": 1, "Trivial": 0}
GROUP_REP_DIM = {"SO2": 2, "SO3": 3, "R1": 1, "Trivial": 1}


# ---------------------------------------------------------------------------
# Encoder heads
# ---------------------------------------------------------------------------

class GroupHead(nn.Module):
    """Per-atom group-element head g_a(x).

    For SO(2): produces a scalar angle θ_a as atan2(W_sin x, W_cos x) — this
    is the *equivariant* head (θ is well-defined modulo 2π and the gradient
    handles wrap-around naturally).
    For SO(3): produces a 3-vector axis-angle ω_a (no normalization; rho_so3
    handles |ω|).
    For Trivial/R1: produces a scalar (unused for Trivial; used as the linear
    coefficient for R1).
    """
    def __init__(self, d_in: int, n_atoms: int, group: GroupName):
        super().__init__()
        self.group = group
        self.n_atoms = n_atoms
        self.d_in = d_in
        gd = GROUP_DIM[group]
        if group == "SO2":
            # Two scalar projections per atom (cos, sin).
            self.W = nn.Parameter(torch.randn(d_in, 2 * n_atoms) / math.sqrt(d_in))
        elif group == "SO3":
            self.W = nn.Parameter(torch.randn(d_in, 3 * n_atoms) / math.sqrt(d_in))
        else:
            self.W = nn.Parameter(torch.zeros(d_in, max(1, gd) * n_atoms))
        # ARD-style log-bandwidth per atom (scales the input projection).
        self.log_bandwidth = nn.Parameter(torch.zeros(n_atoms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns g of shape:
           SO2  -> (B, A)        (scalar angle)
           SO3  -> (B, A, 3)     (axis-angle)
           else -> (B, A)
        """
        B = x.shape[0]; A = self.n_atoms
        scale = torch.exp(self.log_bandwidth).repeat_interleave(
            2 if self.group == "SO2" else (3 if self.group == "SO3" else 1)
        )
        raw = (x @ self.W) * scale.unsqueeze(0)
        if self.group == "SO2":
            raw = raw.view(B, A, 2)
            return torch.atan2(raw[..., 1], raw[..., 0])  # (B, A)
        if self.group == "SO3":
            return raw.view(B, A, 3)
        return raw.view(B, A, 1).squeeze(-1)


class AmplitudeHead(nn.Module):
    """Per-atom amplitude (z_a). Sigmoid-gated softplus by default."""
    def __init__(self, d_in: int, n_atoms: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, n_atoms) / math.sqrt(d_in))
        self.b = nn.Parameter(torch.full((n_atoms,), -2.0))   # initially mostly off
        self.W_amp = nn.Parameter(torch.randn(d_in, n_atoms) / math.sqrt(d_in))

    def forward(self, x: torch.Tensor, tau: float = 1.0, training: bool = True) \
            -> tuple[torch.Tensor, torch.Tensor]:
        gate_logit = x @ self.W + self.b
        if training:
            u = torch.rand_like(gate_logit).clamp(1e-6, 1.0 - 1e-6)
            g_noise = torch.log(u) - torch.log1p(-u)
            gate = torch.sigmoid((gate_logit + g_noise) / tau)
        else:
            gate = torch.sigmoid(gate_logit)
        amp = F.softplus(x @ self.W_amp)
        return gate, amp


# ---------------------------------------------------------------------------
# EquivariantPenalty
# ---------------------------------------------------------------------------

def commutator_residual(
    W: torch.Tensor,   # (A, D, R)   2-frame per atom
    g: torch.Tensor,   # (B, A)      group elt per atom
    z: torch.Tensor,   # (B, A)      amplitude per atom
    group: GroupName = "SO2",
) -> torch.Tensor:
    """½ ‖[ρ(g), W] z‖² — measures how far W's column-span is from being an
    invariant subspace of ρ(g). For an exact irrep, [ρ(g), W] z = 0.

    Implementation detail: ρ acts in R^R (rep space), W in R^{D x R}; the
    commutator in mixed dimensions is the residual

        r  = W ρ(g) e_1 z  −  P_W ρ(g) W e_1 z,
        P_W = W (W^T W)^{-1} W^T  (projection onto W's column-span)

    which is zero iff ρ(g) maps the column-span of W into itself (i.e. W spans
    a ρ-invariant subspace). Equivalently, this is the projection residual of
    the rotated 2-frame onto the original 2-frame.
    """
    if group != "SO2":
        return torch.zeros((), device=W.device, dtype=W.dtype)
    A, D, R = W.shape
    # rho(g): (B, A, 2, 2)
    Rg = rho_so2(g)
    # rotated frame per (b, a): W @ Rg  -> (B, A, D, 2)
    W_rot = torch.einsum("adr,bars->bads", W, Rg)
    # projection onto W's column span using QR-stable formula.
    # WtW: (A, R, R), invWtW: (A, R, R)
    WtW = torch.einsum("adr,ads->ars", W, W) + 1e-6 * torch.eye(R, device=W.device, dtype=W.dtype).unsqueeze(0)
    L = torch.linalg.cholesky(WtW)
    # solve WtW @ X = W^T @ W_rot for X, then proj = W @ X
    # M = W^T W_rot : (B, A, R, R)
    M = torch.einsum("adr,bads->bars", W, W_rot)
    X = torch.cholesky_solve(M.reshape(-1, R, R), L.repeat(M.shape[0], 1, 1).reshape(-1, R, R)).reshape(M.shape)
    proj = torch.einsum("adr,bars->bads", W, X)
    resid = W_rot - proj                                          # (B, A, D, R)
    # ‖resid · z · e_1‖² — pick first column of rep space (canonical phase 0).
    r0 = resid[..., 0]                                            # (B, A, D)
    sq = (r0 ** 2).sum(dim=-1)                                    # (B, A)
    return 0.5 * (z * sq).mean()


# ---------------------------------------------------------------------------
# Full SAE
# ---------------------------------------------------------------------------

@dataclass
class EquivariantSAEConfig:
    d_in: int
    n_so2: int = 64
    n_trivial: int = 448
    aux: str | None = "HSV"       # gauge_companion target
    d_aux_sup: int = 3            # number of supervised SO(2) atoms whose θ tracks HSV-hue/sat/val anchors
    sparsity_weight: float = 1e-3
    eq_weight: float = 1e-2
    ard_weight: float = 1e-4


class EquivariantSAE(nn.Module):
    def __init__(self, config: EquivariantSAEConfig):
        super().__init__()
        self.config = config
        D, A2, A0 = config.d_in, config.n_so2, config.n_trivial

        # SO(2) atoms: per-atom 2-frame W_a ∈ R^{D x 2}.
        # Orthogonal init so the commutator residual starts small.
        W_so2 = torch.empty(A2, D, 2)
        for k in range(A2):
            q, _ = torch.linalg.qr(torch.randn(D, 2))
            W_so2[k] = q
        self.W_so2 = nn.Parameter(W_so2)
        self.group_head = GroupHead(D, A2, "SO2")
        self.amp_head_so2 = AmplitudeHead(D, A2)

        # Trivial atoms: standard linear decoder.
        self.W_triv = nn.Parameter(torch.randn(A0, D) * (1.0 / math.sqrt(D)))
        self.amp_head_triv = AmplitudeHead(D, A0)

        self.b_dec = nn.Parameter(torch.zeros(D))
        self.log_ard_so2 = nn.Parameter(torch.zeros(A2))
        self.log_ard_triv = nn.Parameter(torch.zeros(A0))

    def forward(self, x: torch.Tensor, tau: float = 0.7, training: bool | None = None):
        if training is None:
            training = self.training
        xc = x - self.b_dec

        # ---- SO(2) atoms ----
        theta = self.group_head(xc)                          # (B, A2)
        gate2, amp2 = self.amp_head_so2(xc, tau=tau, training=training)
        z2 = gate2 * amp2 * torch.exp(self.log_ard_so2)      # (B, A2)
        # ρ(θ) · e_1 = (cosθ, sinθ); per atom embed via W (D, 2).
        cs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (B, A2, 2)
        # contribution[b, d] = Σ_a z2[b,a] * Σ_r W_so2[a,d,r] * cs[b,a,r]
        # weighted: w_cs = (z2.unsqueeze(-1)) * cs  -> (B, A2, 2)
        w_cs = z2.unsqueeze(-1) * cs
        # contract atom+rep: einsum
        recon_so2 = torch.einsum("bar,adr->bd", w_cs, self.W_so2)

        # ---- Trivial atoms ----
        gate0, amp0 = self.amp_head_triv(xc, tau=tau, training=training)
        z0 = gate0 * amp0 * torch.exp(self.log_ard_triv)     # (B, A0)
        recon_triv = z0 @ self.W_triv

        recon = recon_so2 + recon_triv + self.b_dec

        return {
            "recon": recon,
            "theta": theta,
            "z_so2": z2,
            "z_triv": z0,
            "gate2": gate2, "gate0": gate0,
        }

    # ----- penalties -----
    def equivariant_penalty(self, theta: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return commutator_residual(self.W_so2, theta, z2, group="SO2")

    def ard_penalty(self) -> torch.Tensor:
        # log(λ + ‖W_a‖²) per atom; pressures dead atoms toward zero.
        s2 = (self.W_so2 ** 2).sum(dim=(1, 2))
        s0 = (self.W_triv ** 2).sum(dim=-1)
        return torch.log(1e-2 + s2).mean() + torch.log(1e-2 + s0).mean()

    def sparsity_penalty(self, gate2: torch.Tensor, gate0: torch.Tensor) -> torch.Tensor:
        return gate2.mean() + gate0.mean()


# ---------------------------------------------------------------------------
# Gauge companion (auto_exp_38 recipe in one helper)
# ---------------------------------------------------------------------------

def gauge_companion_loss(
    theta: torch.Tensor,        # (B, A2)
    hsv: torch.Tensor,          # (B, 3)  ground-truth HSV per row
    d_aux_sup: int = 3,
    weight: float = 1.0,
) -> torch.Tensor:
    """Bake auto_exp_38's HSV gauge-fix into one call.

    Supervises the first 3 SO(2) atoms' angles against HSV channels:
      - atom 0: θ_0 ↔ 2π·H        (hue is cyclic so this is a perfect S^1)
      - atom 1: θ_1 amplitude ↔ S
      - atom 2: θ_2 amplitude ↔ V

    Leaves all remaining (A2 - 3) atoms FREE — they discover name-semantic
    structure unsupervisedly (per auto_exp_38).

    Loss: circular MSE for hue, linear MSE for S/V via |cos θ|.
    """
    if d_aux_sup < 1: return torch.zeros((), device=theta.device, dtype=theta.dtype)
    losses = []
    # hue: circular distance on S^1
    h_rad = hsv[:, 0] * 2 * math.pi
    th0 = theta[:, 0]
    # 1 - cos(θ - h_rad)  is a smooth proxy for circular distance (0 when aligned).
    losses.append((1.0 - torch.cos(th0 - h_rad)).mean())
    if d_aux_sup >= 2 and theta.shape[1] >= 2:
        # encode sat via cos-amplitude alignment: want cos(θ_1) ≈ 2S - 1
        s_target = (2.0 * hsv[:, 1] - 1.0)
        losses.append(((torch.cos(theta[:, 1]) - s_target) ** 2).mean())
    if d_aux_sup >= 3 and theta.shape[1] >= 3:
        v_target = (2.0 * hsv[:, 2] - 1.0)
        losses.append(((torch.cos(theta[:, 2]) - v_target) ** 2).mean())
    return weight * sum(losses) / len(losses)
