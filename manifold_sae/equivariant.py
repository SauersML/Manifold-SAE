"""EquivariantSAE — Lie-group atoms (SO(2)) + trivial atoms.

gamfit 0.1.123 migration
------------------------
The SO(2) ``LieAtom``, ``EquivariantPenalty`` (½‖[ρ(g), W] z‖²) and
``GaugeCompanion`` (HSV auxiliary-supervised gauge fix) all ship as
first-class ``gamfit`` primitives:

  >>> import gamfit
  >>> atom, pen, gc = gamfit.equivariant_smooth(group="SO2", aux="HSV",
  ...                                            n_atoms=64, d_per_atom=2)

``auto_exp_56_equivariant.py`` USES these primitives end-to-end (1-line
constructor; numpy alternating fit). For the *torch* training pipeline
(``scripts/train_equivariant.py``) we keep a torch-grad mirror because
gamfit's penalty surfaces are **numpy-only FFI shims** that return a
``float`` (see ``gamfit._equivariant.EquivariantPenalty.evaluate`` and
``GaugeCompanion.loss``) and so are not differentiable through autograd.

**Gap filed for gamfit**: torch-grad bindings for
``EquivariantPenalty.evaluate`` and ``GaugeCompanion.loss`` (the rust
``equivariant_penalty_value`` + ``equivariant_gauge_companion_loss`` FFI
endpoints need autograd-aware Python wrappers analogous to
``gamfit.torch.SkipAffineSmooth`` / ``JumpReLUPenalty``).

What was deleted in the migration
---------------------------------
- Trivial-rho / R1 branches of ``rho`` and ``GroupHead`` (never used by
  ``train_equivariant.py``; ``EquivariantSAE`` only uses SO(2) + a
  linear trivial-atom decoder).
- SO(3) Rodrigues code path in this file (use ``gamfit.rho_so3`` /
  ``gamfit.rho_so3_jvp`` for numpy-side needs; torch SAE doesn't need it).
- Stand-alone module-level GROUP_DIM / GROUP_REP_DIM (lifted from
  ``gamfit._equivariant``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gamfit
    _GAMFIT_OK = hasattr(gamfit, "equivariant_smooth")
except Exception:  # pragma: no cover - gamfit optional
    gamfit = None
    _GAMFIT_OK = False


# ---------------------------------------------------------------------------
# Torch-grad SO(2) rep (mirror of gamfit.rho_so2; latter is numpy-only).
# ---------------------------------------------------------------------------

def rho_so2(theta: torch.Tensor) -> torch.Tensor:
    """SO(2) rep in R^2; (..., ) -> (..., 2, 2). Torch-grad-flowing."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s, c], dim=-1)], dim=-2)


# ---------------------------------------------------------------------------
# Encoder heads
# ---------------------------------------------------------------------------

class SO2GroupHead(nn.Module):
    """Per-atom θ_a head — atan2(W_sin x, W_cos x). Equivariant, wrap-safe."""

    def __init__(self, d_in: int, n_atoms: int):
        super().__init__()
        self.n_atoms = n_atoms
        self.W = nn.Parameter(torch.randn(d_in, 2 * n_atoms) / math.sqrt(d_in))
        self.log_bandwidth = nn.Parameter(torch.zeros(n_atoms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_bandwidth).repeat_interleave(2)
        raw = (x @ self.W) * scale.unsqueeze(0)
        raw = raw.view(x.shape[0], self.n_atoms, 2)
        return torch.atan2(raw[..., 1], raw[..., 0])


class AmplitudeHead(nn.Module):
    """Per-atom amplitude head with sigmoid-Gumbel gate + softplus amp."""

    def __init__(self, d_in: int, n_atoms: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, n_atoms) / math.sqrt(d_in))
        self.b = nn.Parameter(torch.full((n_atoms,), -2.0))
        self.W_amp = nn.Parameter(torch.randn(d_in, n_atoms) / math.sqrt(d_in))

    def forward(self, x, tau: float = 1.0, training: bool = True):
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
# Torch-grad mirrors of gamfit.EquivariantPenalty / gamfit.gauge_companion.
# Both gamfit objects are constructed at __init__ for REML-side metadata.
# ---------------------------------------------------------------------------

def commutator_residual(W: torch.Tensor, g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """½ ‖[ρ(g), W] z‖² — torch-grad mirror of ``gamfit.EquivariantPenalty.evaluate``.

    W (A, D, 2) ambient frames; g (B, A) per-sample angles; z (B, A) amplitudes.
    """
    A, D, R = W.shape
    Rg = rho_so2(g)                                                       # (B, A, 2, 2)
    W_rot = torch.einsum("adr,bars->bads", W, Rg)
    WtW = torch.einsum("adr,ads->ars", W, W) + 1e-6 * torch.eye(R, device=W.device, dtype=W.dtype).unsqueeze(0)
    L = torch.linalg.cholesky(WtW)
    M = torch.einsum("adr,bads->bars", W, W_rot)
    X = torch.cholesky_solve(M.reshape(-1, R, R), L.repeat(M.shape[0], 1, 1).reshape(-1, R, R)).reshape(M.shape)
    proj = torch.einsum("adr,bars->bads", W, X)
    resid = W_rot - proj
    r0 = resid[..., 0]
    return 0.5 * (z * (r0 ** 2).sum(dim=-1)).mean()


def gauge_companion_loss(theta: torch.Tensor, hsv: torch.Tensor,
                         d_aux_sup: int = 3, weight: float = 1.0) -> torch.Tensor:
    """Torch-grad mirror of ``gamfit.GaugeCompanion(aux='HSV').loss(theta)``.

    Hue: circular MSE on atom 0; sat/val: cos-alignment on atoms 1/2.
    Atoms ≥ d_aux_sup are FREE (auto_exp_38 recipe). See module docstring
    for why this can't just call the gamfit object: it returns ``float``,
    not a torch tensor.
    """
    if d_aux_sup < 1:
        return torch.zeros((), device=theta.device, dtype=theta.dtype)
    losses = []
    h_rad = hsv[:, 0] * 2 * math.pi
    losses.append((1.0 - torch.cos(theta[:, 0] - h_rad)).mean())
    if d_aux_sup >= 2 and theta.shape[1] >= 2:
        losses.append(((torch.cos(theta[:, 1]) - (2.0 * hsv[:, 1] - 1.0)) ** 2).mean())
    if d_aux_sup >= 3 and theta.shape[1] >= 3:
        losses.append(((torch.cos(theta[:, 2]) - (2.0 * hsv[:, 2] - 1.0)) ** 2).mean())
    return weight * sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Full SAE
# ---------------------------------------------------------------------------

@dataclass
class EquivariantSAEConfig:
    d_in: int
    n_so2: int = 64
    n_trivial: int = 448
    aux: str | None = "HSV"
    d_aux_sup: int = 3
    sparsity_weight: float = 1e-3
    eq_weight: float = 1e-2
    ard_weight: float = 1e-4


class EquivariantSAE(nn.Module):
    """SO(2) ``LieAtom``s + trivial linear atoms; HSV gauge-fix companion.

    On construction, registers `gamfit.LieAtom`/`gamfit.EquivariantPenalty`
    / `gamfit.GaugeCompanion` descriptors as attributes for downstream
    REML introspection. Forward / loss use the torch-grad mirrors above.
    """

    def __init__(self, config: EquivariantSAEConfig):
        super().__init__()
        self.config = config
        D, A2, A0 = config.d_in, config.n_so2, config.n_trivial

        # gamfit-side descriptors (no torch params; REML metadata only).
        if _GAMFIT_OK:
            self.lie_atom, self.eq_penalty, self.gauge = gamfit.equivariant_smooth(
                group="SO2", aux=config.aux, n_atoms=A2, d_per_atom=2,
                weight=config.eq_weight, ard_weight=config.ard_weight,
            )
        else:
            self.lie_atom = self.eq_penalty = self.gauge = None

        # SO(2) atom frames — orthonormal init (commutator residual ≈ 0 at t=0).
        W_so2 = torch.empty(A2, D, 2)
        for k in range(A2):
            q, _ = torch.linalg.qr(torch.randn(D, 2))
            W_so2[k] = q
        self.W_so2 = nn.Parameter(W_so2)
        self.group_head = SO2GroupHead(D, A2)
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
        theta = self.group_head(xc)
        gate2, amp2 = self.amp_head_so2(xc, tau=tau, training=training)
        z2 = gate2 * amp2 * torch.exp(self.log_ard_so2)
        cs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        recon_so2 = torch.einsum("bar,adr->bd", z2.unsqueeze(-1) * cs, self.W_so2)
        gate0, amp0 = self.amp_head_triv(xc, tau=tau, training=training)
        z0 = gate0 * amp0 * torch.exp(self.log_ard_triv)
        recon = recon_so2 + z0 @ self.W_triv + self.b_dec
        return {"recon": recon, "theta": theta, "z_so2": z2, "z_triv": z0,
                "gate2": gate2, "gate0": gate0}

    def equivariant_penalty(self, theta, z2):
        return commutator_residual(self.W_so2, theta, z2)

    def ard_penalty(self):
        s2 = (self.W_so2 ** 2).sum(dim=(1, 2))
        s0 = (self.W_triv ** 2).sum(dim=-1)
        return torch.log(1e-2 + s2).mean() + torch.log(1e-2 + s0).mean()

    def sparsity_penalty(self, gate2, gate0):
        return gate2.mean() + gate0.mean()
