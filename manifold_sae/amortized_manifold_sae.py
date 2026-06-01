"""Amortized Manifold-SAE — the gam-native successor to the hand-rolled M-SAE.

Design (see memory: reference_msae_vs_gam_joint_fit, reference_gam_atom_latent_dim):

* **Atoms are gam-native.** Each atom ``k`` is a per-atom ``d``-dimensional manifold
  with a fused decoder block ``B_k`` of shape ``(M, D)`` (embedding + shape fused; no
  separate ``W_k`` plane). The atom's image in ambient ``R^D`` is ``basis(t) @ B_k``.
  ``d`` (``intrinsic_rank``) can be 1 (curve), 2 (surface), … per the gam kernel.

* **Inference is a single feedforward pass (LLM-scale).** A cheap ``F``-wide gate head
  predicts a sparse active set per token (JumpReLU — gam-native, differentiable); a
  coordinate head predicts the on-manifold coordinate ``t`` per atom. No transductive
  per-token solve at inference → scales to ``F~100K`` (sparse decode) over streaming
  tokens. This is what makes it LLM-applicable.

* **Training, now:** end-to-end backprop, ``recon + JumpReLU sparsity`` (+ optional
  isometry / block-orthogonality once wired). gam's JumpReLU penalty is autograd-native
  and works in 0.1.134.

* **Training, intended (behind ``joint_teacher=True``):** gam's joint Arrow-Schur solve
  (:func:`gamfit.sae_manifold_fit`) as a *warm-started teacher* — encoder predicts
  ``(t, assignments)``, the solver refines, and we distill the gap (Predictive-Sparse-
  Decomposition style). Currently DISABLED pending gam#356 (the joint solve panics in
  0.1.134) and gam#357 (it must expose converged per-token coordinates/assignments).

This module deliberately drops M-SAE's lock-snapshot / inference_mode dance, its ``W_k``
planes, and its per-batch closed-form re-derivation — all scaffolding for gamfit gaps
that 0.1.134 closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import torch
from torch import nn

from gamfit.torch import IBPAssignmentPenalty  # gam-native adaptive-K sparsity prior


@dataclass
class AmortizedManifoldSAEConfig:
    input_dim: int                              # ambient D
    n_atoms: int = 512                          # F
    intrinsic_rank: int = 1                     # per-atom manifold dim d (1=curve)
    fourier_harmonics: int = 3                  # circle basis order H -> M = 2H+1
    atom_manifold: Literal["circle"] = "circle"  # v1: circle (d=1); product/sphere TODO
    gate_threshold: float = 0.05                # presence threshold on the gate head
    sparsity_weight: float = 1e-3               # scale on the IBP adaptive-K prior
    ibp_alpha: float = 1.0                      # IBP concentration (expected per-token K)
    incoherence_weight: float = 1e-3            # push atom subspaces apart (μ-incoherence)
    encoder_hidden: int = 0                     # 0 = linear gate/coord heads
    isometry_weight: float = 0.0                # gauge fix: arc-length-isometric θ
    isometry_grid: int = 32                     # θ-grid size for the isometry penalty
    dtype: torch.dtype = torch.float32          # float32: feedforward LM regime

    def __post_init__(self) -> None:
        if self.atom_manifold != "circle":
            raise NotImplementedError(
                "v1 supports atom_manifold='circle' (d=1); product/sphere/higher-d "
                "land once the gam joint teacher (gam#356/#357) is wired."
            )
        if self.intrinsic_rank != 1:
            raise NotImplementedError("v1 circle atoms are d=1; d>1 is the next step.")
        if self.fourier_harmonics < 1:
            raise ValueError("fourier_harmonics must be >= 1")


@dataclass
class AmortizedManifoldSAEOutput:
    x_hat: torch.Tensor          # (N, D) reconstruction
    gate: torch.Tensor           # (N, F) sparse amplitudes (post-JumpReLU)
    theta: torch.Tensor          # (N, F) on-manifold coordinate per atom (radians)
    pre_gate: torch.Tensor       # (N, F) pre-activation gate logits (for the penalty)
    active_fraction: torch.Tensor  # scalar mean fraction of atoms firing


def _circle_basis(theta: torch.Tensor, harmonics: int) -> torch.Tensor:
    """Fourier features on S^1: [1, cos t, sin t, ..., cos Ht, sin Ht] -> (..., 2H+1).

    gamfit 0.1.141 has no autograd-capable Fourier basis evaluator: its periodic
    torch primitives are ``periodic_spline_curve_basis`` (forward-only, no grad
    through ``t``) and ``bspline_basis(periodic=True)`` (a B-spline, not Fourier).
    TODO(gamfit): cut over to ``gamfit.torch.fourier_basis`` once it exists.
    """
    feats = [torch.ones_like(theta)]
    for h in range(1, harmonics + 1):
        feats.append(torch.cos(h * theta))
        feats.append(torch.sin(h * theta))
    return torch.stack(feats, dim=-1)


def _circle_basis_deriv(theta: torch.Tensor, harmonics: int) -> torch.Tensor:
    """d/dtheta of :func:`_circle_basis`: [0, -h sin ht, h cos ht] -> (..., 2H+1)."""
    feats = [torch.zeros_like(theta)]
    for h in range(1, harmonics + 1):
        feats.append(-h * torch.sin(h * theta))
        feats.append(h * torch.cos(h * theta))
    return torch.stack(feats, dim=-1)


class AmortizedManifoldSAE(nn.Module):
    """Feedforward amortized Manifold-SAE on circle (d=1) atoms.

    forward(x) -> AmortizedManifoldSAEOutput. Reconstruction is

        x_hat[n] = sum_k gate[n,k] * ( basis(theta[n,k]) @ B_k )

    with ``gate`` sparse (JumpReLU) and ``theta`` from an atan2 coordinate head
    (guaranteed valid S^1 coordinate, wraps at 2pi).
    """

    def __init__(self, cfg: AmortizedManifoldSAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, F = cfg.input_dim, cfg.n_atoms
        M = 2 * cfg.fourier_harmonics + 1
        self.M = M

        # Fused decoder blocks B_k: (F, M, D) — embedding + shape, gam-native (no W_k).
        self.B = nn.Parameter(torch.randn(F, M, D, dtype=cfg.dtype) * (1.0 / math.sqrt(M * D)))
        self.b_dec = nn.Parameter(torch.zeros(D, dtype=cfg.dtype))

        # Gate head: x -> per-atom pre-activation (sparsified by JumpReLU).
        # Coordinate head: x -> (cos, sin) per atom -> theta = atan2(sin, cos).
        if cfg.encoder_hidden > 0:
            H = cfg.encoder_hidden
            self.gate_head = nn.Sequential(nn.Linear(D, H, dtype=cfg.dtype), nn.GELU(),
                                           nn.Linear(H, F, dtype=cfg.dtype))
            self.coord_head = nn.Sequential(nn.Linear(D, H, dtype=cfg.dtype), nn.GELU(),
                                            nn.Linear(H, 2 * F, dtype=cfg.dtype))
        else:
            self.gate_head = nn.Linear(D, F, dtype=cfg.dtype)
            self.coord_head = nn.Linear(D, 2 * F, dtype=cfg.dtype)

        # Adaptive-K sparsity: an Indian-Buffet-Process prior over the per-token
        # assignment pattern (favours few active atoms, K chosen per token rather
        # than a fixed cap). Acts as the sparsity term on the gate logits.
        self.ibp = IBPAssignmentPenalty(k_max=F, alpha=float(cfg.ibp_alpha), tau=1.0)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (N,D) -> (gate (N,F) sparse amplitudes, theta (N,F), pre_gate (N,F))."""
        F = self.cfg.n_atoms
        pre = self.gate_head(x)                                  # (N, F)
        # Presence gate: keep atoms whose logit clears the threshold; the kept
        # value scales the curve. The IBP prior (in loss) shapes how many clear it.
        gate = pre * (pre > self.cfg.gate_threshold).to(pre.dtype)
        cs = self.coord_head(x).reshape(x.shape[0], F, 2)        # (N, F, 2)
        theta = torch.atan2(cs[..., 1], cs[..., 0])              # (N, F) in (-pi, pi]
        return gate, theta, pre

    def decode(self, gate: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """gate (N,F), theta (N,F) -> x_hat (N,D)."""
        phi = _circle_basis(theta, self.cfg.fourier_harmonics)  # (N, F, M)
        # per-atom curve point: (N,F,M) @ (F,M,D) -> (N,F,D)
        curve = torch.einsum("nfm,fmd->nfd", phi, self.B)
        x_hat = torch.einsum("nf,nfd->nd", gate, curve) + self.b_dec
        return x_hat

    def forward(self, x: torch.Tensor) -> AmortizedManifoldSAEOutput:
        if x.dtype != self.cfg.dtype:
            x = x.to(self.cfg.dtype)
        gate, theta, pre = self.encode(x)
        x_hat = self.decode(gate, theta)
        active = (gate.abs() > 0).float().mean()
        return AmortizedManifoldSAEOutput(
            x_hat=x_hat, gate=gate, theta=theta, pre_gate=pre, active_fraction=active,
        )

    def isometry_penalty(self) -> torch.Tensor:
        """Gauge fix: penalize per-atom variance of arc-length speed ||dg_k/dtheta||
        over a theta-grid, so the coordinate is (up to scale) isometric to the
        manifold's intrinsic metric. Constant speed => identifiable, non-arbitrary
        parametrization (the gauge that auto_exp_38 found enables unsupervised
        factor discovery on the free block). Atom-agnostic to overall scale: we
        normalize each atom's speed by its mean before taking the variance.
        """
        G = max(int(self.cfg.isometry_grid), 4)
        theta = torch.linspace(-math.pi, math.pi, G + 1, dtype=self.cfg.dtype,
                               device=self.B.device)[:-1]            # (G,) no dup endpoint
        dphi = _circle_basis_deriv(theta, self.cfg.fourier_harmonics)  # (G, M)
        # speed_k(theta) = ||dphi @ B_k||  -> (F, G)
        dg = torch.einsum("gm,fmd->fgd", dphi, self.B)               # (F, G, D)
        speed = dg.norm(dim=-1)                                      # (F, G)
        mean = speed.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return (speed / mean).var(dim=-1).mean()                     # scale-free

    def incoherence_penalty(self) -> torch.Tensor:
        """Cross-atom subspace incoherence — the design's load-bearing term.

        Pushes distinct atoms' decoder subspaces toward mutual orthogonality
        (μ-incoherence). This does double duty: it keeps atoms distinct, AND it
        is exactly the precondition under which the per-token sparse
        decomposition is unique/recoverable (the subspace-recovery theorem). We
        penalize the squared off-(atom-block) entries of the Gram matrix of
        L2-normalized decoder rows; the within-atom M×M blocks are left free, so
        an atom's own curvature is unconstrained while different atoms separate.
        """
        F, M, D = self.B.shape
        rows = self.B.reshape(F * M, D)
        rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gram2 = (rows @ rows.t()).pow(2)                       # (F·M, F·M)
        atom_of_row = torch.arange(F * M, device=self.B.device) // M
        cross = atom_of_row.unsqueeze(0) != atom_of_row.unsqueeze(1)  # off-block mask
        return (gram2 * cross).sum() / cross.sum().clamp_min(1)

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor) -> dict:
        if x.dtype != self.cfg.dtype:
            x = x.to(self.cfg.dtype)
        out = self.forward(x)
        recon = (out.x_hat - x).pow(2).mean()
        # Adaptive-K sparsity prior on the gate logits (per-row normalized).
        sparsity = self.ibp(out.pre_gate) / float(out.pre_gate.shape[0])
        incoherence = self.incoherence_penalty()
        total = (
            recon
            + self.cfg.sparsity_weight * sparsity
            + self.cfg.incoherence_weight * incoherence
        )
        iso = self.B.new_zeros(())
        if self.cfg.isometry_weight > 0.0:
            iso = self.isometry_penalty()
            total = total + self.cfg.isometry_weight * iso
        return {
            "loss": total,
            "recon": recon,
            "sparsity": sparsity,
            "incoherence": incoherence.detach(),
            "isometry": iso.detach(),
            "active_fraction": out.active_fraction.detach(),
            "x_hat": out.x_hat,
        }
