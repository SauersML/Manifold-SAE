"""HyperbolicSAE — sparse autoencoder with decoder atoms in the Poincaré ball.

Motivation
----------
Cogito-L40 color activations are naturally hierarchical (perceptual color → name-
semantic modifier_count → template). Hyperbolic geometry's exponential capacity
growth with radius can express such a hierarchy with substantially fewer atoms
than Euclidean SAEs (see arxiv 2505.18973 — Hierarchical Mamba meets Hyperbolic).

Architecture
------------
  x ∈ R^D
    ──► W_enc : R^D → R^{F·d}                    (per-atom tangent vectors at 0)
    ──► exp_0 (Poincaré, per atom)               → atoms_pos : (B, F, d) in B^d
    ──► gates g_k = ReLU(W_gate x + b_gate)      ∈ R^F  (sparse, L1-penalized)

  Decoder atoms: D_atom ∈ R^{F·d} live as points in B^d (one per feature),
    Möbius-scaled by per-sample gate g_k via mobius_add(0, g_k · t_k) where t_k =
    log_0(D_atom_k) (i.e. tangent representative of the learned atom).
  Reconstruction:
    z = Σ_k g_k · log_0(D_atom_k)                (B, d)   — tangent at origin
    x_hat = W_dec_out @ z + b_dec                 (B, D)

Hyperbolic content
------------------
  - atom positions x_k(x) = exp_0(t_k(x))   live in B^d
  - learned atom centers a_k                 live in B^d
  - gate sparsity uses Möbius geodesic distance d_c(x_k(x), a_k) as the "feature
    activation strength" — larger distance ⇒ stronger weighting (rare/specific
    concepts live near boundary, generic concepts near origin)
  - L1 sparsity on g_k

Sufficiently feedforward for MPS / 1M-token deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kernels.poincare import exp_0, log_0, mobius_add, poincare_distance


@dataclass
class HyperbolicSAEConfig:
    input_dim: int
    n_features: int = 128
    ball_dim: int = 32
    curvature: float = 1.0  # magnitude (geometric curvature is -c)
    sparsity_weight: float = 1e-3
    bias: bool = True


class HyperbolicSAE(nn.Module):
    """Sparse autoencoder with decoder atoms in the Poincaré ball B^d.

    Args:
        input_dim:    ambient dimension D
        n_features:   number of atoms F
        ball_dim:     ball dimension d (intrinsic hyperbolic dim)
        curvature:    c > 0; ball radius is 1/sqrt(c). Sign convention: the
                      Riemannian curvature is -c, but all formulas take c.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 128,
        ball_dim: int = 32,
        curvature: float = 1.0,
        sparsity_weight: float = 1e-3,
    ):
        super().__init__()
        self.cfg = HyperbolicSAEConfig(
            input_dim=input_dim,
            n_features=n_features,
            ball_dim=ball_dim,
            curvature=float(curvature),
            sparsity_weight=float(sparsity_weight),
        )
        D, F, d = input_dim, n_features, ball_dim
        self.D, self.F, self.d = D, F, d
        self.c = float(curvature)

        # Encoder: x → F*d tangent components → reshape to (B, F, d)
        self.W_enc = nn.Linear(D, F * d, bias=True)
        # Per-feature gate (scalar sparse activation).
        self.W_gate = nn.Linear(D, F, bias=True)

        # Learned atom centers live in B^d; parameterize via tangent vectors at 0
        # and exp_0 on demand (keeps Adam in Euclidean space — standard "Riemannian
        # via retraction at the origin" trick).
        self.atom_tangents = nn.Parameter(torch.randn(F, d) * 0.05)

        # Tangent → ambient readout.
        self.W_dec_out = nn.Linear(d, D, bias=True)

        # Init: small encoder, identity-ish gate bias 0.
        nn.init.kaiming_uniform_(self.W_enc.weight, a=5 ** 0.5)
        nn.init.zeros_(self.W_enc.bias)
        nn.init.zeros_(self.W_gate.bias)
        nn.init.kaiming_uniform_(self.W_dec_out.weight, a=5 ** 0.5)
        nn.init.zeros_(self.W_dec_out.bias)

    # ------------------------------------------------------------------ helpers

    def atom_positions(self) -> torch.Tensor:
        """(F, d) atom centers in the Poincaré ball."""
        return exp_0(self.atom_tangents, c=self.c)

    # ------------------------------------------------------------------ forward

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x → (per-sample atom positions in B^d, gate activations).

        Returns:
            atoms_pos: (B, F, d)  in B^d
            gates:     (B, F)     ReLU-activated, L1-sparse
        """
        B = x.shape[0]
        t = self.W_enc(x).view(B, self.F, self.d)  # (B, F, d) tangent
        atoms_pos = exp_0(t, c=self.c)  # (B, F, d)
        gates = torch.relu(self.W_gate(x))  # (B, F)
        return atoms_pos, gates

    def decode(self, atoms_pos: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """Decode via tangent-space weighted sum then linear readout.

        atoms_pos: (B, F, d)
        gates:    (B, F)
        """
        # Use the Möbius midpoint approach: tangent at origin = log_0(point)
        # times gate, then linear readout. This is the standard Hyperbolic
        # Networks "log_0 ∘ Mobius-scaling" pattern.
        t = log_0(atoms_pos, c=self.c)  # (B, F, d) tangent at origin
        z = (gates.unsqueeze(-1) * t).sum(dim=1)  # (B, d) — Möbius-additive in tangent
        x_hat = self.W_dec_out(z)  # (B, D)
        return x_hat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        atoms_pos, gates = self.encode(x)
        x_hat = self.decode(atoms_pos, gates)

        # Hyperbolic feature strength: geodesic distance between per-sample
        # atom-position and the learned atom-center. Used both as sparsity
        # "weighting" and for interpretation.
        atom_centers = self.atom_positions()  # (F, d)
        dists = poincare_distance(atoms_pos, atom_centers.unsqueeze(0), c=self.c)  # (B, F)

        l1 = gates.abs().mean() * self.cfg.sparsity_weight
        dist_l1 = (gates * dists).mean() * self.cfg.sparsity_weight  # Möbius-distance gates

        return {
            "x_hat": x_hat,
            "atoms_pos": atoms_pos,
            "gates": gates,
            "dists": dists,
            "l1": l1,
            "dist_l1": dist_l1,
        }

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out = self.forward(x)
        recon = ((out["x_hat"] - x) ** 2).mean()
        sparsity = out["l1"] + out["dist_l1"]
        total = recon + sparsity
        logs = {
            "loss": total.detach(),
            "recon": recon.detach(),
            "l1": out["l1"].detach(),
            "dist_l1": out["dist_l1"].detach(),
            "active_frac": (out["gates"] > 1e-6).float().mean().detach(),
        }
        return total, logs

    # ------------------------------------------------------------------ utils

    @torch.no_grad()
    def feature_norms_in_ball(self) -> torch.Tensor:
        """Distance of each atom from origin in the ball (proxy for hierarchy
        depth: near origin = coarse / generic concept; near boundary = specific
        leaf concept)."""
        pos = self.atom_positions()
        zero = torch.zeros_like(pos)
        return poincare_distance(zero, pos, c=self.c)
