"""HyperbolicSAE — sparse autoencoder with decoder atoms in the Poincaré ball.

gamfit-native rewrite (breaking)
--------------------------------
This module is now a thin composition over ``gamfit.torch.PoincareAtoms``
(Rust-backed Poincaré geometry with analytic backward). All hand-rolled
``exp_0``/``log_0``/``mobius_add``/``poincare_distance`` are gone; the atom
dictionary, the tangent-space decode, and every geodesic diagnostic come from
the primitive.

Curvature sign convention (IMPORTANT, breaking change)
------------------------------------------------------
``PoincareAtoms`` uses the *geometric* sign: the sectional curvature ``c`` is
strictly **negative**. The SAE config still exposes ``curvature`` as a positive
*magnitude* (κ > 0) for ergonomics / continuity with prior runs, and negates it
(``c = -κ``) when constructing the primitive. ``self.c`` holds the magnitude;
``self.atoms_dict.curvature`` holds the negative value handed to Rust.

Architecture
------------
  x ∈ R^D
    ──► gates g = ReLU(W_gate x + b_gate) ∈ R^F   (sparse; JumpReLU / L1 penalty)
    ──► v = Σ_f g_f · log_0(a_f)  ; ball = exp_0(v) ∈ B^{d}_c   (PoincareAtoms.forward)
    ──► x_hat = W_dec_out @ ball + b_dec ∈ R^D     (tangent→ambient readout)

The atoms ``a_f`` are learnable points stored **in the ball** (the primitive's
``.atoms`` parameter); there is no separate encoder producing per-sample atom
positions anymore — the decode is the canonical tangent aggregation of the
fixed dictionary weighted by the gates.

Hyperbolic diagnostics
----------------------
  - ``feature_norms_in_ball()`` — geodesic distance of each atom from the origin
    (hierarchy depth proxy: near origin = coarse, near boundary = specific).
  - the sparsity term ``dist_l1`` weights each gate by its atom's geodesic radius
    so the penalty pushes mass toward shallow (generic) atoms.

References
----------
* Nickel & Kiela, NeurIPS 2017 (arXiv:1705.08039).
* Ganea, Bécigneul, Hofmann, NeurIPS 2018 (arXiv:1805.09112).
* Hierarchical Mamba / hyperbolic, arXiv:2505.18973 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gamfit.torch import JumpReLUPenalty, PoincareAtoms  # gamfit (post-refactor)


@dataclass
class HyperbolicSAEConfig:
    input_dim: int
    n_features: int = 128
    ball_dim: int = 32
    curvature: float = 1.0  # positive MAGNITUDE κ; primitive gets c = -κ
    sparsity_weight: float = 1e-3
    bias: bool = True
    jumprelu_threshold: float = 0.05  # gamfit JumpReLU prior; 0 ⇒ fall back to L1
    lorentz: bool = False  # use boundary-safe Lorentz forward in the primitive
    init_scale: float = 0.05  # atom init std (projected into the ball by primitive)


class HyperbolicSAE(nn.Module):
    """Sparse autoencoder whose decoder atoms live in the Poincaré ball ``B^d``.

    Args:
        input_dim:   ambient dimension D.
        n_features:  number of atoms F.
        ball_dim:    ball dimension d (intrinsic hyperbolic dim).
        curvature:   positive magnitude κ. The geometric curvature handed to the
                     ``PoincareAtoms`` primitive is ``c = -κ`` (strictly < 0).
        sparsity_weight: scales both the JumpReLU/L1 and the geodesic-distance term.
        jumprelu_threshold: per-feature JumpReLU threshold; ``<= 0`` falls back to L1.
        lorentz:     route decode through the primitive's Lorentz forward.
        init_scale:  std of the atom initializer (projected into the ball by the
                     primitive).
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 128,
        ball_dim: int = 32,
        curvature: float = 1.0,
        sparsity_weight: float = 1e-3,
        jumprelu_threshold: float = 0.05,
        lorentz: bool = False,
        init_scale: float = 0.05,
    ):
        super().__init__()
        kappa = float(curvature)
        if kappa <= 0.0:
            raise ValueError(
                "HyperbolicSAE.curvature is a positive magnitude κ "
                f"(the primitive receives c = -κ < 0); got {curvature!r}"
            )
        self.cfg = HyperbolicSAEConfig(
            input_dim=input_dim,
            n_features=n_features,
            ball_dim=ball_dim,
            curvature=kappa,
            sparsity_weight=float(sparsity_weight),
            jumprelu_threshold=float(jumprelu_threshold),
            lorentz=bool(lorentz),
            init_scale=float(init_scale),
        )
        D, F, d = input_dim, n_features, ball_dim
        self.D, self.F, self.d = D, F, d
        # Magnitude κ (positive); the negative curvature is on the primitive.
        self.c = kappa

        # Per-feature sparse gate.
        self.W_gate = nn.Linear(D, F, bias=True)

        # Learnable dictionary of F atoms IN the ball B^d (geometric c = -κ).
        self.atoms_dict = PoincareAtoms(
            F=F,
            ball_dim=d,
            curvature=-kappa,
            lorentz=bool(lorentz),
            init_scale=float(init_scale),
        )

        # Tangent(ball) → ambient readout.
        self.W_dec_out = nn.Linear(d, D, bias=True)

        # gamfit JumpReLU prior on the gate vector (smoothed-L0 surrogate).
        if jumprelu_threshold > 0.0:
            self.jumprelu = JumpReLUPenalty(
                thresholds=torch.full(
                    (F,), float(jumprelu_threshold), dtype=torch.float64
                ),
                weight=1.0,
                smoothing_eps=1e-3,
            )
        else:
            self.jumprelu = None

        # Init.
        nn.init.zeros_(self.W_gate.bias)
        nn.init.kaiming_uniform_(self.W_dec_out.weight, a=5 ** 0.5)
        nn.init.zeros_(self.W_dec_out.bias)

    # ------------------------------------------------------------------ helpers

    def atom_positions(self) -> torch.Tensor:
        """(F, d) atom centers in the Poincaré ball (the primitive's parameter)."""
        return self.atoms_dict.atoms

    # ------------------------------------------------------------------ forward

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x → (atom centers in B^d, gate activations).

        Returns:
            atoms_pos: (F, d)  in B^d — the shared learnable dictionary.
            gates:     (B, F)  ReLU-activated, sparse.

        The atom positions no longer depend on ``x`` (the dictionary is shared);
        the tuple shape is kept for API continuity with downstream callers.
        """
        gates = torch.relu(self.W_gate(x))  # (B, F)
        return self.atoms_dict.atoms, gates

    def decode(self, gates: torch.Tensor) -> torch.Tensor:
        """Decode gates → ambient reconstruction.

        gates: (B, F)
        """
        ball = self.atoms_dict(gates)  # (B, d) — tangent aggregation + exp_0
        x_hat = self.W_dec_out(ball)  # (B, D)
        return x_hat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        atoms_pos, gates = self.encode(x)
        x_hat = self.decode(gates)

        # Hyperbolic feature strength: geodesic radius (distance from origin) of
        # each atom in the ball. Shared across the batch (atoms are a dictionary).
        radii = self.feature_radii()  # (F,)

        # gamfit JumpReLU prior on the gate vector (smoothed-L0 surrogate); if
        # disabled, fall back to mean-L1.
        if self.jumprelu is not None:
            jr_val = self.jumprelu(gates)
            l1 = jr_val / float(gates.numel()) * self.cfg.sparsity_weight
        else:
            l1 = gates.abs().mean() * self.cfg.sparsity_weight
        # Geodesic-weighted sparsity: penalize gate mass on deep (boundary) atoms.
        dist_l1 = (gates * radii.unsqueeze(0)).mean() * self.cfg.sparsity_weight

        return {
            "x_hat": x_hat,
            "atoms_pos": atoms_pos,
            "gates": gates,
            "radii": radii,
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

    def feature_radii(self) -> torch.Tensor:
        """(F,) geodesic distance of each atom from the origin via the primitive."""
        pos = self.atoms_dict.atoms  # (F, d)
        zero = torch.zeros_like(pos)
        return self.atoms_dict.distance(zero, pos)

    @torch.no_grad()
    def feature_norms_in_ball(self) -> torch.Tensor:
        """Geodesic distance of each atom from origin (hierarchy-depth proxy)."""
        return self.feature_radii()


__all__ = ["HyperbolicSAE", "HyperbolicSAEConfig"]
