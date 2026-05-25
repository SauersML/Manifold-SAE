"""Complete Replacement Model (CRM) skeleton.

Chains per-layer SAEs and inter-layer transcoders in series:

    x_0 → SAE_0 → x̂_0 → Tcoder_{0→1} → x̂_1 → SAE_1 → x̂'_1 → ...

For an L-layer stack we have L SAEs and L−1 transcoders. The training loss
is the sum of (a) per-stage reconstruction at each SAE and (b) per-stage
activation match — predicted vs ground-truth layer activations.

This skeleton uses :class:`gamfit.torch.SkipAffineSmooth` for both halves:

* ``rank_skip=0`` ⇒ pure sparse SAE on a single layer (no bypass).
* ``rank_skip>0`` ⇒ paired-residual transcoder with low-rank affine bypass
  (Paulo, Shabalin, Belrose 2025).

The JumpReLU prior + atom dictionary live in the gamfit primitive; we only
own the chaining + per-stage loss + diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn

from gamfit.torch import SkipAffineSmooth  # gamfit >= 0.1.123


def _build_skip_affine(
    in_dim: int,
    out_dim: int,
    n_atoms: int,
    rank_skip: int,
    jumprelu_threshold: float,
    dtype: torch.dtype | None = None,
) -> SkipAffineSmooth:
    """Build a SkipAffineSmooth, working around the gamfit 0.1.123 tied-init
    crash when ``in_dim != out_dim`` (encoder is (in,F) but decoder is (F,out);
    `W_dec.copy_(W_enc.t())` fails). For square cases we use the primitive
    as-shipped; for rectangular cases we use square dummy dims for the
    constructor (which only affects the discarded init) and overwrite the
    parameter tensors with correctly-shaped Kaiming-init weights.
    """
    dt = dtype if dtype is not None else torch.get_default_dtype()
    if in_dim == out_dim:
        return SkipAffineSmooth(
            in_dim=in_dim, out_dim=out_dim, n_atoms=n_atoms,
            rank_skip=rank_skip, jumprelu_threshold=jumprelu_threshold, dtype=dt,
        )
    # Construct under matched dims, then reshape decoder/bypass to (F, out_dim).
    sm = SkipAffineSmooth(
        in_dim=in_dim, out_dim=in_dim, n_atoms=n_atoms,
        rank_skip=min(rank_skip, in_dim), jumprelu_threshold=jumprelu_threshold, dtype=dt,
    )
    sm.out_dim = out_dim
    sm.W_dec = nn.Parameter(torch.empty(n_atoms, out_dim, dtype=dt))
    nn.init.kaiming_uniform_(sm.W_dec, a=5**0.5)
    sm.b_out = nn.Parameter(torch.zeros(out_dim, dtype=dt))
    if rank_skip > 0:
        sm.skip_U = nn.Parameter(torch.empty(out_dim, min(rank_skip, in_dim, out_dim), dtype=dt))
        nn.init.kaiming_uniform_(sm.skip_U, a=5**0.5)
    return sm


@dataclass
class CRMConfig:
    layer_dims: List[int]                       # [d_0, d_1, ..., d_{L-1}]
    n_features_per_sae: int = 512
    transcoder_mid: int = 1024
    transcoder_rank_skip: int = 32
    jumprelu_threshold: float = 0.05
    # Back-compat: pre-0.1.123 CRM exposed `sae_top_k` / `transcoder_top_k`.
    # These are ignored under the gamfit SkipAffineSmooth backend (sparsity is
    # JumpReLU-gated, not hard-K) but accepted so existing call-sites keep
    # working. Drop in a future cleanup pass.
    sae_top_k: int | None = None
    transcoder_top_k: int | None = None


class CompleteReplacementModel(nn.Module):
    """Chains L sparse-SAE smooths and L−1 sparse-transcoder smooths.

    All eight building blocks are :class:`gamfit.torch.SkipAffineSmooth`
    instances. SAEs use ``rank_skip=0`` (no bypass); transcoders use
    ``rank_skip>0`` (Paulo et al. low-rank affine bypass).
    """

    def __init__(self, config: CRMConfig) -> None:
        super().__init__()
        self.config = config
        self.L = len(config.layer_dims)
        assert self.L >= 2, "Need at least 2 layers"

        # Per-layer SAEs: rank_skip=0 → pure sparse code, in_dim == out_dim.
        dt = torch.get_default_dtype()
        self.saes = nn.ModuleList(
            [
                SkipAffineSmooth(
                    in_dim=d,
                    out_dim=d,
                    n_atoms=config.n_features_per_sae,
                    rank_skip=0,
                    jumprelu_threshold=config.jumprelu_threshold,
                    dtype=dt,
                )
                for d in config.layer_dims
            ]
        )
        # Inter-layer transcoders: rank_skip>0 → paired-residual bypass.
        self.transcoders = nn.ModuleList(
            [
                _build_skip_affine(
                    in_dim=config.layer_dims[l],
                    out_dim=config.layer_dims[l + 1],
                    n_atoms=config.transcoder_mid,
                    rank_skip=min(
                        config.transcoder_rank_skip,
                        config.layer_dims[l],
                        config.layer_dims[l + 1],
                    ),
                    jumprelu_threshold=config.jumprelu_threshold,
                )
                for l in range(self.L - 1)
            ]
        )

    def forward(self, xs: Sequence[torch.Tensor]) -> dict:
        """Chained forward; feeds each SAE the previous transcoder's output."""
        assert len(xs) == self.L, (len(xs), self.L)
        recons: list[torch.Tensor] = []
        latents_sae: list[torch.Tensor] = []
        latents_tc: list[torch.Tensor] = []
        prev_x = xs[0]
        for l in range(self.L):
            recon_l, z_l = self.saes[l](prev_x)
            recons.append(recon_l)
            latents_sae.append(z_l)
            if l < self.L - 1:
                tc_out, tc_z = self.transcoders[l](recon_l)
                latents_tc.append(tc_z)
                prev_x = tc_out
        return {"recons": recons, "latents_sae": latents_sae, "latents_tc": latents_tc}

    def loss(
        self,
        xs: Sequence[torch.Tensor],
        recon_weight: float = 1.0,
        match_weight: float = 1.0,
        sparsity_weight: float = 1e-3,
    ) -> dict:
        out = self.forward(xs)
        per_stage = []
        total = xs[0].new_zeros(())
        for l in range(self.L):
            mse_l = (out["recons"][l] - xs[l]).pow(2).mean()
            per_stage.append(mse_l.detach())
            w = recon_weight if l == 0 else match_weight
            total = total + w * mse_l
            # gamfit JumpReLU prior on each SAE latent.
            total = total + sparsity_weight * self.saes[l].jumprelu(out["latents_sae"][l])
        for l, z_tc in enumerate(out["latents_tc"]):
            total = total + sparsity_weight * self.transcoders[l].jumprelu(z_tc)
        return {"loss": total, "per_stage_mse": per_stage, "out": out}

    @torch.no_grad()
    def per_stage_r2(self, xs: Sequence[torch.Tensor]) -> List[float]:
        out = self.forward(xs)
        r2s: list[float] = []
        for l in range(self.L):
            res = (out["recons"][l] - xs[l]).pow(2).mean().item()
            var = xs[l].var().item()
            r2s.append(1.0 - res / max(var, 1e-12))
        return r2s


__all__ = ["CRMConfig", "CompleteReplacementModel"]
