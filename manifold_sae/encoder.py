"""Thin adapter over gamfit's built-in ManifoldSAE encoder (gamfit >= 0.1.241).

The hand-rolled per-feature GELU-MLP encoder has been deleted. ``gamfit.torch.ManifoldSAE``
already ships a built-in encoder (linear when ``encoder_hidden=0``, GELU-MLP otherwise), and
gamfit exposes an amortized/distilled encoder path (``fit.distill_encoder(X)`` +
``gamfit.distill``) for the high-level facade. ``ManifoldEncoder`` is now a light adapter that
constructs that built-in encoder and re-exposes a compatible
``forward(x, y_proj) -> (z_raw, mask_soft, mask_binary)`` triple for the few callers that
import it. Behavior differs from the old bespoke encoder — this is a cutover shim, not a
drop-in numeric replica.
"""

from __future__ import annotations

import torch
from torch import nn

from gamfit.torch import ManifoldSAE, ManifoldSAEConfig


def _manifold_for_rank(rank: int) -> tuple[str, int]:
    """Pick a gamfit ``atom_manifold`` compatible with ``rank``.

    gamfit enforces circle⇒rank 1, sphere⇒rank 2, product⇒rank ≥ 2.
    """
    if rank <= 1:
        return "circle", 1
    if rank == 2:
        return "sphere", 2
    return "product", rank


class ManifoldEncoder(nn.Module):
    """Adapter exposing gamfit's built-in ManifoldSAE encoder.

    The underlying encoder maps ``x`` (N, input_dim) to per-feature coordinate +
    gate logits, shaped (N, (rank + 1) * n_features). We split that into a scalar
    coordinate ``z_raw`` and a gate logit per feature and apply the same
    straight-through TopK gating contract the old encoder used.
    """

    def __init__(
        self,
        intrinsic_rank: int,
        n_features: int,
        input_dim: int,
        hidden_dim: int | None = None,
        top_k: int | None = None,
        shared_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.intrinsic_rank = int(intrinsic_rank)
        self.n_features = int(n_features)
        self.input_dim = int(input_dim)
        self.top_k = top_k
        self.shared_encoder = bool(shared_encoder)
        # encoder_hidden=0 ⇒ linear; otherwise GELU-MLP width. Preserve the old
        # cap so wide residual streams don't allocate an oversized trunk.
        if hidden_dim is None:
            encoder_hidden = min(256, max(64, 8 * self.n_features))
        else:
            encoder_hidden = int(hidden_dim)
        self.hidden_dim = encoder_hidden

        atom_manifold, rank = _manifold_for_rank(self.intrinsic_rank)
        self._rank = rank
        cfg = ManifoldSAEConfig(
            input_dim=self.input_dim,
            n_atoms=self.n_features,
            intrinsic_rank=rank,
            atom_manifold=atom_manifold,
            encoder_hidden=encoder_hidden,
        )
        # gamfit's ManifoldSAE owns the encoder numerics; we only borrow the
        # encoder submodule (built-in linear / GELU-MLP) and drop the rest so we
        # don't register the decoder/basis params as our own.
        self.encoder = ManifoldSAE(cfg).encoder

    def forward(
        self, x: torch.Tensor, y_proj: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, D). ``y_proj`` is accepted for signature compatibility and ignored
        (gamfit's built-in encoder consumes only ``x``).

        Returns ``(z_raw, mask_soft, mask_binary)`` — the same triple shape the old
        encoder produced.
        """
        # gamfit's default dtype is float64; its encoder rejects a dtype mismatch.
        enc_dtype = next(self.encoder.parameters()).dtype
        raw = self.encoder(x.to(dtype=enc_dtype))  # (B, (rank+1)*F)
        B = raw.shape[0]
        F = self.n_features
        per_feat = raw.reshape(B, F, self._rank + 1)
        z_raw = torch.nan_to_num(per_feat[:, :, 0], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        amp_logits = torch.nan_to_num(per_feat[:, :, -1], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        mask_soft = torch.sigmoid(amp_logits)

        if getattr(self, "continuous_amp", False):
            amp_cont = torch.nn.functional.softplus(amp_logits)
            if self.top_k is not None and self.top_k < self.n_features:
                _vals, idx = torch.topk(amp_cont, self.top_k, dim=1)
                gate = torch.zeros_like(amp_cont)
                gate.scatter_(1, idx, 1.0)
                amp_out = amp_cont * gate
            else:
                amp_out = amp_cont
            return z_raw, mask_soft, amp_out

        # Binary straight-through TopK gate (curves absorb magnitude, gate is 0/1).
        if self.top_k is not None and self.top_k < self.n_features:
            _vals, idx = torch.topk(mask_soft, self.top_k, dim=1)
            hard_mask = torch.zeros_like(mask_soft)
            hard_mask.scatter_(1, idx, 1.0)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        else:
            hard_mask = torch.ones_like(mask_soft)
            mask_binary = hard_mask + (mask_soft - mask_soft.detach())
        return z_raw, mask_soft, mask_binary
