"""DAS-SAE: Distributed Alignment Search Sparse Autoencoder (gamfit-native).

Operationalizes "feature i causally encodes concept C" as a SWAP TEST rather
than a correlation. Inspired by Geiger et al. DAS / Boundless-DAS
(arXiv:2303.02536) and the Non-Linear Representation Dilemma
(OpenReview ZYXTLo7kCi).

Causal abstraction targeted: ``hsv_name`` — the cogito-L40 color manifold
factors into a perceptual HSV subspace + a name-semantic subspace
(auto_exp_38, project_cogito_recovery_at_d_aux_3). A subset of SAE features
should align with the HSV (hue) axis such that swapping THOSE feature
activations between two colors A and B produces the activation that the LLM
would emit for the "hue of A on the name of B" hypothetical color.

BREAKING REDESIGN (gamfit-native)
---------------------------------
The decoder + interchange-swap machinery is now the gamfit primitive
:class:`gamfit.torch.InterchangeSwapDecoder`. Two semantic changes flow from
this and are *embraced as the new design*, not shimmed:

1. GATED DECODE. The old ``decode()`` was a plain linear ``z @ W_dec.T``; the
   gate was used only as a swap mask. The primitive gates at decode time::

       x_hat[i, d] = sum_f gate[f] * z[i, f] * W_dec[d, f] + bias[d]

   So the per-feature scalar ``gate`` IS the learned hue gate now — it
   directly scales each atom's contribution to reconstruction. There is no
   longer a separate ``gate_logits``/``sigmoid`` head. ``hue_mask()`` returns
   ``decoder.gate`` (clamped to [0, 1] for the swap-mask threshold and the
   sparsity/entropy diagnostics).

2. BOOLEAN SWAP MASK. ``InterchangeSwapDecoder.swap_decode(z_a, z_b,
   atom_mask)`` takes a BOOLEAN ``(F,)`` mask and fuses swap + gated decode in
   one Rust call. Where ``atom_mask[f]`` is True the column of ``z_a`` is used,
   else ``z_b``. The continuous-mask convex blend of the old design is gone.
   The group-swap test thresholds per-feature scores to bool; the per-feature
   test loops one-hot bool masks.

Training objective::

    L = ||x_a - x̂_a||² + ||x_b - x̂_b||²            (gated recon, both halves)
      + λ_intv · ||swap_decode(z_b, z_a, hue_bool) − target_swap||²
      + λ_gate · ||gate||_1                          (small hue subset)
      + λ_gate_entropy · binary_entropy(gate)        (push gate → {0, 1})
      + λ_l1   · ||z||_1                             (standard SAE sparsity)

target_swap is built in the data layer using the HSV-supervised subspace
(auto_exp_38): x_target = x_b + (hue_a - hue_b) * v_hue / ||v_hue||².

Public surface (stable contract):
  DASSAE(config)
    .forward(x)                      -> DASSAEOutput(z, x_hat, pre_act)
    .encode(x)                       -> (z, pre_act)
    .decode(z)                       -> GATED x_hat
    .swap_decode(z_a, z_b, mask)     -> fused swap + gated decode (BOOL mask)
    .hue_mask()                      -> decoder.gate clamped to [0, 1]
    .gate()                          -> raw decoder.gate (F,)
    .hue_bool_mask(threshold=0.5)    -> bool (F,) mask of gated-on atoms
    .compute_loss(...)               -> dict of losses
    .per_feature_swap_score(...)     -> (F,) one-hot bool swap-R² diagnostic
  build_target_swap(...), fit_hue_direction(...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from gamfit.torch import JumpReLUPenalty  # gamfit >= 0.1.123 (replaces hand L1)
from gamfit.torch import InterchangeSwapDecoder  # gamfit-native gated swap decoder


@dataclass
class DASSAEConfig:
    input_dim: int
    n_features: int = 512
    abstraction: Literal["hsv_name"] = "hsv_name"
    top_k: int | None = None          # if set, use TopK; else L1 (relu)
    tied_weights: bool = False
    init_gate: float = 0.12           # initial value of decoder.gate
    init_scale: float = 0.02          # decoder W_dec init std (primitive default)
    normalize_decoder: bool = True
    jumprelu_threshold: float = 0.0   # >0 ⇒ swap hand-rolled L1 for gamfit JumpReLU


@dataclass
class DASSAEOutput:
    z: torch.Tensor                   # (B, F) sparse latent
    x_hat: torch.Tensor               # (B, D) GATED reconstruction
    pre_act: torch.Tensor             # (B, F) pre-activation logits


class DASSAE(nn.Module):
    """SAE encoder + gamfit ``InterchangeSwapDecoder`` head.

    Adam-owned parameters:
      W_enc : (F, D)         — encoder (omitted if ``tied_weights``)
      b_enc : (F,)           — encoder bias
      b_dec : (D,)           — pre-encoder offset (subtracted before encode)
      decoder.W_dec : (D, F) — gamfit primitive decoder weights
      decoder.gate  : (F,)   — gamfit per-feature scalar gate == hue gate
      decoder.bias  : (D,)   — gamfit decoder bias (post-decode offset)

    The decoder GATES: ``x_hat[i,d] = sum_f gate[f] z[i,f] W_dec[d,f] + bias[d]``.
    The gate is the learned hue gate directly (no sigmoid head). Interchange
    swap uses the primitive's fused boolean-mask ``swap_decode``.
    """

    def __init__(self, config: DASSAEConfig) -> None:
        super().__init__()
        self.config = config
        D = int(config.input_dim)
        F = int(config.n_features)

        # gamfit-native gated swap decoder owns W_dec (D, F), gate (F,), bias (D,).
        self.decoder = InterchangeSwapDecoder(
            D=D,
            F=F,
            swap_mode="scalar_mask",
            bias=True,
            init_scale=float(config.init_scale),
        )
        # Unit-norm decoder columns + tied encoder init (standard SAE recipe).
        with torch.no_grad():
            w = self.decoder.W_dec
            w.div_(w.norm(dim=0, keepdim=True).clamp(min=1e-8))
            # Gate initialised to a small value so the network starts with a
            # weak, sparse-ish hue subset and grows it only when the
            # interchange loss demands.
            self.decoder.gate.fill_(float(config.init_gate))

        if config.tied_weights:
            # Encoder tied to W_dec.T at all times — only one set of params.
            self.register_parameter("W_enc", None)
        else:
            self.W_enc = nn.Parameter(self.decoder.W_dec.detach().t().clone())
        self.b_enc = nn.Parameter(torch.zeros(F))
        # Pre-encoder offset. Decoder post-offset lives in ``decoder.bias``.
        self.b_dec = nn.Parameter(torch.zeros(D))

        # Optional gamfit JumpReLU prior on the SAE latent (smoothed-L0).
        if getattr(config, "jumprelu_threshold", 0.0) > 0.0:
            self.jumprelu = JumpReLUPenalty(
                thresholds=torch.full((F,), float(config.jumprelu_threshold), dtype=torch.float64),
                weight=1.0,
                smoothing_eps=1e-3,
            )
        else:
            self.jumprelu = None

    # ----- accessors -----------------------------------------------------
    def encoder_weight(self) -> torch.Tensor:
        if self.config.tied_weights or getattr(self, "W_enc", None) is None:
            return self.decoder.W_dec.t()
        return self.W_enc

    def gate(self) -> torch.Tensor:
        """Raw per-feature decoder gate (F,)."""
        return self.decoder.gate

    def hue_mask(self) -> torch.Tensor:
        """The learned hue gate, clamped to [0, 1].

        With the gated-decode redesign the gate IS the hue gate, so this is a
        clamped view of ``decoder.gate`` (no sigmoid). Used for the sparsity /
        entropy diagnostics; the bool swap mask is derived from it via
        :meth:`hue_bool_mask`.
        """
        return self.decoder.gate.clamp(0.0, 1.0)

    def hue_bool_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Boolean (F,) swap mask: atoms whose gate exceeds ``threshold``."""
        return self.decoder.gate > float(threshold)

    # ----- forward / swap -----------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        We = self.encoder_weight()
        pre = (x - self.b_dec) @ We.t() + self.b_enc
        if self.config.top_k is not None and self.config.top_k > 0:
            K = int(self.config.top_k)
            # TopK sparse activation (Anthropic-style).
            vals, idx = pre.topk(K, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, idx, torch.relu(vals))
        else:
            z = torch.relu(pre)
        return z, pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """GATED decode via the gamfit primitive.

        ``x_hat[i,d] = sum_f gate[f] z[i,f] W_dec[d,f] + decoder.bias[d]``.
        Note this is gated — it is NOT the old plain ``z @ W_dec.T + b_dec``.
        ``b_dec`` (the pre-encoder offset) is intentionally not re-added here;
        the decoder's own ``bias`` carries the post-decode offset.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> DASSAEOutput:
        z, pre = self.encode(x)
        x_hat = self.decode(z)
        return DASSAEOutput(z=z, x_hat=x_hat, pre_act=pre)

    def swap_decode(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Fused interchange-swap + gated decode via the gamfit primitive.

        For atoms with ``atom_mask[f]`` True, use ``z_a[:, f]``; else
        ``z_b[:, f]``. Then GATED-decode the composed latent. ``atom_mask``
        must be a BOOLEAN ``(F,)`` tensor. If ``None``, defaults to the
        learned hue mask thresholded at ``threshold``.

        Returns a tensor with autograd wired through ``z_a``, ``z_b``,
        ``W_dec``, ``gate`` and ``bias`` (the bool mask carries no gradient).
        """
        if atom_mask is None:
            atom_mask = self.hue_bool_mask(threshold)
        if atom_mask.dtype != torch.bool:
            atom_mask = atom_mask.to(torch.bool)
        return self.decoder.swap_decode(z_a, z_b, atom_mask=atom_mask)

    # ----- loss ---------------------------------------------------------
    def compute_loss(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        target_swap: torch.Tensor,
        lambda_intv: float = 1.0,
        lambda_gate: float = 1e-3,
        lambda_l1: float = 1e-3,
        lambda_gate_entropy: float = 0.0,
        swap_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Gated reconstruction (both halves) + fused interchange loss.

        target_swap : (B, D) — cogito-L40 activation for "hue(a) injected into
                      b" hypothetical color, constructed by the data layer from
                      the auto_exp_38 HSV subspace.

        The interchange term now uses the FUSED boolean-mask ``swap_decode``:
        the hue gate is thresholded to a bool mask, then ``z_a``'s gated hue
        atoms are spliced into ``z_b`` and decoded in a single Rust call.
        Sparsity / binarization pressure acts directly on ``decoder.gate``.
        """
        out_a = self.forward(x_a)
        out_b = self.forward(x_b)

        recon_loss = (
            (out_a.x_hat - x_a).pow(2).mean()
            + (out_b.x_hat - x_b).pow(2).mean()
        )

        # Interchange: take b's latent, splice a's hue-features into it, gated-
        # decode, compare to target_swap (the LLM's "what would cogito-L40 emit
        # if b had a's hue" reference). atom_mask True ⇒ use z_a (the source of
        # the hue). Fused swap + gated decode in one Rust call.
        hue_bool = self.hue_bool_mask(swap_threshold)
        x_swap_hat = self.swap_decode(out_a.z, out_b.z, atom_mask=hue_bool)
        intv_loss = (x_swap_hat - target_swap).pow(2).mean()

        # Gate sparsity / binarization act on the raw decoder gate now.
        gate = self.decoder.gate
        gate_clamped = gate.clamp(0.0, 1.0)
        # Encourage a small, sharp hue subset (L1 on the gate magnitude).
        gate_l1 = gate.abs().sum()
        # Binarization pressure: pull each gate toward 0 or 1.
        gate_entropy = -(
            gate_clamped * (gate_clamped + 1e-8).log()
            + (1 - gate_clamped) * (1 - gate_clamped + 1e-8).log()
        ).mean()

        if self.jumprelu is not None:
            # gamfit smoothed-L0 prior (sum over (B,F)); normalize to mean
            # element-wise so lambda_l1 has the same scale as the L1 path.
            denom = float(out_a.z.numel())
            l1_loss = (self.jumprelu(out_a.z) + self.jumprelu(out_b.z)) / denom
        else:
            l1_loss = (out_a.z.abs().mean() + out_b.z.abs().mean())

        total = (
            recon_loss
            + lambda_intv * intv_loss
            + lambda_gate * gate_l1
            + lambda_l1 * l1_loss
            + lambda_gate_entropy * gate_entropy
        )
        return {
            "loss": total,
            "recon": recon_loss.detach(),
            "intv": intv_loss.detach(),
            "gate_l1": gate_l1.detach(),
            "l1": l1_loss.detach(),
            "gate_entropy": gate_entropy.detach(),
            "n_hue_features_soft": (gate > 0.5).float().sum().detach(),
        }

    # ----- post-hoc swap-test diagnostic --------------------------------
    @torch.no_grad()
    def per_feature_swap_score(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        target_swap: torch.Tensor,
    ) -> torch.Tensor:
        """For each feature i, R² of swap_decode(z_a, z_b, e_i) vs target_swap.

        Returns shape (F,). A feature scoring > 0.7 is "identified" as causally
        encoding the hue concept under this abstraction. Uses the primitive's
        fused boolean one-hot swap_decode per feature (gated decode included),
        so the baseline is the no-swap gated decode of z_b.
        """
        out_a = self.forward(x_a)
        out_b = self.forward(x_b)
        F = out_a.z.shape[-1]
        device = x_a.device
        # Baseline: no swap ⇒ all-False mask ⇒ gated decode(z_b).
        all_false = torch.zeros(F, dtype=torch.bool, device=device)
        baseline = self.swap_decode(out_a.z, out_b.z, atom_mask=all_false)
        var_naive = (target_swap - baseline).pow(2).mean(dim=-1)  # (B,)

        scores = torch.zeros(F, device=device)
        for i in range(F):
            ei = torch.zeros(F, dtype=torch.bool, device=device)
            ei[i] = True
            x_hat = self.swap_decode(out_a.z, out_b.z, atom_mask=ei)
            resid = (target_swap - x_hat).pow(2).mean(dim=-1)
            r2 = 1.0 - resid / var_naive.clamp(min=1e-12)
            scores[i] = r2.mean()
        return scores

    @torch.no_grad()
    def normalize_decoder_columns_(self) -> None:
        """Unit-norm decoder columns (standard SAE post-step)."""
        if not self.config.normalize_decoder:
            return
        with torch.no_grad():
            w = self.decoder.W_dec
            n = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
            w.div_(n)


# ---------------------------------------------------------------------------
# HSV-subspace utilities — used by the data layer to build target_swap.
# ---------------------------------------------------------------------------

def fit_hue_direction(
    X: torch.Tensor,                  # (N, D) cogito activations
    hue: torch.Tensor,                # (N,)   target hue value per row
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Ridge regression: v_hue minimises ||X v - hue_centered||² + ridge·||v||².

    Returns v_hue ∈ R^D such that X @ v_hue ≈ centered hue. The ambient
    direction along which incrementing v_hue increases the LLM's perceived
    hue.
    """
    Xc = X - X.mean(0, keepdim=True)
    hc = hue - hue.mean()
    # Solve (X^T X + ridge I) v = X^T h.
    D = X.shape[1]
    XtX = Xc.t() @ Xc + ridge * torch.eye(D, dtype=X.dtype, device=X.device)
    Xth = Xc.t() @ hc
    v = torch.linalg.solve(XtX, Xth)
    # Normalize so a 1-unit change in v.dot(x) corresponds to a 1-unit
    # change in hue.
    pred = Xc @ v
    scale = (pred * hc).sum() / (pred * pred).sum().clamp(min=1e-8)
    return v * scale


def build_target_swap(
    x_a: torch.Tensor,                # (B, D)
    x_b: torch.Tensor,                # (B, D)
    hue_a: torch.Tensor,              # (B,)
    hue_b: torch.Tensor,              # (B,)
    v_hue: torch.Tensor,              # (D,)
) -> torch.Tensor:
    """Synthesize the "hue(a) injected into b" target activation.

    target = x_b + (hue_a - hue_b) * v_hue / ||v_hue||²
    so that the hue projection of target equals hue_a while every other
    component is inherited from x_b.
    """
    delta = (hue_a - hue_b).unsqueeze(-1)            # (B, 1)
    v_norm_sq = (v_hue * v_hue).sum().clamp(min=1e-8)
    return x_b + delta * v_hue.unsqueeze(0) / v_norm_sq
