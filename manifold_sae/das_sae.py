"""DAS-SAE: Distributed Alignment Search Sparse Autoencoder.

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

Architecture:
  - Standard L1/TopK SAE: x → z = relu(W_e (x - b_dec) + b_enc); x̂ = W_d z + b_dec
  - Learned per-feature hue gate: hue_mask = sigmoid(gate_logits)   ∈ (0, 1)^F
  - swap(z_a, z_b, m) = z_a * (1 - m) + z_b * m

Training objective:
  L = ||x - x̂||²
    + λ_intv · ||W_d · swap(z_a, z_b, hue_mask) + b_dec  −  target_swap||²
    + λ_gate · (||hue_mask||_1)          # encourage a SMALL set of hue features
    + λ_l1   · ||z||_1                   # standard SAE sparsity

target_swap is built in the data layer using the HSV-supervised subspace
(auto_exp_38): x_target = x_b + (hue_a - hue_b) * v_hue, where v_hue is the
ambient-space direction for the hue axis.

API:
  DASSAE(D, F=512, abstraction='hsv_name')
    .forward(x)                       -> SAEOutput(z, x_hat)
    .swap(z_a, z_b, mask=None)        -> z_swapped (defaults to learned hue_mask)
    .decode(z)                        -> x_hat
    .hue_mask()                       -> sigmoid(gate_logits)
    .compute_loss(x_a, x_b, target_swap, lambdas) -> dict of losses

Identification primitive: a feature i passes the HUE-SWAP TEST at threshold τ
if decoding swap(z_a, z_b, e_i)  reproduces target_swap with R² ≥ τ on a held
out set, where e_i is the one-hot mask for feature i. The number of features
that pass at τ=0.7 is the "identified hue feature count".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn


@dataclass
class DASSAEConfig:
    input_dim: int
    n_features: int = 512
    abstraction: Literal["hsv_name"] = "hsv_name"
    top_k: int | None = None          # if set, use TopK; else L1 (relu)
    tied_weights: bool = False
    init_gate_logit: float = -2.0     # sigmoid(-2) ≈ 0.12 — start sparse-ish
    normalize_decoder: bool = True


@dataclass
class DASSAEOutput:
    z: torch.Tensor                   # (B, F) sparse latent
    x_hat: torch.Tensor               # (B, D) reconstruction
    pre_act: torch.Tensor             # (B, F) pre-activation logits


class DASSAE(nn.Module):
    """SAE + interchange-intervention head.

    Adam-owned parameters:
      W_enc : (F, D)    — encoder
      b_enc : (F,)      — encoder bias
      W_dec : (D, F)    — decoder
      b_dec : (D,)      — pre-encoder + post-decoder offset
      gate_logits : (F,) — pre-sigmoid hue gate; sigmoid → hue_mask in (0,1)
    """

    def __init__(self, config: DASSAEConfig) -> None:
        super().__init__()
        self.config = config
        D = int(config.input_dim)
        F = int(config.n_features)

        # Decoder initialized to random unit columns; encoder = decoder.T (tied
        # init) — standard SAE recipe.
        W = torch.randn(D, F) * (1.0 / (D ** 0.5))
        W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec = nn.Parameter(W)
        if config.tied_weights:
            # Encoder tied to W_dec.T at all times — only one set of params.
            self.register_parameter("W_enc", None)
        else:
            self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(F))
        self.b_dec = nn.Parameter(torch.zeros(D))

        # Learned per-feature hue gate. Initialized at sigmoid(init_gate_logit)
        # so the network starts with a small soft hue subset and grows it only
        # when interchange-loss demands.
        self.gate_logits = nn.Parameter(
            torch.full((F,), float(config.init_gate_logit))
        )

    # ----- accessors -----------------------------------------------------
    def encoder_weight(self) -> torch.Tensor:
        if self.config.tied_weights or self.W_enc is None:
            return self.W_dec.t()
        return self.W_enc

    def hue_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

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
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> DASSAEOutput:
        z, pre = self.encode(x)
        x_hat = self.decode(z)
        return DASSAEOutput(z=z, x_hat=x_hat, pre_act=pre)

    def swap(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Interchange-intervention: replace masked features of z_a with z_b's.

        mask ∈ [0, 1]^F.   z = z_a * (1 - mask) + z_b * mask.
        mask=0 (no-swap) ⇒ standard SAE forward.
        mask=1 (full swap) ⇒ z_b.
        Default: learned hue_mask().
        """
        if mask is None:
            mask = self.hue_mask()
        # mask broadcasts over batch.
        return z_a * (1.0 - mask) + z_b * mask

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
    ) -> dict[str, torch.Tensor]:
        """Reconstruction (on both halves of the pair) + interchange loss.

        target_swap : (B, D) — cogito-L40 activation for
                      "hue(a) injected into b" hypothetical color, constructed
                      by the data layer from the auto_exp_38 HSV subspace.
        """
        out_a = self.forward(x_a)
        out_b = self.forward(x_b)

        recon_loss = (
            (out_a.x_hat - x_a).pow(2).mean()
            + (out_b.x_hat - x_b).pow(2).mean()
        )

        # Interchange: take b's latent, splice a's hue-features into it,
        # decode, compare to target_swap (which is the LLM's "what would
        # cogito-L40 emit if b had a's hue" reference).
        mask = self.hue_mask()
        z_swapped = self.swap(out_b.z, out_a.z, mask=mask)
        x_swap_hat = self.decode(z_swapped)
        intv_loss = (x_swap_hat - target_swap).pow(2).mean()

        # Encourage a small, sharp hue subset.
        gate_l1 = mask.sum()
        # Binarization pressure: pull each mask toward 0 or 1.
        gate_entropy = -(mask * (mask + 1e-8).log()
                         + (1 - mask) * (1 - mask + 1e-8).log()).mean()

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
            "n_hue_features_soft": (mask > 0.5).float().sum().detach(),
        }

    # ----- post-hoc swap-test diagnostic --------------------------------
    @torch.no_grad()
    def per_feature_swap_score(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        target_swap: torch.Tensor,
    ) -> torch.Tensor:
        """For each feature i, R² of decode(swap(z_a, z_b, e_i)) vs target_swap.

        Returns shape (F,). A feature scoring > 0.7 is "identified" as
        causally encoding the hue concept under this abstraction.
        """
        out_a = self.forward(x_a)
        out_b = self.forward(x_b)
        F = out_a.z.shape[-1]
        # Baseline: no swap = decode(z_b). Variance of (target - baseline)
        # is the "naive" residual; if a single feature swap brings the
        # residual down by ≥ 0.7×, that feature carries the hue causally.
        baseline = self.decode(out_b.z)
        var_naive = (target_swap - baseline).pow(2).mean(dim=-1)  # (B,)

        scores = torch.zeros(F, device=x_a.device)
        for i in range(F):
            ei = torch.zeros(F, device=x_a.device)
            ei[i] = 1.0
            z_swap = out_b.z * (1.0 - ei) + out_a.z * ei
            x_hat = self.decode(z_swap)
            resid = (target_swap - x_hat).pow(2).mean(dim=-1)
            # Per-batch R²: 1 - resid / var_naive, then averaged.
            r2 = 1.0 - resid / var_naive.clamp(min=1e-12)
            scores[i] = r2.mean()
        return scores

    @torch.no_grad()
    def normalize_decoder_columns_(self) -> None:
        """Unit-norm decoder columns (standard SAE post-step)."""
        if not self.config.normalize_decoder:
            return
        with torch.no_grad():
            n = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_dec.data.div_(n)


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
