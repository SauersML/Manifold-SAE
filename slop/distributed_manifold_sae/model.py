"""ManifoldSAE: K-atom decoder, each atom carries a d_atom-dim Circle latent.

Forward (training):
    x : (B, D)        — residual-stream activations (e.g. cogito-L40, D=7168)
    encoder(x) → logits : (B, K)
    mask  : (B, K)    — IBP-Gumbel sample, per row ~top_k active
    amp   : (B, K)    — non-negative amplitudes (softplus of logits)
    For each active (b, k):
        θ_{b,k}  ∈ Circle = S^1 ⊂ R^2                — learned via amortized head
        tangent direction d_k(θ) = anchor_k + T_k · θ_{b,k}
            anchor_k ∈ R^D   (atom anchor in residual stream)
            T_k      ∈ R^{D × d_atom}   (per-atom tangent frame)
    recon = b_dec + Σ_k amp_{b,k} · mask_{b,k} · d_k(θ_{b,k})

Design notes
------------
* "Circle topology": each atom's latent lives on S^1 (so d_atom=2 with
  unit-norm constraint), preserved by Riemannian projection in the
  ManifoldOptimizer wrapper (see train.py).
* Anchor + tangent factorisation comes from auto_exp_44/47: separating
  the bulk direction (anchor) from the manifold variation (T·θ) gives
  cleaner ARD pruning and matches the gauge-fix-companion result from
  the cogito recovery experiment (auto_exp_38).
* For K=1M, parameters are dominated by anchors (K·D ≈ 7.2B floats) and
  tangent frames (K·D·d_atom ≈ 14.3B). FSDP sharding across N=4 GPUs is
  mandatory; see train.py's FSDP wrap policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ManifoldSAEConfig:
    input_dim: int = 7168            # cogito-L40 hidden size
    n_atoms: int = 1_000_000         # K
    d_atom: int = 2                  # Circle latent dimension
    top_k: int = 32                  # per-row active atoms (IBP-Gumbel target)
    encoder_hidden: int = 4096
    gumbel_tau_init: float = 1.0
    gumbel_tau_final: float = 0.1
    use_grad_checkpoint: bool = True
    # Manifold geometry
    manifold: str = "circle"         # "circle" | "sphere" | "euclidean"
    # Initialization
    anchor_init_scale: float = 0.02
    tangent_init_scale: float = 0.01


# ---------------------------------------------------------------------------
# Encoder: amortized inference q(z | x). Outputs (logits, theta_raw).
# ---------------------------------------------------------------------------
class AmortizedEncoder(nn.Module):
    """Two-layer MLP producing per-atom logits and per-atom Circle coordinates.

    For K=1M, the projection to (K * (1 + d_atom)) is ~3M-wide. This is
    sharded by FSDP at the column level in the train script. Here we keep
    it as a single Linear; FSDP handles the partition.
    """

    def __init__(self, cfg: ManifoldSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.input_dim, cfg.encoder_hidden)
        self.act = nn.GELU()
        # logits: (K,) ; theta_raw: (K, d_atom)  — packed into one matmul
        out_dim = cfg.n_atoms * (1 + cfg.d_atom)
        self.out_proj = nn.Linear(cfg.encoder_hidden, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (B, D)
        h = self.act(self.in_proj(x))                    # (B, H)
        z = self.out_proj(h)                             # (B, K*(1+d))
        B = x.shape[0]
        K, d = self.cfg.n_atoms, self.cfg.d_atom
        z = z.view(B, K, 1 + d)
        logits = z[..., 0]                               # (B, K)
        theta_raw = z[..., 1:]                           # (B, K, d)
        return logits, theta_raw


# ---------------------------------------------------------------------------
# IBP-Gumbel sparse assignment.
# ---------------------------------------------------------------------------
def ibp_gumbel_mask(
    logits: torch.Tensor,
    tau: float,
    top_k: int,
    hard: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row top-k Gumbel-softmax via relaxed Bernoulli + sort-truncation.

    Returns (mask_soft, mask_hard). During training we typically use the
    straight-through estimator: forward uses mask_hard, backward uses mask_soft.
    """
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    g = -torch.log(-torch.log(u))
    relaxed = torch.sigmoid((logits + g) / tau)          # (B, K) in (0,1)
    # Per-row top-k hard mask
    topv, topi = relaxed.topk(top_k, dim=-1)
    hard_mask = torch.zeros_like(relaxed)
    hard_mask.scatter_(-1, topi, 1.0)
    if hard:
        # straight-through: forward uses hard 0/1 mask, backward flows through soft.
        mask = (hard_mask - relaxed).detach() + relaxed
        return mask, relaxed
    return relaxed, relaxed


# ---------------------------------------------------------------------------
# Manifold projection: Circle (S^1) retraction.
# ---------------------------------------------------------------------------
def circle_project(theta_raw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project R^2 vectors onto unit Circle by L2 normalization."""
    n = theta_raw.norm(dim=-1, keepdim=True).clamp_min(eps)
    return theta_raw / n


# ---------------------------------------------------------------------------
# Main module.
# ---------------------------------------------------------------------------
class ManifoldSAE(nn.Module):
    """Manifold-SAE with K-atom decoder, anchor + tangent factorisation.

    The decoder parameters (anchors, tangents) are stored as plain Parameters;
    FSDP at the module level shards them across ranks. Forward uses
    gather-on-demand via the active-atom indices to avoid materializing
    the full (K, D) matrices per step.
    """

    def __init__(self, cfg: ManifoldSAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = AmortizedEncoder(cfg)

        # Decoder parameters: anchor and tangent frame per atom.
        # Stored as (K, D) and (K, D, d_atom). For K=1M this is large
        # but FSDP handles sharding.
        self.anchor = nn.Parameter(
            torch.randn(cfg.n_atoms, cfg.input_dim) * cfg.anchor_init_scale
        )
        self.tangent = nn.Parameter(
            torch.randn(cfg.n_atoms, cfg.input_dim, cfg.d_atom) * cfg.tangent_init_scale
        )
        self.b_dec = nn.Parameter(torch.zeros(cfg.input_dim))

        # Gumbel temperature (buffer, set by training loop schedule).
        self.register_buffer("tau", torch.tensor(cfg.gumbel_tau_init))

    # ------------------------------------------------------------------
    def _decode_active(
        self,
        mask: torch.Tensor,
        amp: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Decode only the active atoms per row.

        mask  : (B, K)  straight-through binary
        amp   : (B, K)  non-negative
        theta : (B, K, d_atom) on the manifold

        Returns recon_no_bias : (B, D)

        Implementation: we gather the per-row top-k atoms via topk on mask,
        then index anchor[K, D] and tangent[K, D, d] for just those indices.
        This costs O(B * top_k * D) per step regardless of K.
        """
        B, K = mask.shape
        D, d = self.cfg.input_dim, self.cfg.d_atom
        k = self.cfg.top_k

        # Top-k indices per row.
        topv, topi = mask.topk(k, dim=-1)                # (B, k)
        # Effective weight: amp * mask (straight-through preserves grad to amp).
        w = (amp * mask).gather(-1, topi)                # (B, k)

        # Gather active anchors and tangents.
        # anchor: (K, D) -> (B, k, D)
        anc = self.anchor[topi]                          # (B, k, D)
        # tangent: (K, D, d) -> (B, k, D, d)
        tan = self.tangent[topi]                         # (B, k, D, d)
        th = theta.gather(1, topi.unsqueeze(-1).expand(-1, -1, d))  # (B, k, d)

        # Per-atom direction d_k(θ) = anchor_k + tangent_k @ θ
        # (B, k, D, d) @ (B, k, d, 1) -> (B, k, D, 1)
        manifold_part = torch.einsum("bkdj,bkj->bkd", tan, th)
        direction = anc + manifold_part                  # (B, k, D)

        # Weighted sum.
        recon = (w.unsqueeze(-1) * direction).sum(dim=1)  # (B, D)
        return recon

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        cfg = self.cfg

        if cfg.use_grad_checkpoint and self.training:
            logits, theta_raw = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            logits, theta_raw = self.encoder(x)

        # Amplitudes: softplus(logits) gives non-negative magnitude.
        amp = F.softplus(logits)
        # Mask: IBP-Gumbel per-row top-k.
        mask_hard, mask_soft = ibp_gumbel_mask(
            logits, tau=float(self.tau), top_k=cfg.top_k, hard=True
        )

        # Manifold projection of theta.
        if cfg.manifold == "circle":
            theta = circle_project(theta_raw)
        elif cfg.manifold == "euclidean":
            theta = theta_raw
        elif cfg.manifold == "sphere":
            theta = circle_project(theta_raw)            # same retraction for unit-sphere
        else:
            raise ValueError(f"Unknown manifold: {cfg.manifold}")

        recon_no_bias = self._decode_active(mask_hard, amp, theta)
        recon = recon_no_bias + self.b_dec

        return {
            "recon": recon,
            "logits": logits,
            "amp": amp,
            "mask_soft": mask_soft,
            "mask_hard": mask_hard,
            "theta": theta,
            "theta_raw": theta_raw,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def riemannian_retract(self) -> None:
        """Project the underlying parameters back to the manifold.

        Anchor + tangent live in R^D, not on a manifold themselves — the
        Circle constraint applies to the per-sample theta, which we already
        project in forward. This hook exists so the optimizer step can call
        it to keep auxiliary geometric invariants (e.g. tangent orthogonality
        to anchor, or unit-norm tangent columns).

        Default implementation: per-atom orthonormalize the d_atom tangent
        columns via QR. This enforces ‖T_k v‖ = ‖v‖ for v on S^1, so the
        manifold-induced metric matches the unit Circle.
        """
        if self.cfg.manifold not in ("circle", "sphere"):
            return
        K, D, d = self.cfg.n_atoms, self.cfg.input_dim, self.cfg.d_atom
        # Batched QR on (K, D, d) - small d so this is fine.
        # Memory: this allocates a (K, D, d) copy. For K=1M, D=7168, d=2,
        # that's ~57 GB fp32. In practice we shard over atoms and call this
        # on each rank's local shard. See train.py.
        q, _ = torch.linalg.qr(self.tangent.data, mode="reduced")
        self.tangent.data.copy_(q)
