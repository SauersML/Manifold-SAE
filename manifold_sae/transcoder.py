"""Skip-Transcoder trainer for cogito-style paired-residual data.

Encodes layer L_in residual with a JumpReLU-gated sparse code, decodes layer
L_out residual; adds a low-rank affine bypass A · x_in. Reference:

  Paulo, Shabalin, Belrose. "Transcoders Beat Sparse Autoencoders for
  Interpretability." arXiv:2501.18823, 2025.

Why skip-transcoder over plain SAE
----------------------------------
A plain SAE forces every output bit to be reconstructed by the sparse code,
even bits that are *linearly preserved* between L_in and L_out. The skip
bypass A absorbs the linear-preservation residual; the sparse dictionary
then specializes on the *nonlinear circuit* the deep network adds between
layers. This makes each atom a CIRCUIT PRIMITIVE — feature i at L_in causally
predicts feature j at L_out via the trained decoder + skip Jacobian — which
is the foundation of Anthropic-style attribution graphs.

Architecture
------------
This trainer is a thin wrapper around :class:`gamfit.torch.SkipAffineSmooth`.
The gamfit primitive carries the JumpReLU prior + low-rank bypass; we provide
the optimizer loop, top-k logging, snapshot, and the interp-scoring metrics
the SAE comparison suite expects.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gamfit.torch import SkipAffineSmooth  # gamfit >= 0.1.123 required


@dataclass
class TranscoderConfig:
    in_dim: int
    out_dim: int
    n_atoms: int = 512
    rank_skip: int = 64
    jumprelu_threshold: float = 0.03
    smoothing_eps: float = 1e-3
    learnable_threshold: bool = True
    lambda_sparse: float = 1e-3
    lambda_skip_l2: float = 1e-4
    lr: float = 3e-4
    epochs: int = 15
    batch_size: int = 512
    device: str = "mps"
    seed: int = 0


@dataclass
class TranscoderTrainOutput:
    smooth: SkipAffineSmooth
    history: list[dict]
    final_mse: float
    final_sparsity: float
    final_explained_variance: float


class TranscoderTrainer:
    """Adam trainer for the skip-transcoder smooth."""

    def __init__(self, cfg: TranscoderConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        # Apple-silicon MPS path; falls back to CPU if MPS unavailable.
        if cfg.device == "mps" and not torch.backends.mps.is_available():
            cfg.device = "cpu"
        self.device = torch.device(cfg.device)
        self.smooth = SkipAffineSmooth(
            in_dim=cfg.in_dim,
            out_dim=cfg.out_dim,
            n_atoms=cfg.n_atoms,
            rank_skip=cfg.rank_skip,
            jumprelu_threshold=cfg.jumprelu_threshold,
            learnable_threshold=cfg.learnable_threshold,
            smoothing_eps=cfg.smoothing_eps,
            device=self.device,
            dtype=torch.float32,
        )

    def _loss(self, x_in: torch.Tensor, y_out: torch.Tensor) -> dict[str, torch.Tensor]:
        y_hat, z = self.smooth(x_in)
        mse = (y_hat - y_out).pow(2).mean()
        # Sparsity: L0-surrogate via JumpReLU penalty value (analytic).
        sparsity_penalty = self.smooth.jumprelu(z)
        skip_l2 = torch.tensor(0.0, device=x_in.device)
        if self.smooth.skip_U is not None and self.smooth.skip_V is not None:
            skip_l2 = self.smooth.skip_U.pow(2).mean() + self.smooth.skip_V.pow(2).mean()
        loss = (
            mse
            + self.cfg.lambda_sparse * sparsity_penalty
            + self.cfg.lambda_skip_l2 * skip_l2
        )
        return {
            "loss": loss,
            "mse": mse.detach(),
            "sparsity_penalty": sparsity_penalty.detach(),
            "skip_l2": skip_l2.detach(),
            "frac_alive": (z.abs() > 1e-8).float().mean().detach(),
        }

    def fit(self, X_in: torch.Tensor, Y_out: torch.Tensor) -> TranscoderTrainOutput:
        cfg = self.cfg
        N = X_in.shape[0]
        assert Y_out.shape[0] == N, "paired residuals must have matching N"
        X_in = X_in.to(self.device, dtype=torch.float32)
        Y_out = Y_out.to(self.device, dtype=torch.float32)
        opt = torch.optim.Adam(self.smooth.parameters(), lr=cfg.lr)

        history: list[dict] = []
        for epoch in range(cfg.epochs):
            perm = torch.randperm(N, device=self.device)
            running = {"loss": 0.0, "mse": 0.0, "sparsity": 0.0, "frac_alive": 0.0}
            n_batches = 0
            for start in range(0, N, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                xb = X_in[idx]
                yb = Y_out[idx]
                stats = self._loss(xb, yb)
                opt.zero_grad()
                stats["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.smooth.parameters(), max_norm=5.0)
                opt.step()
                running["loss"] += float(stats["loss"].detach().item())
                running["mse"] += float(stats["mse"].item())
                running["sparsity"] += float(stats["frac_alive"].item())
                running["frac_alive"] += float(stats["frac_alive"].item())
                n_batches += 1
            for k in running:
                running[k] /= max(1, n_batches)
            running["epoch"] = int(epoch)
            history.append(running)
            print(
                f"[transcoder epoch {epoch:2d}] "
                f"loss={running['loss']:.5f} mse={running['mse']:.5f} "
                f"frac_alive={running['frac_alive']:.4f}"
            )

        # Final eval pass.
        with torch.no_grad():
            y_hat, z = self.smooth(X_in)
            mse = float((y_hat - Y_out).pow(2).mean().item())
            sparsity = float((z.abs() > 1e-8).float().mean().item())
            var_y = float(Y_out.var().item())
            ev = 1.0 - mse / max(var_y, 1e-12)
        return TranscoderTrainOutput(
            smooth=self.smooth,
            history=history,
            final_mse=mse,
            final_sparsity=sparsity,
            final_explained_variance=ev,
        )


# ---------------------------------------------------------------------------
# Interpretability score (matched to L1 SAE / Manifold SAE evaluators)
# ---------------------------------------------------------------------------


def interp_score_hsv_coherence(
    smooth: SkipAffineSmooth,
    X_in: torch.Tensor,
    hsv: torch.Tensor,
    top_k: int = 20,
) -> dict[str, float]:
    """For each atom, top-20 firing rows; report (a) HSV coherence —
    1/(1 + circular_std(H)) — and (b) xkcd compactness — mean intra-cluster
    cosine within top-20. Aggregated across atoms by mean.

    Higher is better. The Manifold-SAE and L1 SAE eval suites use the same
    operational definition so the numbers compose head-to-head.
    """
    with torch.no_grad():
        _, z = smooth(X_in.to(next(smooth.parameters()).device, dtype=torch.float32))
        z = z.cpu()
    F = z.shape[1]
    H = hsv[:, 0].cpu()                                       # hue in [0, 1]
    hsv_cpu = hsv.cpu()
    per_atom_hue = []
    per_atom_compact = []
    for k in range(F):
        zk = z[:, k]
        if (zk.abs() > 1e-8).sum() < top_k:
            continue
        top_idx = zk.topk(top_k).indices
        hues = H[top_idx]
        angles = 2 * torch.pi * hues
        sin_m = float(torch.sin(angles).mean().item())
        cos_m = float(torch.cos(angles).mean().item())
        R = (sin_m * sin_m + cos_m * cos_m) ** 0.5
        circ_std = (-2.0 * torch.log(torch.tensor(max(R, 1e-8)))) ** 0.5
        per_atom_hue.append(1.0 / (1.0 + float(circ_std.item())))
        sub = hsv_cpu[top_idx]
        sub = sub / sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
        mean_dir = sub.mean(dim=0, keepdim=True)
        mean_dir = mean_dir / mean_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
        per_atom_compact.append(float((sub @ mean_dir.t()).mean().item()))
    return {
        "hue_coherence_mean": float(sum(per_atom_hue) / max(1, len(per_atom_hue))),
        "xkcd_compactness_mean": float(sum(per_atom_compact) / max(1, len(per_atom_compact))),
        "n_atoms_scored": len(per_atom_hue),
        "combined_interp_score": float(
            0.5 * sum(per_atom_hue) / max(1, len(per_atom_hue))
            + 0.5 * sum(per_atom_compact) / max(1, len(per_atom_compact))
        ),
    }


__all__ = [
    "TranscoderConfig",
    "TranscoderTrainer",
    "TranscoderTrainOutput",
    "interp_score_hsv_coherence",
]
