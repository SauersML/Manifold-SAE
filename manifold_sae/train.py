"""Training loop for the gamfit-native ManifoldSAE.

Cutover semantics
-----------------
The gamfit-native :class:`gamfit.torch.ManifoldSAE` trains its encoder, anchor,
and ``log_lambda`` by backprop on the reconstruction + regularizers, and uses
the closed-form Rust REML solve (:meth:`ManifoldSAE.fit`) to refresh the decoder
blocks / smoothing λ at a configurable cadence. Deploy is via
:meth:`ManifoldSAE.lock_snapshot` (replaces the old ``update_snapshot`` /
``inference_mode`` pair).

The training step:
  1. forward(x)            -> ManifoldSAEOutput (encoder + current decoder)
  2. total_loss(out, x, sae)  -> backprop through encoder/anchor/log_lambda
  3. optimizer.step()
  4. every ``fit_every`` steps (and once at the end): sae.fit(batch) to refresh
     the closed-form decoder blocks + REML λ.

Call :meth:`ManifoldSAE.lock_snapshot` after training to freeze for inference.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable

import torch

from .diagnostics import (
    dead_feature_mask,
    position_amplitude_grad_ratio,
    position_variance,
)
from .losses import total_loss
from .sae import ManifoldSAE


def build_optimizer(
    sae: ManifoldSAE,
    lr: float = 1e-3,
    lr_groups: dict[str, float] | None = None,
) -> torch.optim.Optimizer:
    """Single Adam over the backprop-trained parameters.

    ``lr_groups`` may supply a distinct lr for the encoder vs the rest
    (``{"encoder": ..., "default": ...}``); the gamfit encoder is a single
    ``nn.Linear`` (or a small Sequential), so we group by parameter-name prefix.
    """
    lr_groups = lr_groups or {}
    if not lr_groups:
        return torch.optim.Adam(sae.parameters(), lr=lr)

    enc_lr = lr_groups.get("encoder", lr_groups.get("positions", lr))
    other_lr = lr_groups.get("default", lr)

    enc_params, other_params = [], []
    for name, p in sae.named_parameters():
        if name.startswith("encoder."):
            enc_params.append(p)
        else:
            other_params.append(p)
    groups = [
        {"params": other_params, "lr": other_lr},
        {"params": enc_params, "lr": enc_lr},
    ]
    return torch.optim.Adam(groups)


def train(
    sae: ManifoldSAE,
    data_loader: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    log_every: int = 50,
    fit_every: int = 0,
) -> dict[str, list[Any]]:
    """Backprop loop with optional periodic closed-form REML refresh.

    ``fit_every <= 0`` disables the periodic ``sae.fit`` refresh (decoder blocks
    are then trained purely by backprop). Set ``fit_every > 0`` to interleave the
    closed-form solve every ``fit_every`` steps.
    """
    history: dict[str, list[Any]] = {
        "step": [],
        "mse": [],
        "sparsity": [],
        "position_variance_mean": [],
        "dead_feature_count": [],
        "grad_ratio_mean": [],
    }

    it = iter(data_loader)
    step = 0
    while step < n_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(data_loader)
            batch = next(it)

        device = next(sae.parameters()).device
        batch = batch.to(device=device, dtype=sae.cfg.dtype)
        optimizer.zero_grad(set_to_none=True)
        out = sae(batch)
        losses = total_loss(out, batch, sae)
        loss = losses["total"]

        if step % log_every == 0:
            ratio_info = position_amplitude_grad_ratio(
                loss, out.positions, out.amplitudes
            )
            pos_var = position_variance(out)
            dead = dead_feature_mask(out)
            print(
                f"[step {step:6d}] "
                f"mse={losses['mse'].item():.4e} "
                f"sparsity={losses['sparsity'].item():.4e} "
                f"pos_var={pos_var.mean().item():.4e} "
                f"dead={int(dead.sum().item())} "
                f"grad_ratio={ratio_info['ratio'].mean().item():.4e}"
            )
            history["step"].append(step)
            history["mse"].append(losses["mse"].item())
            history["sparsity"].append(losses["sparsity"].item())
            history["position_variance_mean"].append(pos_var.mean().item())
            history["dead_feature_count"].append(int(dead.sum().item()))
            history["grad_ratio_mean"].append(ratio_info["ratio"].mean().item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        optimizer.step()

        if fit_every > 0 and step > 0 and step % fit_every == 0 and not sae.is_locked:
            with torch.no_grad():
                sae.fit(batch)

        step += 1

    return history
