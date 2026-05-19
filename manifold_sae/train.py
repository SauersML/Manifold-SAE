"""Training loop for ManifoldSAE."""

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
    """Single Adam; lr_groups allows different lr for position-head vs the rest."""
    lr_groups = lr_groups or {}
    if not lr_groups:
        return torch.optim.Adam(sae.parameters(), lr=lr)

    pos_head_lr = lr_groups.get("positions", lr)
    other_lr = lr_groups.get("default", lr)

    pos_params = []
    other_params = []
    for name, p in sae.named_parameters():
        if name in ("encoder.fc2.weight", "encoder.fc2.bias"):
            # Split fc2 into position rows vs amplitude rows; achieved by adding the
            # full tensor to the appropriate group based on whether 'positions' key set.
            pos_params.append(p)
        else:
            other_params.append(p)
    groups = [
        {"params": other_params, "lr": other_lr},
        {"params": pos_params, "lr": pos_head_lr},
    ]
    return torch.optim.Adam(groups)


def train(
    sae: ManifoldSAE,
    data_loader: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    log_every: int = 50,
) -> dict[str, list[Any]]:
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

        # Move batch to whatever device the SAE lives on.
        device = next(sae.parameters()).device
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = sae(batch)
        losses = total_loss(out, batch, sae.config)
        loss = losses["total"]

        if step % log_every == 0:
            ratio_info = position_amplitude_grad_ratio(loss, out.positions, out.amplitudes)
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
        step += 1

    return history
