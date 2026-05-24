"""Atom interpretability evaluation. STUB — basic R² + activeness only.

A complete eval would include:
  * per-atom max-activating example mining
  * cogito-color manifold projection (HSV/name-token axis correlations)
  * tangent-frame alignment with semantic axes
  * dead-atom rate, atom-degeneracy clustering
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int = 50) -> dict:
    model.eval()
    ss_res = 0.0
    ss_tot = 0.0
    activeness = torch.zeros(model.cfg.n_atoms, device=device)
    n_rows = 0

    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        out = model(x)
        recon = out["recon"]
        ss_res += float(((x - recon) ** 2).sum())
        ss_tot += float(((x - x.mean(dim=0, keepdim=True)) ** 2).sum())
        activeness += out["mask_hard"].sum(dim=0)
        n_rows += x.shape[0]

    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    rate = activeness / max(n_rows, 1)
    dead_frac = float((rate == 0).float().mean())
    return {
        "r2": r2,
        "dead_atom_frac": dead_frac,
        "mean_activation_rate": float(rate.mean()),
    }
