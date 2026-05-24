"""Causal steering of SAE-atom activations to verify behavioral probes.

Anthropic-style representation engineering protocol applied to SAE atoms:

  1. Train a behavioral probe (refusal / sycophancy / hedging) on SAE
     atom activations a in R^F.
  2. Identify top-k atoms by |probe weight|.
  3. For held-out activations, push those atoms by +alpha (a -> a + alpha * e_k
     scaled by the sign of the probe weight, so the perturbation always moves
     atoms IN the direction that the probe says encodes the behavior).
  4. Measure Delta-P(behavior) on the SAME probe. A large positive Delta
     supports a causal interpretation (intervening on the atoms moves the
     behavior probability), distinguishing it from a spurious / correlational
     probe.

This is the SAE-atom analog of Arditi 2024's residual-stream ablation: the
key difference is that we steer the *sparse, interpretable* atom layer
rather than the ambient direction. If the probe is genuinely picking up the
behavior-relevant subspace, steering should move the behavior. If the probe
is over-fit noise, steering should not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .probes import BehavioralProbe


@dataclass
class SteerResult:
    target: str
    alphas: list[float]
    top_atoms: list[int]                  # atom indices used for the steer
    baseline_p_mean: float                # mean P(behavior) on held-out, alpha=0
    steered_p_mean: dict[float, float]    # alpha -> mean P(behavior)
    delta_p: dict[float, float]           # alpha -> mean P(behavior) - baseline
    flip_rate: dict[float, float]         # fraction of examples whose binary decision flipped


def steer_activations(
    atoms: np.ndarray | torch.Tensor,
    *,
    atom_indices: Iterable[int],
    signs: Iterable[float] | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Return atoms with the given indices pushed by +alpha * sign.

    Parameters
    ----------
    atoms : (B, F)
    atom_indices : iterable[int]
        Which atoms to perturb.
    signs : iterable[float] or None
        Sign per atom — pass the sign of the probe weight so the perturbation
        moves the atom *in the probe-positive direction*. Default: all +1.
    alpha : float
        Step size in raw-activation units. Comparable to TopK SAE activations,
        which are typically O(1) for firing atoms.
    """
    out = torch.as_tensor(np.asarray(atoms), dtype=torch.float32).clone()
    idx = list(int(i) for i in atom_indices)
    if signs is None:
        signs = [1.0] * len(idx)
    signs = list(float(s) for s in signs)
    for i, s in zip(idx, signs):
        # Move atom in the +sign direction; sign(s) so weight magnitudes
        # don't double-count (the alpha is the user's step size).
        out[:, i] = out[:, i] + alpha * (1.0 if s >= 0 else -1.0)
    return out


def causal_steer_eval(
    probe: BehavioralProbe,
    atoms_holdout: np.ndarray | torch.Tensor,
    *,
    top_k: int = 10,
    alphas: Iterable[float] = (1.0, 2.0, 5.0),
    device: str | torch.device = "cpu",
) -> SteerResult:
    """Steer top-k atoms on held-out activations and measure probe response."""
    probe = probe.to(device).eval()
    A = torch.as_tensor(np.asarray(atoms_holdout), dtype=torch.float32, device=device)
    top = probe.top_k_atoms(k=top_k)
    atom_idx = [i for i, _ in top]
    signs = [w for _, w in top]

    with torch.no_grad():
        base_p = probe(A).cpu().numpy()
        base_decision = (base_p > 0.5).astype(np.int32)

    steered_means: dict[float, float] = {}
    deltas: dict[float, float] = {}
    flips: dict[float, float] = {}
    for alpha in alphas:
        A_steer = steer_activations(A.cpu(), atom_indices=atom_idx, signs=signs, alpha=alpha).to(device)
        with torch.no_grad():
            p = probe(A_steer).cpu().numpy()
        steered_means[float(alpha)] = float(np.mean(p))
        deltas[float(alpha)] = float(np.mean(p) - np.mean(base_p))
        flips[float(alpha)] = float(np.mean((p > 0.5).astype(np.int32) != base_decision))

    return SteerResult(
        target=probe.target,
        alphas=[float(a) for a in alphas],
        top_atoms=atom_idx,
        baseline_p_mean=float(np.mean(base_p)),
        steered_p_mean=steered_means,
        delta_p=deltas,
        flip_rate=flips,
    )
