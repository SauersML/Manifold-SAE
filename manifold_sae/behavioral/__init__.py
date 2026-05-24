"""Behavioral-interp probes on Manifold-SAE atom activations.

References:
  - Arditi et al. 2024 (Refusal in LLMs is mediated by a single direction)
  - Sharma et al. 2023 (Towards understanding sycophancy in LMs)
  - Zou et al. 2023 (Representation engineering)

Public API:
  - BehavioralProbe: linear (logistic) probe over SAE-atom activations.
  - top_atoms_for: ranked list of behavior-encoding atoms.
  - causal_steer_eval: push top atoms by +alpha and measure Delta-P(behavior).
"""
from .probes import BehavioralProbe, top_atoms_for, cross_correlation
from .causal_steer import causal_steer_eval, steer_activations

__all__ = [
    "BehavioralProbe",
    "top_atoms_for",
    "cross_correlation",
    "causal_steer_eval",
    "steer_activations",
]
