"""concept_manifold_steering — transferable HSV-style gauge-fix + anchor-offset
steering recipe for arbitrary LLM activation manifolds.

This package generalises the validated cogito-L40 color-manifold pipeline
(Manifold-SAE auto_exp_38, _44, _49, _52, _53, _54) into a model-agnostic
toolkit:

  1. Harvest hidden-state activations at any layer of any HuggingFace
     decoder-only LM with a list of prompts (see :mod:`harvest`).
  2. Fit a *gauge-fixed* linear subspace to those activations whose
     coordinates align with a user-supplied target table (e.g.
     ``{"hue": [...], "saturation": [...]}``).  See :class:`GaugeFix`.
  3. Steer the live model along *anchor offset* directions (concept_to -
     concept_from) rather than tangent vectors of the manifold
     (auto_exp_44 falsified the tangent hypothesis).  See
     :class:`ManifoldSteerer`.
  4. Diagnose locality / curvature / null-baselines BEFORE you ship a
     steerer to production users.  See :mod:`diagnostics`.

Public API
----------
>>> from concept_manifold_steering import (
...     GaugeFix, ManifoldSteerer, harvest_activations, plot_diagnostics,
... )

Typical 10-line use::

    from concept_manifold_steering import harvest_activations, GaugeFix, ManifoldSteerer

    X = harvest_activations("cogito-v1", prompts, layer=40)
    gauge = GaugeFix(targets=["hue", "saturation", "value"]).fit(X, labels)
    steerer = ManifoldSteerer(gauge, server_url="http://localhost:8000")
    steerer.steer("The fire engine was painted", concept="red", alpha=2.0)
"""

from __future__ import annotations

from .gauge import GaugeFix
from .steer import ManifoldSteerer
from .harvest import harvest_activations
from .diagnostics import (
    per_anchor_curvature,
    null_topology_control,
    variance_vs_locality,
    plot_diagnostics,
)

__all__ = [
    "GaugeFix",
    "ManifoldSteerer",
    "harvest_activations",
    "per_anchor_curvature",
    "null_topology_control",
    "variance_vs_locality",
    "plot_diagnostics",
]

__version__ = "0.1.0"
