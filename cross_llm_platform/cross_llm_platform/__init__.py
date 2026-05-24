"""Cross-LLM concept interpretability platform.

Five-line path:

``harvest_activations`` -> ``label_prompts`` -> ``fit_gauge`` ->
``register_anchor`` -> ``ConceptSteerer``.
"""

from .concepts import ConceptSpec, REGISTRY, label_prompts
from .diagnostics import (
    null_topology_control,
    per_anchor_curvature,
    validated_diagnostics,
    variance_vs_concept_locality,
)
from .gauge import GaugeFit, fit_gauge
from .ingest import HarvestResult, harvest_activations, load_prompts
from .steer import ConceptSteerer, SteeringRequest, SteeringResult

__all__ = [
    "ConceptSpec",
    "REGISTRY",
    "label_prompts",
    "null_topology_control",
    "per_anchor_curvature",
    "validated_diagnostics",
    "variance_vs_concept_locality",
    "GaugeFit",
    "fit_gauge",
    "HarvestResult",
    "harvest_activations",
    "load_prompts",
    "ConceptSteerer",
    "SteeringRequest",
    "SteeringResult",
]
