"""Cross-LLM Universal-SAE infrastructure.

Local-only harvest + shared-encoder / per-model-decoder SAE for cross-model
universality experiments (Universal-SAE, arXiv:2502.03714; Atlas-Alignment,
arXiv:2510.27413).

Cluster access is BANNED — all harvests run locally on MPS / CPU via the
``transformers`` library.
"""

from .universal_sae import UniversalSAE, UniversalSAEConfig  # noqa: F401
from .harvest_local import (  # noqa: F401
    harvest,
    build_color_prompts,
    load_xkcd_colors,
    TEMPLATES,
)
