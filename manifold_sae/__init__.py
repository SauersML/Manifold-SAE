"""Manifold-SAE: sparse autoencoders whose features are smooth 1D curves in the residual stream.

The decoder is a vector-output GAM wrapping the Gaussian REML primitive from the ``gamfit``
package (via ``gamfit.torch``, autograd-native). Each feature carries a scalar manifold
coordinate, and reconstruction sums smooth vector-valued functions of those coordinates.
"""

from __future__ import annotations

__version__ = "0.0.1"

from .encoder import ManifoldEncoder
from .sae import (
    ManifoldSAE,
    ManifoldSAEConfig,
    ManifoldSAEOutput,
    extract_feature_curves,
)

__all__ = [
    "ManifoldEncoder",
    "ManifoldSAE",
    "ManifoldSAEConfig",
    "ManifoldSAEOutput",
    "extract_feature_curves",
]
