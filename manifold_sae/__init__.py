"""Manifold-SAE: sparse autoencoders whose features are smooth 1D curves in the residual stream.

The decoder is a vector-output GAM wrapping the Gaussian REML primitive from the ``gamfit``
package. Each feature carries a scalar manifold coordinate, and reconstruction sums smooth
vector-valued functions of those coordinates.
"""

from __future__ import annotations

__version__ = "0.0.1"

from .encoder import ManifoldEncoder
from .gamfit_glue import BasisSpec, ManifoldFit, manifold_fit
from .sae import ManifoldSAE, ManifoldSAEConfig, ManifoldSAEOutput

__all__ = [
    "BasisSpec",
    "ManifoldEncoder",
    "ManifoldFit",
    "ManifoldSAE",
    "ManifoldSAEConfig",
    "ManifoldSAEOutput",
    "manifold_fit",
]
