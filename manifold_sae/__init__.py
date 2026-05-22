"""Manifold-SAE: sparse autoencoders whose features are smooth 1D curves in the residual stream.

The decoder is a vector-output GAM wrapping the Gaussian REML primitive from the ``gamfit``
package (via ``gamfit.torch``, autograd-native). Each feature carries a scalar manifold
coordinate, and reconstruction sums smooth vector-valued functions of those coordinates.
"""

from __future__ import annotations

__version__ = "0.0.1"

# Lazy submodule imports — sae.py requires the new gamfit (≥0.1.99) Duchon
# class, which is not present in the cluster's currently-pinned gamfit
# 0.1.98. We don't want a `_cluster_bridge` import (used by every LLM
# experiment driver) to fail because of an unrelated module's broken
# dependency. Resolve sae/encoder lazily on first attribute access.

__all__ = [
    "ManifoldEncoder",
    "ManifoldSAE",
    "ManifoldSAEConfig",
    "ManifoldSAEOutput",
    "extract_feature_curves",
]


def __getattr__(name):
    if name == "ManifoldEncoder":
        from .encoder import ManifoldEncoder
        return ManifoldEncoder
    if name in {"ManifoldSAE", "ManifoldSAEConfig", "ManifoldSAEOutput",
                "extract_feature_curves"}:
        from . import sae as _sae
        return getattr(_sae, name)
    raise AttributeError(f"module 'manifold_sae' has no attribute {name!r}")
