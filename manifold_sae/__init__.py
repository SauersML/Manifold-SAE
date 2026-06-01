"""Manifold-SAE: sparse autoencoders whose features are smooth 1D curves in the residual stream.

The decoder is a vector-output GAM wrapping the Gaussian REML primitive from the ``gamfit``
package (via ``gamfit.torch``, autograd-native). Each feature carries a scalar manifold
coordinate, and reconstruction sums smooth vector-valued functions of those coordinates.
"""

from __future__ import annotations

__version__ = "0.0.1"

# Lazy submodule imports — sae.py is a thin wrapper over gamfit's
# ``gamfit.torch.ManifoldSAE`` (requires gamfit >= 0.1.134, which ships the
# curve-atom SAE / AdaptiveTopK / InterchangeSwapDecoder / PoincareAtoms
# primitives). We resolve sae/encoder lazily so a `_cluster_bridge` import
# (used by every LLM experiment driver) doesn't fail because of an unrelated
# module's heavy dependency.

# Re-exported straight from gamfit via .sae, plus Manifold-SAE-side helpers.
_SAE_EXPORTS = {
    "ManifoldSAE",
    "ManifoldSAEConfig",
    "ManifoldSAEOutput",
    "DecoderConfig",
    "RemlConfig",
    "SparsityConfig",
    "extract_feature_curves",
    "lift_atom_curve",
    "load_sae",
}

__all__ = ["ManifoldEncoder", *sorted(_SAE_EXPORTS)]


def __getattr__(name):
    if name == "ManifoldEncoder":
        from .encoder import ManifoldEncoder
        return ManifoldEncoder
    if name in _SAE_EXPORTS:
        from . import sae as _sae
        return getattr(_sae, name)
    raise AttributeError(f"module 'manifold_sae' has no attribute {name!r}")
