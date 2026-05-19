"""Manifold-SAE: sparse autoencoders whose features are smooth 1D curves in the residual stream.

The decoder is a vector-output GAM wrapping the Gaussian REML primitive from the ``gamfit``
package. Each feature carries a scalar manifold coordinate, and reconstruction sums smooth
vector-valued functions of those coordinates.

This ``__init__`` re-exports the main public symbols. Each import is guarded by
``try/except ImportError`` so the package still imports cleanly during parallel development
when some sibling modules may not yet exist. Missing symbols simply won't be exposed at the
top level — import them directly from their submodules once available.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__: list[str] = []

try:
    from .gamfit_glue import BasisSpec, ManifoldFit, manifold_fit  # noqa: F401
except ImportError:
    pass
else:
    __all__ += ["BasisSpec", "ManifoldFit", "manifold_fit"]

try:
    from .encoder import ManifoldEncoder  # noqa: F401
except ImportError:
    pass
else:
    __all__ += ["ManifoldEncoder"]

try:
    from .sae import ManifoldSAE, ManifoldSAEConfig  # noqa: F401
except ImportError:
    pass
else:
    __all__ += ["ManifoldSAE", "ManifoldSAEConfig"]
