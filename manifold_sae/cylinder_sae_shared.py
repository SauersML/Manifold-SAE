"""CylinderSAESharedEnc — alias for ``CylinderSAE`` (kept for back-compat).

Post gamfit 0.1.123 migration, the canonical ``CylinderSAE`` is itself
shared-encoder (single Linear → F·4 heads, ~F× memory reduction over the
old per-feature MLP). This module preserves the old import path.
"""
from __future__ import annotations

from .cylinder_sae import CylinderSAE as CylinderSAESharedEnc
from .cylinder_sae import CylinderSAEConfig as CylinderSAESharedEncConfig

__all__ = ["CylinderSAESharedEnc", "CylinderSAESharedEncConfig"]
