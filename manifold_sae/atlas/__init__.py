"""SAE atom-set post-hoc geometric atlas (PHATE + persistent H1 + Mapper).

Inspired by NN manifold diffusion-geometry (arXiv:2411.12626): rather than
analysing input activations, we analyse the GEOMETRY OF THE LEARNED ATOM SET
itself. A Manifold-SAE trained on a 1-D color hue ring should produce atoms
that ALSO sit on a 1-manifold; PHATE + Vietoris-Rips H1 reveals this.
"""

from .phate_atlas import (
    atom_atlas,
    persistent_h1,
    mapper_atlas,
    extract_atom_directions,
)

__all__ = [
    "atom_atlas",
    "persistent_h1",
    "mapper_atlas",
    "extract_atom_directions",
]
