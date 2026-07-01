"""Cellular sheaf cohomology for cross-layer SAE consistency.

Reference: arxiv 2511.11092, "Sheaf cohomology of predictive coding nets".

Idea
----
Treat the layer-l SAE's atom-space as the stalk F(v_l) at a vertex v_l of a
cellular sheaf over a 1-D chain graph (v_1 — v_2 — ... — v_L). For each
adjacent pair (l, l+1), a *restriction map* F_{l ⊳ e} : F(v_l) → F(e) and
F_{l+1 ⊳ e} : F(v_{l+1}) → F(e) glue the two stalks across edge e. Here
both stalks land in the same edge stalk R^F (number of atoms is identical
across layers), and the restriction maps are the cross-layer transcoders.

The (un-weighted) sheaf coboundary δ : C^0(F) → C^1(F) reads, on edge
e = (l, l+1):

    (δs)(e) = F_{l+1 ⊳ e} s_{l+1}  −  F_{l ⊳ e} s_l

The sheaf-Laplacian L = δ^* δ is symmetric PSD. ‖δs‖² = ⟨s, L s⟩ measures
*global inconsistency* of an L-tuple of atom activations s = (s_1, …, s_L).

For a paired SAE stack:
  - vertices = per-layer SAEs (each has F atoms)
  - edges    = adjacent (l, l+1) pairs
  - restriction maps = learned (F × F) transcoders T_{l→l+1}; we take
    F_{l ⊳ e} = T_{l→l+1}, F_{l+1 ⊳ e} = I_F (i.e. compare the predicted
    next-layer code to the actually-encoded one).

Harmonic 0-cochains (ker L) are atoms that propagate exactly through the
transcoder stack — i.e. globally consistent circuit features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Pure-numpy / torch sheaf object
# ---------------------------------------------------------------------------


@dataclass
class CellularSheaf:
    """A cellular sheaf over a 1-D chain of SAEs.

    Parameters
    ----------
    layer_saes : list
        Length-L list of SAE objects. Only their ``F`` (atom count) is read;
        we don't depend on the SAE class internals.
    restriction_maps : dict[(int, int), np.ndarray]
        Map (l, l+1) → (F_target, F_source) transcoder restriction matrix.
        Adjacency-only edges supported; non-adjacent keys raise.
    """

    layer_saes: list
    restriction_maps: dict[tuple[int, int], np.ndarray]

    def __post_init__(self) -> None:
        self.n_layers = len(self.layer_saes)
        def _F_of(s):
            if hasattr(s, "F"):
                return int(s.F)
            if hasattr(s, "n_atoms"):
                return int(s.n_atoms)
            raise AttributeError(f"SAE-like {type(s).__name__} has neither .F nor .n_atoms")

        self.F_per_layer = [_F_of(s) for s in self.layer_saes]
        for (a, b) in self.restriction_maps:
            if b != a + 1:
                raise ValueError(f"Only adjacent edges supported, got ({a},{b})")
            T = self.restriction_maps[(a, b)]
            if T.shape != (self.F_per_layer[b], self.F_per_layer[a]):
                raise ValueError(
                    f"Restriction map ({a},{b}) shape {T.shape} != "
                    f"({self.F_per_layer[b]}, {self.F_per_layer[a]})"
                )

    # ------------------------------------------------------------------
    # Coboundary and Laplacian
    # ------------------------------------------------------------------

    def coboundary(self, s: list[np.ndarray]) -> list[np.ndarray]:
        """(δs)(l, l+1) = s_{l+1} − T_{l→l+1} s_l, one array per edge."""
        out: list[np.ndarray] = []
        for l in range(self.n_layers - 1):
            T = self.restriction_maps[(l, l + 1)]
            out.append(s[l + 1] - T @ s[l])
        return out

    def coboundary_matrix(self) -> np.ndarray:
        """Dense (E·F, sum F_l) coboundary matrix δ when all F_l equal.

        Returned in block form so the sheaf Laplacian L = δ^T δ is
        immediately diagonalizable.
        """
        if len(set(self.F_per_layer)) != 1:
            raise ValueError("coboundary_matrix requires equal F per layer")
        F = self.F_per_layer[0]
        L = self.n_layers
        E = L - 1
        D = np.zeros((E * F, L * F), dtype=np.float64)
        for l in range(E):
            T = self.restriction_maps[(l, l + 1)].astype(np.float64)
            # edge row l acts on s_l and s_{l+1}
            D[l * F : (l + 1) * F, l * F : (l + 1) * F] = -T
            D[l * F : (l + 1) * F, (l + 1) * F : (l + 2) * F] = np.eye(F)
        return D

    def laplacian(self) -> np.ndarray:
        """Sheaf Laplacian L = δ^T δ, shape (L·F, L·F). PSD."""
        D = self.coboundary_matrix()
        return D.T @ D


def sheaf_laplacian_energy(
    s: list[np.ndarray],
    restriction_maps: dict[tuple[int, int], np.ndarray] | None = None,
    sheaf: CellularSheaf | None = None,
) -> float:
    """‖δs‖² for an L-tuple of activation vectors.

    Either pass an explicit ``sheaf`` or supply ``restriction_maps`` directly.
    Shapes: s[l] is (F_l,) or (B, F_l); a batched s yields per-row energy
    summed across the batch.
    """
    if sheaf is None:
        if restriction_maps is None:
            raise ValueError("Pass either sheaf or restriction_maps")
        # build minimal stub
        L = len(s)
        stub = [type("_Stub", (), {"F": int(np.asarray(s[l]).shape[-1])})() for l in range(L)]
        sheaf = CellularSheaf(stub, restriction_maps)

    total = 0.0
    for l in range(sheaf.n_layers - 1):
        T = sheaf.restriction_maps[(l, l + 1)]
        a, b = np.asarray(s[l]), np.asarray(s[l + 1])
        diff = b - a @ T.T if a.ndim == 2 else b - T @ a
        total += float(np.sum(diff * diff))
    return total


# ---------------------------------------------------------------------------
# Differentiable training loss
# ---------------------------------------------------------------------------


class SheafConsistencyHead(nn.Module):
    """Learned transcoder restriction maps + differentiable sheaf energy.

    Owns the (F, F) transcoder for each adjacent pair. Pairs with an
    arbitrary list of SAE encoder modules — we only require each to expose
    ``.encode(x)`` returning a (B, F) code.
    """

    def __init__(self, n_layers: int, F: int) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.F = int(F)
        # Initialize transcoders near identity so initial energy is just
        # cross-layer code disagreement, not random noise.
        self.transcoders = nn.ParameterList(
            [nn.Parameter(torch.eye(F) + 0.01 * torch.randn(F, F)) for _ in range(n_layers - 1)]
        )

    def restriction_dict(self) -> dict[tuple[int, int], np.ndarray]:
        return {
            (l, l + 1): self.transcoders[l].detach().cpu().numpy().copy()
            for l in range(self.n_layers - 1)
        }

    def energy(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Mean-over-batch ‖δz‖² for the supplied list of per-layer codes."""
        if len(codes) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} code tensors, got {len(codes)}")
        total = codes[0].new_zeros(())
        for l in range(self.n_layers - 1):
            T = self.transcoders[l]
            pred = codes[l] @ T.T            # (B, F)
            diff = codes[l + 1] - pred
            total = total + (diff * diff).sum(dim=-1).mean()
        return total

    def coboundary_matrix_torch(self) -> torch.Tensor:
        F = self.F
        L = self.n_layers
        E = L - 1
        device = self.transcoders[0].device
        D = torch.zeros(E * F, L * F, device=device, dtype=self.transcoders[0].dtype)
        eye = torch.eye(F, device=device, dtype=self.transcoders[0].dtype)
        for l in range(E):
            D[l * F : (l + 1) * F, l * F : (l + 1) * F] = -self.transcoders[l]
            D[l * F : (l + 1) * F, (l + 1) * F : (l + 2) * F] = eye
        return D


def sheaf_consistency_loss(
    head: SheafConsistencyHead,
    saes: list,
    x_layers: list[torch.Tensor],
) -> torch.Tensor:
    """Encode each layer with its SAE, then return ‖δz‖² (differentiable).

    Each SAE must have ``.encode(x)`` returning (B, F). For SAEs whose
    encode signature differs we wrap call sites.
    """
    codes: list[torch.Tensor] = []
    for sae, x in zip(saes, x_layers, strict=True):
        z = sae.encode(x)
        if isinstance(z, tuple):
            z = z[0]
        codes.append(z)
    return head.energy(codes)


# ---------------------------------------------------------------------------
# Harmonic atoms (kernel of the sheaf Laplacian)
# ---------------------------------------------------------------------------


def harmonic_atoms(
    sheaf: CellularSheaf,
    tol: float = 1e-4,
    return_eigenvalues: bool = False,
):
    """Return atoms whose sheaf-Laplacian eigenvalue is ≤ tol.

    These are *global sections* — features that survive every restriction
    map without distortion. Concretely we diagonalize L = δ^T δ and return:

        modes : (n_harmonic, L·F)  rows = harmonic 0-cochains
        atom_indices : np.ndarray of atom indices (column-projected back to
                       per-layer atoms) whose total mass in the harmonic
                       subspace exceeds 1/F.

    The kernel always contains at least the dim-F space "atoms that the
    transcoders carry through exactly" (when transcoders are identities,
    every concatenated copy of a single basis vector is harmonic).
    """
    L_mat = sheaf.laplacian()
    eigvals, eigvecs = np.linalg.eigh(L_mat)  # ascending
    mask = eigvals <= tol
    modes = eigvecs[:, mask].T  # (n_harm, L·F)

    F = sheaf.F_per_layer[0]
    L = sheaf.n_layers
    # Project each mode to per-layer atom space; sum |coef|² over layers,
    # call atom k "harmonic-supported" if its total mass across the kernel
    # exceeds threshold 1/F.
    per_atom_mass = np.zeros(F)
    for mode in modes:
        m = mode.reshape(L, F)
        per_atom_mass += (m * m).sum(axis=0)
    threshold = 1.0 / F
    atom_indices = np.where(per_atom_mass > threshold)[0]

    if return_eigenvalues:
        return modes, atom_indices, eigvals
    return modes, atom_indices
