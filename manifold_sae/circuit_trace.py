"""Circuit tracing across crosscoder atoms via per-layer Jacobians.

Given a trained :class:`manifold_sae.crosscoder.Crosscoder` and a batch of
multi-layer activations, this module computes a directed graph

    atom i (layer l)  --w_ij-->  atom j (layer l+1)

where ``w_ij = ⟨z̃_l[:, i], z̃_{l+1}[:, j]⟩ / N``, with ``z̃_l`` the
crosscoder's sparse code REPROJECTED through only layer l's decoder (so we
isolate "what layer l would have said on its own"):

    z̃_l[:, k] = z[:, k] · ||W_l[k, :]||_2          # decoder-weighted activity at layer l

This is the simplest tractable surrogate for
``∂z_{l+1, j} / ∂z_{l, i}`` aggregated across a dataset — under the standard
crosscoder forward (z shared, decoders per-layer), the cross-layer
Jacobian is exactly this co-activation matrix once you normalize for the
per-layer "amount of explanation" each atom contributes.

Output
------
Writes a DOT graph to ``runs/crosscoder/circuit.dot``. Each layer is a
``cluster`` subgraph; edges are top-K per (l, atom_i) by weight.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .crosscoder import Crosscoder


def _per_layer_decoder_norms(model: Crosscoder) -> list[np.ndarray]:
    return [model.decoders[l].detach().norm(dim=1).cpu().numpy() for l in range(model.n_layers)]


@torch.no_grad()
def per_layer_attribution(
    model: Crosscoder, x_layers: list[torch.Tensor]
) -> np.ndarray:
    """Per-atom per-layer "active contribution" matrix Z̃ of shape (N, F, L).

    Z̃[n, k, l] = z[n, k] · ||W_l[k, :]||  — how much atom k contributed
    to the reconstruction of layer l on row n.
    """
    device = next(model.parameters()).device
    x_concat = torch.cat([x.to(device) for x in x_layers], dim=-1)
    z = model.encode(x_concat)  # (N, F)
    dec_norms = torch.stack(
        [model.decoders[l].norm(dim=1) for l in range(model.n_layers)], dim=-1
    )  # (F, L)
    z_tilde = z.unsqueeze(-1) * dec_norms.unsqueeze(0)  # (N, F, L)
    return z_tilde.cpu().numpy()


def build_circuit(
    z_tilde: np.ndarray,
    *,
    top_k_per_atom: int = 3,
    min_weight: float = 1e-3,
) -> list[tuple[int, int, int, int, float]]:
    """Return list of (l_src, i_src, l_dst, j_dst, weight) edges, l_dst = l_src+1."""
    N, F, L = z_tilde.shape
    # Standardize per (atom, layer) so weights are correlations, not magnitudes.
    mean = z_tilde.mean(axis=0, keepdims=True)
    std = z_tilde.std(axis=0, keepdims=True).clip(min=1e-6)
    z_std = (z_tilde - mean) / std

    edges: list[tuple[int, int, int, int, float]] = []
    for l in range(L - 1):
        # M[i, j] = mean_n z_std[n, i, l] * z_std[n, j, l+1]
        M = (z_std[:, :, l].T @ z_std[:, :, l + 1]) / N  # (F, F)
        # Top-K per source atom.
        for i in range(F):
            row = M[i]
            order = np.argsort(-np.abs(row))
            taken = 0
            for j in order:
                w = float(row[j])
                if abs(w) < min_weight:
                    break
                edges.append((l, int(i), l + 1, int(j), w))
                taken += 1
                if taken >= top_k_per_atom:
                    break
    return edges


def write_dot(
    edges: list[tuple[int, int, int, int, float]],
    out_path: str | Path,
    *,
    n_layers: int,
    atom_labels: dict[int, str] | None = None,
    keep_atoms: Iterable[int] | None = None,
) -> Path:
    """Write a Graphviz DOT file. ``keep_atoms`` restricts to a subset of atoms."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = set(int(a) for a in keep_atoms) if keep_atoms is not None else None
    atom_labels = atom_labels or {}

    lines: list[str] = []
    lines.append("digraph crosscoder_circuit {")
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=circle, fontsize=9, style=filled, fillcolor="#eef"];')

    # Subgraph per layer.
    atoms_per_layer: dict[int, set[int]] = {l: set() for l in range(n_layers)}
    for l_src, i, l_dst, j, _w in edges:
        if keep is None or (i in keep and j in keep):
            atoms_per_layer[l_src].add(i)
            atoms_per_layer[l_dst].add(j)

    for l in range(n_layers):
        lines.append(f'  subgraph cluster_l{l} {{')
        lines.append(f'    label="layer {l+1}";')
        lines.append('    style=dashed;')
        for a in sorted(atoms_per_layer[l]):
            label = atom_labels.get(a, str(a))
            lines.append(f'    "l{l}_a{a}" [label="{label}"];')
        lines.append('  }')

    for l_src, i, l_dst, j, w in edges:
        if keep is not None and not (i in keep and j in keep):
            continue
        color = "#cc3333" if w < 0 else "#225588"
        width = 0.4 + 4.0 * min(abs(w), 1.0)
        lines.append(
            f'  "l{l_src}_a{i}" -> "l{l_dst}_a{j}" '
            f'[penwidth={width:.2f}, color="{color}", '
            f'label="{w:.2f}", fontsize=7];'
        )
    lines.append("}")

    out_path.write_text("\n".join(lines))
    return out_path


def trace_and_save(
    model: Crosscoder,
    x_layers: list[torch.Tensor],
    out_path: str | Path,
    *,
    top_k_per_atom: int = 3,
    min_weight: float = 0.1,
    keep_atoms: Iterable[int] | None = None,
) -> tuple[Path, list[tuple[int, int, int, int, float]]]:
    """End-to-end: compute z̃, build circuit, dump DOT. Returns (path, edges)."""
    z_tilde = per_layer_attribution(model, x_layers)
    edges = build_circuit(z_tilde, top_k_per_atom=top_k_per_atom, min_weight=min_weight)
    path = write_dot(edges, out_path, n_layers=model.n_layers, keep_atoms=keep_atoms)
    return path, edges
