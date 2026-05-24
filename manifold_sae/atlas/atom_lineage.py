"""Atom lineage: cross-architecture concept matching.

For each Manifold-SAE atom, find the 3 nearest TopK and L1 atoms by cosine
similarity of the *top-20-activating-color HSV centroid*. This is the
"concept-fingerprint" matching key — what each atom fires on, not what it
points to in feature space.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from manifold_sae.atlas.semantic_atlas import build_semantic_atlas, AtomCard


def _hsv_to_unit_xyz(hsv: np.ndarray) -> np.ndarray:
    """Map (H, S, V) -> 3-D point on unit-ish sphere where hue is angular.

    Use cylinder coords (V·S·cos(2πH), V·S·sin(2πH), V) and L2-normalise.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    x = v * s * np.cos(2 * np.pi * h)
    y = v * s * np.sin(2 * np.pi * h)
    z = v
    out = np.stack([x, y, z], -1)
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(n, 1e-9)


def fingerprint(cards: list[AtomCard]) -> np.ndarray:
    """(F, 3) fingerprint matrix."""
    out = np.zeros((len(cards), 3), dtype=np.float64)
    for i, c in enumerate(cards):
        if c.n_active == 0:
            continue
        out[i] = np.asarray(c.hsv_centroid)
    return _hsv_to_unit_xyz(out)


def nearest_atoms(
    src_cards: list[AtomCard],
    tgt_cards: list[AtomCard],
    k: int = 3,
) -> list[list[dict]]:
    """For each src card, k nearest tgt cards by HSV-fingerprint cosine sim."""
    src_fp = fingerprint(src_cards)
    tgt_fp = fingerprint(tgt_cards)
    # mask dead atoms
    src_alive = np.array([c.n_active > 0 for c in src_cards])
    tgt_alive = np.array([c.n_active > 0 for c in tgt_cards])
    sim = src_fp @ tgt_fp.T
    sim[~tgt_alive[None, :].repeat(sim.shape[0], 0)] = -2.0
    out: list[list[dict]] = []
    for i, c in enumerate(src_cards):
        if not src_alive[i]:
            out.append([])
            continue
        order = np.argsort(-sim[i])[:k]
        out.append([
            {
                "atom_id": int(j),
                "cosine": float(sim[i, j]),
                "category": tgt_cards[j].category,
                "top_color": tgt_cards[j].top_color,
                "hsv_centroid": list(tgt_cards[j].hsv_centroid),
            }
            for j in order if sim[i, j] > -1.5
        ])
    return out


def build_lineage(
    *,
    manifold_cards: list[AtomCard],
    topk_cards: list[AtomCard],
    l1_cards: list[AtomCard],
    k: int = 3,
) -> dict[str, list[dict]]:
    nn_topk = nearest_atoms(manifold_cards, topk_cards, k=k)
    nn_l1 = nearest_atoms(manifold_cards, l1_cards, k=k)
    out = {"k": k, "n_manifold_atoms": len(manifold_cards), "atoms": []}
    for i, c in enumerate(manifold_cards):
        out["atoms"].append({
            "manifold_atom_id": int(c.atom_id),
            "category": c.category,
            "top_color": c.top_color,
            "hsv_centroid": list(c.hsv_centroid),
            "nearest_topk": nn_topk[i],
            "nearest_l1": nn_l1[i],
        })
    return out
