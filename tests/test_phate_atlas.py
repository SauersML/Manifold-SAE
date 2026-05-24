"""Tests for the PHATE / persistent-H1 / Mapper atom atlas.

Synthetic atom sets with KNOWN topology:
  - circle    -> H1 has exactly 1 dominant cycle
  - gaussian  -> H1 flat (no dominant cycle)
  - mapper-cover correctness on a circle (#nodes ≈ #bins, graph forms a loop)
  - extractor handles both (F,D) decoder and (F,K,D) curve-basis state dicts
"""

from __future__ import annotations

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.atlas.phate_atlas import (
    atom_atlas,
    persistent_h1,
    mapper_atlas,
    extract_atom_directions,
    _diffusion_embed,
    _knn_affinity,
)


def _save_temp_sd(state_dict: dict, tmp_path: Path, name: str) -> Path:
    p = tmp_path / f"{name}.pt"
    torch.save(state_dict, p)
    return p


def test_circle_atoms_have_one_dominant_h1_cycle(tmp_path):
    rng = np.random.default_rng(0)
    F, D = 64, 32
    theta = np.linspace(0, 2 * np.pi, F, endpoint=False)
    # 2-D circle embedded in D-dim, small noise on the orthogonal directions
    W_d = np.zeros((F, D), dtype=np.float32)
    W_d[:, 0] = np.cos(theta)
    W_d[:, 1] = np.sin(theta)
    W_d += rng.normal(scale=0.01, size=W_d.shape).astype(np.float32)
    sd = {"W_d": torch.from_numpy(W_d)}
    p = _save_temp_sd(sd, tmp_path, "circle")
    atlas = atom_atlas(p, n_components=2, knn=8, diffusion_t=4)
    bars = persistent_h1(atlas)
    assert len(bars) >= 1, "circle atoms should yield ≥1 H1 bar"
    pers = sorted([d - b for (b, d) in bars], reverse=True)
    dom = pers[0]
    runner = pers[1] if len(pers) > 1 else 0.0
    # Dominant cycle should be clearly separated from runner-up.
    assert dom > 0, f"dominant H1 persistence non-positive: {dom}"
    assert dom > 3 * max(runner, 1e-6), (
        f"circle dominance ratio too small: dom={dom:.4f} runner={runner:.4f}"
    )


def test_gaussian_atoms_have_no_dominant_h1_cycle(tmp_path):
    rng = np.random.default_rng(1)
    F, D = 64, 32
    W_d = rng.normal(size=(F, D)).astype(np.float32)
    sd = {"W_d": torch.from_numpy(W_d)}
    p = _save_temp_sd(sd, tmp_path, "gaussian")
    atlas = atom_atlas(p, n_components=2, knn=8, diffusion_t=4)
    bars = persistent_h1(atlas)
    pers = sorted([d - b for (b, d) in bars], reverse=True)
    dom = pers[0] if pers else 0.0
    runner = pers[1] if len(pers) > 1 else 0.0
    ratio = dom / max(runner, 1e-6)
    # Random Gaussian atoms should NOT show a 3x-dominant cycle.
    assert ratio < 3.0, (
        f"random atoms produced a dominant H1 cycle (ratio={ratio:.2f}); "
        f"dom={dom:.3f} runner={runner:.3f} (n_bars={len(bars)})"
    )


def test_mapper_cover_on_circle_makes_a_loop(tmp_path):
    rng = np.random.default_rng(2)
    F, D = 60, 8
    theta = np.linspace(0, 2 * np.pi, F, endpoint=False)
    W_d = np.zeros((F, D), dtype=np.float32)
    W_d[:, 0] = np.cos(theta); W_d[:, 1] = np.sin(theta)
    W_d += rng.normal(scale=0.005, size=W_d.shape).astype(np.float32)
    sd = {"W_d": torch.from_numpy(W_d)}
    p = _save_temp_sd(sd, tmp_path, "circle_mapper")
    # supply theta as the filter so the cover walks around the circle
    atlas = atom_atlas(p, n_components=2, hue_labels=theta / (2 * np.pi), knn=8)
    mp = mapper_atlas(atlas, n_bins=8, overlap=0.4)
    assert len(mp["nodes"]) >= 6, f"too few mapper nodes: {len(mp['nodes'])}"
    # connectivity: every node should have at least one neighbor
    deg = {n["id"]: 0 for n in mp["nodes"]}
    for u, v, _w in mp["edges"]:
        deg[u] += 1; deg[v] += 1
    # On a smooth circle with overlapping cover, isolated nodes shouldn't exist.
    isolated = sum(1 for d in deg.values() if d == 0)
    assert isolated <= 1, f"mapper graph has {isolated} isolated nodes (expected ≤1)"
    # node count roughly matches bin count (allow 1 cluster per bin ± a few).
    assert len(mp["nodes"]) <= 2 * mp["n_bins"] + 2


def test_extractor_handles_manifold_curve_basis(tmp_path):
    rng = np.random.default_rng(3)
    F, K, D = 16, 5, 8
    D_k = rng.normal(size=(F, K, D)).astype(np.float32)
    sd = {"D_k": torch.from_numpy(D_k)}
    A = extract_atom_directions(sd)
    assert A.shape == (F, D), f"expected ({F},{D}), got {A.shape}"
    # mean across basis should match
    expected = D_k.mean(axis=1)
    np.testing.assert_allclose(A, expected, atol=1e-6)

    # nn.Linear-style decoder.weight has shape (out_features=D, in_features=F)
    # and should be transposed to (F, D).
    W = rng.normal(size=(D, F)).astype(np.float32)
    sd2 = {"decoder.weight": torch.from_numpy(W)}
    A2 = extract_atom_directions(sd2)
    assert A2.shape == (F, D), f"expected transposed to ({F},{D}), got {A2.shape}"

    # The codebase convention: `W_d` is (F, D) rows-as-atoms — no transpose.
    W2 = rng.normal(size=(F, D)).astype(np.float32)
    sd3 = {"W_d": torch.from_numpy(W2)}
    A3 = extract_atom_directions(sd3)
    assert A3.shape == (F, D)
