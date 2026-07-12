"""Exact analytic-geometry and joint-superposition checks for the AMM zoo."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.amm_zoo.amm import (
    GEOMETRY_REVISION,
    HELIX_T,
    embed_unit,
    generate_amm,
)


def test_closed_manifold_seams_and_defining_equations() -> None:
    tau = 2.0 * np.pi

    circle = embed_unit("circle", np.array([[0.0], [tau]]))
    np.testing.assert_allclose(circle[0], circle[1], atol=2.0e-15)
    np.testing.assert_allclose(np.sum(circle * circle, axis=1), 1.0, atol=2.0e-15)

    torus = embed_unit("torus", np.array([[0.0, 0.0], [tau, tau]]))
    np.testing.assert_allclose(torus[0], torus[1], atol=2.0e-15)
    np.testing.assert_allclose(np.sum(torus[:, :2] ** 2, axis=1), 0.5, atol=2.0e-15)
    np.testing.assert_allclose(np.sum(torus[:, 2:] ** 2, axis=1), 0.5, atol=2.0e-15)

    sphere = embed_unit(
        "sphere",
        np.array([[0.0, 0.0], [np.pi / 2.0, 0.0], [np.pi, tau]]),
    )
    np.testing.assert_allclose(np.sum(sphere * sphere, axis=1), 1.0, atol=2.0e-15)
    assert np.ptp(sphere[:, 2]) == pytest.approx(2.0)


def test_mobius_has_one_orientation_reversing_seam() -> None:
    widths = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    left = embed_unit("mobius", np.column_stack([np.zeros_like(widths), widths]))
    right = embed_unit(
        "mobius",
        np.column_stack([np.full_like(widths, 2.0 * np.pi), -widths]),
    )
    np.testing.assert_allclose(left, right, atol=2.0e-15)
    assert not np.allclose(
        left,
        embed_unit(
            "mobius",
            np.column_stack([np.full_like(widths, 2.0 * np.pi), widths]),
        ),
    )


def test_helix_is_open_and_has_axial_extent() -> None:
    helix = embed_unit("helix", np.array([[0.0], [HELIX_T]]))
    np.testing.assert_allclose(helix[0, :2], helix[1, :2], atol=2.0e-15)
    assert helix[1, 2] > helix[0, 2]
    assert not np.allclose(helix[0], helix[1])


@pytest.mark.parametrize(
    ("topology", "coords"),
    [
        ("circle", np.array([[-0.1]])),
        ("torus", np.array([[0.0, 2.0 * np.pi + 0.1]])),
        ("sphere", np.array([[np.pi + 0.1, 0.0]])),
        ("helix", np.array([[HELIX_T + 0.1]])),
        ("mobius", np.array([[0.0, 1.1]])),
    ],
)
def test_coordinate_domains_are_enforced(topology: str, coords: np.ndarray) -> None:
    with pytest.raises(ValueError):
        embed_unit(topology, coords)


def test_corpus_is_one_exact_joint_superposition() -> None:
    dataset = generate_amm(
        seed=7,
        sigma_frac=0.0,
        n_train=320,
        n_test=128,
        d=32,
        k=3,
    )
    assert dataset.G == 28
    assert np.all(dataset.train.active.sum(axis=1) == 3)
    assert np.all(dataset.test.active.sum(axis=1) == 3)
    for factor, frame in zip(dataset.factors, dataset.frames, strict=True):
        np.testing.assert_allclose(
            frame.T @ frame,
            np.eye(factor.block_dim),
            atol=2.0e-12,
        )
    reconstructed = sum(
        (dataset.contribution("test", g) for g in range(dataset.G)),
        start=np.zeros_like(dataset.test.x),
    )
    np.testing.assert_allclose(dataset.test.x, reconstructed, atol=2.0e-12)


def test_persistence_is_float64_and_rejects_stale_geometry(tmp_path) -> None:
    dataset = generate_amm(
        seed=11,
        sigma_frac=0.05,
        n_train=64,
        n_test=32,
        d=16,
        k=3,
    )
    dataset.save(tmp_path)
    with np.load(tmp_path / "amm.npz") as saved:
        assert saved["x_train"].dtype == np.float64
        assert saved["coords_train"].dtype == np.float64
        assert all(saved[f"V_{g}"].dtype == np.float64 for g in range(dataset.G))

    loaded = type(dataset).load(tmp_path)
    assert loaded.train.x.dtype == np.float64
    np.testing.assert_array_equal(loaded.train.x, dataset.train.x)
    np.testing.assert_array_equal(loaded.test.coords, dataset.test.coords)

    metadata_path = tmp_path / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["geometry_revision"] == GEOMETRY_REVISION
    metadata["geometry_revision"] = "stale-loop-placeholder"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="stale or quantized"):
        type(dataset).load(tmp_path)
