"""Synthetic ground-truth manifold dataset for Manifold-SAE recovery validation.

Generates an activation-stream-like dataset whose latent structure is *known by
construction*: each feature is a smooth 1D curve in :math:`\\mathbb{R}^{d_{\\text{ambient}}}`,
obtained by mapping a low-dimensional shape (circle, line, helix, lissajous, parabola)
through a random orthogonal projection. Each sample is a sparse sum of these curves,
evaluated at random parameter values, scaled by random amplitudes, plus Gaussian noise.

The dataset returns both the activation tensor (for training) and a ``ground_truth``
dictionary holding the per-feature curve callables and a dense reference t-grid; the
synthetic-recovery experiment uses the latter to evaluate whether the trained SAE has
rediscovered the planted curves up to reparameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Curve primitives: each takes a parameter t in [0, 1] (vectorized) and returns
# a (T, d_intrinsic) array. d_intrinsic is small; the orthogonal projection
# lifts each curve into the ambient space and provides gauge.
# ---------------------------------------------------------------------------


def _circle(t: np.ndarray) -> np.ndarray:
    # Closed curve in R^2.
    theta = 2.0 * np.pi * t
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def _line(t: np.ndarray) -> np.ndarray:
    # Straight ramp in R^1; we still return a (T, 1) for shape consistency.
    return np.stack([2.0 * t - 1.0], axis=-1)


def _helix(t: np.ndarray) -> np.ndarray:
    # Two coiled turns in R^3.
    theta = 2.0 * np.pi * 2.0 * t
    return np.stack([np.cos(theta), np.sin(theta), 2.0 * t - 1.0], axis=-1)


def _lissajous(t: np.ndarray) -> np.ndarray:
    # 3:2 Lissajous in R^2 (non-closed unless integer ratio with phase 0).
    return np.stack(
        [np.sin(2.0 * np.pi * 3.0 * t), np.sin(2.0 * np.pi * 2.0 * t + np.pi / 4)],
        axis=-1,
    )


def _parabola(t: np.ndarray) -> np.ndarray:
    # Open arc in R^2.
    u = 2.0 * t - 1.0
    return np.stack([u, u**2 - 1.0 / 3.0], axis=-1)


CURVE_TYPES: list[tuple[str, Callable[[np.ndarray], np.ndarray], int, bool]] = [
    ("circle", _circle, 2, True),
    ("line", _line, 1, False),
    ("helix", _helix, 3, False),
    ("lissajous", _lissajous, 2, False),
    ("parabola", _parabola, 2, False),
]


def _random_orthogonal_projection(
    d_intrinsic: int, d_ambient: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a (d_intrinsic, d_ambient) row-orthonormal matrix.

    Built by QR-decomposing a Gaussian matrix; the rows form an orthonormal basis
    for a random d_intrinsic-dimensional subspace of R^d_ambient. The curve's
    intrinsic coordinates are right-multiplied into this matrix.
    """
    g = rng.standard_normal(size=(d_ambient, d_intrinsic))
    q, _ = np.linalg.qr(g)
    # q is (d_ambient, d_intrinsic) with orthonormal columns.
    return q.T  # (d_intrinsic, d_ambient)


@dataclass
class FeatureCurve:
    """A planted ground-truth curve in ambient space.

    Attributes
    ----------
    name:
        Human-readable curve type ("circle", "helix", ...).
    periodic:
        Whether the curve is cyclic (used to pick the matching basis spec at eval).
    intrinsic_fn:
        Callable mapping ``t in [0, 1]`` (vectorized, shape ``(T,)``) to intrinsic
        coordinates with shape ``(T, d_intrinsic)``.
    projection:
        ``(d_intrinsic, d_ambient)`` row-orthonormal matrix mapping intrinsic to
        ambient coordinates.
    """

    name: str
    periodic: bool
    intrinsic_fn: Callable[[np.ndarray], np.ndarray]
    projection: np.ndarray

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Return ambient-space points for parameter values ``t`` (shape ``(T,)``)."""
        intrinsic = self.intrinsic_fn(np.asarray(t, dtype=np.float64))
        return intrinsic @ self.projection  # (T, d_ambient)


class SyntheticDataset(Dataset):
    """Sparse sum of planted 1D-manifold features in :math:`\\mathbb{R}^{d_{ambient}}`.

    Constructor parameters
    ----------------------
    d_ambient:
        Ambient dimension (mimics the residual stream width).
    n_features:
        Number of planted ground-truth features. Curve types cycle through
        circle, line, helix, lissajous, parabola.
    n_samples:
        Number of activation vectors to generate.
    sparsity:
        Bernoulli activation rate per feature per sample. At least one feature is
        forced active.
    noise:
        Standard deviation of additive Gaussian noise on the ambient activation.
    seed:
        RNG seed.

    The dataset exposes ``ground_truth``: a dict with keys ``"features"`` (list of
    :class:`FeatureCurve`), ``"t_grid"`` (dense reference grid), and
    ``"curve_points"`` (per-feature ``(T_grid, d_ambient)`` reference point clouds).
    Each ``__getitem__`` returns a single sample tensor of shape ``(d_ambient,)``.
    """

    def __init__(
        self,
        d_ambient: int = 64,
        n_features: int = 5,
        n_samples: int = 8192,
        sparsity: float = 0.3,
        noise: float = 0.05,
        seed: int = 0,
        t_grid_size: int = 256,
    ) -> None:
        if d_ambient < 4:
            raise ValueError(f"d_ambient={d_ambient} too small; need >=4")
        if n_features < 1:
            raise ValueError(f"n_features must be >=1; got {n_features}")
        if not 0.0 < sparsity <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1]; got {sparsity}")
        if noise < 0:
            raise ValueError(f"noise must be >=0; got {noise}")

        self.d_ambient = d_ambient
        self.n_features = n_features
        self.n_samples = n_samples
        self.sparsity = sparsity
        self.noise = noise
        self.seed = seed
        self.t_grid_size = t_grid_size

        rng = np.random.default_rng(seed)

        # Build per-feature ground-truth curves, cycling through curve types.
        self.features: list[FeatureCurve] = []
        for k in range(n_features):
            name, fn, d_intrinsic, periodic = CURVE_TYPES[k % len(CURVE_TYPES)]
            # Bump d_intrinsic to ensure projection rank; for a 1D line use a 2D
            # padding so the projection isn't degenerate.
            d_proj = max(d_intrinsic, 2)
            if d_intrinsic < d_proj:
                def fn_padded(t: np.ndarray, fn=fn, d_intrinsic=d_intrinsic, d_proj=d_proj) -> np.ndarray:
                    base = fn(t)
                    pad = np.zeros((base.shape[0], d_proj - d_intrinsic), dtype=base.dtype)
                    return np.concatenate([base, pad], axis=-1)
                intrinsic_fn = fn_padded
            else:
                intrinsic_fn = fn
            proj = _random_orthogonal_projection(d_proj, d_ambient, rng)
            self.features.append(
                FeatureCurve(
                    name=name,
                    periodic=periodic,
                    intrinsic_fn=intrinsic_fn,
                    projection=proj,
                )
            )

        # Precompute a dense reference point cloud per feature on a fixed t-grid.
        t_grid = np.linspace(0.0, 1.0, t_grid_size, endpoint=False)
        # Use endpoint=False so circles don't double-count the wrap point; for
        # open curves this still covers [0, 1) which is the canonical evaluation
        # domain (the right endpoint is implicit).
        curve_points = np.stack(
            [feat.evaluate(t_grid) for feat in self.features], axis=0
        )  # (F, T_grid, d_ambient)

        # Sample activation data: per-sample mask + per-active t + per-active amp.
        active = rng.random(size=(n_samples, n_features)) < sparsity
        # Force at least one active feature per sample.
        no_active_rows = ~active.any(axis=1)
        if no_active_rows.any():
            forced = rng.integers(0, n_features, size=int(no_active_rows.sum()))
            active[no_active_rows, forced] = True

        ts = rng.random(size=(n_samples, n_features))  # uniform in [0, 1)
        amps = np.abs(rng.standard_normal(size=(n_samples, n_features)))

        x = np.zeros((n_samples, d_ambient), dtype=np.float64)
        for k, feat in enumerate(self.features):
            mask = active[:, k]
            if not mask.any():
                continue
            pts = feat.evaluate(ts[mask, k])  # (n_active, d_ambient)
            x[mask] += amps[mask, k : k + 1] * pts

        if noise > 0:
            x = x + noise * rng.standard_normal(size=x.shape)

        # Respect torch's default dtype so the dataset slots into either an
        # f32 training run (the common case) or an f64 test/gradcheck regime
        # without manual casting at the dataloader boundary.
        default_dtype = torch.get_default_dtype()
        self._x = torch.from_numpy(x).to(default_dtype)
        self._active = torch.from_numpy(active)
        self._ts = torch.from_numpy(ts).to(default_dtype)
        self._amps = torch.from_numpy(amps).to(default_dtype)

        self.ground_truth: dict = {
            "features": self.features,
            "t_grid": t_grid,
            "curve_points": curve_points,  # (F, T_grid, d_ambient) float64
            "active": self._active,
            "ts": self._ts,
            "amps": self._amps,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._x[idx]

    @property
    def x(self) -> torch.Tensor:
        """Full (n_samples, d_ambient) activation tensor."""
        return self._x


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds in R^d.

    ``a`` has shape ``(N_a, d)`` and ``b`` has shape ``(N_b, d)``. Returns the mean
    of the mean nearest-neighbor distance from a->b and b->a. This is gauge-free
    over reparameterizations of the underlying curves (it only depends on the
    point cloud, not the order or scaling of the parameter axis).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Pairwise squared distances; works fine for small clouds (T ~ 256).
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    d_ab = np.sqrt(d2.min(axis=1)).mean()
    d_ba = np.sqrt(d2.min(axis=0)).mean()
    return 0.5 * (float(d_ab) + float(d_ba))


__all__ = ["SyntheticDataset", "FeatureCurve", "CURVE_TYPES", "chamfer_distance"]
