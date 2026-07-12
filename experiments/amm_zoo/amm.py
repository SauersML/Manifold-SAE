"""Additive Manifold Mixture (AMM) generator — BSF paper Appendix-H zoo.

Def 2.1 (additive manifold mixture): a corpus of ``d``-dim tokens is a sparse
additive sum of per-factor manifold contributions plus isotropic noise::

    x_i = Σ_{g ∈ S_i} m_g(θ_{ig}) + σ ε_i ,   |S_i| = k ,   ε_i ~ N(0, I_d)

Each factor ``g`` is a manifold embedded in a ``b_g``-dim subspace ``V_g`` of
``R^d`` (``V_g`` is ``d × b_g`` column-orthonormal). The **zoo** plants a
controlled mix of topologies so a featurizer can be scored not just on
reconstruction but on whether it recovers the RIGHT geometry:

    8 circles   (intrinsic 1, ambient b=2)      m(θ)      = r[cosθ, sinθ]
    4 tori      (intrinsic 2, ambient b=4)      m(θ1,θ2)  = r[cosθ1,sinθ1,cosθ2,sinθ2]/√2
    4 spheres   (intrinsic 2, ambient b=3)      m(φ,λ)    = r[sinφcosλ, sinφsinλ, cosφ]
    4 arcs      (intrinsic 1, ambient b=2)      m(θ)      = r[cosθ, sinθ], θ∈[lo,hi]⊊[0,2π)
    2 helices   (intrinsic 1, ambient b=3)      m(t)      = r[cos t, sin t, p(t-2π)]
    2 Möbius    (intrinsic 2, ambient b=3)      m(θ,w)    = standard one-twist strip
    4 linear    (intrinsic b, ambient b=2)      m(c)      = r V_g c,  c ~ N(0, I)   [CONTROL]

The 4 **linear** factors are the control: they carry NO curvature, so a curved
chart must NOT beat a linear/block featurizer on them (a chart that "wins" there
is overfitting noise, the failure the R²-vs-σ curves expose).

A **subspace-coherence** knob ``coherence ∈ [0,1)`` interpolates each ``V_g``
from independent (near-orthogonal in ``d=128``) toward a shared low-dim subspace,
so the benchmark can sweep how entangled the factors are; the achieved minimum
principal angle between factor subspaces is measured and stored.

Ground truth (frames ``V_g``, per-token intrinsic coords, active membership,
radii, topology) is saved **for scoring only** — no arm ever sees it. Dense
per-factor contributions are NOT stored (``n × G × d`` is huge); they are
recomputed on demand from ``(V_g, coords)`` via :meth:`AMMDataset.contribution`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

# Topology registry: (name, intrinsic_dim d_i, ambient block dim b, count).
# helix = non-closed curved 1-D; mobius = orientation-reversing 2-D strip — the two
# cases where topology typing (vs a plain subspace) matters most.
ZOO = [
    ("circle", 1, 2, 8),
    ("torus", 2, 4, 4),
    ("sphere", 2, 3, 4),
    ("arc", 1, 2, 4),
    ("helix", 1, 3, 2),
    ("mobius", 2, 3, 2),
    ("linear", 2, 2, 4),
]
MAX_INTRINSIC = 2  # >=2-intrinsic factors carry 2 coords; 1-D carry 1 (2nd = NaN)
HELIX_T = 4.0 * np.pi  # helix parameter range (2 turns), non-closed
HELIX_PITCH = 0.16  # z-rise so the axial extent ≈ the ring radius
GEOMETRY_REVISION = "amm-analytic-v2-float64"


@dataclass(frozen=True)
class Factor:
    """One planted manifold factor (ground truth, scoring-only)."""

    fid: int
    topology: str
    intrinsic_dim: int
    block_dim: int  # b_g (ambient subspace dim)
    radius: float
    arc_lo: float  # arcs only (else 0.0)
    arc_hi: float  # arcs only (else 2π)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Topology embeddings: intrinsic coords (n, <=2) -> unit ambient block (n, b).
# --------------------------------------------------------------------------- #
def _embed_circle(coords: np.ndarray) -> np.ndarray:
    th = coords[:, 0]
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def _embed_arc(coords: np.ndarray) -> np.ndarray:
    return _embed_circle(coords)


def _embed_torus(coords: np.ndarray) -> np.ndarray:
    t1, t2 = coords[:, 0], coords[:, 1]
    inv = 1.0 / np.sqrt(2.0)  # keep ‖m‖ = r (unit before radius/frame)
    return inv * np.stack([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)], axis=1)


def _embed_sphere(coords: np.ndarray) -> np.ndarray:
    phi, lam = coords[:, 0], coords[:, 1]  # phi in [0,pi], lam in [0,2pi)
    return np.stack([np.sin(phi) * np.cos(lam), np.sin(phi) * np.sin(lam), np.cos(phi)], axis=1)


def _embed_linear(coords: np.ndarray) -> np.ndarray:
    # Gaussian coords already ~N(0,1); no curvature. Block dim b=2.
    return coords[:, :2]


def _embed_helix(coords: np.ndarray) -> np.ndarray:
    t = coords[:, 0]
    z = HELIX_PITCH * (t - HELIX_T / 2.0)  # axial rise, centred so z ~ [-1, 1]
    return np.stack([np.cos(t), np.sin(t), z], axis=1)


def _embed_mobius(coords: np.ndarray) -> np.ndarray:
    theta, w = coords[:, 0], coords[:, 1]  # theta in [0,2pi), w in [-1,1] (width)
    half = 0.5 * w
    rad = 1.0 + half * np.cos(theta / 2.0)  # the θ/2 half-twist -> orientation reversal
    return np.stack([rad * np.cos(theta), rad * np.sin(theta), half * np.sin(theta / 2.0)], axis=1)


_EMBED = {
    "circle": _embed_circle,
    "arc": _embed_arc,
    "torus": _embed_torus,
    "sphere": _embed_sphere,
    "helix": _embed_helix,
    "mobius": _embed_mobius,
    "linear": _embed_linear,
}


def _validated_coords(topology: str, coords: np.ndarray) -> np.ndarray:
    """Return float64 coordinates in the closed domain used for seam evaluation.

    Random samples use half-open periodic intervals; accepting the right endpoint
    here permits exact seam tests without changing the sampling distribution.
    """
    intrinsic_dims = {
        "circle": 1,
        "arc": 1,
        "torus": 2,
        "sphere": 2,
        "helix": 1,
        "mobius": 2,
        "linear": 2,
    }
    if topology not in intrinsic_dims:
        raise ValueError(f"unknown topology {topology!r}")
    values = np.asarray(coords, dtype=np.float64)
    needed = intrinsic_dims[topology]
    if values.ndim != 2 or values.shape[1] < needed:
        raise ValueError(
            f"{topology} coordinates must have shape (n, >= {needed}), got {values.shape}"
        )
    intrinsic = values[:, :needed]
    if not np.isfinite(intrinsic).all():
        raise ValueError(f"{topology} coordinates contain non-finite values")

    tolerance = 32.0 * np.finfo(np.float64).eps

    def require_interval(column: int, lower: float, upper: float, name: str) -> None:
        value = intrinsic[:, column]
        if np.any(value < lower - tolerance) or np.any(value > upper + tolerance):
            raise ValueError(f"{topology} {name} must lie in [{lower}, {upper}]")

    if topology in {"circle", "arc", "helix"}:
        upper = HELIX_T if topology == "helix" else 2.0 * np.pi
        require_interval(0, 0.0, upper, "angle")
    elif topology == "torus":
        require_interval(0, 0.0, 2.0 * np.pi, "first angle")
        require_interval(1, 0.0, 2.0 * np.pi, "second angle")
    elif topology == "sphere":
        require_interval(0, 0.0, np.pi, "polar angle")
        require_interval(1, 0.0, 2.0 * np.pi, "longitude")
    elif topology == "mobius":
        require_interval(0, 0.0, 2.0 * np.pi, "angle")
        require_interval(1, -1.0, 1.0, "width")
    return values


def _validated_factor_coords(factor: Factor, coords: np.ndarray) -> np.ndarray:
    values = _validated_coords(factor.topology, coords)
    if factor.topology == "arc":
        tolerance = 32.0 * np.finfo(np.float64).eps
        angle = values[:, 0]
        if np.any(angle < factor.arc_lo - tolerance) or np.any(angle > factor.arc_hi + tolerance):
            raise ValueError(f"arc coordinates must lie in [{factor.arc_lo}, {factor.arc_hi}]")
    return values


def embed_unit(topology: str, coords: np.ndarray) -> np.ndarray:
    """Unit-scale ambient block ``(n, b)`` for a topology from intrinsic coords."""
    coords = _validated_coords(topology, coords)
    embedded = _EMBED[topology](coords)
    if not np.isfinite(embedded).all():
        raise ValueError(f"{topology} embedding returned non-finite values")
    tolerance = 2.0e-12
    if topology in {"circle", "arc"}:
        residual = embedded[:, 0] ** 2 + embedded[:, 1] ** 2 - 1.0
    elif topology == "torus":
        inv_sq = 0.5
        residual = np.maximum(
            np.abs(embedded[:, 0] ** 2 + embedded[:, 1] ** 2 - inv_sq),
            np.abs(embedded[:, 2] ** 2 + embedded[:, 3] ** 2 - inv_sq),
        )
    elif topology == "sphere":
        residual = np.sum(embedded * embedded, axis=1) - 1.0
    elif topology == "helix":
        radius_error = embedded[:, 0] ** 2 + embedded[:, 1] ** 2 - 1.0
        height_error = embedded[:, 2] - HELIX_PITCH * (coords[:, 0] - HELIX_T / 2.0)
        residual = np.maximum(np.abs(radius_error), np.abs(height_error))
    elif topology == "mobius":
        theta = coords[:, 0]
        half_width = 0.5 * coords[:, 1]
        radius = 1.0 + half_width * np.cos(theta / 2.0)
        expected = np.stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                half_width * np.sin(theta / 2.0),
            ],
            axis=1,
        )
        residual = embedded - expected
    elif topology == "linear":
        residual = embedded - coords[:, :2]
    else:
        raise ValueError(f"unknown topology {topology!r}")
    error = float(np.max(np.abs(residual), initial=0.0))
    if error > tolerance:
        raise ValueError(f"{topology} defining-equation error {error:.3e} exceeds {tolerance:.3e}")
    return embedded


def _sample_coords(topology: str, n: int, factor: Factor, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` intrinsic coordinates for a factor; returns ``(n, MAX_INTRINSIC)``
    with unused slots = NaN (so ground-truth coords stay a rectangular array)."""
    out = np.full((n, MAX_INTRINSIC), np.nan, dtype=np.float64)
    if topology in ("circle",):
        out[:, 0] = rng.uniform(0.0, 2.0 * np.pi, n)
    elif topology == "arc":
        out[:, 0] = rng.uniform(factor.arc_lo, factor.arc_hi, n)
    elif topology == "torus":
        out[:, 0] = rng.uniform(0.0, 2.0 * np.pi, n)
        out[:, 1] = rng.uniform(0.0, 2.0 * np.pi, n)
    elif topology == "sphere":
        # Uniform on S^2: cos φ ~ U(-1,1), λ ~ U(0,2π).
        out[:, 0] = np.arccos(rng.uniform(-1.0, 1.0, n))
        out[:, 1] = rng.uniform(0.0, 2.0 * np.pi, n)
    elif topology == "helix":
        out[:, 0] = rng.uniform(0.0, HELIX_T, n)
    elif topology == "mobius":
        out[:, 0] = rng.uniform(0.0, 2.0 * np.pi, n)
        out[:, 1] = rng.uniform(-1.0, 1.0, n)
    elif topology == "linear":
        out[:, 0] = rng.standard_normal(n)
        out[:, 1] = rng.standard_normal(n)
    else:  # pragma: no cover
        raise ValueError(f"unknown topology {topology!r}")
    return out


# --------------------------------------------------------------------------- #
# Subspace frames with a controlled coherence knob.
# --------------------------------------------------------------------------- #
def _make_frames(
    factors: list[Factor], d: int, coherence: float, rng: np.random.Generator
) -> tuple[list[np.ndarray], float]:
    """Column-orthonormal ``V_g`` (d × b_g) per factor. ``coherence ∈ [0,1)`` mixes
    each frame's raw Gaussian span toward a SHARED random subspace ``S`` (dim
    ``s = max b_g``), so the factor subspaces overlap more as coherence rises.
    Returns ``(frames, measured_min_principal_angle_deg)``."""
    coherence = float(coherence)
    if not 0.0 <= coherence < 1.0:
        raise ValueError(f"coherence must lie in [0, 1), got {coherence}")
    s = max(f.block_dim for f in factors)
    shared, _ = np.linalg.qr(rng.standard_normal((d, s)))  # d × s shared basis
    frames = []
    for f in factors:
        b = f.block_dim
        raw = rng.standard_normal((d, b))
        if coherence > 0.0:
            mix = shared @ rng.standard_normal((s, b))
            raw = (1.0 - coherence) * raw + coherence * mix
        q, _ = np.linalg.qr(raw)
        frames.append(np.ascontiguousarray(q[:, :b]))
    return frames, _min_principal_angle_deg(frames)


def _min_principal_angle_deg(frames: list[np.ndarray]) -> float:
    """Minimum principal angle (degrees) over all factor-subspace pairs — the
    coherence the generator actually achieved (0° = a shared direction)."""
    worst = 90.0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            # Largest singular value of Vi^T Vj is cos(min principal angle).
            sv = np.linalg.svd(frames[i].T @ frames[j], compute_uv=False)
            cos_min = float(np.clip(sv.max(), 0.0, 1.0))
            worst = min(worst, np.degrees(np.arccos(cos_min)))
    return round(worst, 3)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class AMMSplit:
    """One split (train or test) of an AMM corpus."""

    x: np.ndarray  # (n, d) noisy tokens
    active: np.ndarray  # (n, G) bool factor-active mask
    coords: np.ndarray  # (n, G, MAX_INTRINSIC) intrinsic coords, NaN where inactive

    @property
    def n(self) -> int:
        return self.x.shape[0]


class AMMDataset:
    """A planted AMM corpus + scoring-only ground truth.

    Attributes
    ----------
    factors: list[Factor]        the G planted factors
    frames:  list[np.ndarray]    V_g (d × b_g) column-orthonormal, per factor
    train, test: AMMSplit
    d, G, k, sigma_frac, sigma, signal_rms, coherence, min_principal_angle_deg, seed
    """

    def __init__(
        self,
        factors: list[Factor],
        frames: list[np.ndarray],
        train: AMMSplit,
        test: AMMSplit,
        *,
        d: int,
        k: int,
        sigma_frac: float,
        sigma: float,
        signal_rms: float,
        coherence: float,
        min_principal_angle_deg: float,
        seed: int,
    ) -> None:
        self.factors = factors
        self.frames = frames
        self.train = train
        self.test = test
        self.d = d
        self.G = len(factors)
        self.k = k
        self.sigma_frac = sigma_frac
        self.sigma = sigma
        self.signal_rms = signal_rms
        self.coherence = coherence
        self.min_principal_angle_deg = min_principal_angle_deg
        self.seed = seed

    # -- scoring helpers ---------------------------------------------------- #
    def contribution(self, split: str, g: int) -> np.ndarray:
        """Recompute the TRUE per-token contribution ``m_g`` of factor ``g`` on a
        split: ``(n, d)``, zero on tokens where ``g`` is inactive. Cheap
        recomputation from ``(V_g, radius, coords)`` — never stored densely."""
        sp = self.train if split == "train" else self.test
        f = self.factors[g]
        out = np.zeros((sp.n, self.d), dtype=np.float64)
        rows = np.nonzero(sp.active[:, g])[0]
        if rows.size == 0:
            return out
        coords = _validated_factor_coords(f, sp.coords[rows, g, :])
        emb = embed_unit(f.topology, coords)  # (n_g, b)
        out[rows] = f.radius * (emb @ self.frames[g].T)
        return out

    def true_intrinsic(self, split: str, g: int) -> tuple[np.ndarray, np.ndarray]:
        """Active rows and their true intrinsic coords ``(rows, coords[n_g, d_i])``
        for factor ``g`` (for circular-corr / geodesic-Spearman scoring)."""
        sp = self.train if split == "train" else self.test
        rows = np.nonzero(sp.active[:, g])[0]
        di = self.factors[g].intrinsic_dim
        return rows, sp.coords[rows, g, :di]

    def validate_exact_geometry(self) -> None:
        """Reject malformed, quantized, or non-analytic persisted ground truth."""
        expected_registry = [
            (topology, intrinsic_dim, block_dim)
            for topology, intrinsic_dim, block_dim, count in ZOO
            for _ in range(count)
        ]
        if self.G != len(expected_registry) or len(self.frames) != self.G:
            raise ValueError("AMM factor registry has the wrong size")
        if not 1 <= self.k <= self.G:
            raise ValueError("AMM sparsity is outside the factor registry")
        for g, (factor, frame, expected) in enumerate(
            zip(self.factors, self.frames, expected_registry, strict=True)
        ):
            if factor.fid != g:
                raise ValueError("AMM factor ids must be contiguous and ordered")
            if (factor.topology, factor.intrinsic_dim, factor.block_dim) != expected:
                raise ValueError("AMM factor registry does not match the analytic zoo")
            if not np.isfinite(factor.radius) or factor.radius <= 0.0:
                raise ValueError("AMM factor radii must be finite and positive")
            if factor.topology == "arc" and not (
                0.0 <= factor.arc_lo < factor.arc_hi <= 2.0 * np.pi
            ):
                raise ValueError("AMM arc bounds are invalid")
            if frame.dtype != np.float64 or frame.shape != (self.d, factor.block_dim):
                raise ValueError("AMM frame has the wrong dtype or shape")
            gram = frame.T @ frame
            if not np.allclose(gram, np.eye(factor.block_dim), rtol=0.0, atol=2.0e-12):
                raise ValueError("AMM frame is not an isometry")

        for name, split in (("train", self.train), ("test", self.test)):
            if split.x.dtype != np.float64 or split.coords.dtype != np.float64:
                raise ValueError(f"AMM {name} geometry must be float64")
            if split.active.dtype != np.bool_:
                raise ValueError(f"AMM {name} active mask must be boolean")
            if split.x.ndim != 2 or split.x.shape[1] != self.d:
                raise ValueError(f"AMM {name} tokens have the wrong shape")
            n = split.x.shape[0]
            if split.active.shape != (n, self.G):
                raise ValueError(f"AMM {name} active mask has the wrong shape")
            if split.coords.shape != (n, self.G, MAX_INTRINSIC):
                raise ValueError(f"AMM {name} coordinates have the wrong shape")
            if not np.isfinite(split.x).all():
                raise ValueError(f"AMM {name} tokens contain non-finite values")
            if not np.all(split.active.sum(axis=1) == self.k):
                raise ValueError(f"AMM {name} rows do not have exactly k active factors")
            if not np.isnan(split.coords[~split.active]).all():
                raise ValueError(f"AMM {name} inactive coordinates must be NaN")
            for g, factor in enumerate(self.factors):
                rows = np.flatnonzero(split.active[:, g])
                coords = split.coords[rows, g]
                _validated_factor_coords(factor, coords)
                if factor.intrinsic_dim == 1 and not np.isnan(coords[:, 1]).all():
                    raise ValueError("unused intrinsic coordinates must be NaN")
                embed_unit(factor.topology, coords)

    # -- persistence -------------------------------------------------------- #
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "geometry_revision": GEOMETRY_REVISION,
            "d": self.d,
            "G": self.G,
            "k": self.k,
            "sigma_frac": self.sigma_frac,
            "sigma": self.sigma,
            "signal_rms": self.signal_rms,
            "coherence": self.coherence,
            "min_principal_angle_deg": self.min_principal_angle_deg,
            "seed": self.seed,
            "n_train": self.train.n,
            "n_test": self.test.n,
            "factors": [f.to_json() for f in self.factors],
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        arrays: dict[str, np.ndarray] = {
            "x_train": np.ascontiguousarray(self.train.x, dtype=np.float64),
            "active_train": self.train.active,
            "coords_train": np.ascontiguousarray(self.train.coords, dtype=np.float64),
            "x_test": np.ascontiguousarray(self.test.x, dtype=np.float64),
            "active_test": self.test.active,
            "coords_test": np.ascontiguousarray(self.test.coords, dtype=np.float64),
        }
        for g, v in enumerate(self.frames):
            arrays[f"V_{g}"] = np.ascontiguousarray(v, dtype=np.float64)
        np.savez_compressed(out / "amm.npz", **arrays)

    @classmethod
    def load(cls, out_dir: str | Path) -> "AMMDataset":
        out = Path(out_dir)
        meta = json.loads((out / "meta.json").read_text())
        if meta.get("geometry_revision") != GEOMETRY_REVISION:
            raise ValueError("saved AMM corpus uses a stale or quantized geometry revision")
        factors = [
            Factor(
                fid=int(fj["fid"]),
                topology=str(fj["topology"]),
                intrinsic_dim=int(fj["intrinsic_dim"]),
                block_dim=int(fj["block_dim"]),
                radius=float(fj["radius"]),
                arc_lo=float(fj["arc_lo"]),
                arc_hi=float(fj["arc_hi"]),
            )
            for fj in meta["factors"]
        ]
        expected_float = {
            "x_train",
            "coords_train",
            "x_test",
            "coords_test",
            *(f"V_{g}" for g in range(len(factors))),
        }
        expected_bool = {"active_train", "active_test"}
        with np.load(out / "amm.npz") as saved:
            if set(saved.files) != expected_float | expected_bool:
                raise ValueError("saved AMM corpus has an unexpected array schema")
            if any(saved[name].dtype != np.float64 for name in expected_float):
                raise ValueError("saved AMM geometry must be persisted directly as float64")
            if any(saved[name].dtype != np.bool_ for name in expected_bool):
                raise ValueError("saved AMM active masks must be boolean")
            frames = [np.ascontiguousarray(saved[f"V_{g}"]) for g in range(len(factors))]
            train = AMMSplit(
                x=np.ascontiguousarray(saved["x_train"]),
                active=np.ascontiguousarray(saved["active_train"]),
                coords=np.ascontiguousarray(saved["coords_train"]),
            )
            test = AMMSplit(
                x=np.ascontiguousarray(saved["x_test"]),
                active=np.ascontiguousarray(saved["active_test"]),
                coords=np.ascontiguousarray(saved["coords_test"]),
            )
        dataset = cls(
            factors,
            frames,
            train,
            test,
            d=int(meta["d"]),
            k=int(meta["k"]),
            sigma_frac=float(meta["sigma_frac"]),
            sigma=float(meta["sigma"]),
            signal_rms=float(meta["signal_rms"]),
            coherence=float(meta["coherence"]),
            min_principal_angle_deg=float(meta["min_principal_angle_deg"]),
            seed=int(meta["seed"]),
        )
        if dataset.train.n != int(meta["n_train"]) or dataset.test.n != int(meta["n_test"]):
            raise ValueError("saved AMM split sizes disagree with metadata")
        if dataset.G != int(meta["G"]):
            raise ValueError("saved AMM factor count disagrees with metadata")
        dataset.validate_exact_geometry()
        return dataset


def build_factors(rng: np.random.Generator) -> list[Factor]:
    """Instantiate the 28-factor zoo with per-factor radii and arc ranges."""
    factors: list[Factor] = []
    fid = 0
    for topology, di, b, count in ZOO:
        for _ in range(count):
            radius = float(rng.uniform(0.8, 1.2))
            arc_lo, arc_hi = 0.0, 2.0 * np.pi
            if topology == "arc":
                # An open arc spanning ~[0.6π, 1.4π] of the circle (never wraps),
                # centred at a random phase — genuinely a manifold with boundary.
                span = float(rng.uniform(0.6 * np.pi, 1.4 * np.pi))
                lo = float(rng.uniform(0.0, 2.0 * np.pi - span))
                arc_lo, arc_hi = lo, lo + span
            factors.append(Factor(fid, topology, di, b, radius, arc_lo, arc_hi))
            fid += 1
    return factors


def _make_split(
    n: int,
    factors: list[Factor],
    frames: list[np.ndarray],
    d: int,
    k: int,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[AMMSplit, float]:
    """Build one split; returns ``(split, sum_sq)`` where ``sum_sq`` is the total
    noiseless signal energy (for the shared ``signal_rms`` estimate)."""
    G = len(factors)
    # Active set: k distinct factors per token, uniformly sampled. Partial
    # selection avoids sorting all G keys when k << G.
    active = np.zeros((n, G), dtype=bool)
    order = np.argpartition(rng.random((n, G)), k - 1, axis=1)[:, :k]
    np.put_along_axis(active, order, True, axis=1)

    coords = np.full((n, G, MAX_INTRINSIC), np.nan, dtype=np.float64)
    x_clean = np.zeros((n, d), dtype=np.float64)
    for g, f in enumerate(factors):
        rows = np.nonzero(active[:, g])[0]
        if rows.size == 0:
            continue
        c = _sample_coords(f.topology, rows.size, f, rng)  # (n_g, MAX_INTRINSIC)
        _validated_factor_coords(f, c)
        coords[rows, g, :] = c
        emb = embed_unit(f.topology, c)  # (n_g, b)
        x_clean[rows] += f.radius * (emb @ frames[g].T)

    sum_sq = float(np.einsum("ij,ij->", x_clean, x_clean, optimize=True))
    _add_noise_in_place(x_clean, sigma, rng)
    return AMMSplit(x=x_clean, active=active, coords=coords), sum_sq


def _add_noise_in_place(
    x: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    *,
    target_chunk_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Add Gaussian noise without allocating another corpus-sized matrix."""
    if sigma == 0.0:
        return
    if sigma < 0.0 or not np.isfinite(sigma):
        raise ValueError(f"noise standard deviation must be finite and nonnegative, got {sigma}")
    row_bytes = x.shape[1] * x.dtype.itemsize
    chunk_rows = max(1, target_chunk_bytes // row_bytes)
    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        noise = rng.standard_normal((stop - start, x.shape[1]))
        noise *= sigma
        x[start:stop] += noise


def generate_amm(
    *,
    seed: int = 0,
    sigma_frac: float = 0.05,
    coherence: float = 0.0,
    n_train: int = 200_000,
    n_test: int = 50_000,
    d: int = 128,
    k: int = 3,
) -> AMMDataset:
    """Generate an AMM zoo corpus (Def 2.1). ``sigma_frac`` is the noise std as a
    fraction of the per-coordinate signal RMS (measured on the TRAIN split so the
    same absolute ``σ`` applies to both splits — a matched noise floor)."""
    rng = np.random.default_rng(seed)
    factors = build_factors(rng)
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must both be positive")
    if not np.isfinite(sigma_frac) or sigma_frac < 0.0:
        raise ValueError(f"sigma_frac must be finite and nonnegative, got {sigma_frac}")
    if not 1 <= k <= len(factors):
        raise ValueError(f"k must lie in [1, {len(factors)}], got {k}")
    max_span = max(factor.block_dim for factor in factors)
    if d < max_span:
        raise ValueError(f"ambient dimension {d} cannot contain native AMM span {max_span}")
    frames, min_angle = _make_frames(factors, d, coherence, rng)

    # Generate the training realization exactly once. Its clean energy fixes σ,
    # then independent Gaussian noise is added in cache-sized blocks in place.
    train, sum_sq = _make_split(
        n_train, factors, frames, d, k, 0.0, np.random.default_rng(seed + 1)
    )
    signal_rms = float(np.sqrt(sum_sq / (n_train * d)))
    sigma = sigma_frac * signal_rms
    _add_noise_in_place(train.x, sigma, np.random.default_rng(seed + 10_001))
    test, _ = _make_split(n_test, factors, frames, d, k, sigma, np.random.default_rng(seed + 2))

    return AMMDataset(
        factors,
        frames,
        train,
        test,
        d=d,
        k=k,
        sigma_frac=sigma_frac,
        sigma=sigma,
        signal_rms=signal_rms,
        coherence=coherence,
        min_principal_angle_deg=min_angle,
        seed=seed,
    )


# Standard benchmark sweeps.
SIGMA_GRID = [0.02, 0.05, 0.1, 0.2]
SEEDS = [0, 1, 2, 3, 4]


if __name__ == "__main__":
    # Smoke: a small corpus, verify shapes + ground-truth round-trip + signal.
    ds = generate_amm(seed=0, sigma_frac=0.05, n_train=2000, n_test=500)
    print(
        f"factors={ds.G} d={ds.d} k={ds.k} signal_rms={ds.signal_rms:.4f} "
        f"sigma={ds.sigma:.4f} min_angle={ds.min_principal_angle_deg}deg"
    )
    print(f"topologies: {[f.topology for f in ds.factors]}")
    print(
        f"train x {ds.train.x.shape}, active/token = {ds.train.active.sum(1).mean():.2f} (k={ds.k})"
    )
    # contribution reconstructs the clean sum:
    clean = sum(ds.contribution("test", g) for g in range(ds.G))
    resid = ds.test.x - clean
    print(f"noise std check: {resid.std():.4f} vs sigma {ds.sigma:.4f}")
