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
ZOO = [
    ("circle", 1, 2, 8),
    ("torus", 2, 4, 4),
    ("sphere", 2, 3, 4),
    ("arc", 1, 2, 4),
    ("linear", 2, 2, 4),
]
MAX_INTRINSIC = 2  # torus/sphere/linear carry 2 intrinsic coords; circle/arc carry 1


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
    return np.stack(
        [np.sin(phi) * np.cos(lam), np.sin(phi) * np.sin(lam), np.cos(phi)], axis=1
    )


def _embed_linear(coords: np.ndarray) -> np.ndarray:
    # Gaussian coords already ~N(0,1); no curvature. Block dim b=2.
    return coords[:, :2]


_EMBED = {
    "circle": _embed_circle,
    "arc": _embed_arc,
    "torus": _embed_torus,
    "sphere": _embed_sphere,
    "linear": _embed_linear,
}


def embed_unit(topology: str, coords: np.ndarray) -> np.ndarray:
    """Unit-scale ambient block ``(n, b)`` for a topology from intrinsic coords."""
    return _EMBED[topology](coords)


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
    coherence = float(np.clip(coherence, 0.0, 0.98))
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
        emb = embed_unit(f.topology, sp.coords[rows, g, :])  # (n_g, b)
        out[rows] = f.radius * (emb @ self.frames[g].T)
        return out

    def true_intrinsic(self, split: str, g: int) -> tuple[np.ndarray, np.ndarray]:
        """Active rows and their true intrinsic coords ``(rows, coords[n_g, d_i])``
        for factor ``g`` (for circular-corr / geodesic-Spearman scoring)."""
        sp = self.train if split == "train" else self.test
        rows = np.nonzero(sp.active[:, g])[0]
        di = self.factors[g].intrinsic_dim
        return rows, sp.coords[rows, g, :di]

    # -- persistence -------------------------------------------------------- #
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
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
            "x_train": self.train.x.astype(np.float32),
            "active_train": self.train.active,
            "coords_train": self.train.coords.astype(np.float32),
            "x_test": self.test.x.astype(np.float32),
            "active_test": self.test.active,
            "coords_test": self.test.coords.astype(np.float32),
        }
        for g, v in enumerate(self.frames):
            arrays[f"V_{g}"] = v.astype(np.float32)
        np.savez_compressed(out / "amm.npz", **arrays)

    @classmethod
    def load(cls, out_dir: str | Path) -> "AMMDataset":
        out = Path(out_dir)
        meta = json.loads((out / "meta.json").read_text())
        z = np.load(out / "amm.npz")
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
        frames = [np.ascontiguousarray(z[f"V_{g}"], dtype=np.float64) for g in range(len(factors))]
        train = AMMSplit(
            x=np.ascontiguousarray(z["x_train"], dtype=np.float64),
            active=np.ascontiguousarray(z["active_train"]),
            coords=np.ascontiguousarray(z["coords_train"], dtype=np.float64),
        )
        test = AMMSplit(
            x=np.ascontiguousarray(z["x_test"], dtype=np.float64),
            active=np.ascontiguousarray(z["active_test"]),
            coords=np.ascontiguousarray(z["coords_test"], dtype=np.float64),
        )
        return cls(
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


def build_factors(rng: np.random.Generator) -> list[Factor]:
    """Instantiate the 24-factor zoo with per-factor radii and arc ranges."""
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
            factors.append(
                Factor(fid, topology, di, b, radius, arc_lo, arc_hi)
            )
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
    # Active set: k distinct factors per token, uniform (argsort of noise = shuffle).
    active = np.zeros((n, G), dtype=bool)
    order = np.argsort(rng.random((n, G)), axis=1)[:, :k]
    np.put_along_axis(active, order, True, axis=1)

    coords = np.full((n, G, MAX_INTRINSIC), np.nan, dtype=np.float64)
    x_clean = np.zeros((n, d), dtype=np.float64)
    for g, f in enumerate(factors):
        rows = np.nonzero(active[:, g])[0]
        if rows.size == 0:
            continue
        c = _sample_coords(f.topology, rows.size, f, rng)  # (n_g, MAX_INTRINSIC)
        coords[rows, g, :] = c
        emb = embed_unit(f.topology, c)  # (n_g, b)
        x_clean[rows] += f.radius * (emb @ frames[g].T)

    sum_sq = float((x_clean ** 2).sum())
    noise = sigma * rng.standard_normal((n, d))
    x = x_clean + noise
    return AMMSplit(x=x, active=active, coords=coords), sum_sq


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
    frames, min_angle = _make_frames(factors, d, coherence, rng)

    # First a σ=0 train pass to fix the signal RMS, then set σ and regenerate both
    # splits at that absolute noise level (deterministic: fresh child RNGs).
    probe_rng = np.random.default_rng(seed + 101)
    probe, sum_sq = _make_split(n_train, factors, frames, d, k, 0.0, probe_rng)
    signal_rms = float(np.sqrt(sum_sq / (n_train * d)))
    sigma = sigma_frac * signal_rms

    train, _ = _make_split(n_train, factors, frames, d, k, sigma, np.random.default_rng(seed + 1))
    test, _ = _make_split(n_test, factors, frames, d, k, sigma, np.random.default_rng(seed + 2))
    del probe

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
    print(f"factors={ds.G} d={ds.d} k={ds.k} signal_rms={ds.signal_rms:.4f} "
          f"sigma={ds.sigma:.4f} min_angle={ds.min_principal_angle_deg}deg")
    print(f"topologies: {[f.topology for f in ds.factors]}")
    print(f"train x {ds.train.x.shape}, active/token = {ds.train.active.sum(1).mean():.2f} (k={ds.k})")
    # contribution reconstructs the clean sum:
    clean = sum(ds.contribution('test', g) for g in range(ds.G))
    resid = ds.test.x - clean
    print(f"noise std check: {resid.std():.4f} vs sigma {ds.sigma:.4f}")
