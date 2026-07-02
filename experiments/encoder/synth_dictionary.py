"""Synthetic composed dictionaries for WS-E encoder validation.

Builds a *composed* dictionary in the shape SAC produces: T1 linear atoms
(straight affine directions) plus T2 curved atoms (planted circles). The
encoder harness is dictionary-agnostic, so a synthetic planted fit is a stand-in
for a real SAC-composed dictionary until WS-A emits one.

Two planting regimes:
  * ``planted_circles(...)``: K circles in mutually orthogonal 2-planes with
    STRONGLY separated per-atom variances so the top principal components align
    with the planted circles per atom â€” the regime where even the joint cold
    fit converges (SAC_PLAN Part 1: "on planted data the top PCs *are* the
    planted circles"). Used to build a K>=2 composed dictionary without invoking
    the failing joint algorithm on real-shaped data.
  * ``planted_single_circle(...)``: one circle, the K=1 dictionary that always
    fits (the proven machinery), for the fast local smoke.

Token-frequency metadata is synthesized alongside (a Zipf draw over a synthetic
vocabulary) so the per-decile fallback breakdown can be exercised before the
real WS-D corpus manifest is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SyntheticCorpus:
    X_train: np.ndarray          # (N_train, p)
    X_eval: np.ndarray           # (N_eval, p)
    token_id_eval: np.ndarray    # (N_eval,) synthetic vocab id per eval row
    token_freq_eval: np.ndarray  # (N_eval,) corpus frequency of that token
    ambient_dim: int
    n_circles: int
    n_linear: int
    description: str


def _orthonormal_frame(p: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((p, cols)))
    return q[:, :cols]


def _zipf_tokens(n: int, vocab: int, rng: np.random.Generator, s: float = 1.1) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.arange(1, vocab + 1, dtype=np.float64)
    probs = ranks ** (-s)
    probs /= probs.sum()
    ids = rng.choice(vocab, size=n, p=probs)
    freq = probs[ids]
    return ids.astype(np.int64), freq.astype(np.float64)


def planted_circles(
    *,
    n_train: int = 4000,
    n_eval: int = 4000,
    ambient_dim: int = 64,
    n_circles: int = 2,
    n_linear: int = 1,
    variances: tuple[float, ...] | None = None,
    noise: float = 0.01,
    vocab: int = 512,
    random_state: int = 0,
) -> SyntheticCorpus:
    """Planted circles (separated variances) + straight linear atoms.

    Each row draws an independent phase per circle and an independent scalar per
    linear atom; the ambient signal is the sum of the decoded atoms plus small
    isotropic noise. Circle ``k`` gets variance ``variances[k]`` (default a
    geometric ladder 4:1 between adjacent atoms) so PC ordering matches atom
    ordering â€” the clean regime for a joint cold fit.
    """
    rng = np.random.default_rng(random_state)
    n = n_train + n_eval
    cols = 2 * n_circles + n_linear
    if ambient_dim < cols:
        raise ValueError(f"ambient_dim {ambient_dim} < required {cols} planted columns")
    frame = _orthonormal_frame(ambient_dim, cols, rng)
    if variances is None:
        variances = tuple(4.0 ** (n_circles - 1 - k) for k in range(n_circles))
    if len(variances) != n_circles:
        raise ValueError("variances must have one entry per circle")

    X = np.zeros((n, ambient_dim), dtype=np.float64)
    for k in range(n_circles):
        amp = float(np.sqrt(variances[k]))
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        plane = frame[:, 2 * k : 2 * k + 2]      # (p, 2)
        X += amp * (np.c_[np.cos(theta), np.sin(theta)] @ plane.T)
    for j in range(n_linear):
        s = rng.standard_normal(n)
        direction = frame[:, 2 * n_circles + j]  # (p,)
        X += s[:, None] * direction[None, :]
    X += noise * rng.standard_normal((n, ambient_dim))

    tok_id, tok_freq = _zipf_tokens(n, vocab, rng)
    return SyntheticCorpus(
        X_train=np.ascontiguousarray(X[:n_train], dtype=np.float32),
        X_eval=np.ascontiguousarray(X[n_train:], dtype=np.float32),
        token_id_eval=tok_id[n_train:],
        token_freq_eval=tok_freq[n_train:],
        ambient_dim=ambient_dim,
        n_circles=n_circles,
        n_linear=n_linear,
        description=(
            f"{n_circles} planted circles (var ladder {variances}) + "
            f"{n_linear} linear atoms in R^{ambient_dim}, noise={noise}"
        ),
    )


def planted_single_circle(
    *,
    n_train: int = 1200,
    n_eval: int = 800,
    ambient_dim: int = 16,
    noise: float = 0.01,
    vocab: int = 256,
    random_state: int = 0,
) -> SyntheticCorpus:
    """One planted circle â€” the K=1 dictionary that always fits (local smoke)."""
    return planted_circles(
        n_train=n_train,
        n_eval=n_eval,
        ambient_dim=ambient_dim,
        n_circles=1,
        n_linear=0,
        variances=(1.0,),
        noise=noise,
        vocab=vocab,
        random_state=random_state,
    )


def fit_dictionary(
    corpus: SyntheticCorpus,
    *,
    n_iter: int = 40,
    random_state: int = 0,
    d_atom: int = 1,
) -> Any:
    """Fit a ManifoldSAE on the planted corpus (the composed dictionary).

    K = n_circles + n_linear atoms. Circles use ``atom_topology="circle"`` under
    the IBP-MAP gate (the proven curved path). When there are linear atoms the
    fit is heterogeneous, so we fall back to all-circle topology (a circle atom
    subsumes a straight image via the hybrid split's LINEAR verdict) unless the
    corpus is purely linear.
    """
    import gamfit

    k = corpus.n_circles + corpus.n_linear
    return gamfit.sae_manifold_fit(
        corpus.X_train,
        K=k,
        d_atom=d_atom,
        atom_topology="circle",
        assignment="ibp_map",
        n_iter=int(n_iter),
        random_state=int(random_state),
    )
