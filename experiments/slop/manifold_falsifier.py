"""Keystone falsifier: does the joint manifold-SAE recover KNOWN manifolds as
coherence rises, scored where the blind metrics are blind?

The design session established that the joint solve fails along a single
identifiability coordinate, and that the metrics we usually report cannot see
the failure. Reconstruction R^2 is dominated by the additive sum and stays high
even when two superposed circles are split *wrongly* per token; decoder-direction
cosine matches the recovered subspace but is silent about whether each token's
coordinate landed on the right atom. The quantity that actually governs whether
the per-token split is well-posed is the conditioning of the local tangent
frame: when the active atoms' tangent directions become colinear, the share-out
of a token between atoms is underdetermined and the coordinate error blows up
even though R^2 barely moves.

This experiment plants ground truth with TWO orthogonal knobs and scores with a
SPLIT-SENSITIVE metric:

  * COHERENCE  -- two planted circles whose 2D planes share a tangent direction
    at a controllable angle theta. theta = 90deg is orthogonal/incoherent (the
    easy regime); sweeping theta -> 0 drives the planes colinear and the split
    ill-posed. Exposed as ``coherence`` in [0, 1] (0 -> orthogonal, 1 -> colinear).

  * COVERAGE   -- the co-active fraction (tokens where BOTH atoms fire, the only
    tokens that exercise the split) and a count of disambiguating-only tokens
    (exactly one atom fires, which pin each atom's gauge). Additive superposition:
        x = sum_k gate_k * amp_k * circle_k(t_k) @ plane_k.T + noise.

The score is per-token COORDINATE error up to each circle's isometry group
(``circ_procrustes_r2``: best rotation+reflection alignment of the recovered
angle onto the planted angle), plus the IDENTIFIABILITY COORDINATE itself:
``sigma_min`` of the stacked active-atom tangent matrix per co-active token
(its distribution across co-active tokens governs the split). Atoms are matched
to planted manifolds by Hungarian assignment on principal-angle subspace overlap.

The canonical assignment is IBP (``assignment="ibp"``, the gam default):
adaptive atom count with true zeros, not softmax+top_k. The fit is currently
BLOCKED at K>=2 (the multi-atom joint solve diverges upstream; fix in progress).
So the harness self-gates: if the multi-atom fit returns reconstruction_r2 < 0.5
it prints a BLOCKED line and skips recovery scoring rather than reporting
garbage. This file is the regression test that goes green when the solver fix +
the incoherence (decoder block-orthogonality) fix land.

Because the scoring must be trustworthy BEFORE the fit unblocks, ``--selftest``
scores the PLANTED coordinates against themselves (circ_procrustes_r2 ~ 1.0) and
sweeps coherence to show sigma_min decreasing toward 0 as the planes go colinear
-- proving the conditioning metric is the right identifiability coordinate.

Run:  .venv/bin/python -m experiments.manifold_falsifier --selftest
      .venv/bin/python -m experiments.manifold_falsifier   # full (self-gates on #629)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

import gamfit

DIVERGED_R2 = 0.5  # below this the multi-atom fit is treated as non-converged (gam#629)


@dataclass
class Config:
    # ground-truth data-generation knobs (the genuine experiment-design choices)
    n: int = 300                 # total tokens
    d_ambient: int = 8           # ambient activation dimension
    noise: float = 0.02          # additive Gaussian noise std
    coherence: float = 0.0       # 0 -> orthogonal planes; 1 -> colinear (ill-posed)
    coactive_fraction: float = 0.6  # fraction of tokens where BOTH atoms fire
    n_disambiguating: int = 60   # tokens where exactly ONE atom fires (gauge pins)
    seed: int = 0
    # the one fit knob that is a real choice, not a default to discover: the
    # gauge weight, verified load-bearing (isometry_weight=0 collapses periodic).
    isometry_weight: float = 0.1
    n_iter: int = 25


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _circle(t: np.ndarray) -> np.ndarray:
    """Unit circle embedding: t in [0,1] -> (cos, sin)(2*pi*t), shape (n, 2)."""
    a = 2 * np.pi * t
    return np.c_[np.cos(a), np.sin(a)]


def _circle_tangent(t: np.ndarray) -> np.ndarray:
    """d/dt circle(t) = 2*pi*(-sin, cos)(2*pi*t), shape (n, 2)."""
    a = 2 * np.pi * t
    return 2 * np.pi * np.c_[-np.sin(a), np.cos(a)]


def _coherent_planes(d: int, coherence: float, rng: np.random.Generator):
    """Two orthonormal 2D planes (each d x 2) sharing one tangent direction at
    an angle theta, with coherence in [0,1]: 0 -> theta=90deg (orthogonal
    in the shared coordinate), 1 -> theta=0 (colinear shared axis).

    Plane A is random. Plane B reuses A's first axis tilted toward A: its first
    basis vector is cos(theta)*a0' + sin(theta)*a_perp for an independent random
    a0' and a fresh perpendicular direction; coherence interpolates the angle.
    """
    theta = (1.0 - coherence) * (np.pi / 2.0)  # coherence=0 -> 90deg; =1 -> 0deg
    A = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    # B starts random, then we force its first axis to make angle theta with A[:,0].
    B = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    perp = B[:, 0] - (B[:, 0] @ A[:, 0]) * A[:, 0]
    nperp = np.linalg.norm(perp)
    perp = perp / nperp if nperp > 1e-12 else B[:, 1]
    b0 = np.cos(theta) * A[:, 0] + np.sin(theta) * perp
    b0 = b0 / np.linalg.norm(b0)
    # second axis: orthonormalize B's original second axis against b0.
    b1 = B[:, 1] - (B[:, 1] @ b0) * b0
    nb1 = np.linalg.norm(b1)
    b1 = b1 / nb1 if nb1 > 1e-12 else perp
    B = np.c_[b0, b1]
    return A, B


def circ_procrustes_r2(t_hat: np.ndarray, t_true: np.ndarray) -> float:
    """Coordinate-recovery score for a circle up to its isometry group O(2).

    Embed both angle sequences on the unit circle and find the best
    rotation+reflection (the circle's gauge freedom) aligning the recovered
    embedding onto the true one, then report 1 - residual / total. Returns ~1.0
    when t_hat equals t_true up to a constant phase shift and/or orientation
    flip, and degrades with per-token coordinate error. This is the metric the
    blind scores miss: it is sensitive to *which* token got *which* coordinate.
    """
    t_hat = np.asarray(t_hat, float).ravel()
    t_true = np.asarray(t_true, float).ravel()
    Yh = _circle(t_hat)          # (n, 2)
    Yt = _circle(t_true)         # (n, 2)
    # both are centered on the circle (mean ~0 for full coverage); center anyway
    Yh = Yh - Yh.mean(0)
    Yt = Yt - Yt.mean(0)
    # best orthogonal map R (2x2, det +-1) minimizing ||Yt - Yh R||: R = V U^T
    # from SVD of Yh^T Yt (orthogonal Procrustes, reflection allowed).
    M = Yh.T @ Yt
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    resid = float(np.sum((Yt - Yh @ R) ** 2))
    total = float(np.sum(Yt ** 2))
    if total < 1e-12:
        return float("nan")
    return 1.0 - resid / total


def tangent_sigma_min(t_a: float, t_b: float, plane_a: np.ndarray,
                      plane_b: np.ndarray) -> float:
    """Smallest singular value of the stacked active-atom tangent matrix at a
    co-active token. Each atom's ambient tangent is its planar tangent mapped
    through its plane: tangent_k = circle_tangent(t_k) @ plane_k.T, a vector in
    R^D. Stacking the active atoms column-wise gives a (D, n_active) matrix; its
    smallest singular value is 0 iff the tangents are colinear -- the split is
    then underdetermined. This is the identifiability coordinate.
    """
    ta = (_circle_tangent(np.array([t_a]))[0]) @ plane_a.T  # (D,)
    tb = (_circle_tangent(np.array([t_b]))[0]) @ plane_b.T  # (D,)
    # normalize each tangent so sigma_min measures geometric (angular)
    # conditioning, not amplitude; (-sin,cos) already has constant norm 2*pi,
    # so this only divides out the shared 2*pi scale.
    T = np.c_[ta / np.linalg.norm(ta), tb / np.linalg.norm(tb)]  # (D, 2)
    return float(np.linalg.svd(T, compute_uv=False).min())


def subspace_overlap(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean cosine of principal angles between two d-planes (orthonormal cols)."""
    sv = np.linalg.svd(P.T @ Q, compute_uv=False)
    return float(np.clip(sv, 0.0, 1.0).mean())


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def plant(cfg: Config) -> dict:
    """Plant two circles in coherent planes with controllable coverage.

    Token gating, given n total tokens:
      * ``n_disambiguating`` tokens fire exactly ONE atom (half each), pinning
        each atom's gauge in isolation;
      * of the remainder, ``coactive_fraction`` fire BOTH atoms (the tokens that
        exercise the split), the rest fire one atom at random.
    Amplitudes are random positive gains; superposition is additive.
    """
    rng = np.random.default_rng(cfg.seed)
    A, B = _coherent_planes(cfg.d_ambient, cfg.coherence, rng)
    planes = [A, B]

    n = cfg.n
    t = rng.uniform(0.0, 1.0, (2, n))           # planted coordinate per atom
    amp = np.zeros((n, 2))
    gate = np.zeros((n, 2), dtype=bool)

    idx = rng.permutation(n)
    n_dis = min(cfg.n_disambiguating, n)
    dis = idx[:n_dis]
    rest = idx[n_dis:]
    # disambiguating-only tokens: alternate which single atom fires
    for j, i in enumerate(dis):
        gate[i, j % 2] = True
    # remainder: coactive_fraction fire both, the others fire one at random
    n_co = int(round(cfg.coactive_fraction * len(rest)))
    co = rest[:n_co]
    single = rest[n_co:]
    gate[co, :] = True
    for i in single:
        gate[i, rng.integers(0, 2)] = True

    amp[gate] = 0.6 + 0.8 * rng.uniform(size=int(gate.sum()))

    X = np.zeros((n, cfg.d_ambient))
    for k in range(2):
        contrib = (_circle(t[k]) @ planes[k].T)        # (n, D)
        X += (amp[:, k] * gate[:, k])[:, None] * contrib
    X += cfg.noise * rng.standard_normal(X.shape)

    coactive = gate.all(axis=1)
    return dict(X=X, planes=planes, t=t, gate=gate, amp=amp, coactive=coactive)


# ---------------------------------------------------------------------------
# sigma_min distribution over co-active tokens
# ---------------------------------------------------------------------------

def coactive_sigma_min(gt: dict) -> np.ndarray:
    """sigma_min of the active-atom tangent frame at every co-active token."""
    t, planes, coactive = gt["t"], gt["planes"], gt["coactive"]
    out = []
    for i in np.flatnonzero(coactive):
        out.append(tangent_sigma_min(t[0, i], t[1, i], planes[0], planes[1]))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Fit + split-sensitive recovery scoring (gated on gam#629)
# ---------------------------------------------------------------------------

def fit(X: np.ndarray, cfg: Config):
    # CANONICAL assignment = IBP (adaptive atom count, true zeros), the gam
    # default and the design-decided answer. NOT softmax+top_k.
    return gamfit.sae_manifold_fit(
        X, K=2, d_atom=1, atom_topology="circle", assignment="ibp",
        ard_per_atom=False, alpha="auto",
        sparsity_weight=0.01, smoothness_weight=0.01,
        isometry_weight=cfg.isometry_weight, learning_rate=1.0,
        n_iter=cfg.n_iter, random_state=cfg.seed)


def _atom_plane(atom) -> np.ndarray:
    """Recover an atom's 2D ambient plane from its periodic decoder. The circle
    harmonic design is [1, cos(2*pi*t), sin(2*pi*t)] (3 coeffs, matching the
    decoder_coefficients row count); B @ coeffs gives the per-atom ambient
    reconstruction whose top-2 right singular vectors span the recovered plane.
    """
    coeffs = np.asarray(atom.decoder_coefficients)          # (3, D)
    tc = np.asarray(atom.coords)[:, 0]
    Bdes = np.c_[np.ones_like(tc), np.cos(2 * np.pi * tc), np.sin(2 * np.pi * tc)]
    recon = Bdes @ coeffs
    recon = recon - recon.mean(0)
    _, _, Vt = np.linalg.svd(recon, full_matrices=False)
    return Vt[:2].T                                          # (D, 2)


def score_recovery(model, gt: dict) -> dict:
    """Match recovered atoms to planted manifolds (Hungarian on subspace
    overlap), then score per-token coordinate recovery up to each circle's
    isometry group, evaluated on the tokens where that atom actually fired."""
    planes_true = gt["planes"]
    atoms = list(model.atoms)
    planes_hat = [_atom_plane(a) for a in atoms]
    coords_hat = [np.asarray(a.coords)[:, 0] for a in atoms]

    # cost = -overlap so Hungarian maximizes subspace agreement
    K = len(planes_true)
    cost = np.zeros((K, len(atoms)))
    for i in range(K):
        for j in range(len(atoms)):
            cost[i, j] = -subspace_overlap(planes_true[i], planes_hat[j])
    row, col = linear_sum_assignment(cost)

    per_atom = []
    for i, j in zip(row, col):
        fired = gt["gate"][:, i]
        # convert recovered circle coordinate to a t in [0,1) for the metric
        t_hat = (coords_hat[j][fired] % 1.0)
        t_true = gt["t"][i][fired]
        r2 = circ_procrustes_r2(t_hat, t_true)
        per_atom.append(dict(planted=i, atom=j,
                             overlap=-cost[i, j], coord_r2=r2,
                             n_fired=int(fired.sum())))
    return dict(matches=per_atom)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_selftest(cfg: Config) -> None:
    print("=== SELFTEST: scoring functions on ground truth (no fit) ===\n")

    # (a) circ_procrustes_r2 of planted coords against themselves, and against
    #     a gauge-transformed copy (phase shift + reflection) -> must be ~1.0;
    #     against shuffled (wrong per-token split) -> must collapse.
    gt = plant(cfg)
    t0 = gt["t"][0]
    rng = np.random.default_rng(cfg.seed + 1)
    self_r2 = circ_procrustes_r2(t0, t0)
    gauged = (-t0 + 0.37) % 1.0                       # reflection + phase shift
    gauge_r2 = circ_procrustes_r2(gauged, t0)
    shuffled = rng.permutation(t0)                     # destroys per-token split
    shuf_r2 = circ_procrustes_r2(shuffled, t0)
    print("  circ_procrustes_r2 (split-sensitive coordinate metric):")
    print(f"    self (t vs t)                  = {self_r2:.4f}   (expect ~1.0)")
    print(f"    gauge-transformed (flip+shift) = {gauge_r2:.4f}   (expect ~1.0; isometry-invariant)")
    print(f"    shuffled (wrong per-token)     = {shuf_r2:.4f}   (expect << 1; blind metrics miss this)")

    # (b) sigma_min across a coherence sweep -> must DECREASE toward 0 as the
    #     planes go colinear (coherence -> 1, theta -> 0).
    print("\n  sigma_min of the active-atom tangent frame across co-active tokens")
    print("  (the identifiability coordinate; theta = (1-coherence)*90deg):\n")
    print(f"    {'coherence':>9s} {'theta_deg':>9s} {'sigma_min_med':>13s} {'sigma_min_p10':>13s}")
    meds = []
    for coh in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99):
        g = plant(Config(**{**vars(cfg), "coherence": coh}))
        sm = coactive_sigma_min(g)
        med = float(np.median(sm))
        p10 = float(np.percentile(sm, 10))
        meds.append(med)
        theta_deg = (1.0 - coh) * 90.0
        print(f"    {coh:9.2f} {theta_deg:9.1f} {med:13.4f} {p10:13.4f}")
    decreasing = all(meds[i] >= meds[i + 1] - 1e-9 for i in range(len(meds) - 1))

    ok_self = self_r2 > 0.999 and gauge_r2 > 0.999 and shuf_r2 < 0.5
    print("\n  CHECKS:")
    print(f"    [{'PASS' if ok_self else 'FAIL'}] coordinate metric is isometry-invariant & split-sensitive")
    print(f"    [{'PASS' if decreasing else 'FAIL'}] median sigma_min decreases monotonically as coherence -> 1 (colinear)")
    if ok_self and decreasing:
        print("\n  SELFTEST PASSED: scoring is trustworthy before the fit unblocks.")
    else:
        print("\n  SELFTEST FAILED.")


def run_full(cfg: Config) -> None:
    print("=== FULL: plant -> joint fit (K=2) -> split-sensitive recovery ===\n")
    gt = plant(cfg)
    sm = coactive_sigma_min(gt)
    n_co = int(gt["coactive"].sum())
    print(f"  data: n={cfg.n}, D={cfg.d_ambient}, coherence={cfg.coherence}, "
          f"theta={ (1-cfg.coherence)*90:.1f}deg, noise={cfg.noise}")
    print(f"  coverage: {n_co} co-active tokens, "
          f"{cfg.n_disambiguating} disambiguating-only tokens")
    print(f"  conditioning: sigma_min over co-active tokens  "
          f"median={np.median(sm):.4f}  p10={np.percentile(sm,10):.4f}\n")

    model = fit(gt["X"], cfg)
    r2 = float(model.reconstruction_r2)
    print(f"  fit reconstruction_r2 = {r2:.4f}")

    if r2 < DIVERGED_R2:
        print("\n  -> BLOCKED on gam#629 (multi-atom fit diverged): the cold-start")
        print("     assignment logits init to a uniform symmetric saddle, so the K=2")
        print("     joint solve collapses. Skipping recovery scoring (would be garbage).")
        print("     This harness is the regression test; it goes green once #629 and the")
        print("     incoherence fix land. Conditioning (sigma_min) above is fit-independent.")
        return

    res = score_recovery(model, gt)
    print("\n  split-sensitive recovery (Hungarian-matched atoms):")
    print(f"    {'planted':>7s} {'atom':>4s} {'overlap':>7s} {'coord_r2':>8s} {'n_fired':>7s}")
    for m in res["matches"]:
        print(f"    {m['planted']:7d} {m['atom']:4d} {m['overlap']:7.3f} "
              f"{m['coord_r2']:8.3f} {m['n_fired']:7d}")
    worst = min(m["coord_r2"] for m in res["matches"])
    print(f"\n  worst-atom coordinate R2 = {worst:.3f}  "
          f"(this is what tracks sigma_min, not reconstruction_r2={r2:.3f})")


def _add_args(p: argparse.ArgumentParser) -> None:
    for f, v in vars(Config()).items():
        p.add_argument(f"--{f}", type=type(v), default=v)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="score planted coords against themselves + sigma_min "
                        "coherence sweep (runs now; no fit needed)")
    _add_args(p)
    args = p.parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items() if k != "selftest"})
    print(f"gamfit {gamfit.__version__}  |  {cfg}\n")
    if args.selftest:
        run_selftest(cfg)
    else:
        run_full(cfg)


if __name__ == "__main__":
    main()
