"""RUN 4 -- per-atom TRUST SCORE + LEVEL-0 misspecification (the honest deliverable).

A typed manifold atom (e.g. "this region is a circle with coordinate t") is only
worth reporting if (a) its per-token split is well-conditioned, (b) it is not
geometrically confusable with a neighbour, (c) the data actually supports its
declared topology, (d) it is covered/visited enough to pin its gauge, and (e) a
type-agnostic reference cannot reconstruct it dramatically better (i.e. it is not
misspecified). The TRUST SCORE folds these into one number in [0, 1], and the
LEVEL-0 misspecification test flags atoms that should be reported as UNTYPED
rather than forced onto the typed menu.

EVERYTHING HERE IS SOLVER-INDEPENDENT. The five trust ingredients and the
Level-0 reference are computed from the activation points and the (known or
nonparametrically-estimated) local geometry -- they do NOT require the typed
``sae_manifold_fit`` to converge. This matters because the gam K>=1 joint solve
is currently BLOCKED (RemlConvergenceError on the K=1 circle smoke, gamfit
0.1.151), so the deliverable is authored and calibrated on synthetic ground
truth and on real LLM activations using a NONPARAMETRIC coordinate estimator in
place of the (blocked) typed fit. When the solver unblocks, the same trust score
consumes the typed atoms' coords/planes directly (see ``trust_from_typed_atom``).

Trust ingredients (each mapped to [0,1], 1 = trustworthy):
  sigma_min      conditioning of the local tangent frame at co-active points.
  coherence      1 - max subspace overlap with other atoms (incoherence is good).
  topo_margin    evidence the declared topology beats the next-best alternative.
  coverage       fraction of the coordinate period visited * a sample-count floor.
  level0         1 - (typed_residual - nonparam_residual) gap (misspecification).

TRUST = geometric mean of the five sub-scores (a single weak link tanks trust;
that is the intended behaviour -- an atom is only as trustworthy as its weakest
qualification). Each sub-score is reported alongside the aggregate so the failure
mode is legible.

CALIBRATION (the whole point):
  * SYNTHETIC (known truth): LOW trust must predict HIGH coordinate error. We
    plant circles spanning the conditioning/coverage/type axes, score trust, and
    correlate trust vs a nonparametric coordinate-recovery error against planted
    truth (Spearman, plus a low-vs-high-trust error contrast).
  * REAL (LLM activations): LOW trust must predict cross-seed DISAGREEMENT. We
    harvest real activations for cyclic vs linear categories, bootstrap the
    nonparametric coordinate estimate across seeds, and correlate trust vs
    cross-seed coordinate disagreement.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Geometry primitives (shared semantics with manifold_falsifier.circ_procrustes_r2
# and tangent_sigma_min, re-expressed here so the module is self-contained and
# runnable even where the falsifier's gamfit import would pull in the blocked
# solver path).
# ---------------------------------------------------------------------------

def _circle(t: np.ndarray) -> np.ndarray:
    a = 2 * np.pi * np.asarray(t, float).ravel()
    return np.c_[np.cos(a), np.sin(a)]


def circ_procrustes_r2(t_hat, t_true) -> float:
    """Coordinate recovery for a circle up to its isometry group O(2): best
    rotation+reflection aligning recovered angles onto true ones, 1 - resid/total.
    Split-sensitive (which token got which coordinate), unlike reconstruction R2.
    (Identical semantics to experiments.manifold_falsifier.circ_procrustes_r2.)"""
    Yh = _circle(t_hat); Yt = _circle(t_true)
    Yh = Yh - Yh.mean(0); Yt = Yt - Yt.mean(0)
    U, _, Vt = np.linalg.svd(Yh.T @ Yt)
    R = U @ Vt
    resid = float(np.sum((Yt - Yh @ R) ** 2))
    total = float(np.sum(Yt ** 2))
    return float("nan") if total < 1e-12 else 1.0 - resid / total


def subspace_overlap(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean cosine of principal angles between two subspaces (orthonormal cols)."""
    sv = np.linalg.svd(P.T @ Q, compute_uv=False)
    return float(np.clip(sv, 0.0, 1.0).mean())


def local_pca_plane(P: np.ndarray, d: int = 2) -> np.ndarray:
    """Top-d right singular vectors of the mean-removed point cloud -> the local
    tangent plane the data lives in, with NO type prior (the Level-0 frame)."""
    Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    return Vt[:d].T  # (D, d)


def nonparam_circle_coord(P: np.ndarray) -> np.ndarray:
    """Type-AGNOSTIC coordinate estimate: angle in the local-PCA 2-plane. This is
    the stand-in for the (blocked) typed solver's per-token coordinate -- it uses
    only PCA, no circle prior beyond reading off an angle in the dominant plane.
    Returns t in [0,1)."""
    plane = local_pca_plane(P, 2)
    proj = (P - P.mean(0)) @ plane            # (n, 2)
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    return (ang / (2 * np.pi)) % 1.0


# ---------------------------------------------------------------------------
# The five trust ingredients
# ---------------------------------------------------------------------------

def plane_conditioning(P: np.ndarray) -> float:
    """sigma_2 / sigma_1 of the mean-removed cloud: the conditioning of the
    embedding 2-plane the typed circle lives in. =1 for a perfectly circular ring
    (the two harmonic directions are equally expressed, so the angular coordinate
    is well-determined everywhere); ->0 for an eccentric/near-1D embedding, where
    one harmonic dominates and the angle is ill-conditioned near the thin axis.

    This is the single-atom analogue of tangent_sigma_min: there the smallest
    singular value of the STACKED co-active tangent frame governs the per-token
    split; for a lone typed atom the split that must be well-posed is the angle
    within its own plane, whose conditioning is exactly this ratio. (When the
    typed solver unblocks and a co-active partner exists, trust_from_typed_atom
    feeds the stacked-frame sigma_min through this same slot.)"""
    s = np.linalg.svd(P - P.mean(0), compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 1e-12 else 0.0


def sub_sigma_min(P: np.ndarray) -> float:
    """Conditioning sub-score in [0,1]: the embedding-plane conditioning. A clean
    circular ring -> ~1; an eccentric / near-colinear region -> ~0."""
    return float(np.clip(plane_conditioning(P), 0.0, 1.0))


def sub_coherence(plane: np.ndarray, neighbor_planes) -> float:
    """Incoherence sub-score in [0,1]: 1 - max subspace overlap with neighbours.
    1 = orthogonal to every neighbour (identifiable split); 0 = shares a full
    subspace with some neighbour (the split with it is ill-posed)."""
    if not neighbor_planes:
        return 1.0
    worst = max(subspace_overlap(plane, Q) for Q in neighbor_planes)
    return float(np.clip(1.0 - worst, 0.0, 1.0))


def sub_topo_margin(P: np.ndarray) -> float:
    """Topology-evidence margin in [0,1]: does the data support a CLOSED loop
    (circle) over an open arc/line? We read the angular coverage in the local
    plane and the second-eigenvalue mass. A true ring fills the angle uniformly
    AND has strong 2D mass; a line fills a narrow angular wedge / has tiny second
    eigenvalue. Margin = min(angular-uniformity, twoD-mass-ratio)."""
    plane = local_pca_plane(P, 2)
    Pc = P - P.mean(0)
    proj = Pc @ plane
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    # angular uniformity via resultant length R: R~0 => uniform ring (good),
    # R~1 => concentrated wedge/line (bad). uniformity = 1 - R.
    R = np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))
    uniformity = 1.0 - R
    # 2D mass: ratio of second to first singular value of the full cloud.
    s = np.linalg.svd(Pc, full_matrices=False)[1]
    twoD = s[1] / s[0] if s[0] > 1e-12 else 0.0
    return float(np.clip(min(uniformity, twoD), 0.0, 1.0))


def sub_coverage(t_est: np.ndarray, n_floor: int = 60) -> float:
    """Coverage/frequency sub-score in [0,1]: how much of the [0,1) period is
    visited (binned occupancy) times a soft sample-count floor. A region seen
    over only 90deg, or with very few tokens, scores low -- its gauge is
    underdetermined even if locally well-conditioned."""
    n = len(t_est)
    bins = 24
    occ = len(np.unique(np.clip((t_est * bins).astype(int), 0, bins - 1))) / bins
    count = min(1.0, n / n_floor)
    return float(np.clip(occ, 0.0, 1.0) * count)


def sub_level0(P: np.ndarray, typed_recon: np.ndarray | None = None) -> tuple:
    """Level-0 misspecification sub-score in [0,1] AND the untyped flag.

    Fit a NONPARAMETRIC reference (local-PCA 2-plane, no type prior) and a TYPED
    reconstruction (circle harmonics on the nonparametric angle). Compare residual
    fractions. If the typed model reconstructs MUCH worse than the flexible
    reference, the atom is misspecified -> flag UNTYPED and tank the sub-score.

    typed_recon (optional): when the real typed solver is available, pass its
    per-point reconstruction to compare against the nonparam reference instead of
    the harmonic stand-in. (Blocked today -> stand-in used.)"""
    Pc = P - P.mean(0)
    total = float(np.sum(Pc ** 2)) + 1e-12
    # nonparametric reference: project onto local 2-plane (best rank-2, no type).
    plane = local_pca_plane(P, 2)
    ref = (Pc @ plane) @ plane.T
    ref_resid = float(np.sum((Pc - ref) ** 2)) / total
    # typed reconstruction: circle harmonics [1,cos,sin] regressed on nonparam t.
    if typed_recon is None:
        t = nonparam_circle_coord(P)
        Bdes = np.c_[np.ones_like(t), np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]
        coef, *_ = np.linalg.lstsq(Bdes, Pc, rcond=None)
        typed = Bdes @ coef
    else:
        typed = typed_recon - typed_recon.mean(0)
    typed_resid = float(np.sum((Pc - typed) ** 2)) / total
    # misspecification: how many times worse the typed (circle) reconstruction is
    # than the type-AGNOSTIC rank-2 reference. A well-specified ring leaves about
    # as little residual as the free 2-plane (ratio ~1-3, both ~noise floor). A 2D
    # blob, a line, or an eccentric ellipse all leave the typed model many times
    # worse because a single angle cannot capture their extra (radial / linear)
    # degree of freedom. The ratio is scale-robust where an absolute gap is not.
    ratio = typed_resid / (ref_resid + 1e-9)
    gap = max(0.0, typed_resid - ref_resid)
    untyped = ratio > 4.0  # typed >4x worse than the free 2-plane -> misspecified
    # sub-score decays with log-ratio so trust drops smoothly as type gets worse.
    score = 1.0 / (1.0 + max(0.0, np.log(max(ratio, 1.0)) / np.log(4.0)))
    return float(np.clip(score, 0.0, 1.0)), bool(untyped), dict(
        ref_resid=ref_resid, typed_resid=typed_resid, gap=gap, ratio=float(ratio))


# ---------------------------------------------------------------------------
# Aggregate trust
# ---------------------------------------------------------------------------

@dataclass
class TrustReport:
    trust: float
    sigma_min: float
    coherence: float
    topo_margin: float
    coverage: float
    level0: float
    untyped: bool
    detail: dict


def trust_score(P: np.ndarray, neighbor_planes=None, typed_recon=None) -> TrustReport:
    """Per-atom trust from the five solver-independent ingredients. P is the atom's
    ambient point cloud (the tokens the atom fired on). neighbor_planes is the list
    of OTHER atoms' local planes (for the coherence/incoherence term)."""
    plane = local_pca_plane(P, 2)
    t_est = nonparam_circle_coord(P)
    s_sig = sub_sigma_min(P)
    s_coh = sub_coherence(plane, neighbor_planes or [])
    s_topo = sub_topo_margin(P)
    s_cov = sub_coverage(t_est)
    s_l0, untyped, l0det = sub_level0(P, typed_recon)
    subs = np.array([s_sig, s_coh, s_topo, s_cov, s_l0])
    # geometric mean: a single weak link tanks trust (intended).
    trust = float(np.exp(np.mean(np.log(np.clip(subs, 1e-6, 1.0)))))
    return TrustReport(trust, s_sig, s_coh, s_topo, s_cov, s_l0, untyped,
                       dict(level0=l0det))


def trust_from_typed_atom(atom, P, neighbor_planes=None) -> TrustReport:
    """Once the typed solver unblocks: build trust from a real fitted atom. Uses
    the atom's decoder plane + coords for coherence/coverage, and the atom's own
    per-point reconstruction for the Level-0 gap. (Wired but not exercised today
    because the K>=1 joint fit is BLOCKED in gamfit 0.1.151.)"""
    coeffs = np.asarray(atom.decoder_coefficients)
    tc = np.asarray(atom.coords)[:, 0]
    Bdes = np.c_[np.ones_like(tc), np.cos(2 * np.pi * tc), np.sin(2 * np.pi * tc)]
    typed_recon = Bdes @ coeffs
    return trust_score(P, neighbor_planes, typed_recon=typed_recon)
