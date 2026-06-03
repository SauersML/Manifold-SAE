"""Gauge-free shape-fidelity metric for recovered 1D curves.

A recovered curve is only defined up to: reparameterization, curve reversal, a
cyclic phase shift (closed curves), and a similarity transform within its
subspace (rotation/reflection/scale — the SAE can't fix a curve's orientation or
overall scale). ``rel_rms`` quotients all of these out: it arc-length-resamples
both curves, then minimizes RMS over {reversal, cyclic phase, Procrustes}, and
normalizes by the ground-truth diameter. ~0.01 means ~1% (near pixel-perfect).

Searching cyclic phase is essential for closed non-symmetric curves (an ellipse
resampled from a different start point cannot be aligned by rotation alone) — a
phase-blind metric reports large spurious errors for them.
"""
from __future__ import annotations

import numpy as np


def _arclen(C, M, closed=False):
    if closed:
        C = np.vstack([C, C[:1]])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if d[-1] < 1e-12:
        return np.repeat(C[:1], M, 0)
    return np.stack([np.interp(np.linspace(0, d[-1], M, endpoint=not closed), d, C[:, j])
                     for j in range(C.shape[1])], 1)


def _procrustes_rms(G, L):
    Gc = G - G.mean(0)
    Lc = L - L.mean(0)
    H = Lc.T @ Gc
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    s = S.sum() / max((Lc * Lc).sum(), 1e-12)
    return np.sqrt(((Gc - (Lc * s) @ R) ** 2).sum(1).mean())


def rel_rms(gt, learned, M=240, n_phase=40):
    """Relative RMS shape error (fraction of GT diameter), gauge-free over
    reversal + cyclic phase + similarity transform."""
    G = _arclen(gt, M)
    diam = np.linalg.norm(G - G.mean(0), axis=1).max() * 2 + 1e-12
    L0 = _arclen(learned, M)
    best = np.inf
    step = max(M // n_phase, 1)
    for Lr in (L0, L0[::-1]):
        for sh in range(0, M, step):
            best = min(best, _procrustes_rms(G, np.roll(Lr, sh, axis=0)))
    return best / diam
