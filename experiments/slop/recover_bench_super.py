"""Superposed benchmark — the hard case real LLMs actually have.

Unlike recover_bench (one feature per token, orthogonal blocks), here each token
is a SPARSE SUM of several active features, the feature subspaces are NON-
orthogonal (optionally overcomplete: 2K > D), and the amplitude is multiplicative
and sparse:  x = sum_{k active} a_k * g_k(t_k) + noise,  a_k >= 0.

This is the regime where moment / co-activation / ISA methods are expected to
fail (the covariance is no longer block-diagonal, co-activation is dense, and ISA
needs independence + undercompleteness). It is the regime an SAE exists for.
"""
from __future__ import annotations

import numpy as np

from experiments.shape_metrics import rel_rms
from experiments.recover_bench import shape_library
from manifold_sae.data_synthetic import chamfer_distance


def _subspaces(K, D, orthogonal, seed):
    rng = np.random.default_rng(seed + 100)
    if orthogonal:
        Q, _ = np.linalg.qr(rng.standard_normal((D, 2 * K)))
        return [Q[:, 2 * i:2 * i + 2].T for i in range(K)]   # (2,D) orthonormal, disjoint
    # non-orthogonal: each feature an independent random 2D subspace (overlap if 2K>D)
    blocks = []
    for _ in range(K):
        B, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        blocks.append(B.T)                                    # (2,D) row-orthonormal, NOT mutually orth
    return blocks


def make_data(shapes, D=16, N=4000, n_active=3, noise=0.03, multiplicative=True,
              orthogonal=False, seed=0):
    names = list(shapes)
    K = len(names)
    rng = np.random.default_rng(seed)
    B = _subspaces(K, D, orthogonal, seed)
    gts = [shapes[n] @ B[i] for i, n in enumerate(names)]     # GT curves in R^D
    Tn = len(shapes[names[0]])
    X = np.zeros((N, D))
    for t in range(N):
        n_a = max(1, rng.poisson(n_active))                  # how many features fire
        active = rng.choice(K, size=min(n_a, K), replace=False)
        for i in active:
            a = abs(rng.standard_normal()) if multiplicative else 1.0
            X[t] += a * (shapes[names[i]][rng.integers(0, Tn)] @ B[i])
    X += noise * rng.standard_normal((N, D))
    return X + 0.0, gts, names


def evaluate(recover, seeds=(0, 1, 2), D=16, N=4000, n_active=3, noise=0.03,
             multiplicative=True, orthogonal=False, verbose=True):
    shapes = shape_library()
    names = list(shapes)
    seed_means = []
    for seed in seeds:
        X, gts, _ = make_data(shapes, D=D, N=N, n_active=n_active, noise=noise,
                              multiplicative=multiplicative, orthogonal=orthogonal, seed=seed)
        curves = recover(X, len(names))
        used, errs = set(), []
        for i in range(len(names)):
            cand = [k for k in range(len(curves)) if k not in used and curves[k] is not None]
            if not cand:
                errs.append(2.0); continue
            best = min(cand, key=lambda k: chamfer_distance(gts[i], curves[k]))
            used.add(best)
            errs.append(rel_rms(gts[i], curves[best]))
        seed_means.append(float(np.mean(errs)))
        if verbose:
            print(f"  seed {seed}: " + " ".join(f"{n[:4]}={100*e:.0f}" for n, e in zip(names, errs))
                  + f"  MEAN={100*np.mean(errs):.1f}%", flush=True)
    m = float(np.mean(seed_means))
    if verbose:
        cfg = f"n_active={n_active} orthogonal={orthogonal} mult={multiplicative} D={D}"
        print(f"OVERALL mean={100*m:.1f}%  ({cfg})", flush=True)
    return m
