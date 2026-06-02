"""Shared benchmark for manifold-recovery methods.

A method is a function ``recover(X, K) -> list[np.ndarray]`` returning K curves
(each ``(G, D)`` polyline in the ambient space). ``evaluate`` plants a diverse
bag of shapes in orthogonal subspaces, samples noisy points (single active
feature per token), runs the method, matches atoms to ground truth, and reports
phase-aware relative-RMS shape error.

CONSTRAINTS for methods benchmarked here: NO clustering (k-means / K-subspaces),
NO k-NN graphs, NO trained SAE/encoder. Use principled global structure
(algebraic / moment / spectral-of-moments / harmonic / optimal-transport / ...).
"""
from __future__ import annotations

import numpy as np

from experiments.shape_metrics import rel_rms
from manifold_sae.data_synthetic import chamfer_distance


def shape_library(n=400):
    T = np.linspace(0, 1, n, endpoint=False)
    th = 2 * np.pi * T
    u = 2 * T - 1
    C = lambda A: A - A.mean(0)
    return {
        "ellipse": C(np.c_[2 * np.cos(th), np.sin(th)]),
        "offset_circle": C(np.c_[2.2 + np.cos(th), np.sin(th)]),
        "cardioid": C(np.c_[(1 + 0.6 * np.cos(th)) * np.cos(th), (1 + 0.6 * np.cos(th)) * np.sin(th)]),
        "Scurve": C(np.c_[u, np.tanh(3 * u)]),
        "parabola": C(np.c_[u, u ** 2 - 0.33]),
        "spiral": C(np.c_[(0.3 + T) * np.cos(4 * np.pi * T), (0.3 + T) * np.sin(4 * np.pi * T)]),
        "semicircle": C(np.c_[np.cos(np.pi * T), np.sin(np.pi * T)]),
        "wave": C(np.c_[u, 0.5 * np.sin(3 * np.pi * T)]),
    }


def orthogonal_blocks(K, D, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, 2 * K)))
    return [Q[:, 2 * i:2 * i + 2].T for i in range(K)]   # each (2, D), mutually orthogonal


def make_data(shapes, D=20, N=2400, noise=0.015, seed=0):
    names = list(shapes)
    rng = np.random.default_rng(seed)
    B = orthogonal_blocks(len(names), D, seed + 100)
    gts = [shapes[n] @ B[i] for i, n in enumerate(names)]
    Tn = len(shapes[names[0]])
    lab = rng.integers(0, len(names), N)
    X = np.zeros((N, D))
    for t in range(N):
        i = lab[t]
        X[t] = shapes[names[i]][rng.integers(0, Tn)] @ B[i]
    return X + noise * rng.standard_normal((N, D)), gts, names


def evaluate(recover, seeds=(0, 1, 2), D=20, N=2400, noise=0.015, verbose=True):
    """recover(X, K) -> list of K (G,D) curves. Returns mean rel-RMS over seeds."""
    shapes = shape_library()
    names = list(shapes)
    seed_means = []
    for seed in seeds:
        X, gts, _ = make_data(shapes, D=D, N=N, noise=noise, seed=seed)
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
            print(f"  seed {seed}: " + " ".join(f"{n[:4]}={100*e:.1f}" for n, e in zip(names, errs))
                  + f"  MEAN={100*np.mean(errs):.1f}%", flush=True)
    m = float(np.mean(seed_means))
    if verbose:
        print(f"OVERALL mean={100*m:.2f}%  worst-seed={100*max(seed_means):.1f}%", flush=True)
    return m
