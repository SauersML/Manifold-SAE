"""Benchmark for HIGHER-DIM manifold features (2D surfaces, 3D manifolds).

Same setup as recover_bench but each feature is an intrinsic-d manifold (d=2 or 3)
embedded in its own orthogonal block of R^D. Each token is one point on one
manifold + noise. A method returns, per feature, a point cloud sampling the
recovered manifold; error is symmetric Chamfer (gauge-free for surfaces — no 1D
ordering/reparam exists) normalized by the GT diameter.
"""
from __future__ import annotations

import numpy as np

from manifold_sae.data_synthetic import chamfer_distance


def manifold_library_2d(n=900):
    """Intrinsic-2D surfaces, each as (n, d_intrinsic_embedding) points."""
    rng = np.random.default_rng(0)
    uv = rng.random((n, 2))
    u, v = uv[:, 0], uv[:, 1]
    C = lambda A: A - A.mean(0)
    return {
        "sphere": C(np.c_[np.sin(np.pi * u) * np.cos(2 * np.pi * v),
                          np.sin(np.pi * u) * np.sin(2 * np.pi * v),
                          np.cos(np.pi * u)]),                         # 2D surface in R^3
        "torus": C(np.c_[(2 + np.cos(2 * np.pi * v)) * np.cos(2 * np.pi * u),
                         (2 + np.cos(2 * np.pi * v)) * np.sin(2 * np.pi * u),
                         np.sin(2 * np.pi * v)]),                      # 2D surface in R^3
        "saddle": C(np.c_[2 * u - 1, 2 * v - 1, (2 * u - 1) ** 2 - (2 * v - 1) ** 2]),
        "swiss_roll": C(np.c_[(0.5 + u) * np.cos(4 * np.pi * u),
                              2 * v,
                              (0.5 + u) * np.sin(4 * np.pi * u)]),     # 2D manifold in R^3
    }


def manifold_library_3d(n=1200):
    """Intrinsic-3D manifolds embedded in R^4."""
    rng = np.random.default_rng(1)
    uvw = rng.random((n, 3))
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    C = lambda A: A - A.mean(0)
    return {
        "3sphere_patch": C(np.c_[np.sin(np.pi * u) * np.cos(np.pi * v),
                                 np.sin(np.pi * u) * np.sin(np.pi * v) * np.cos(np.pi * w),
                                 np.sin(np.pi * u) * np.sin(np.pi * v) * np.sin(np.pi * w),
                                 np.cos(np.pi * u)]),
        "cube_warp": C(np.c_[2 * u - 1, 2 * v - 1, 2 * w - 1,
                             (2 * u - 1) * (2 * v - 1) + 0.5 * (2 * w - 1) ** 2]),
    }


def orthogonal_blocks(dims, D, seed):
    """One row-orthonormal block per feature, mutually orthogonal. `dims[i]` is the
    embedding dim of feature i (3 for the R^3 surfaces, 4 for the R^4 3-manifolds)."""
    total = sum(dims)
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, total)))
    out, c = [], 0
    for d in dims:
        out.append(Q[:, c:c + d].T)  # (d, D)
        c += d
    return out


def make_data(library, D=24, N=4000, noise=0.02, seed=0):
    names = list(library)
    dims = [library[n].shape[1] for n in names]
    rng = np.random.default_rng(seed)
    B = orthogonal_blocks(dims, D, seed + 100)
    gts = [library[n] @ B[i] for i, n in enumerate(names)]
    counts = [len(library[n]) for n in names]
    lab = rng.integers(0, len(names), N)
    X = np.zeros((N, D))
    for t in range(N):
        i = lab[t]
        X[t] = library[names[i]][rng.integers(0, counts[i])] @ B[i]
    return X + noise * rng.standard_normal((N, D)), gts, names


def evaluate(recover, kind="2d", seeds=(0,), D=24, N=4000, noise=0.02, verbose=True):
    lib = manifold_library_2d() if kind == "2d" else manifold_library_3d()
    names = list(lib)
    seed_means = []
    for seed in seeds:
        X, gts, _ = make_data(lib, D=D, N=N, noise=noise, seed=seed)
        clouds = recover(X, len(names))
        used, errs = set(), []
        for i in range(len(names)):
            cand = [k for k in range(len(clouds)) if k not in used and clouds[k] is not None]
            if not cand:
                errs.append(2.0); continue
            best = min(cand, key=lambda k: chamfer_distance(gts[i], clouds[k]))
            used.add(best)
            diam = np.linalg.norm(gts[i] - gts[i].mean(0), axis=1).max() * 2 + 1e-12
            errs.append(chamfer_distance(gts[i], clouds[best]))  # already scale-normalized in chamfer
        seed_means.append(float(np.mean(errs)))
        if verbose:
            print(f"  seed {seed}: " + " ".join(f"{n[:6]}={100*e:.1f}" for n, e in zip(names, errs))
                  + f"  MEAN={100*np.mean(errs):.1f}%", flush=True)
    m = float(np.mean(seed_means))
    if verbose:
        print(f"OVERALL ({kind}) mean={100*m:.2f}%", flush=True)
    return m
