"""Unsupervised recovery of an arbitrary bag of shapes — end-to-end demo.

Plants a diverse set of 1D shapes (open & closed, star & non-star: ellipse,
offset-circle, cardioid, S-curve, parabola, spiral, semicircle, sine wave) in
mutually-orthogonal subspaces, samples points on them with additive noise (no
labels), and recovers every curve with the shape-agnostic graph method
(``graph_curve.recover_manifolds``: K-subspaces clustering + MST ordering +
topology-from-MST + spline). Reports phase-aware relative-RMS error and plots.

Run: ``python -m experiments.unsup_demo``  (or import ``main``).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from experiments.graph_curve import recover_manifolds
from experiments.shape_metrics import rel_rms
from manifold_sae.data_synthetic import chamfer_distance

torch.set_default_dtype(torch.float64)


def shape_library(n=400):
    """A diverse bag of 2D shapes (each centered)."""
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


def _orthogonal_blocks(K, D, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, 2 * K)))
    return [Q[:, 2 * i:2 * i + 2].T for i in range(K)]   # each (2, D), mutually orthogonal


def make_data(shapes, D=20, N=2400, noise=0.015, seed=0):
    """Each token = a point on one randomly chosen shape, lifted to its block,
    plus Gaussian noise. Returns (X, ground_truth_curves)."""
    names = list(shapes)
    rng = np.random.default_rng(seed)
    B = _orthogonal_blocks(len(names), D, seed + 100)
    gts = [shapes[n] @ B[i] for i, n in enumerate(names)]
    Tn = len(shapes[names[0]])
    lab = rng.integers(0, len(names), N)
    X = np.zeros((N, D))
    for t in range(N):
        i = lab[t]
        X[t] = shapes[names[i]][rng.integers(0, Tn)] @ B[i]
    return X + noise * rng.standard_normal((N, D)), gts


@dataclass
class Config:
    d_ambient: int = 20
    n_samples: int = 2400
    noise: float = 0.015
    seed: int = 0
    knn: int = 12
    plot: bool = True
    output_dir: str = "runs/unsup_demo"


def main(cfg: Config = Config()):
    from pathlib import Path
    shapes = shape_library()
    names = list(shapes)
    X, gts = make_data(shapes, D=cfg.d_ambient, N=cfg.n_samples, noise=cfg.noise, seed=cfg.seed)
    curves, asg, B = recover_manifolds(X, K=len(names), seed=cfg.seed, k=cfg.knn)

    used, rows = set(), []
    for i, n in enumerate(names):
        cand = [k for k in range(len(names)) if k not in used and curves[k] is not None]
        best = min(cand, key=lambda k: chamfer_distance(gts[i], curves[k])) if cand else None
        if best is None:
            rows.append((n, None, 2.0)); continue
        used.add(best)
        rows.append((n, best, rel_rms(gts[i], curves[best])))

    print(f"{'shape':<14}{'rel_RMS':>9}")
    for n, _, e in rows:
        print(f"  {n:<12}{100 * e:>7.2f}%")
    mean = float(np.mean([e for _, _, e in rows]))
    print(f"  {'MEAN':<12}{100 * mean:>7.2f}%")

    if cfg.plot:
        out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
        _plot(out / "recovery.png", gts, curves, rows, names)
    return mean


def _plot(path, gts, curves, rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.shape_metrics import _arclen, _procrustes_rms
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (n, best, e) in zip(axes.ravel(), rows):
        i = names.index(n)
        g = _arclen(gts[i], 240); l = _arclen(curves[best], 240) if best is not None else g * 0
        gn = (g - g.mean(0)); gn /= max(np.linalg.norm(gn), 1e-9)
        ln = (l - l.mean(0)); ln /= max(np.linalg.norm(ln), 1e-9)
        _, _, vt = np.linalg.svd(np.concatenate([gn, ln]), full_matrices=False)
        pc = vt[:2]; g2 = gn @ pc.T; l2 = ln @ pc.T
        bestL, be = l2, 1e9
        for Lr in (l2, l2[::-1]):
            for sh in range(0, 240, 6):
                Ls = np.roll(Lr, sh, axis=0); M = Ls.T @ g2
                U, _, Vt = np.linalg.svd(M); La = Ls @ (U @ Vt); err = ((g2 - La) ** 2).sum()
                if err < be:
                    be, bestL = err, La
        ax.plot(g2[:, 0], g2[:, 1], "-", c="C0", lw=5, alpha=.4, label="GT")
        ax.plot(bestL[:, 0], bestL[:, 1], "-", c="C1", lw=1.5, label="recovered")
        ax.set_title(f"{n}  {100 * e:.1f}%"); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Unsupervised recovery of an arbitrary shape bag — graph method, no shape priors")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    main()
