"""Which SHAPE is the color manifold? Fit the color representation distance matrix
to candidate manifolds (constrained MDS with optimal global scale), comparing
topologies at matched intrinsic dimension + a leave-one-color-out held-out test:

  dim 1 : line R^1            | circle S^1
  dim 2 : plane/sheet R^2     | sphere S^2 | torus T^2 | cylinder S^1xR^1
  dim 3 : R^3 | S^3 | T^3 | JOINT hue-ring x sheet  S^1xR^2
  dim 5 : R^5 | S^5 | T^5     (hyperobjects)

Lower normalized stress = better fit; held-out stress guards against overfitting
with more dims (noise is expected — this is fitting, not interpolation). Reports
the best topology per dim, the stress-vs-dim elbow (effective dim), and held-out.
Read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RNG = np.random.RandomState(0)


def _unit(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def _arccos(Z):
    U = _unit(Z); d = np.clip(U @ U.T, -1, 1); return np.arccos(d)


def _euclid(Z):
    diff = Z[:, None] - Z[None]; return np.sqrt((diff ** 2).sum(-1) + 1e-12)


# each topology: params-per-point p, and geodesic(flat params) -> (n,n)
def make_geo(name):
    if name == "lineR1":
        return 1, lambda Z: _euclid(Z)
    if name == "circleS1":
        return 2, lambda Z: _arccos(Z)
    if name == "planeR2":
        return 2, lambda Z: _euclid(Z)
    if name == "sphereS2":
        return 3, lambda Z: _arccos(Z)
    if name == "torusT2":
        return 4, lambda Z: np.sqrt(_arccos(Z[:, :2]) ** 2 + _arccos(Z[:, 2:]) ** 2)
    if name == "cylS1xR1":
        return 3, lambda Z: np.sqrt(_arccos(Z[:, :2]) ** 2 + _euclid(Z[:, 2:]) ** 2)
    if name == "R3":
        return 3, lambda Z: _euclid(Z)
    if name == "S3":
        return 4, lambda Z: _arccos(Z)
    if name == "T3":
        return 6, lambda Z: np.sqrt(_arccos(Z[:, :2]) ** 2 + _arccos(Z[:, 2:4]) ** 2 + _arccos(Z[:, 4:]) ** 2)
    if name == "jointS1xR2":   # hue ring x lightness/sat sheet
        return 4, lambda Z: np.sqrt(_arccos(Z[:, :2]) ** 2 + _euclid(Z[:, 2:]) ** 2)
    if name == "R5":
        return 5, lambda Z: _euclid(Z)
    if name == "S5":
        return 6, lambda Z: _arccos(Z)
    raise ValueError(name)


DIMS = {"lineR1": 1, "circleS1": 1, "planeR2": 2, "sphereS2": 2, "torusT2": 2,
        "cylS1xR1": 2, "R3": 3, "S3": 3, "T3": 3, "jointS1xR2": 3, "R5": 5, "S5": 5}


def opt_scale_stress(geo, D):
    iu = np.triu_indices(len(D), 1); g = geo[iu]; d = D[iu]
    s = (g @ d) / (g @ g + 1e-12)
    return float(np.sqrt(((s * g - d) ** 2).sum() / (d @ d)))


def fit_topology(D, name, restarts=4, maxiter=300):
    n = len(D); p, geo = make_geo(name)

    def stress(x):
        return opt_scale_stress(geo(x.reshape(n, p)), D)
    best = None
    for _ in range(restarts):
        x0 = RNG.randn(n * p) * 0.5
        r = minimize(stress, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, best.x.reshape(n, p)


def heldout_stress(D, name, restarts=2, maxiter=200):
    """Leave-one-color-out: fit on n-1, place the held-out point, measure its
    distance-prediction error to the others. Averaged."""
    n = len(D); p, geo = make_geo(name)
    errs = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        Dk = D[np.ix_(keep, keep)]
        _, Zk = fit_topology(Dk, name, restarts=restarts, maxiter=maxiter)
        # optimal scale from training
        gk = geo(Zk); iu = np.triu_indices(len(Dk), 1)
        s = (gk[iu] @ Dk[iu]) / (gk[iu] @ gk[iu] + 1e-12)
        dtarget = D[i, keep]

        def perr(z):
            Z2 = np.vstack([Zk, z.reshape(1, p)])
            gg = geo(Z2)[-1, :-1]
            return float(((s * gg - dtarget) ** 2).mean())
        rr = minimize(perr, RNG.randn(p) * 0.5, method="L-BFGS-B", options={"maxiter": 150})
        errs.append(np.sqrt(rr.fun / (dtarget ** 2).mean()))
    return float(np.mean(errs))


def color_D(ckpt, layer):
    extra = Path(ckpt) / "extra"
    X = np.load(extra / "activations.npy"); recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    by, fr = {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i)
    Hd = H.copy()
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)
    cols = list(by); V = np.stack([Hd[by[c]].mean(0) for c in cols])
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    return 1 - Vn @ Vn.T, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir")
    ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()
    D, cols = color_D(args.ckpt_dir, args.layer)
    print(f"checkpoint {args.ckpt_dir}  ({len(cols)} colors, L{args.layer})")
    order = ["lineR1", "circleS1", "planeR2", "sphereS2", "torusT2", "cylS1xR1",
             "R3", "S3", "T3", "jointS1xR2", "R5", "S5"]
    print("%-12s dim  in-sample-stress  held-out" % "topology")
    res = {}
    for nm in order:
        ins, _ = fit_topology(D, nm)
        ho = heldout_stress(D, nm) if args.heldout else float("nan")
        res[nm] = (DIMS[nm], ins, ho)
        print("%-12s %d    %.3f             %s" % (nm, DIMS[nm], ins, ("%.3f" % ho) if args.heldout else "-"))
    # best per dim (in-sample)
    print("\nbest topology per dimension (lower stress = better fit):")
    for dd in sorted(set(DIMS.values())):
        cand = [(nm, res[nm][1]) for nm in order if DIMS[nm] == dd]
        bnm, bs = min(cand, key=lambda t: t[1])
        print("  dim %d: %-12s stress=%.3f   (all: %s)" % (dd, bnm, bs, ", ".join("%s=%.3f" % (n, s) for n, s in cand)))


if __name__ == "__main__":
    main()
