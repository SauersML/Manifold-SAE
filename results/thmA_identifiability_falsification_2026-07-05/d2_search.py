"""
Theorem A uniqueness cliff for a d=2 curved atom (generalize past ellipses).

Dictionary {d=2, d=1}:
  atom1 = curved 2-surface (SPHERE PATCH): g1(u,v) = c1 + A1 @ s(u,v),
          s(u,v) = (cos u cos v, cos u sin v, sin u),  A1: p x 3  (generic curvature)
  atom2 = ellipse (d=1):               g2(t)   = c2 + A2 @ (cos t, sin t), A2: p x 2

Cones C_k = {a g_k(.)}. Sum set Sigma = {a1 g1(u,v) + a2 g2(t)}.
Parse unknowns: x = (u, v, t, log a1, log a2)  -> 5 params.  Residual: p eqs.
Boundary: Sum_k(d_k+1) = 3 + 2 = 5 <= p-1  =>  cliff between p=5 (slack -1,
ambiguous) and p=6 (slack 0, unique). Sweep p in {4,5,6,8}.

Prediction (Theorem A generalizes): double-parse fraction = 1 for p<=5, ->0
(measure-zero) for p>=6. Positive control at underdetermined p must still detect
multiplicity (proves the d=2 search sees ambiguity when it is present).

Gauge quotient: cluster converged parses by the RECONSTRUCTED atom points
p1 = a1 g1(u,v), p2 = a2 g2(t) in R^p (labeled atoms), same as the d=1 lane.
"""
import json
import numpy as np
from scipy.optimize import least_squares


def sphere(u, v):
    return np.array([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)])


def g1(u, v, A1, c1):
    return c1 + A1 @ sphere(u, v)


def g2(t, A2, c2):
    return c2 + A2 @ np.array([np.cos(t), np.sin(t)])


class Dict2:
    def __init__(self, rng, p, curvature=1.0, center_scale=1.0, rotate=True):
        self.p = p
        A1 = rng.standard_normal((p, 3))
        A2 = rng.standard_normal((p, 2))
        # curvature knob shrinks the last axis of each atom toward a flat patch
        A1[:, 2] *= curvature
        A2[:, 1] *= curvature
        self.c1 = center_scale * rng.standard_normal(p)
        self.c2 = center_scale * rng.standard_normal(p)
        if rotate:
            Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
            A1 = Q @ A1
            A2 = Q @ A2
            self.c1 = Q @ self.c1
            self.c2 = Q @ self.c2
        self.A1, self.A2 = A1, A2

    def synth(self, rng):
        u = rng.uniform(-np.pi / 2 + 0.2, np.pi / 2 - 0.2)  # avoid sphere poles
        v = rng.uniform(-np.pi, np.pi)
        t = rng.uniform(-np.pi, np.pi)
        a1 = rng.uniform(0.3, 2.0)
        a2 = rng.uniform(0.3, 2.0)
        z = a1 * g1(u, v, self.A1, self.c1) + a2 * g2(t, self.A2, self.c2)
        return z, (u, v, t, a1, a2)

    def residual(self, x, z):
        u, v, t, la1, la2 = x
        pred = np.exp(la1) * g1(u, v, self.A1, self.c1) \
            + np.exp(la2) * g2(t, self.A2, self.c2)
        return pred - z

    def atom_points(self, x):
        u, v, t, la1, la2 = x
        p1 = np.exp(la1) * g1(u, v, self.A1, self.c1)
        p2 = np.exp(la2) * g2(t, self.A2, self.c2)
        return np.concatenate([p1, p2])


def find_parses(D, z, rng, n_starts=80, tol=1e-8):
    sols = []
    method = "lm" if D.p >= 5 else "trf"  # need m>=n for lm; 5 params
    for _ in range(n_starts):
        x0 = np.array([
            rng.uniform(-1.2, 1.2),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-1.5, 1.0),
            rng.uniform(-1.5, 1.0),
        ])
        res = least_squares(D.residual, x0, args=(z,), method=method,
                            max_nfev=500, xtol=1e-14, ftol=1e-14)
        cost = np.sqrt(2 * res.cost)
        if cost < tol:
            sols.append(D.atom_points(res.x))
    return sols


def count_distinct(sols, cluster_tol=1e-4):
    reps = []
    for feat in sols:
        if not any(np.linalg.norm(feat - r) < cluster_tol for r in reps):
            reps.append(feat)
    return len(reps), reps


def run_trial(seed, p, curvature=1.0, center_scale=1.0, n_z=100, n_starts=80):
    rng = np.random.default_rng(seed)
    D = Dict2(rng, p, curvature=curvature, center_scale=center_scale)
    double = 0
    valid = 0
    for _ in range(n_z):
        z, _ = D.synth(rng)
        sols = find_parses(D, z, rng, n_starts=n_starts)
        if not sols:
            continue
        valid += 1
        ndist, _ = count_distinct(sols)
        if ndist >= 2:
            double += 1
    return dict(p=p, curvature=curvature, valid=valid, double=double,
                frac=double / max(valid, 1))


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if mode == "sanity":
        # seconds-scale: unique regime + underdetermined positive control
        r6 = run_trial(seed=0, p=6, n_z=25, n_starts=40)
        r4 = run_trial(seed=0, p=4, n_z=25, n_starts=40)
        print("SANITY p=6 (slack0):", json.dumps(r6))
        print("SANITY p=4 (underdet ctrl):", json.dumps(r4))
    elif mode == "sweep":
        seeds = list(range(int(sys.argv[2]))) if len(sys.argv) > 2 else list(range(4))
        out = []
        for p in [4, 5, 6, 8]:
            fracs = []
            for s in seeds:
                r = run_trial(seed=500 + s, p=p, n_z=120, n_starts=80)
                fracs.append(r["frac"])
            row = dict(p=p, slack=p - 1 - 5, fracs=fracs,
                       mean_frac=float(np.mean(fracs)),
                       max_frac=float(np.max(fracs)))
            out.append(row)
            print("D2-PTRANS", json.dumps(row), flush=True)
        json.dump(out, open("d2_sweep_results.json", "w"), indent=2)
        print("D2-DONE", flush=True)
