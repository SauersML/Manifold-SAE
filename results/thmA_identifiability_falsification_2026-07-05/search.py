"""
Theorem A falsification search.

Theorem A (Superposed Geometry): generic uniqueness of sparse manifold-atom
decompositions. Concretely for TWO curved 1-manifolds (d=1) in R^p, the sparse-2
sum set  Sigma = {a1 g1(t1) + a2 g2(t2) : a>0}  should admit, for a.e. z in Sigma,
a UNIQUE parse (up to each atom's own ray gauge). Hypothesis regime:
sum_k (d_k+1) = 4 <= p-1, so p >= 5.

We numerically SEARCH for a positive-measure set of z with >=2 distinct parses.
Pure numpy/scipy, CPU.

Atoms are curved cones C_k = {a * g_k(t)}. g_k is an ellipse (curved 1-manifold):
    g_k(t) = c_k + A_k @ (cos t, sin t),   A_k: p x 2,  c_k: p.
Optionally add ambient rotation Q (does not change identifiability).

A "parse" of z solves  a1 g1(t1) + a2 g2(t2) = z  with a1,a2>0.
Unknowns: (t1, t2, a1, a2)  -> 4 unknowns, p residual eqs (overdetermined for p>=5).

Gauge to quotient out: for a fixed atom k, the ray {a g_k(t)} can coincide for
two different t if the curve g_k crosses the same ray twice. For a generic
OFF-CENTER ellipse each ray through origin hits the ellipse curve in at most a
finite set; a*g_k(t)=a'*g_k(t') as a POINT means same cone point. We compare
parses by the RECONSTRUCTED atom points  p_k = a_k g_k(t_k) in R^p, not by
(t,a) coords -- this automatically quotients the ray gauge. Two parses are
"the same" iff {p1,p2} match as an unordered... no: atoms are labeled (g1!=g2),
so p1 must match p1', p2 match p2'.
"""
import numpy as np
from scipy.optimize import least_squares


def make_atom(rng, p, curved=True, center_scale=1.0):
    A = rng.standard_normal((p, 2))
    if not curved:
        # near-flat: shrink the second column so the ellipse degenerates toward
        # a line segment -> curvature -> 0 (a straight chord, a flat 1-manifold)
        pass
    c = center_scale * rng.standard_normal(p)
    return A, c


def g(t, A, c):
    # t scalar or (n,)
    t = np.asarray(t)
    basis = np.stack([np.cos(t), np.sin(t)], axis=-1)  # (...,2)
    return c + basis @ A.T


def gdot(t, A):
    basis = np.stack([-np.sin(t), np.cos(t)], axis=-1)
    return basis @ A.T


class Dictionary:
    def __init__(self, rng, p, curvature=1.0, center_scale=1.0, rotate=True):
        self.p = p
        A1 = rng.standard_normal((p, 2))
        A2 = rng.standard_normal((p, 2))
        # curvature knob: scale the 2nd axis of each ellipse. curvature->0 gives a
        # flat chord (straight line direction), the near-flat degenerate regime.
        A1[:, 1] *= curvature
        A2[:, 1] *= curvature
        self.A1, self.A2 = A1, A2
        self.c1 = center_scale * rng.standard_normal(p)
        self.c2 = center_scale * rng.standard_normal(p)
        if rotate:
            # random rotation via QR
            Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
            self.A1 = Q @ self.A1
            self.A2 = Q @ self.A2
            self.c1 = Q @ self.c1
            self.c2 = Q @ self.c2

    def synth(self, rng):
        t1 = rng.uniform(-np.pi, np.pi)
        t2 = rng.uniform(-np.pi, np.pi)
        a1 = rng.uniform(0.3, 2.0)
        a2 = rng.uniform(0.3, 2.0)
        z = a1 * g(t1, self.A1, self.c1) + a2 * g(t2, self.A2, self.c2)
        return z, (t1, t2, a1, a2)

    def residual(self, x, z):
        t1, t2, la1, la2 = x
        a1 = np.exp(la1)  # enforce a>0 via log param
        a2 = np.exp(la2)
        pred = a1 * g(t1, self.A1, self.c1) + a2 * g(t2, self.A2, self.c2)
        return pred - z

    def atom_points(self, x):
        t1, t2, la1, la2 = x
        p1 = np.exp(la1) * g(t1, self.A1, self.c1)
        p2 = np.exp(la2) * g(t2, self.A2, self.c2)
        return p1, p2


def find_parses(D, z, rng, n_starts=60, tol=1e-8):
    """Multistart least squares; return list of converged (x, cost, (p1,p2))."""
    sols = []
    method = "lm" if D.p >= 4 else "trf"
    for _ in range(n_starts):
        x0 = np.array([
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-1.5, 1.0),
            rng.uniform(-1.5, 1.0),
        ])
        res = least_squares(D.residual, x0, args=(z,), method=method,
                            max_nfev=400, xtol=1e-14, ftol=1e-14)
        cost = np.sqrt(2 * res.cost)  # residual L2 norm
        if cost < tol:
            p1, p2 = D.atom_points(res.x)
            sols.append((res.x, cost, np.concatenate([p1, p2])))
    return sols


def count_distinct(sols, cluster_tol=1e-4):
    """Cluster solutions by reconstructed atom points (quotients ray gauge)."""
    reps = []
    for x, cost, feat in sols:
        matched = False
        for r in reps:
            if np.linalg.norm(feat - r) < cluster_tol:
                matched = True
                break
        if not matched:
            reps.append(feat)
    return len(reps), reps


def run_trial(seed, p, curvature, center_scale, n_z, n_starts):
    rng = np.random.default_rng(seed)
    D = Dictionary(rng, p, curvature=curvature, center_scale=center_scale)
    double = 0
    total_valid = 0
    margins = []
    for _ in range(n_z):
        z, truth = D.synth(rng)
        sols = find_parses(D, z, rng, n_starts=n_starts)
        if len(sols) == 0:
            continue  # solver failed to recover even the true parse; skip
        ndist, reps = count_distinct(sols)
        total_valid += 1
        if ndist >= 2:
            double += 1
            # margin: min pairwise distance between distinct parse features
            dmin = np.inf
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    dmin = min(dmin, np.linalg.norm(reps[i] - reps[j]))
            margins.append(dmin)
    frac = double / max(total_valid, 1)
    return dict(p=p, curvature=curvature, center_scale=center_scale,
                n_z=n_z, total_valid=total_valid, double=double, frac=frac,
                median_margin=float(np.median(margins)) if margins else None)


if __name__ == "__main__":
    import sys, json
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if mode == "sanity":
        r = run_trial(seed=0, p=5, curvature=1.0, center_scale=1.0,
                      n_z=40, n_starts=40)
        print(json.dumps(r, indent=2))
