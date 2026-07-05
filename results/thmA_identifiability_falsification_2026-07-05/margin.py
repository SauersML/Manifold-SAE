"""
Theorem B quantitative margin experiments.

The p=5+ regime has a UNIQUE parse (frac=0 everywhere), so the identifiability
"margin" is not a distance between double parses -- it is the CONDITIONING of
the unique parse. We measure sigma_min of the parse Jacobian
    J = d residual / d(t1, t2, a1, a2)   at the true parse.
sigma_min -> 0 signals the parse becoming ill-posed (local non-identifiability,
the linearized Davis-Kahan co-collapse the engine fights).

Two knobs:
 (A) curvature: flatten each atom individually (2nd axis *= curv).
 (B) co-collapse: interpolate atom 2's generator toward atom 1 (align subspaces
     and centers). alpha=0 -> atoms independent (well separated);
     alpha=1 -> atoms identical (fully co-collapsed, parse ambiguous).
"""
import json
import numpy as np


def atom(rng, p):
    A = rng.standard_normal((p, 2))
    c = rng.standard_normal(p)
    return A, c


def g(t, A, c):
    return c + np.array([np.cos(t), np.sin(t)]) @ A.T


def gdot(t, A):
    return np.array([-np.sin(t), np.cos(t)]) @ A.T


def parse_jacobian(t1, t2, a1, a2, A1, c1, A2, c2):
    """Columns: d/dt1, d/dt2, d/da1, d/da2 of  a1 g1(t1)+a2 g2(t2)."""
    col_t1 = a1 * gdot(t1, A1)
    col_t2 = a2 * gdot(t2, A2)
    col_a1 = g(t1, A1, c1)
    col_a2 = g(t2, A2, c2)
    return np.stack([col_t1, col_t2, col_a1, col_a2], axis=1)  # p x 4


def sigma_min_stats(rng, p, curv=1.0, collapse_alpha=0.0, n=400):
    A1, c1 = atom(rng, p)
    A2, c2 = atom(rng, p)
    A1[:, 1] *= curv
    A2[:, 1] *= curv
    # co-collapse: interpolate atom 2 toward atom 1
    A2 = (1 - collapse_alpha) * A2 + collapse_alpha * A1
    c2 = (1 - collapse_alpha) * c2 + collapse_alpha * c1
    smins = []
    for _ in range(n):
        t1 = rng.uniform(-np.pi, np.pi)
        t2 = rng.uniform(-np.pi, np.pi)
        a1 = rng.uniform(0.3, 2.0)
        a2 = rng.uniform(0.3, 2.0)
        J = parse_jacobian(t1, t2, a1, a2, A1, c1, A2, c2)
        s = np.linalg.svd(J, compute_uv=False)
        smins.append(s[-1])
    smins = np.array(smins)
    return dict(p=p, curv=curv, collapse_alpha=collapse_alpha,
                median_sigma_min=float(np.median(smins)),
                p05_sigma_min=float(np.percentile(smins, 5)),
                min_sigma_min=float(np.min(smins)))


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    out = {"curvature_conditioning": [], "cocollapse": []}
    # (A) conditioning vs curvature (each atom flattened), well separated
    for curv in [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        rows = [sigma_min_stats(np.random.default_rng(100 + s), p=5, curv=curv)
                for s in range(8)]
        med = float(np.median([r["median_sigma_min"] for r in rows]))
        p05 = float(np.median([r["p05_sigma_min"] for r in rows]))
        out["curvature_conditioning"].append(
            dict(curv=curv, median_sigma_min=med, p05_sigma_min=p05))
        print("CURV-COND", json.dumps(out["curvature_conditioning"][-1]), flush=True)
    # (B) conditioning vs co-collapse (atom2 -> atom1)
    for alpha in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]:
        rows = [sigma_min_stats(np.random.default_rng(200 + s), p=5, curv=1.0,
                                collapse_alpha=alpha) for s in range(8)]
        med = float(np.median([r["median_sigma_min"] for r in rows]))
        p05 = float(np.median([r["p05_sigma_min"] for r in rows]))
        out["cocollapse"].append(
            dict(collapse_alpha=alpha, median_sigma_min=med, p05_sigma_min=p05))
        print("COLLAPSE", json.dumps(out["cocollapse"][-1]), flush=True)
    with open("margin_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("MARGIN-DONE", flush=True)
