"""Qualia/entity manifold — INTRINSIC DIMENSION (the honest tool for entities).

Unlike color (which has an external ground-truth coordinate, RGB), entities have NO
external coordinate, so a non-circular gamfit topology fit isn't identifiable (any
"predict rep from rep-derived MDS coords" is circular — the core reason the old
expand_topology.py qualia claim was retracted). The well-posed question that needs
no coordinate is: what is the INTRINSIC DIMENSION of the entity-rep point cloud?

Three estimators (geometric, coordinate-free), each with a bootstrap CI over entities:
  * TwoNN (Facco et al. 2017): slope of the log-ratio of 2nd/1st NN distances.
  * Levina-Bickel MLE (k-NN), averaged over k.
  * participation ratio of the covariance spectrum (a linear upper-bound notion).
These are DESCRIPTIVE estimators, not manifold fits, so they don't fall under the
"fit via gamfit" rule; any actual smoothing still goes through gamfit elsewhere.

Run across layers to show depth-dependence. Read-only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def twonn(X, frac=0.9):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(X).query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = (r2[ok] / r1[ok])
    mu = np.sort(mu)
    n = len(mu)
    cut = int(frac * n)
    mu = mu[:cut]
    F = np.arange(1, cut + 1) / n
    x = np.log(mu); y = -np.log(1 - F)
    d_est = float(np.sum(x * y) / np.sum(x * x))
    return d_est


def levina_bickel(X, ks=(5, 10, 15, 20)):
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    est = []
    for k in ks:
        d, _ = tree.query(X, k=k + 1)
        d = d[:, 1:]
        with np.errstate(divide="ignore"):
            logs = np.log(d[:, -1][:, None] / d[:, :-1])
        m = (k - 2) / logs.sum(1)
        est.append(float(np.mean(m[np.isfinite(m)])))
    return float(np.mean(est))


def participation_ratio(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float((s.sum() ** 2) / (s ** 2).sum())


def load_pairs(ckpt_dir, layer):
    X = np.load(Path(ckpt_dir) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(ckpt_dir) / "prompts.jsonl") if l.strip()]
    role = np.array([r["role"] for r in recs])
    L = min(layer, X.shape[1] - 1)
    P = X[:, L, :][np.where(role == "pair")[0]].astype(np.float64)
    return P


def boot_ci(fn, X, nboot=200, seed=0):
    rng = np.random.RandomState(seed); n = len(X)
    vals = []
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        try:
            vals.append(fn(X[idx]))
        except Exception:
            pass
    vals = np.array(vals)
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dirs", nargs="+")
    ap.add_argument("--layers", default="25")
    ap.add_argument("--nboot", type=int, default=200)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    for ck in args.ckpt_dirs:
        print(f"\n######## {ck} ########", flush=True)
        for layer in layers:
            P = load_pairs(ck, layer)
            twn = boot_ci(twonn, P, args.nboot)
            lb = boot_ci(levina_bickel, P, args.nboot)
            pr = boot_ci(participation_ratio, P, args.nboot)
            print(f"L{layer} (n={len(P)}, ambient={P.shape[1]}):", flush=True)
            print("  TwoNN          d=%5.1f  95%%CI[%.1f, %.1f]" % twn, flush=True)
            print("  Levina-Bickel  d=%5.1f  95%%CI[%.1f, %.1f]" % lb, flush=True)
            print("  participation  d=%5.1f  95%%CI[%.1f, %.1f]  (linear upper-bound)" % pr, flush=True)


if __name__ == "__main__":
    main()
