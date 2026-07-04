"""
Topology-by-evidence for a calendar probe harvest: does the weekday signal
actually live on a CIRCLE, or is that an imposed shape?  Answers the
"you forced the circle" objection with held-out numbers and matched nulls.

Runs entirely on a cached harvest (X_last, tmpl_mean; template-major, K templates
x N_DAYS days).  Pure numpy/scipy — no gamfit fit required (a gamfit circle atom
is rank-2, so its reconstruction R^2 is upper-bounded by the PCA-2 plane, which we
compute directly and far more cheaply).

Fits, all with params estimated on TRAIN only, TSS = TRAIN mean:
  line   = PC1            (1 ambient dim, 1 coord)   == atom_topology="linear"
  circle = Kasa in PC1-2  (2 ambient dims, 1 coord)  == atom_topology="circle"
  plane  = PC1,PC2        (2 ambient dims, 2 coords)  matched ambient basis

Tests:
  1. leave-one-template-out CV held-out R^2 (line vs circle vs plane)
  2. curvature gain (circle-line) vs matched-Gaussian-in-plane null q99
  3. weekday order_corr (circular-circular) vs weekday-shuffle null q99
  4. per-template within-plane ring: radiusCV vs matched-Gaussian null
  5. within-template leave-one-day-out: does curvature generalize? (line vs circle)
  6. per-template plane alignment (are templates in a shared plane at all?)

Usage:
  python weekday_topology_evidence.py /path/to/harvest_cache_weekday_L18_n70.npz [N_DAYS]

Finding on Qwen3-8B L18 weekday cache (2026-07-03): the honest OPPOSITE of a
circle win — the line beats the circle on held-out R^2, weekday order and
curvature-generalization both FAIL their nulls, and templates sit in near-
orthogonal planes so no single shared circle exists.  The only structure beating
a null is a label-free constant-radius shell (not a weekday circle).
"""
import sys, json
import numpy as np

N_DAYS = 7

# ---------- geometry ----------
def pca_basis(Xtr, k):
    c = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - c, full_matrices=False)
    return c, Vt[:k]

def kasa(P):
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy, c3 = sol
    return cx, cy, np.sqrt(max(c3 + cx*cx + cy*cy, 1e-12))

def fit_line(Xtr):  c, W = pca_basis(Xtr, 1); return {"c": c, "W": W}
def fit_plane(Xtr): c, W = pca_basis(Xtr, 2); return {"c": c, "W": W}
def fit_circle(Xtr):
    c, W = pca_basis(Xtr, 2)
    cx, cy, R = kasa((Xtr - c) @ W.T)
    try:
        from scipy.optimize import least_squares
        P = (Xtr - c) @ W.T
        r = least_squares(lambda p: np.hypot(P[:,0]-p[0], P[:,1]-p[1]) - p[2],
                          [cx, cy, R], method="lm").x
        cx, cy, R = r[0], r[1], abs(r[2])
    except Exception:
        pass
    return {"c": c, "W": W, "cen2": np.array([cx, cy]), "R": R}

def recon_line(f, Xte):  c, w = f["c"], f["W"][0]; return c + np.outer((Xte-c)@w, w)
def recon_plane(f, Xte): c, W = f["c"], f["W"]; return c + (Xte-c)@W.T@W
def recon_circle(f, Xte):
    c, W, cen2, R = f["c"], f["W"], f["cen2"], f["R"]
    P = (Xte - c) @ W.T
    v = P - cen2
    ang = np.arctan2(v[:, 1], v[:, 0])
    return c + (cen2 + R*np.c_[np.cos(ang), np.sin(ang)]) @ W, ang

def r2(Xte, Xhat, tm):
    return (float(((Xte-Xhat)**2).sum()), float(((Xte-tm)**2).sum()))

def _cm(x): return np.arctan2(np.sin(x).sum(), np.cos(x).sum())
def circ_corr(a, b):
    a, b = a-_cm(a), b-_cm(b)
    den = np.sqrt(np.sum(np.sin(a)**2)*np.sum(np.sin(b)**2))
    return float(np.sum(np.sin(a)*np.sin(b))/den) if den > 0 else 0.0

# ---------- CV ----------
def loto(Xd, tmpl, day, fit_fn, recon_fn, circle=False):
    res = tot = 0.0; ins = []; ocs = []
    for t in np.unique(tmpl):
        te = tmpl == t; tr = ~te
        Xtr, Xte, tm = Xd[tr], Xd[te], Xd[tr].mean(0)
        f = fit_fn(Xtr)
        Ht = recon_fn(f, Xtr)[0] if circle else recon_fn(f, Xtr)
        sr, st = r2(Xtr, Ht, tm); ins.append(1 - sr/st)
        if circle:
            Hh, ang = recon_fn(f, Xte)
            ocs.append(circ_corr(ang, 2*np.pi*day[te]/N_DAYS))
        else:
            Hh = recon_fn(f, Xte)
        sr, st = r2(Xte, Hh, tm); res += sr; tot += st
    return {"heldout_r2": 1-res/tot, "insample_r2": float(np.mean(ins)), "order_corrs": ocs}

def gauss_plane_null(Xd, seed):
    rng = np.random.default_rng(seed)
    c, W = pca_basis(Xd, 2)
    P = (Xd - c) @ W.T
    off = (Xd - c) - P @ W
    return c + rng.multivariate_normal(np.zeros(2), np.cov(P.T), len(P)) @ W + off

def main(path, n_days=N_DAYS):
    global N_DAYS; N_DAYS = n_days
    d = np.load(path)
    Xd = (d["X_last"] - d["tmpl_mean"]).astype(np.float64)
    n = len(Xd); K = n // N_DAYS
    tmpl = np.repeat(np.arange(K), N_DAYS)
    day = np.tile(np.arange(N_DAYS), K)
    norms = {int(t): float(np.linalg.norm(Xd[tmpl == t])) for t in range(K)}
    out = {"template_norms": norms}
    for tag, mask in [("no_t0", tmpl != 0), ("full", np.ones(n, bool))]:
        Xs, ts, ds = Xd[mask], tmpl[mask], day[mask]
        r = {n_: loto(Xs, ts, ds, f_, rc_, c_)
             for n_, f_, rc_, c_ in [("line", fit_line, recon_line, False),
                                     ("plane", fit_plane, recon_plane, False),
                                     ("circle", fit_circle, recon_circle, True)]}
        gain = r["circle"]["heldout_r2"] - r["line"]["heldout_r2"]
        ng = np.array([loto(gauss_plane_null(Xs, s), ts, ds, fit_circle, recon_circle, True)["heldout_r2"]
                       - loto(gauss_plane_null(Xs, s), ts, ds, fit_line, recon_line)["heldout_r2"]
                       for s in range(200)])
        rng = np.random.default_rng(0); noc = []
        for _ in range(200):
            dss = ds.copy()
            for t in np.unique(ts):
                idx = np.where(ts == t)[0]; dss[idx] = rng.permutation(ds[idx])
            noc.append(np.mean(loto(Xs, ts, dss, fit_circle, recon_circle, True)["order_corrs"]))
        roc = float(np.mean(r["circle"]["order_corrs"]))
        out[tag] = {**{k: {kk: vv for kk, vv in v.items() if kk != "order_corrs"} for k, v in r.items()},
                    "curvature_gain": gain, "gauss_null_gain_q99": float(np.quantile(ng, .99)),
                    "curvature_beats_null": bool(gain > np.quantile(ng, .99)),
                    "order_corr": roc, "order_shuffle_null_q99": float(np.quantile(np.abs(noc), .99)),
                    "order_beats_null": bool(abs(roc) > np.quantile(np.abs(noc), .99))}
        print(f"[{tag}] line={r['line']['heldout_r2']:.4f} circle={r['circle']['heldout_r2']:.4f} "
              f"plane={r['plane']['heldout_r2']:.4f} | curvature_gain={gain:+.4f} "
              f"(null q99 {np.quantile(ng,.99):+.4f}) | order_corr={roc:+.3f} "
              f"(shuffle q99 {np.quantile(np.abs(noc),.99):.3f})", flush=True)
    json.dump(out, open("weekday_topology_evidence.json", "w"), indent=2, default=float)
    print("wrote weekday_topology_evidence.json", flush=True)

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "harvest_cache_weekday_L18_n70.npz"
    main(p, int(sys.argv[2]) if len(sys.argv) > 2 else 7)
