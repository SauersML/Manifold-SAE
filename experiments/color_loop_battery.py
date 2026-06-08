"""A 'shitton of loops' through the color representation, via many gamfit features.

ONE config (RL3.1 final, L44). 24 vivid colors (no brown/beige/white/black/grey/tan).
Uses the NOISY per-prompt replicates (6 per color) so gamfit's penalty does the denoising
(gam handles noise). Fits a closed loop through the colors several different ways:

  fixed-position curve fitters (loop ORDER from a TSP arc-length tour in top-3 PC space,
  no hue assumed; each replicate inherits its color's position t in [0,1)):
    * periodic cubic B-spline   gamfit.periodic_spline_curve_basis   (native closed curve)
    * periodic Duchon (m=2)     gamfit.duchon_basis(periodic_per_axis=[1.0])
    * Fourier / harmonic        first-K harmonics  cos/sin(2 pi k t)
  unsupervised latent manifolds (gamfit recovers the coordinate itself):
    * circle  S^1   gaussian_reml_optimize_latent(manifold='circle')   (see gam#876)
    * sphere  S^2   gaussian_reml_optimize_latent(manifold='sphere')
    * torus   T^2   gaussian_reml_optimize_latent(manifold='torus')
  native model selection:
    * gamfit.select_topology  ranks candidate topologies by REML evidence.

For each fitter we report fit R^2 against the denoised color means and (for curves) the
per-replicate R^2 (the achievable noise floor). Builds an interactive plotly figure with
every loop overlaid (toggle in legend). gamfit-only for all manifold/curve fits. Read-only.
"""
from __future__ import annotations
import argparse, json, colorsys
from pathlib import Path
import numpy as np
import gamfit

VIVID = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan", "magenta",
         "teal", "turquoise", "violet", "indigo", "crimson", "gold", "maroon", "lime",
         "coral", "salmon", "peach", "mint", "lavender", "navy", "olive"}


def load(extra: Path, layer: int):
    X = np.load(extra / "activations.npy")
    recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    H = X[:, min(layer, X.shape[1] - 1), :].astype(np.float64)
    by, fr, rgb = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i)
        fr.setdefault(r["frame"], []).append(i)
        rgb[r["color"]] = np.array(r["rgb"], float)
    Hd = H.copy()                                       # frame-demean (remove prompt-template axis)
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)
    cols = [c for c in by if c in VIVID]
    reps = {c: Hd[by[c]] for c in cols}                 # 6 noisy replicates per color
    rgbm = {c: rgb[c] for c in cols}
    return cols, reps, rgbm


def tsp_order(M):
    from scipy.spatial.distance import cdist
    D = cdist(M, M); n = len(M); tour = [0]; rem = set(range(1, n))
    while rem:
        j = min(rem, key=lambda k: D[tour[-1], k]); tour.append(j); rem.discard(j)
    tl = lambda t: sum(D[t[i], t[(i + 1) % n]] for i in range(n))
    imp = True
    while imp:
        imp = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                nt = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
                if tl(nt) < tl(tour) - 1e-9:
                    tour = nt; imp = True
    return tour, D


def gcv(B, P, Y):
    n = len(Y); best = None
    for lam in np.logspace(-4, 4, 50):
        A = B @ np.linalg.solve(B.T @ B + lam * P, B.T); tr = np.trace(A)
        rss = ((Y - A @ Y) ** 2).sum() / Y.shape[1]; g = n * rss / (n - tr) ** 2
        if best is None or g < best[0]:
            best = (g, lam)
    lam = best[1]; coef = np.linalg.solve(B.T @ B + lam * P, B.T @ Y)
    return coef, lam, np.trace(B @ np.linalg.solve(B.T @ B + lam * P, B.T))


def r2(Y, F):
    return float(1 - ((Y - F) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra"); ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--html", default="/tmp/loops_all.html")
    ap.add_argument("--pcs", type=int, default=3)
    a = ap.parse_args()
    cols, reps, rgbm = load(Path(a.extra), a.layer)
    n = len(cols)
    mean = np.stack([reps[c].mean(0) for c in cols])            # 24 x 5120 denoised means
    # common low-d frame: top PCs of the means (lossless rotation for compute)
    Mc = mean - mean.mean(0); U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    P3 = Vt[:a.pcs]                                             # projection to display space
    Ym = Mc @ P3.T                                             # 24 x pcs means
    # noisy replicates projected to same frame
    Xr = np.concatenate([reps[c] for c in cols]); cidx = np.concatenate([[i] * len(reps[c]) for i, c in enumerate(cols)])
    Yr = (Xr - mean.mean(0)) @ P3.T
    print(f"# {n} vivid colors, L{a.layer}, {len(Yr)} noisy replicates -> top-{a.pcs} PC display frame")

    tour, D = tsp_order(Ym)
    seg = np.array([np.linalg.norm(Ym[tour[i]] - Ym[tour[(i + 1) % n]]) for i in range(n)])
    tpos = np.zeros(n); tpos[tour] = np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum()
    tr_rep = tpos[cidx]                                         # each replicate -> its color's loop position
    tgrid = np.linspace(0, 1, 400)

    loops = {}; rows = []

    # ---- 1. native periodic cubic B-spline (fit on noisy replicates) ----
    B, Pn = gamfit.periodic_spline_curve_basis(tr_rep, 12); B = np.asarray(B); Pn = np.asarray(Pn)
    coef, lam, edf = gcv(B, Pn, Yr)
    Bg = np.asarray(gamfit.periodic_spline_curve_basis(tgrid, 12)[0])
    loops["periodic B-spline"] = Bg @ coef
    fit_means = np.asarray(gamfit.periodic_spline_curve_basis(tpos, 12)[0]) @ coef
    rows.append(("periodic B-spline", r2(Ym, fit_means), r2(Yr, B @ coef), edf, lam))

    # ---- 2. periodic Duchon ----
    C = np.linspace(0, 1, 12, endpoint=False).reshape(-1, 1)
    Bd = np.asarray(gamfit.duchon_basis(tr_rep.reshape(-1, 1), C, m=2, periodic_per_axis=[1.0]))
    coef2, lam2, edf2 = gcv(Bd, np.eye(Bd.shape[1]), Yr)
    Bdg = np.asarray(gamfit.duchon_basis(tgrid.reshape(-1, 1), C, m=2, periodic_per_axis=[1.0]))
    loops["periodic Duchon"] = Bdg @ coef2
    fdm = np.asarray(gamfit.duchon_basis(tpos.reshape(-1, 1), C, m=2, periodic_per_axis=[1.0])) @ coef2
    rows.append(("periodic Duchon", r2(Ym, fdm), r2(Yr, Bd @ coef2), edf2, lam2))

    # ---- 3. Fourier / harmonic loop ----
    def fourier(t, K=5):
        cols_ = [np.ones_like(t)]
        for k in range(1, K + 1):
            cols_ += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
        return np.stack(cols_, 1)
    Bf = fourier(tr_rep); coef3, lam3, edf3 = gcv(Bf, np.eye(Bf.shape[1]), Yr)
    loops["Fourier (5 harm.)"] = fourier(tgrid) @ coef3
    rows.append(("Fourier (5 harm.)", r2(Ym, fourier(tpos) @ coef3), r2(Yr, Bf @ coef3), edf3, lam3))

    # ---- 4-6. unsupervised latent manifolds (gamfit recovers coordinate) ----
    def latent(manifold, dim):
        rng = np.random.RandomState(0)
        if dim == 1:
            C_ = np.linspace(0, 1, 12).reshape(-1, 1)
        elif manifold == "sphere":
            c = rng.randn(16, 3); C_ = c / np.linalg.norm(c, 1, keepdims=True) if False else c / np.linalg.norm(c, axis=1, keepdims=True)
        else:
            C_ = np.array([[x, y] for x in np.linspace(0, 1, 4) for y in np.linspace(0, 1, 4)], float)
        kw = dict(y=Ym.astype(float), n_obs=n, latent_dim=(3 if manifold == "sphere" else dim),
                  centers=C_.astype(float), penalty=np.eye(len(C_)), m=2, manifold=manifold,
                  basis_kind="duchon", max_iter=120, seed=0)
        if manifold == "sphere":
            t0 = rng.randn(n, 3); kw["t"] = (t0 / np.linalg.norm(t0, axis=1, keepdims=True)).reshape(-1)
        r = gamfit.gaussian_reml_optimize_latent(**kw)
        t = np.asarray(r.get("t", r.get("latent"))).reshape(n, -1)
        fitted = np.asarray(r.get("fitted")).reshape(n, -1) if r.get("fitted") is not None else None
        return t, r.get("converged"), float(r.get("grad_t_norm", np.nan)), fitted
    for manifold, dim in [("circle", 1), ("sphere", 2), ("torus", 2)]:
        try:
            t, conv, gnorm, fitted = latent(manifold, dim)
            ev = r2(Ym, fitted) if fitted is not None and fitted.shape == Ym.shape else float("nan")
            tstd = float(np.std(t))
            rows.append((f"latent {manifold}", ev, float("nan"), float("nan"), float("nan")))
            print(f"   latent {manifold:6s}: converged={conv} ||grad||={gnorm:.1e} t_std={tstd:.3f} recon_R2={ev:.3f}")
            if manifold == "sphere":
                np.save("/tmp/sphere_t.npy", t)                 # unit positions on S^2 for the sphere figure
        except Exception as e:
            print(f"   latent {manifold:6s}: ERR {str(e)[:70]}")

    # ---- 7. native select_topology ranking ----
    try:
        import pandas as pd
        df = pd.DataFrame(Ym, columns=[f"y{i}" for i in range(a.pcs)])
        st = gamfit.select_topology(df, response=[f"y{i}" for i in range(a.pcs)] if a.pcs > 1 else "y0",
                                    score="reml", score_scale="per_observation")
        print("\n select_topology ranking:", getattr(st, "ranking", st))
    except Exception as e:
        print(f"\n select_topology: ERR {str(e)[:120]}")

    print("\n loop fit quality (curve fitters, on NOISY replicates):")
    print(f"   {'method':20s} {'R2(means)':>10s} {'R2(replic)':>11s} {'edf':>6s} {'lambda':>9s}")
    for nm, rm, rr, edf_, lam_ in rows:
        print(f"   {nm:20s} {rm:10.3f} {rr if rr==rr else float('nan'):11.3f} {edf_ if edf_==edf_ else float('nan'):6.1f} {lam_ if lam_==lam_ else float('nan'):9.2g}")

    # ---- interactive figure (only valid in 3D) ----
    if a.pcs == 3:
        import plotly.graph_objects as go
        rgbs = [f"rgb({int(rgbm[c][0])},{int(rgbm[c][1])},{int(rgbm[c][2])})" for c in cols]
        fig = go.Figure()
        palette = {"periodic B-spline": "rgba(255,255,255,0.95)", "periodic Duchon": "rgba(120,200,255,0.85)",
                   "Fourier (5 harm.)": "rgba(255,180,90,0.85)"}
        for nm, L in loops.items():
            fig.add_trace(go.Scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="lines",
                          line=dict(width=5, color=palette.get(nm, "grey")), name=nm))
        fig.add_trace(go.Scatter3d(x=Ym[:, 0], y=Ym[:, 1], z=Ym[:, 2], mode="markers+text",
                      marker=dict(size=14, color=rgbs, line=dict(width=1.5, color="white")),
                      text=cols, textfont=dict(color="white", size=10), name="colors"))
        fig.update_layout(template="plotly_dark", title=f"a shitton of gamfit loops through {n} vivid colors (L{a.layer} RL3.1)",
                          scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
        fig.write_html(a.html); print("\nwrote", a.html)


if __name__ == "__main__":
    main()
