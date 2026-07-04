"""Deliverable-1 controlled test: flat-space circle fit absorbs norm variation
into spurious curvature; the LN-sphere-ambient fit does not.

Generative model (LN geometry): the behaviorally-relevant object is a DIRECTION
on the unit sphere tracing a great circle,
    v(theta) = cos(theta) e1 + sin(theta) e2      (unit norm, pure harmonic-1),
and each token's activation is that direction scaled by a nuisance norm r_i that
is INDEPENDENT of theta (post-LN the model reads only v; r is quotiented away):
    x_i = r_i * v(theta_i),   log r_i ~ N(0, s^2).

Two fits of a K=1 periodic-harmonic circle chart gamma(theta) = Phi(theta) B:
  * FLAT   : reconstruct x_i in R^p under the Euclidean metric (jointly over
             theta and B) -- what an ambient-flat circle atom does.
  * SPHERE : reconstruct u_i = x_i/||x_i|| (the LN projection) instead.

Spurious curvature is measured two ways, both zero for the true generator:
  (A) higher-harmonic energy fraction of the fitted decoder B (the true curve is
      pure harmonic 1, so ANY energy in harmonics >=2 is spurious);
  (B) radial residual fraction -- the share of reconstruction-residual variance
      lying ALONG the local curve direction v(theta) (a certificate-violating,
      theta-rotating, rank-1 heteroscedastic term). The flat residual is
      dominated by it; the sphere residual has none by construction.

We sweep the norm-variation scale s and show (A),(B) grow with s for FLAT and
stay flat-zero for SPHERE. Pure numpy; deterministic.
"""
import json
import numpy as np

RNG = np.random.default_rng(0)


def harmonic_design(theta, H):
    """Periodic harmonic basis Phi(theta): [1, cos t, sin t, ..., cos Ht, sin Ht]."""
    cols = [np.ones_like(theta)]
    for h in range(1, H + 1):
        cols.append(np.cos(h * theta))
        cols.append(np.sin(h * theta))
    return np.stack(cols, 1)  # (n, 2H+1)


def fit_decoder(theta, X, H):
    """Least-squares decoder B: X ~ Phi(theta) B."""
    Phi = harmonic_design(theta, H)
    B, *_ = np.linalg.lstsq(Phi, X, rcond=None)
    return B, Phi


def refine_theta(theta, X, B, iters=200, lr=0.3):
    """Coordinate descent on theta_i to minimize ||x_i - Phi(theta_i) B||^2.
    Gauss-Newton per-row on the scalar theta_i (the circle-term inner solve)."""
    H = (B.shape[0] - 1) // 2
    t = theta.copy()
    for _ in range(iters):
        # gamma(t) and d gamma/dt per row.
        cols = [np.ones_like(t)]
        dcols = [np.zeros_like(t)]
        for h in range(1, H + 1):
            cols += [np.cos(h * t), np.sin(h * t)]
            dcols += [-h * np.sin(h * t), h * np.cos(h * t)]
        Phi = np.stack(cols, 1)
        dPhi = np.stack(dcols, 1)
        g = Phi @ B          # (n,p)
        dg = dPhi @ B        # (n,p)
        resid = X - g        # (n,p)
        # Newton step on scalar t: num = <resid, dg>, den = <dg,dg> - <resid, d2g>.
        num = (resid * dg).sum(1)
        den = (dg * dg).sum(1) + 1e-9
        t = t + lr * num / den
    return np.mod(t, 2 * np.pi)


def joint_fit(theta0, X, H, sweeps=12):
    """Alternate (fit B | theta) and (refine theta | B) -- the circle-atom fit."""
    t = theta0.copy()
    B = None
    for _ in range(sweeps):
        B, _ = fit_decoder(t, X, H)
        t = refine_theta(t, X, B, iters=60, lr=0.5)
    B, Phi = fit_decoder(t, X, H)
    return t, B, Phi


def higher_harmonic_fraction(B):
    """Fraction of decoder energy (excl. the constant row 0) in harmonics >=2."""
    e = (B ** 2).sum(1)  # per-basis-row energy
    h1 = e[1] + e[2]
    hhi = e[3:].sum()
    return float(hhi / (h1 + hhi + 1e-30))


def radial_residual_fraction(theta, X, B):
    """Share of residual variance along the RADIAL (norm) direction of the fitted
    curve point gamma(t)/||gamma(t)|| -- the certificate-violating, theta-rotating
    rank-1 term that pure norm variation injects into a flat fit. On the LN sphere
    (X already unit-norm) this direction carries no free variance."""
    H = (B.shape[0] - 1) // 2
    Phi = harmonic_design(theta, H)
    g = Phi @ B
    resid = X - g
    rad = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-30)  # radial unit dir
    along = (resid * rad).sum(1)
    n = X.shape[0]
    radial_rms = float(np.sqrt((along ** 2).sum() / n))
    total_rms = float(np.sqrt((resid ** 2).sum() / n))
    return radial_rms, total_rms


def run(s, n=280, p=8, H=4, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    # true direction: great circle in the e1-e2 plane (pure harmonic-1, unit norm)
    V = np.zeros((n, p))
    V[:, 0] = np.cos(theta)
    V[:, 1] = np.sin(theta)
    r = np.exp(s * rng.standard_normal(n))            # nuisance norm, LN scale
    X = r[:, None] * V
    # deterministic seed from PC phase (what the demo does), noised so it's honest
    seed_t = np.mod(np.arctan2(X[:, 1], X[:, 0]) + 0.15 * rng.standard_normal(n),
                    2 * np.pi)
    out = {}
    # FLAT ambient fit
    tf, Bf, _ = joint_fit(seed_t, X, H)
    out["flat_hh_frac"] = higher_harmonic_fraction(Bf)
    # normalize residual RMS by mean norm so flat & sphere are comparable scales
    scale = float(np.mean(np.linalg.norm(X, axis=1)))
    rr, tr = radial_residual_fraction(tf, X, Bf)
    out["flat_radial_rms"] = rr / scale
    out["flat_total_rms"] = tr / scale
    # SPHERE (LN) ambient fit: project to unit sphere first
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    ts, Bs, _ = joint_fit(seed_t, U, H)
    out["sphere_hh_frac"] = higher_harmonic_fraction(Bs)
    rr, tr = radial_residual_fraction(ts, U, Bs)
    out["sphere_radial_rms"] = rr  # U already unit-norm
    out["sphere_total_rms"] = tr
    out["norm_cv"] = float(np.std(r) / np.mean(r))
    return out


def main():
    rows = []
    for s in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # average a few seeds for stability
        accs = [run(s, seed=k) for k in range(4)]
        agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
        agg["s"] = s
        rows.append(agg)
        print(f"s={s:.2f} normCV={agg['norm_cv']:.3f} | "
              f"FLAT hh={agg['flat_hh_frac']:.4f} radRMS={agg['flat_radial_rms']:.4f} "
              f"totRMS={agg['flat_total_rms']:.4f} | "
              f"SPHERE hh={agg['sphere_hh_frac']:.4f} radRMS={agg['sphere_radial_rms']:.5f} "
              f"totRMS={agg['sphere_total_rms']:.4f}")
    with open("/private/tmp/claude-501/-Users-user/"
              "402ec9d9-07ac-42f4-87a0-73d65d949d5b/scratchpad/"
              "spurious_curvature_result.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
