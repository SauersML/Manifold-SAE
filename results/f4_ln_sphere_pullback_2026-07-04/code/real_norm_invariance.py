"""Deliverable-1, real-data confirmation: on the REAL Qwen3-8B L18 weekday
activations, injecting per-token norm variation bends the flat-space circle
decoder but leaves the LN-sphere decoder (near-)invariant.

Real activations already have intrinsic norm spread; we can't call their intrinsic
higher-harmonic content 'spurious' (the real weekday manifold need not be pure
harmonic-1). So we test the CLEAN causal claim instead: the LN projection u=x/||x||
is invariant to any radial rescaling, so multiplying each row by an independent
lognormal factor must NOT change the sphere fit, while it DOES bend the flat fit.
That isolates the norm-variation -> spurious-curvature channel on real data.
"""
import json
import numpy as np

R = "/projects/standard/hsiehph/sauer354"
NPZ = f"{R}/dose_qwen8b_out/harvest_cache_weekday_L18_n70.npz"
OUT = f"{R}/dose_qwen8b_out/real_norm_invariance_result.json"
H = 4
RNG = np.random.default_rng(0)

WEEKDAYS = 7
TEMPLATES = 10


def harmonic_design(theta, H):
    cols = [np.ones_like(theta)]
    for h in range(1, H + 1):
        cols += [np.cos(h * theta), np.sin(h * theta)]
    return np.stack(cols, 1)


def fit_decoder(theta, X, H):
    Phi = harmonic_design(theta, H)
    B, *_ = np.linalg.lstsq(Phi, X, rcond=None)
    return B


def refine_theta(theta, X, B, iters, lr):
    Hn = (B.shape[0] - 1) // 2
    t = theta.copy()
    for _ in range(iters):
        cols = [np.ones_like(t)]
        dcols = [np.zeros_like(t)]
        for h in range(1, Hn + 1):
            cols += [np.cos(h * t), np.sin(h * t)]
            dcols += [-h * np.sin(h * t), h * np.cos(h * t)]
        Phi = np.stack(cols, 1)
        dPhi = np.stack(dcols, 1)
        g = Phi @ B
        dg = dPhi @ B
        resid = X - g
        num = (resid * dg).sum(1)
        den = (dg * dg).sum(1) + 1e-9
        t = t + lr * num / den
    return np.mod(t, 2 * np.pi)


def joint_fit(theta0, X, H, sweeps=15):
    t = theta0.copy()
    B = None
    for _ in range(sweeps):
        B = fit_decoder(t, X, H)
        t = refine_theta(t, X, B, 60, 0.5)
    return t, fit_decoder(t, X, H)


def hh_frac(B):
    e = (B ** 2).sum(1)
    h1 = e[1] + e[2]
    hi = e[3:].sum()
    return float(hi / (h1 + hi + 1e-30))


def radial_rms(theta, X, B):
    Phi = harmonic_design(theta, (B.shape[0] - 1) // 2)
    g = Phi @ B
    resid = X - g
    rad = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-30)
    along = (resid * rad).sum(1)
    return float(np.sqrt((along ** 2).mean())), float(np.sqrt((resid ** 2).mean()))


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = d["X_last"].astype(np.float64)  # (70, 4096)
    n = X.shape[0]
    # Row order = 10 templates x 7 weekdays (per weekday_probe_harvest).
    template_ids = np.repeat(np.arange(TEMPLATES), WEEKDAYS)[:n]
    # W7 recipe: subtract each template's mean over its 7 weekday tokens.
    Xd = X.copy()
    for t in range(TEMPLATES):
        m = template_ids == t
        if m.sum() > 0:
            Xd[m] -= Xd[m].mean(0, keepdims=True)
    # Reduce to the leading PCs where the weekday circle lives (tractable p).
    U, S, Vt = np.linalg.svd(Xd, full_matrices=False)
    p = 6
    Z = Xd @ Vt[:p].T  # (n, p) projected activations
    zn = np.linalg.norm(Z, axis=1)
    base_norm_cv = float(np.std(zn) / np.mean(zn))
    # seed from phase of the top-2 PCs
    seed = np.mod(np.arctan2(Z[:, 1], Z[:, 0]), 2 * np.pi)

    rows = []
    sphere_decoders = []
    for inj in [0.0, 0.2, 0.4]:
        rng = np.random.default_rng(42)
        factor = np.exp(inj * rng.standard_normal(n))
        Xi = Z * factor[:, None]
        # FLAT
        tf, Bf = joint_fit(seed, Xi, H)
        f_hh = hh_frac(Bf)
        f_rad, f_tot = radial_rms(tf, Xi, Bf)
        # SPHERE (LN projection)
        Ui = Xi / np.linalg.norm(Xi, axis=1, keepdims=True)
        ts, Bs = joint_fit(seed, Ui, H)
        s_hh = hh_frac(Bs)
        s_rad, s_tot = radial_rms(ts, Ui, Bs)
        sphere_decoders.append(Bs)
        cv = float(np.std(np.linalg.norm(Xi, axis=1)) / np.mean(np.linalg.norm(Xi, axis=1)))
        rows.append(dict(inject=inj, norm_cv=cv,
                         flat_hh=f_hh, flat_radial_rms=f_rad, flat_total_rms=f_tot,
                         sphere_hh=s_hh, sphere_radial_rms=s_rad, sphere_total_rms=s_tot))
        print(f"inj={inj:.2f} normCV={cv:.3f} | FLAT hh={f_hh:.4f} radial={f_rad:.4f} | "
              f"SPHERE hh={s_hh:.4f} radial={s_rad:.5f}", flush=True)

    # Sphere-decoder invariance: sphere decoder should be ~identical across
    # injection scales (LN quotients the injected norm out).
    d0 = sphere_decoders[0]
    invariance = []
    for k in range(1, len(sphere_decoders)):
        num = np.linalg.norm(sphere_decoders[k] - d0)
        den = np.linalg.norm(d0) + 1e-30
        invariance.append(float(num / den))
    print(f"sphere decoder rel-change across injection: {invariance}", flush=True)
    result = dict(base_norm_cv=base_norm_cv, rows=rows,
                  sphere_decoder_rel_change=invariance,
                  flat_decoder_bends=(rows[-1]["flat_hh"] - rows[0]["flat_hh"]))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
