"""SPEC-compliant re-derivation of the D2 synthetic headline: the circle fits now
run through the Rust engine (gamfit.sae_manifold_fit), not numpy. Only the fits
move to Rust; the sphere-tangent chart is a deterministic linear map (data prep)
and the scoring (circular corr, pullback EV, KL calibration r²) is measurement on
the Rust-fitted coordinate — numpy is a permitted cross-check there.

Compares behavior-first pullback vs activation-first, same synthetic as
pullback_synthetic.py (shared latent theta; behavior expresses the feature).
"""
import numpy as np
import gamfit

RNG = np.random.default_rng(0)
N, P_X, V = 140, 6, 6


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def householder(a):
    Vn = a.size; p = int(np.argmax(np.abs(a))); w = -a.copy(); w[p] += 1
    n = np.linalg.norm(w); w = w / n if n > 0 else w * 0
    return np.stack([np.eye(Vn)[j] - 2 * w[j] * w for j in range(Vn) if j != p], 1)


def sphere_tangent(P):  # deterministic half-density chart (data prep, not a fit)
    q = np.sqrt(P / P.sum(1, keepdims=True)); qb = q.mean(0); qb /= np.linalg.norm(qb)
    return np.sqrt(2.0) * (q @ householder(qb))


def exact_kl(a, b):
    m = a > 0; return float(np.sum(a[m] * np.log(a[m] / np.clip(b[m], 1e-300, None))))


def gamfit_circle_coord(X):
    """The FIT — Rust REML circle via gamfit; returns per-token coordinate (rad)."""
    m = gamfit.sae_manifold_fit(X=np.ascontiguousarray(X), K=1, d_atom=1,
                                atom_topology="circle", isometry_weight=0.0,
                                n_iter=40, random_state=0)
    return np.asarray(m.coords)[0, :, 0]


def circ_corr(a, b):
    a = a - np.angle(np.mean(np.exp(1j * a))); b = b - np.angle(np.mean(np.exp(1j * b)))
    return float(np.sum(np.sin(a) * np.sin(b)) /
                 (np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)) + 1e-30))


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi); return np.minimum(d, 2 * np.pi - d)


def hdesign(t, H=3):
    c = [np.ones_like(t)]
    for h in range(1, H + 1):
        c += [np.cos(h * t), np.sin(h * t)]
    return np.stack(c, 1)


def calib(t, P, pairs):
    d = np.array([circ_dist(t[i], t[j]) for i, j in pairs])
    meas = np.array([exact_kl(P[i], P[j]) for i, j in pairs])
    m = meas > 1e-9
    a = np.sum(d[m] ** 2 * meas[m]) / np.sum(d[m] ** 4 + 1e-30)
    pred = a * d ** 2
    lp, lm = np.log(pred[m]), np.log(meas[m])
    A = np.stack([lm, np.ones_like(lm)], 1); coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    return float(1 - np.sum((lp - A @ coef) ** 2) / np.sum((lp - lp.mean()) ** 2))


def main():
    theta = RNG.uniform(0, 2 * np.pi, N)
    Z = np.zeros((N, P_X))
    Z[:, 0] = np.cos(theta); Z[:, 1] = np.sin(theta)
    Z[:, 2] = 0.4 * np.cos(2 * theta); Z[:, 3] = 0.4 * np.sin(2 * theta)
    Z += 0.05 * RNG.standard_normal((N, P_X))
    Zs = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    L = np.stack([1.3 * np.cos(theta), 1.3 * np.sin(theta),
                  0.5 * np.cos(2 * theta), 0.5 * np.sin(2 * theta),
                  0.2 * np.ones_like(theta), np.zeros_like(theta)], 1)
    P = softmax(L)

    # FITS via gamfit (Rust).
    t_act = gamfit_circle_coord(Zs)
    Y = sphere_tangent(P)
    t_beh = gamfit_circle_coord(Y)
    # pullback: reconstruct activations from the behavioral coordinate (lstsq is a
    # readout of the already-fitted coordinate, not a manifold fit — measurement).
    Bp, *_ = np.linalg.lstsq(hdesign(t_beh), Zs, rcond=None)
    pull_ev = float(1 - np.sum((Zs - hdesign(t_beh) @ Bp) ** 2) /
                    np.sum((Zs - Zs.mean(0)) ** 2))
    pairs = [(int(i), int(j)) for _ in range(2000)
             for i, j in [RNG.choice(N, 2, replace=False)]]
    print("SPEC-compliant (gamfit-fitted) D2 synthetic:")
    print(f"  circ_corr(act,beh)   = {circ_corr(t_act, t_beh):.3f}")
    print(f"  circ_corr(beh,truth) = {circ_corr(t_beh, theta):.3f}")
    print(f"  circ_corr(act,truth) = {circ_corr(t_act, theta):.3f}")
    print(f"  pullback_activation_ev = {pull_ev:.3f}")
    print(f"  cal_beh_r2 = {calib(t_beh, P, pairs):.3f}")
    print(f"  cal_act_r2 = {calib(t_act, P, pairs):.3f}")


if __name__ == "__main__":
    main()
