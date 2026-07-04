"""Deliverable-2, method check: when the behavioral summary DOES express the
feature, the behavior-first pullback recovers the same circle as the activation
fit AND calibrates KL tighter -- the boldest inversion works. (The real-weekday
run shows the complementary null: the next-token readout is template-dominated,
so it does NOT carry the weekday, and there the pullback cannot recover it.)

Plant one shared latent theta. Activation = a curved image on the LN sphere.
Behavior = a softmax whose logits rotate with theta (so the next-token law moves
along the SAME coordinate). Fit the behavior manifold FIRST on the sqrt-p sphere,
pull back to activations, and compare to the activation-first circle and truth.
"""
import json
import numpy as np

RNG = np.random.default_rng(0)
H = 3
N, P_X, V = 140, 6, 6


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def householder_tangent_basis(axis):
    Vn = axis.size
    piv = int(np.argmax(np.abs(axis)))
    w = -axis.copy(); w[piv] += 1.0
    nrm = np.linalg.norm(w); w = w / nrm if nrm > 0 else w * 0
    cols = []
    for j in range(Vn):
        if j == piv:
            continue
        e = np.zeros(Vn); e[j] = 1.0
        cols.append(e - 2 * w[j] * w)
    return np.stack(cols, 1)


def sphere_tangent(Pm):
    q = np.sqrt(Pm / Pm.sum(1, keepdims=True))
    qbar = q.mean(0); qbar /= np.linalg.norm(qbar)
    E = householder_tangent_basis(qbar)
    return np.sqrt(2.0) * (q @ E)


def exact_kl(a, b):
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / np.clip(b[m], 1e-300, None))))


def hdesign(t, H):
    cols = [np.ones_like(t)]
    for h in range(1, H + 1):
        cols += [np.cos(h * t), np.sin(h * t)]
    return np.stack(cols, 1)


def refine(t, X, B, it=80, lr=0.5):
    Hn = (B.shape[0] - 1) // 2
    t = t.copy()
    for _ in range(it):
        cols = [np.ones_like(t)]; dc = [np.zeros_like(t)]
        for h in range(1, Hn + 1):
            cols += [np.cos(h * t), np.sin(h * t)]
            dc += [-h * np.sin(h * t), h * np.cos(h * t)]
        g = np.stack(cols, 1) @ B; dg = np.stack(dc, 1) @ B
        r = X - g
        t = t + lr * (r * dg).sum(1) / ((dg * dg).sum(1) + 1e-9)
    return np.mod(t, 2 * np.pi)


def circle_fit(X, seed, H, sweeps=20):
    t = seed.copy(); B = None
    for _ in range(sweeps):
        B, *_ = np.linalg.lstsq(hdesign(t, H), X, rcond=None)
        t = refine(t, X, B)
    B, *_ = np.linalg.lstsq(hdesign(t, H), X, rcond=None)
    return t, B


def circ_corr(a, b):
    a = a - np.angle(np.mean(np.exp(1j * a))); b = b - np.angle(np.mean(np.exp(1j * b)))
    return float(np.sum(np.sin(a) * np.sin(b)) /
                 (np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)) + 1e-30))


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def calib(t, Pm, pairs):
    d = np.array([circ_dist(t[i], t[j]) for i, j in pairs])
    meas = np.array([exact_kl(Pm[i], Pm[j]) for i, j in pairs])
    m = meas > 1e-9
    a = np.sum(d[m] ** 2 * meas[m]) / np.sum(d[m] ** 4 + 1e-30)
    pred = a * d ** 2
    lp, lm = np.log(pred[m]), np.log(meas[m])
    A = np.stack([lm, np.ones_like(lm)], 1)
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    r2 = 1 - np.sum((lp - A @ coef) ** 2) / np.sum((lp - lp.mean()) ** 2)
    return float(r2), float(np.median(pred[m] / meas[m]))


def main():
    theta = RNG.uniform(0, 2 * np.pi, N)
    # activation: curved image (harmonics 1&2) in first dims, on LN sphere + noise
    Z = np.zeros((N, P_X))
    Z[:, 0] = np.cos(theta); Z[:, 1] = np.sin(theta)
    Z[:, 2] = 0.4 * np.cos(2 * theta); Z[:, 3] = 0.4 * np.sin(2 * theta)
    Z += 0.05 * RNG.standard_normal((N, P_X))
    Zs = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    # behavior: softmax logits rotating with theta -> next-token law on the circle
    L = np.stack([1.3 * np.cos(theta), 1.3 * np.sin(theta),
                  0.5 * np.cos(2 * theta), 0.5 * np.sin(2 * theta),
                  0.2 * np.ones_like(theta), np.zeros_like(theta)], 1)
    Pm = softmax(L)

    # activation-first circle
    seed_a = np.mod(np.arctan2(Zs[:, 1], Zs[:, 0]), 2 * np.pi)
    t_act, _ = circle_fit(Zs, seed_a, H)
    # behavior-first circle (M_y on sqrt-p sphere), then pull back to activations
    Y = sphere_tangent(Pm)
    Yc = Y - Y.mean(0, keepdims=True)
    _, _, Vy = np.linalg.svd(Yc, full_matrices=False)
    Yp = Yc @ Vy[:2].T
    seed_b = np.mod(np.arctan2(Yp[:, 1], Yp[:, 0]), 2 * np.pi)
    t_beh, _ = circle_fit(Y, seed_b, H)
    Bpull, *_ = np.linalg.lstsq(hdesign(t_beh, H), Zs, rcond=None)
    recon = hdesign(t_beh, H) @ Bpull
    pull_ev = float(1 - np.sum((Zs - recon) ** 2) / np.sum((Zs - Zs.mean(0)) ** 2))

    pairs = [(int(i), int(j)) for _ in range(2000)
             for i, j in [RNG.choice(N, 2, replace=False)]]
    r2_beh, mr_beh = calib(t_beh, Pm, pairs)
    r2_act, mr_act = calib(t_act, Pm, pairs)

    out = dict(
        circ_corr_act_beh=circ_corr(t_act, t_beh),
        circ_corr_beh_truth=circ_corr(t_beh, theta),
        circ_corr_act_truth=circ_corr(t_act, theta),
        pullback_activation_ev=pull_ev,
        cal_beh_r2=r2_beh, cal_beh_medratio=mr_beh,
        cal_act_r2=r2_act, cal_act_medratio=mr_act,
    )
    print(json.dumps(out, indent=2))
    with open("/private/tmp/claude-501/-Users-user/"
              "402ec9d9-07ac-42f4-87a0-73d65d949d5b/scratchpad/"
              "pullback_synthetic_result.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
