"""Behavior-first pullback vs activation-first, on a readout that CARRIES weekday.

Predict-the-weekday probe (harvest_predict_weekday.py): each prompt makes the model
produce a weekday, so the next-token distribution over the 7 weekday tokens is the
weekday identity itself -- a point on the sqrt-7 sphere where squared tangent
distance is KL in nats. The TARGET weekday traces the 7-cycle, so both the L18
activation and the behavioral readout should trace the weekday circle.

Reports, comparing behavior-first pullback vs activation-first circle:
  (1) agreement    -- circular correlation of the two per-token coordinates,
  (2) pullback EV  -- how much of the L18 activation the behavioral coord explains,
  (3) calibration  -- which coordinate better predicts the real pairwise weekday KL
                      (dose-flavored, same-template pairs isolate the weekday move).
Transport-invariance is inherited from XPORT's metric_transport (J-propagation of
the output-Fisher metric matches direct harvest to 0.1%); the pullback atom is a
preimage of that metric, so it composes with J the same way.
"""
import json
import numpy as np

R = "/projects/standard/hsiehph/sauer354"
NPZ = f"{R}/dose_qwen8b_out/predict_weekday_L18.npz"
OUT = f"{R}/dose_qwen8b_out/predict_weekday_pullback_result.json"
H = 3


def householder(axis):
    V = axis.size; piv = int(np.argmax(np.abs(axis)))
    w = -axis.copy(); w[piv] += 1.0
    nrm = np.linalg.norm(w); w = w / nrm if nrm > 0 else w * 0
    cols = [np.eye(V)[j] - 2 * w[j] * w for j in range(V) if j != piv]
    return np.stack(cols, 1)


def sphere_tangent(P):
    q = np.sqrt(P / P.sum(1, keepdims=True))
    qbar = q.mean(0); qbar /= np.linalg.norm(qbar)
    return np.sqrt(2.0) * (q @ householder(qbar))


def exact_kl(a, b):
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / np.clip(b[m], 1e-300, None))))


def hdesign(t, H):
    cols = [np.ones_like(t)]
    for h in range(1, H + 1):
        cols += [np.cos(h * t), np.sin(h * t)]
    return np.stack(cols, 1)


def refine(t, X, B, it=80, lr=0.5):
    Hn = (B.shape[0] - 1) // 2; t = t.copy()
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
    d = np.abs(a - b) % (2 * np.pi); return np.minimum(d, 2 * np.pi - d)


def calib(t, P, pairs):
    d = np.array([circ_dist(t[i], t[j]) for i, j in pairs])
    meas = np.array([exact_kl(P[i], P[j]) for i, j in pairs])
    m = meas > 1e-9
    a = np.sum(d[m] ** 2 * meas[m]) / np.sum(d[m] ** 4 + 1e-30)
    pred = a * d ** 2
    lp, lm = np.log(pred[m]), np.log(meas[m])
    A = np.stack([lm, np.ones_like(lm)], 1); coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    r2 = 1 - np.sum((lp - A @ coef) ** 2) / np.sum((lp - lp.mean()) ** 2)
    return float(r2), float(np.median(pred[m] / meas[m]))


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = d["X_last"].astype(np.float64)
    P7 = d["probs7"].astype(np.float64)
    Pu = d["probs_union"].astype(np.float64); Pu = Pu / Pu.sum(1, keepdims=True)
    target = d["target_label"]; base = d["base_label"]; tmpl = d["template_ids"]
    wk_mass = d["weekday_mass"]
    n = X.shape[0]
    n_tmpl = int(tmpl.max()) + 1
    print(f"n={n} weekday_mass[{wk_mass.min():.3f},{wk_mass.max():.3f}] "
          f"pred_acc={(P7.argmax(1)==target).mean():.3f}", flush=True)

    # W7 demean activations per template, top-PC, LN sphere.
    Xd = X.copy()
    for t in range(n_tmpl):
        msk = tmpl == t
        Xd[msk] -= Xd[msk].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xd, full_matrices=False)
    Z = Xd @ Vt[:6].T
    Zs = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    seed_a = np.mod(np.arctan2(Zs[:, 1], Zs[:, 0]), 2 * np.pi)
    t_act, _ = circle_fit(Zs, seed_a, H)

    # behavior-first on the sqrt-7 sphere (weekday identity), demeaned per template.
    Y = sphere_tangent(P7)
    Yw = Y.copy()
    for t in range(n_tmpl):
        msk = tmpl == t
        Yw[msk] -= Yw[msk].mean(0, keepdims=True)
    Yc = Yw - Yw.mean(0, keepdims=True)
    _, _, Vy = np.linalg.svd(Yc, full_matrices=False)
    Yp = Yc @ Vy[:2].T
    seed_b = np.mod(np.arctan2(Yp[:, 1], Yp[:, 0]), 2 * np.pi)
    t_beh, _ = circle_fit(Yw, seed_b, H)
    Bpull, *_ = np.linalg.lstsq(hdesign(t_beh, H), Zs, rcond=None)
    pull_ev = float(1 - np.sum((Zs - hdesign(t_beh, H) @ Bpull) ** 2) /
                    np.sum((Zs - Zs.mean(0)) ** 2))

    # within/across-target KL diagnostic on the WEEKDAY-restricted readout P7 (the
    # object the behavior circle is fit on): same-target should be CLOSE.
    def mkl(Pm, fn):
        v = [exact_kl(Pm[i], Pm[j]) for i in range(n) for j in range(i + 1, n) if fn(i, j)]
        return float(np.mean(v)) if v else float("nan")
    kl_same_target = mkl(P7, lambda i, j: target[i] == target[j])
    kl_same_tmpl = mkl(P7, lambda i, j: tmpl[i] == tmpl[j])
    kl_all = mkl(P7, lambda i, j: True)

    # Calibration on P7-KL (weekday identity, the behavioral object). All pairs,
    # since the weekday move is the whole signal here (offset varies across pairs).
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    r2_beh, mr_beh = calib(t_beh, P7, pairs)
    r2_act, mr_act = calib(t_act, P7, pairs)

    # ordering: does each coordinate sort tokens by the TARGET weekday?
    def order_purity(t):
        order = target[np.argsort(t)]
        # fraction of adjacent pairs (cyclically) with |Δweekday| in {0,1,6}
        adj = np.abs(np.diff(np.concatenate([order, order[:1]])))
        return float(np.mean([a in (0, 1, 6) for a in adj]))

    out = dict(
        n=n, weekday_mass=[float(wk_mass.min()), float(wk_mass.max())],
        pred_accuracy=float((P7.argmax(1) == target).mean()),
        circ_corr_act_vs_beh=circ_corr(t_act, t_beh),
        pullback_activation_ev=pull_ev,
        kl_same_target=kl_same_target, kl_same_template=kl_same_tmpl, kl_all=kl_all,
        order_purity_activation=order_purity(t_act),
        order_purity_behavior=order_purity(t_beh),
        calib_behavior_first=dict(r2=r2_beh, median_ratio=mr_beh),
        calib_activation_first=dict(r2=r2_act, median_ratio=mr_act),
        n_pairs=len(pairs),
    )
    print(json.dumps(out, indent=2), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
