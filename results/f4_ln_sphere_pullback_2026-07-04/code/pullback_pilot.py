"""Deliverable-2 pilot: behavior-first pullback vs activation-first circle.

The boldest inversion: instead of fitting a circle to ACTIVATIONS and hoping its
coordinate is behaviorally meaningful, we fit the behavioral manifold M_y FIRST on
the sqrt-p sphere (where squared tangent distance IS KL in nats, by construction of
the SphereTangentEmbedding chart), get a behavioral coordinate per token, then PULL
BACK to activations: the atom is the preimage of the behavioral circle, and its
activation-space shape is read off by least squares. We then ask:

  (1) AGREEMENT  -- do the behavior-first and activation-first coordinates order the
      tokens the same way around the circle? (circular correlation)
  (2) CALIBRATION -- whose coordinate better predicts the REAL pairwise behavioral
      KL (exact, from the harvested next-token distributions)? For a circle with
      unit-speed-in-nats parametrization, KL(i,j) ~ (arc length between t_i, t_j)^2;
      we fit predicted = a * d(t_i,t_j)^2 and report the log-log calibration
      (slope, R^2, median ratio) in the SAME format as the activation-first dose
      calibration stats. The behavior-first chart is built to make this tight; the
      question is whether the activation-first circle -- the incumbent -- matches it.

All geometry mirrors crates/gam-sae/src/manifold/behavior.rs (SphereTangentEmbedding)
and the periodic-harmonic circle fit the two-block demo runs; pure numpy.
"""
import json
import numpy as np

R = "/projects/standard/hsiehph/sauer354"
ACT_NPZ = f"{R}/dose_qwen8b_out/harvest_cache_weekday_L18_n70.npz"
BEH_NPZ = f"{R}/dose_qwen8b_out/behavior_nexttoken_weekday_L18_n70.npz"
OUT = f"{R}/dose_qwen8b_out/pullback_pilot_result.json"
H = 3
WEEKDAYS = 7
TEMPLATES = 10


# ---- SphereTangentEmbedding (faithful port of behavior.rs) -----------------
def sphere_tangent_fit(P):
    """P: (n,V) non-neg rows. Return (Y nats-unit tangent (n,V-1), qbar, E, q)."""
    q = np.sqrt(P / P.sum(1, keepdims=True))          # half-densities on S^{V-1}
    mean = q.mean(0)
    qbar = mean / np.linalg.norm(mean)                # extrinsic-mean basepoint
    E = householder_tangent_basis(qbar)               # (V, V-1)
    Y = np.sqrt(2.0) * (q @ E)                         # nats-unit tangent coords
    return Y, qbar, E, q


def householder_tangent_basis(axis):
    V = axis.size
    pivot = int(np.argmax(np.abs(axis)))
    w = -axis.copy()
    w[pivot] += 1.0
    nrm = np.linalg.norm(w)
    if nrm > 0:
        w /= nrm
    else:
        w[:] = 0.0
    cols = []
    for j in range(V):
        if j == pivot:
            continue
        e = np.zeros(V)
        e[j] = 1.0
        cols.append(e - 2.0 * w[j] * w)
    return np.stack(cols, 1)                            # (V, V-1)


def exact_kl(pa, pb):
    m = pa > 0
    return float(np.sum(pa[m] * np.log(pa[m] / np.clip(pb[m], 1e-300, None))))


# ---- periodic-harmonic circle fit (the two-block demo's atom) --------------
def harmonic_design(t, H):
    cols = [np.ones_like(t)]
    for h in range(1, H + 1):
        cols += [np.cos(h * t), np.sin(h * t)]
    return np.stack(cols, 1)


def refine_theta(t, X, B, iters=80, lr=0.5):
    Hn = (B.shape[0] - 1) // 2
    t = t.copy()
    for _ in range(iters):
        cols = [np.ones_like(t)]
        dcols = [np.zeros_like(t)]
        for h in range(1, Hn + 1):
            cols += [np.cos(h * t), np.sin(h * t)]
            dcols += [-h * np.sin(h * t), h * np.cos(h * t)]
        g = np.stack(cols, 1) @ B
        dg = np.stack(dcols, 1) @ B
        resid = X - g
        num = (resid * dg).sum(1)
        den = (dg * dg).sum(1) + 1e-9
        t = t + lr * num / den
    return np.mod(t, 2 * np.pi)


def circle_fit(X, seed, H, sweeps=20):
    t = seed.copy()
    B = None
    for _ in range(sweeps):
        B, *_ = np.linalg.lstsq(harmonic_design(t, H), X, rcond=None)
        t = refine_theta(t, X, B)
    B, *_ = np.linalg.lstsq(harmonic_design(t, H), X, rcond=None)
    return t, B


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def circular_corr(a, b):
    """Correlation of two angular variables (Fisher-Lee circular correlation)."""
    a = a - np.angle(np.mean(np.exp(1j * a)))
    b = b - np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a) * np.sin(b))
    den = np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return float(num / (den + 1e-30))


def calibration_stats(pred, meas):
    """log-log slope/R^2/median-ratio of predicted vs measured KL (dose format)."""
    m = (pred > 1e-9) & (meas > 1e-9)
    lp, lm = np.log(pred[m]), np.log(meas[m])
    A = np.stack([lm, np.ones_like(lm)], 1)
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    resid = lp - A @ coef
    r2 = float(1 - np.sum(resid ** 2) / np.sum((lp - lp.mean()) ** 2))
    ratio = pred[m] / meas[m]
    return dict(n=int(m.sum()), log_slope=slope, log_intercept=intercept,
                log_r2=r2, ratio_median=float(np.median(ratio)),
                mean_abs_log_ratio=float(np.mean(np.abs(np.log(ratio)))))


def pairwise_calibration(t, P, pairs):
    """Predicted arc^2 (fit scale a) vs measured exact KL, over token pairs."""
    d = np.array([circ_dist(t[i], t[j]) for i, j in pairs])
    meas = np.array([exact_kl(P[i], P[j]) for i, j in pairs])
    # unit-speed circle: predicted nats = a * d^2; a by least squares in raw units.
    mask = meas > 1e-9
    a = float(np.sum((d[mask] ** 2) * meas[mask]) / np.sum(d[mask] ** 4 + 1e-30))
    pred = a * d ** 2
    return calibration_stats(pred, meas), a


def main():
    da = np.load(ACT_NPZ, allow_pickle=True)
    db = np.load(BEH_NPZ, allow_pickle=True)
    X = da["X_last"].astype(np.float64)                # (70, 4096)
    P = db["probs"].astype(np.float64)                 # (70, V), rows sum to captured
    captured = db["captured_mass"]
    P = P / P.sum(1, keepdims=True)                     # renormalize the restricted set
    labels = db["labels"]
    n = X.shape[0]
    template_ids = np.repeat(np.arange(TEMPLATES), WEEKDAYS)[:n]
    print(f"n={n} V={P.shape[1]} captured_mass[{captured.min():.4f},"
          f"{captured.max():.4f}]", flush=True)

    # W7 demean + top-PC activation projection.
    Xd = X.copy()
    for t in range(TEMPLATES):
        msk = template_ids == t
        Xd[msk] -= Xd[msk].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xd, full_matrices=False)
    Z = Xd @ Vt[:6].T
    Zsphere = Z / np.linalg.norm(Z, axis=1, keepdims=True)  # LN-sphere (deliverable 1)

    # ---- activation-first circle (incumbent): fit circle to activations.
    seed_act = np.mod(np.arctan2(Z[:, 1], Z[:, 0]), 2 * np.pi)
    t_act, _ = circle_fit(Zsphere, seed_act, H)         # fit on LN sphere (fair)

    # Diagnostic: is the behavioral next-token variation about WEEKDAY or TEMPLATE?
    # Mean exact KL within-same-weekday, within-same-template, and across-all.
    def mean_kl(mask_fn):
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                if mask_fn(i, j):
                    vals.append(exact_kl(P[i], P[j]))
        return float(np.mean(vals)) if vals else float("nan")
    kl_same_weekday = mean_kl(lambda i, j: labels[i] == labels[j])
    kl_same_template = mean_kl(lambda i, j: template_ids[i] == template_ids[j])
    kl_all = mean_kl(lambda i, j: True)

    # ---- behavior-first: fit M_y on the sqrt-p sphere, get behavioral coord.
    Y, qbar, E, q = sphere_tangent_fit(P)               # nats-unit tangent (n,V-1)
    # W7 on the BEHAVIOR side: demean the nats-unit tangent per template, matching
    # the activation pipeline that demeans per template to expose the weekday
    # feature (template continuation dominates the raw next-token distribution).
    Yw = Y.copy()
    for t in range(TEMPLATES):
        msk = template_ids == t
        if msk.sum() > 0:
            Yw[msk] -= Yw[msk].mean(0, keepdims=True)
    # seed from the phase of the top-2 PRINCIPAL directions of the (template-
    # demeaned) behavioral tangent, matching the activation seed's top-2-PC
    # treatment -- NOT arbitrary Householder columns.
    Yc = Yw - Yw.mean(0, keepdims=True)
    _, _, Vy = np.linalg.svd(Yc, full_matrices=False)
    Yp = Yc @ Vy[:2].T
    seed_beh = np.mod(np.arctan2(Yp[:, 1], Yp[:, 0]), 2 * np.pi)
    t_beh, C = circle_fit(Yw, seed_beh, H)              # M_y circle in nats units
    beh_top2_ev = float((Yp ** 2).sum() / (Yc ** 2).sum())
    # PULL BACK: activation-space atom as a function of the BEHAVIORAL coordinate.
    B_pull, *_ = np.linalg.lstsq(harmonic_design(t_beh, H), Zsphere, rcond=None)
    act_recon = harmonic_design(t_beh, H) @ B_pull
    pull_ev = float(1 - np.sum((Zsphere - act_recon) ** 2) /
                    np.sum((Zsphere - Zsphere.mean(0)) ** 2))

    # ---- (1) agreement between the two coordinates.
    cc = circular_corr(t_act, t_beh)
    # order tokens by each coordinate; compare weekday label sequences.
    order_act = labels[np.argsort(t_act)]
    order_beh = labels[np.argsort(t_beh)]

    # ---- (2) calibration: predict real pairwise behavioral KL from each coord.
    # Same-template pairs isolate WEEKDAY behavioral variation (template held
    # constant), matching the per-template-demeaned circles. This is the weekday
    # dose-flavored calibration: does the coordinate predict the real behavioral
    # KL between two weekdays under the same template?
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if template_ids[i] == template_ids[j]]
    cal_beh, a_beh = pairwise_calibration(t_beh, P, pairs)
    cal_act, a_act = pairwise_calibration(t_act, P, pairs)

    result = dict(
        n=n, vocab=int(P.shape[1]),
        captured_mass=[float(captured.min()), float(captured.max())],
        circular_corr_act_vs_beh=cc,
        behavior_tangent_top2_ev=beh_top2_ev,
        pullback_activation_ev=pull_ev,
        kl_same_weekday=kl_same_weekday,
        kl_same_template=kl_same_template,
        kl_all=kl_all,
        n_calibration_pairs=len(pairs),
        order_by_activation=order_act.tolist(),
        order_by_behavior=order_beh.tolist(),
        calibration_behavior_first=cal_beh,
        calibration_activation_first=cal_act,
    )
    print(json.dumps(result, indent=2), flush=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
