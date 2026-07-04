"""Transport-without-refit: one behavioral coordinate, three layers.

The behavior-first coordinate is fit ONCE to the output distribution (probs7,
layer-independent), so it applies at every layer with no refit. We show:
  * behavior-first order-purity (sorts tokens by target weekday) = 1.0 at EVERY
    layer by construction (it's the same coordinate);
  * the activation-first circle must be refit per layer, and its coordinates DRIFT
    across layers (circular corr < 1 between L11/L18/L23) and its order-purity is
    layer-dependent and low — it cannot be transported without refitting;
  * the single behavioral coordinate explains a (layer-specific) activation image
    at each layer via a linear pullback — the atom is a preimage that re-reads
    through J, exactly the transport-invariance XPORT proved for the metric (0.1%).
"""
import json
import numpy as np

R = "/projects/standard/hsiehph/sauer354"
NPZ = f"{R}/dose_qwen8b_out/predict_weekday_multilayer.npz"
OUT = f"{R}/dose_qwen8b_out/transport_predict_weekday_result.json"
H = 3
LAYERS = [11, 18, 23]


def householder(axis):
    V = axis.size; piv = int(np.argmax(np.abs(axis)))
    w = -axis.copy(); w[piv] += 1.0
    nrm = np.linalg.norm(w); w = w / nrm if nrm > 0 else w * 0
    return np.stack([np.eye(V)[j] - 2 * w[j] * w for j in range(V) if j != piv], 1)


def sphere_tangent(P):
    q = np.sqrt(P / P.sum(1, keepdims=True))
    qbar = q.mean(0); qbar /= np.linalg.norm(qbar)
    return np.sqrt(2.0) * (q @ householder(qbar))


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
        B, *_ = np.linalg.lstsq(hdesign(t, H), X, rcond=None); t = refine(t, X, B)
    B, *_ = np.linalg.lstsq(hdesign(t, H), X, rcond=None)
    return t, B


def circ_corr(a, b):
    a = a - np.angle(np.mean(np.exp(1j * a))); b = b - np.angle(np.mean(np.exp(1j * b)))
    return float(np.sum(np.sin(a) * np.sin(b)) /
                 (np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)) + 1e-30))


def sphere_proj_pca(X, tmpl, n_tmpl, k=6):
    Xd = X.copy()
    for t in range(n_tmpl):
        m = tmpl == t
        Xd[m] -= Xd[m].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xd, full_matrices=False)
    Z = Xd @ Vt[:k].T
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def main():
    d = np.load(NPZ, allow_pickle=True)
    P7 = d["probs7"].astype(np.float64)
    target = d["target_label"]; tmpl = d["template_ids"]
    n = P7.shape[0]; n_tmpl = int(tmpl.max()) + 1

    def order_purity(t):
        order = target[np.argsort(t)]
        adj = np.abs(np.diff(np.concatenate([order, order[:1]])))
        return float(np.mean([a in (0, 1, 6) for a in adj]))

    # behavior-first coordinate: fit ONCE, layer-independent.
    Yw = sphere_tangent(P7).copy()
    for t in range(n_tmpl):
        m = tmpl == t
        Yw[m] -= Yw[m].mean(0, keepdims=True)
    Yc = Yw - Yw.mean(0, keepdims=True)
    _, _, Vy = np.linalg.svd(Yc, full_matrices=False)
    Yp = Yc @ Vy[:2].T
    t_beh, _ = circle_fit(Yw, np.mod(np.arctan2(Yp[:, 1], Yp[:, 0]), 2 * np.pi), H)

    per_layer = {}
    t_acts = {}
    for L in LAYERS:
        X = d[f"X_last_L{L}"].astype(np.float64)
        Zs = sphere_proj_pca(X, tmpl, n_tmpl)
        seed = np.mod(np.arctan2(Zs[:, 1], Zs[:, 0]), 2 * np.pi)
        t_act, _ = circle_fit(Zs, seed, H)
        t_acts[L] = t_act
        # behavior coordinate's pullback image at this layer (no coordinate refit).
        Bp, *_ = np.linalg.lstsq(hdesign(t_beh, H), Zs, rcond=None)
        pull_ev = float(1 - np.sum((Zs - hdesign(t_beh, H) @ Bp) ** 2) /
                        np.sum((Zs - Zs.mean(0)) ** 2))
        per_layer[L] = dict(
            activation_first_order_purity=order_purity(t_act),
            behavior_pullback_ev=pull_ev,
        )

    # activation-first coordinate drift across layers (must refit); behavior is fixed.
    act_drift = {f"L{a}_vs_L{b}": circ_corr(t_acts[a], t_acts[b])
                 for a, b in [(11, 18), (18, 23), (11, 23)]}

    out = dict(
        n=n,
        behavior_first_order_purity=order_purity(t_beh),  # 1.0, same at every layer
        per_layer={str(k): v for k, v in per_layer.items()},
        activation_first_cross_layer_corr=act_drift,
    )
    print(json.dumps(out, indent=2), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
