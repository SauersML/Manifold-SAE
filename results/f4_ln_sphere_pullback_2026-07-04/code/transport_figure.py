"""The transport-without-refit figure: one behavioral coordinate holds across
layers while the activation circle is refit and drifts each hop.

Top-left panel: the behavior-first coordinate (fit once to the layer-independent
output) -- tokens placed at their behavioral angle, colored by target weekday,
perfectly ordered (order-purity 1.00). Labeled "same at every layer".
Right three panels: the activation-first coordinate refit at L11/L18/L23 -- the
same tokens scramble (order-purity 0.47/0.41/0.36), and the coordinates disagree
across layers (cross-layer corr ~0.3). Bottom bar: order-purity per method/layer.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/projects/standard/hsiehph/sauer354"
NPZ = f"{R}/dose_qwen8b_out/predict_weekday_multilayer.npz"
FIG = f"{R}/dose_qwen8b_out/transport_figure.png"
H = 3
LAYERS = [11, 18, 23]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def householder(a):
    V = a.size; p = int(np.argmax(np.abs(a))); w = -a.copy(); w[p] += 1
    nrm = np.linalg.norm(w); w = w / nrm if nrm > 0 else w * 0
    return np.stack([np.eye(V)[j] - 2 * w[j] * w for j in range(V) if j != p], 1)


def sphere_tangent(P):
    q = np.sqrt(P / P.sum(1, keepdims=True)); qb = q.mean(0); qb /= np.linalg.norm(qb)
    return np.sqrt(2.0) * (q @ householder(qb))


def hdesign(t, H):
    c = [np.ones_like(t)]
    for h in range(1, H + 1):
        c += [np.cos(h * t), np.sin(h * t)]
    return np.stack(c, 1)


def refine(t, X, B, it=80, lr=.5):
    Hn = (B.shape[0] - 1) // 2; t = t.copy()
    for _ in range(it):
        c = [np.ones_like(t)]; d = [np.zeros_like(t)]
        for h in range(1, Hn + 1):
            c += [np.cos(h * t), np.sin(h * t)]; d += [-h * np.sin(h * t), h * np.cos(h * t)]
        g = np.stack(c, 1) @ B; dg = np.stack(d, 1) @ B; r = X - g
        t = t + lr * (r * dg).sum(1) / ((dg * dg).sum(1) + 1e-9)
    return np.mod(t, 2 * np.pi)


def fit(X, seed, sweeps=20):
    t = seed.copy()
    for _ in range(sweeps):
        B, *_ = np.linalg.lstsq(hdesign(t, H), X, rcond=None); t = refine(t, X, B)
    return t


def sphere_pca(X, tmpl, nt, k=6):
    Xd = X.copy()
    for t in range(nt):
        m = tmpl == t; Xd[m] -= Xd[m].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xd, full_matrices=False)
    Z = Xd @ Vt[:k].T
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def purity(t, target):
    o = target[np.argsort(t)]
    adj = np.abs(np.diff(np.concatenate([o, o[:1]])))
    return float(np.mean([a in (0, 1, 6) for a in adj]))


def main():
    d = np.load(NPZ, allow_pickle=True)
    P7 = d["probs7"].astype(np.float64); target = d["target_label"]; tmpl = d["template_ids"]
    n = P7.shape[0]; nt = int(tmpl.max()) + 1

    Yw = sphere_tangent(P7).copy()
    for t in range(nt):
        m = tmpl == t; Yw[m] -= Yw[m].mean(0, keepdims=True)
    Yc = Yw - Yw.mean(0, keepdims=True)
    _, _, Vy = np.linalg.svd(Yc, full_matrices=False); Yp = Yc @ Vy[:2].T
    t_beh = fit(Yw, np.mod(np.arctan2(Yp[:, 1], Yp[:, 0]), 2 * np.pi))

    t_acts = {}
    for L in LAYERS:
        Zs = sphere_pca(d[f"X_last_L{L}"].astype(np.float64), tmpl, nt)
        t_acts[L] = fit(Zs, np.mod(np.arctan2(Zs[:, 1], Zs[:, 0]), 2 * np.pi))

    cmap = plt.get_cmap("hsv")
    colors = [cmap(w / 7.0) for w in target]
    fig = plt.figure(figsize=(13, 4.2))

    def circ_panel(ax, coord, title, pur):
        ax.scatter(np.cos(coord), np.sin(coord), c=colors, s=70,
                   edgecolors="k", linewidths=0.4, zorder=3)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="0.8", lw=1, zorder=1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
        ax.set_title(f"{title}\norder-purity {pur:.2f}", fontsize=10)

    ax0 = fig.add_subplot(1, 4, 1)
    circ_panel(ax0, t_beh, "behavior-first\n(one coord, ALL layers)", purity(t_beh, target))
    for k, L in enumerate(LAYERS):
        ax = fig.add_subplot(1, 4, 2 + k)
        circ_panel(ax, t_acts[L], f"activation-first L{L}\n(refit each layer)",
                   purity(t_acts[L], target))
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(w / 7.),
                      markeredgecolor='k', markersize=7, label=WEEKDAYS[w]) for w in range(7)]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Transport without refit: the behavioral coordinate holds across "
                 "layers; the activation circle is refit and scrambles (Qwen3-8B, "
                 "predict-weekday probe)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIG, dpi=130, bbox_inches="tight")
    print("WROTE", FIG, flush=True)


if __name__ == "__main__":
    main()
