"""Color-manifold TOPOLOGY + DIMENSIONALITY selection across training.

For each checkpoint's color reps (frame-demeaned, L44), reduced to top color-rep
PCs, we ask which latent TOPOLOGY best PREDICTS the representation via leave-one-
color-out cross-validation (held-out R^2 — a cross-validated likelihood proxy;
noise is expected, this is fitting/smoothing not interpolation):
  * hue_circle  : 1D ring, circular harmonics [cos kθ, sin kθ] (k<=3)
  * rgb_linear  : 3D linear RGB
  * rgb_quad    : 3D RGB + quadratic (curved 3-manifold)
  * hsv         : hue(circular)+S+V
  * lab         : perceptual CIELAB (3D)
The winner = topology the color manifold actually has. We also report the
likelihood-based intrinsic dimension (Levina-Bickel MLE) and participation ratio.
Runs across all stages -> CSV + trajectory plot. Read-only.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

STAGE_ORDER = [("pretrain", "OLMO3_32B_TRAJ"), ("SFT", "OLMO3_32B_TRAJ_SFT"),
               ("DPO", "OLMO3_32B_TRAJ_DPO"), ("RL3.0", "OLMO3_32B_TRAJ_RL"),
               ("RL3.1", "OLMO3_32B_TRAJ_RL31")]


def _key(n):
    m = re.search(r"stage(\d+).*?step(\d+)", n) or re.search(r"step[_-]?(\d+)", n)
    return (int(m.group(1)), int(m.group(2))) if (m and m.lastindex == 2) else ((0, int(m.group(1))) if m else (0, 0))


def _rgb_to_lab(rgb):
    rgb = rgb / 255.0
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = rgb @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, xyz ** (1 / 3), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def latents(rgb_by_color, colors):
    rgb = np.stack([rgb_by_color[c] for c in colors])
    hsv = np.stack([colorsys.rgb_to_hsv(*(rgb_by_color[c] / 255.0)) for c in colors])
    h = hsv[:, 0]
    def harm(theta, K):
        cols = [np.ones_like(theta)]
        for k in range(1, K + 1):
            cols += [np.cos(2 * np.pi * k * theta), np.sin(2 * np.pi * k * theta)]
        return np.stack(cols, 1)
    lab = np.stack([_rgb_to_lab(rgb_by_color[c]) for c in colors])
    def withbias(X):
        return np.concatenate([np.ones((len(X), 1)), (X - X.mean(0)) / (X.std(0) + 1e-9)], 1)
    return {
        "hue_circle": harm(h, 3),
        "rgb_linear": withbias(rgb.astype(float)),
        "rgb_quad": np.concatenate([withbias(rgb.astype(float)),
                                    withbias((rgb.astype(float) ** 2))[:, 1:]], 1),
        "hsv": np.concatenate([harm(h, 3), withbias(hsv[:, 1:])[:, 1:]], 1),
        "lab": withbias(lab),
    }


def loo_r2(Lz, Y, ridge=1.0):
    """Leave-one-row-out ridge held-out R^2 (variance-weighted over Y columns)."""
    n = len(Y)
    pred = np.zeros_like(Y)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        X = Lz[tr]; A = X.T @ X + ridge * np.eye(X.shape[1])
        W = np.linalg.solve(A, X.T @ Y[tr])
        pred[i] = Lz[i] @ W
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    var = Y.var(0)
    return float(np.average(1 - ss_res / ss_tot, weights=var))


def intrinsic_dim_mle(V, k=8):
    """Levina-Bickel MLE intrinsic dimension."""
    from numpy.linalg import norm
    D = np.sqrt(((V[:, None] - V[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    md = []
    for i in range(len(V)):
        d = np.sort(D[i])[:k]
        d = d[d > 0]
        if len(d) < 3:
            continue
        Tk = d[-1]
        m = (np.log(Tk / d[:-1])).mean()
        if m > 0:
            md.append(1.0 / m)
    return float(np.mean(md)) if md else float("nan")


def color_topology(extra, layer, npc=10):
    X = np.load(Path(extra) / "activations.npy"); recs = [json.loads(l) for l in open(Path(extra) / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    by, fr, rgb = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i); rgb[r["color"]] = np.asarray(r["rgb"], float)
    Hd = H.copy()
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)
    colors = list(by); V = np.stack([Hd[by[c]].mean(0) for c in colors])
    Vc = V - V.mean(0)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    Y = U[:, :npc] * S[:npc]                      # top color-rep PCs as targets
    Lz = latents(rgb, colors)
    out = {"color_layer": L,
           "intrinsic_dim_mle": intrinsic_dim_mle(V, k=8),
           "participation_ratio": float((S ** 2).sum() ** 2 / (S ** 4).sum())}
    for name, Lmat in Lz.items():
        out[f"r2_{name}"] = loo_r2(Lmat, Y)
    out["best_topology"] = max(Lz, key=lambda nm: out[f"r2_{nm}"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs"); ap.add_argument("--out", default="runs/ANALYSIS")
    ap.add_argument("--layer", type=int, default=44)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []; gi = 0
    for label, sd in STAGE_ORDER:
        base = Path(args.runs_root) / sd
        if not base.exists():
            continue
        for d in sorted([x for x in glob.glob(f"{base}/*/") if os.path.exists(os.path.join(x, "done.json"))],
                        key=lambda x: _key(Path(x).name)):
            ex = Path(d) / "extra"
            if not (ex / "activations.npy").exists():
                continue
            try:
                row = {"global_idx": gi, "stage": label, "checkpoint": Path(d).name}
                row.update(color_topology(ex, args.layer))
                rows.append(row); gi += 1
                print("  %-26s best=%-10s r2: hue=%.2f rgb=%.2f rgbq=%.2f hsv=%.2f lab=%.2f | dim_mle=%.1f PR=%.1f"
                      % (row["checkpoint"], row["best_topology"], row["r2_hue_circle"], row["r2_rgb_linear"],
                         row["r2_rgb_quad"], row["r2_hsv"], row["r2_lab"], row["intrinsic_dim_mle"], row["participation_ratio"]))
            except Exception as e:
                print("fail", d, e)
    if not rows:
        return
    keys = ["global_idx", "stage", "checkpoint", "color_layer", "best_topology",
            "r2_hue_circle", "r2_rgb_linear", "r2_rgb_quad", "r2_hsv", "r2_lab",
            "intrinsic_dim_mle", "participation_ratio"]
    with open(out / "manifold_topology.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: r[k] for k in keys} for r in rows])
    print(f"[topo] {len(rows)} ckpts -> {out/'manifold_topology.csv'}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = [r["global_idx"] for r in rows]; stages = [r["stage"] for r in rows]
        bounds = [i for i in range(1, len(rows)) if stages[i] != stages[i - 1]]
        fig, a = plt.subplots(2, 1, figsize=(max(9, len(rows) * 0.28), 8), sharex=True, constrained_layout=True)
        for nm, c in [("hue_circle", "#1f6feb"), ("rgb_linear", "#d62728"), ("rgb_quad", "#9467bd"),
                      ("hsv", "#2ca02c"), ("lab", "#e08e0b")]:
            a[0].plot(x, [r[f"r2_{nm}"] for r in rows], "o-", ms=3, lw=1.2, c=c, label=nm)
        a[0].axhline(0, color="0.8", lw=0.7); a[0].set_ylabel("held-out R²\n(topology fit)"); a[0].legend(fontsize=7, ncol=3)
        a[0].set_title("Color manifold: which topology predicts the representation (held-out CV) + intrinsic dim, across training")
        a[1].plot(x, [r["intrinsic_dim_mle"] for r in rows], "o-", ms=3, c="#333", label="MLE intrinsic dim")
        a[1].plot(x, [r["participation_ratio"] for r in rows], "s-", ms=3, c="#e08e0b", label="participation ratio")
        a[1].set_ylabel("dimensionality"); a[1].legend(fontsize=7)
        for b in bounds:
            for ax in a:
                ax.axvline(b - 0.5, color="0.6", lw=0.8, ls="--")
        a[1].set_xticks(x); a[1].set_xticklabels([r["checkpoint"][:12] for r in rows], rotation=90, fontsize=5)
        fig.savefig(out / "manifold_topology.png", dpi=160); plt.close(fig)
        print(f"[topo] wrote {out/'manifold_topology.png'}")
    except Exception as e:
        print("plot skip", e)


if __name__ == "__main__":
    main()
