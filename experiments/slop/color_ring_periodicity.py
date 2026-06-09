"""Confirm the color hue-RING two ways (gamfit circle + harmonic periodicity).

1) RESPONSE-DIM SWEEP: gamfit's unsupervised circle (gaussian_reml_optimize_latent,
   manifold='circle') recovers true hue at ~0.65 circular-corr robustly when fed 4-24
   response rep-PCs, but COLLAPSES to ~0.27 if the noisy tail PCs (25-29) are included --
   i.e. the earlier "gamfit can't find hue" was a noise-tail artifact, not a real failure.
2) PERIODICITY: per rep-PC, leave-one-color-out R^2 of a 1st-harmonic regression
   [cos(hue), sin(hue)] vs a linear-in-hue regression. PC3/PC4 fit PERIODICALLY (harmonic
   R^2 > 0, linear R^2 < 0) and the 2nd harmonic adds nothing -> a clean fundamental-frequency
   circle. So the color manifold is genuinely a hue ring. gamfit-only for the manifold fit.
Read-only.
"""
from __future__ import annotations
import argparse, json, colorsys
from pathlib import Path
import numpy as np
import gamfit


def load(extra: Path, layer: int):
    X = np.load(extra / "activations.npy")
    recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    H = X[:, min(layer, X.shape[1] - 1), :].astype(np.float64)
    by, fr, rgb = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i)
        rgb[r["color"]] = np.array(r["rgb"], float)
    Hd = H.copy()
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)
    cols = list(by); V = np.stack([Hd[by[c]].mean(0) for c in cols])
    hue = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255))[0] for c in cols])
    return V, hue


def cc(a, b):
    am = np.angle(np.mean(np.exp(1j * a))); bm = np.angle(np.mean(np.exp(1j * b)))
    sa = np.sin(a - am); sb = np.sin(b - bm)
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def circle_hue(Yb, hr):
    n = len(Yb); C = np.linspace(0, 1, 12).reshape(-1, 1)
    r = gamfit.gaussian_reml_optimize_latent(y=Yb.astype(float), n_obs=n, latent_dim=1, centers=C,
                                             penalty=np.eye(12), m=2, manifold="circle", basis_kind="duchon",
                                             max_iter=80, seed=0)
    t = np.asarray(r.get("t", r.get("latent"))).reshape(n, -1)
    return max(abs(cc(t[:, 0], hr)), abs(cc(-t[:, 0], hr)))


def loo_r2(y, Xd):
    n = len(y); pred = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        b, *_ = np.linalg.lstsq(Xd[m], y[m], rcond=None); pred[i] = Xd[i] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra_dirs", nargs="+"); ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--label", nargs="*", default=None)
    a = ap.parse_args()
    for k, d in enumerate(a.extra_dirs):
        lab = (a.label[k] if a.label and k < len(a.label) else Path(d).parent.name)
        V, hue = load(Path(d), a.layer); hr = hue * 2 * np.pi
        Vc = V - V.mean(0); U, S, _ = np.linalg.svd(Vc, full_matrices=False); Y = U * S
        print(f"\n######## {lab} (L{a.layer}) ########")
        print("response-dim sweep (gamfit circle hue recovery):")
        for kk in [4, 8, 12, 16, 20, 24, 29]:
            print(f"   {kk:2d} PCs: {circle_hue(Y[:, :kk], hr):.3f}")
        print("periodicity (LOO R^2): 1st-harmonic vs 2nd-harmonic vs linear, per rep-PC:")
        n = len(hue)
        H1 = np.c_[np.ones(n), np.cos(hr), np.sin(hr)]
        H2 = np.c_[H1, np.cos(2 * hr), np.sin(2 * hr)]
        Lin = np.c_[np.ones(n), hue]
        for pc in range(4):
            y = Y[:, pc]
            print(f"   PC{pc+1}: harm1={loo_r2(y, H1):+.2f}  harm2={loo_r2(y, H2):+.2f}  linear={loo_r2(y, Lin):+.2f}")


if __name__ == "__main__":
    main()
