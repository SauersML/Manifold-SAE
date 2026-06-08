"""GAMFIT-ONLY hue-ring recovery via dominant-PC deflation.

Finding: on raw color reps a gamfit circle recovers hue only ~0.27, because gamfit's
reconstruction-driven REML fixates on a dominant NON-hue direction (PC1, ~lightness).
The hue ring is variance-sub-dominant, living in the PC2xPC3 plane (circ-corr to true
hue ~0.68, the ceiling). Deflating PC1 (an unsupervised, label-free step) lets the gamfit
circle recover hue at ~0.65 — near the ceiling. So gamfit CAN find the ring once the
dominant nuisance direction is removed.

Reports, per checkpoint extra/ dir: variance per PC, what each top PC encodes
(hue/sat/lightness), the hue-ring PC-plane, and gamfit-circle hue recovery for
raw / deflate-PC1 / PC-plane. gamfit-only for the manifold fit; PCA used only as a
lossless rotation + deflation. Read-only.
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
    hsv = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255)) for c in cols])
    light = np.array([(0.3 * rgb[c][0] + 0.59 * rgb[c][1] + 0.11 * rgb[c][2]) / 255 for c in cols])
    return V, hsv[:, 0] * 2 * np.pi, hsv[:, 1], light


def cc(a, b):
    am = np.angle(np.mean(np.exp(1j * a))); bm = np.angle(np.mean(np.exp(1j * b)))
    sa = np.sin(a - am); sb = np.sin(b - bm)
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def gamfit_circle_hue(Yb, hue):
    n = len(Yb); C = np.linspace(0, 1, 12).reshape(-1, 1)
    res = gamfit.gaussian_reml_optimize_latent(y=Yb.astype(float), n_obs=n, latent_dim=1, centers=C,
                                               penalty=np.eye(12), m=2, manifold="circle", basis_kind="duchon",
                                               max_iter=90, seed=0)
    t = np.asarray(res.get("t", res.get("latent"))).reshape(n, -1)
    return max(abs(cc(t[:, 0], hue)), abs(cc(-t[:, 0], hue)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra_dirs", nargs="+"); ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--label", nargs="*", default=None)
    a = ap.parse_args()
    for k, d in enumerate(a.extra_dirs):
        lab = (a.label[k] if a.label and k < len(a.label) else Path(d).parent.name)
        V, hue, sat, light = load(Path(d), a.layer)
        Vc = V - V.mean(0); U, S, _ = np.linalg.svd(Vc, full_matrices=False); Y = U * S
        var = (S ** 2 / (S ** 2).sum())
        print(f"\n######## {lab} (L{a.layer}) ########")
        print("var/PC:", np.round(var[:6], 3))
        # hue-ring plane
        best = (None, 0.0)
        for i in range(6):
            for j in range(i + 1, 6):
                ang = np.arctan2(Y[:, j], Y[:, i]); v = max(abs(cc(ang, hue)), abs(cc(-ang, hue)))
                if v > best[1]:
                    best = ((i, j), v)
        i, j = best[0]
        print(f"hue-ring plane = PC{i+1}xPC{j+1} (oracle circ-corr {best[1]:.3f})")
        print(f"gamfit circle hue recovery:")
        print(f"   raw (all PCs)        : {gamfit_circle_hue(Y[:, :20], hue):.3f}")
        print(f"   deflate PC1          : {gamfit_circle_hue((U[:, 1:] * S[1:])[:, :20], hue):.3f}")
        print(f"   PC{i+1}xPC{j+1} plane only  : {gamfit_circle_hue(Y[:, [i, j]], hue):.3f}")
        print(f"   (oracle plane ceiling: {best[1]:.3f})")


if __name__ == "__main__":
    main()
