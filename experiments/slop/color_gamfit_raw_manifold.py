"""GAMFIT manifold fitting on the RAW color reps — NO PCA at all.

Feeds the full frame-demeaned 5120-d color representation matrix (30 colors) directly to
gamfit's unsupervised latent solver. Finding: on raw reps a single CIRCLE recovers hue only
~0.31 (the dominant non-hue variation corrupts it), but a TORUS (two angles) recovers hue at
~0.65 — one angle locks onto the hue ring while the other absorbs the dominant nuisance,
implicitly deflating it. So no PCA is needed; the right manifold (torus) finds the ring on
the raw data. (sphere ~0.04; line/plane ~0.2-0.3; 'cylinder' is rejected by the latent solver
though listed as a topology name elsewhere.) gamfit-only. Read-only.
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
    cols = list(by); V = np.stack([Hd[by[c]].mean(0) for c in cols])           # raw 5120-d, no PCA
    hue = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255))[0] for c in cols]) * 2 * np.pi
    return V, hue


def cc(a, b):
    am = np.angle(np.mean(np.exp(1j * a))); bm = np.angle(np.mean(np.exp(1j * b)))
    sa = np.sin(a - am); sb = np.sin(b - bm)
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def fit(V, hue, manifold, dim):
    n = len(V); rng = np.random.RandomState(0)
    if dim == 1:
        centers = np.linspace(0, 1, 12).reshape(-1, 1)
    elif manifold == "sphere":
        c = rng.randn(16, 3); centers = c / np.linalg.norm(c, axis=1, keepdims=True)
    else:
        centers = np.array([[a, b] for a in np.linspace(0, 1, 4) for b in np.linspace(0, 1, 4)], float)
    kw = dict(y=V.astype(float), n_obs=n, latent_dim=(3 if manifold == "sphere" else dim),
              centers=centers.astype(float), penalty=np.eye(len(centers)), m=2,
              manifold=manifold, basis_kind="duchon", max_iter=90, seed=0)
    if manifold == "sphere":
        t0 = rng.randn(n, 3); kw["t"] = (t0 / np.linalg.norm(t0, axis=1, keepdims=True)).reshape(-1)
    res = gamfit.gaussian_reml_optimize_latent(**kw)
    t = np.asarray(res.get("t", res.get("latent"))).reshape(n, -1)
    ang = np.arctan2(t[:, 1], t[:, 0]) if t.shape[1] >= 2 else t[:, 0]
    return max(abs(cc(ang, hue)), abs(cc(-ang, hue))), float(res.get("reml_score", np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra_dirs", nargs="+"); ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--label", nargs="*", default=None)
    a = ap.parse_args()
    for k, d in enumerate(a.extra_dirs):
        lab = (a.label[k] if a.label and k < len(a.label) else Path(d).parent.name)
        V, hue = load(Path(d), a.layer)
        print(f"\n######## {lab} (L{a.layer}, raw {V.shape[1]}-d, NO PCA) ########")
        for name, mani, dim in [("line", "euclidean", 1), ("circle", "circle", 1),
                                ("plane", "euclidean", 2), ("torus", "torus", 2), ("sphere", "sphere", 2)]:
            try:
                hc, reml = fit(V, hue, mani, dim)
                print(f"   {name:7s} hue_cc={hc:+.3f}  reml={reml:.0f}")
            except Exception as e:
                print(f"   {name:7s} ERR {str(e)[:60]}")


if __name__ == "__main__":
    main()
