"""GAMFIT-ONLY unsupervised color-manifold fitting + model selection.

No external coordinates, no sklearn (PCA only as a lossless rotation of the 30-color
data into its <=29-dim span, for compute). For each candidate latent manifold we run
gamfit's REML latent-coordinate optimizer (gaussian_reml_optimize_latent) — it jointly
recovers the per-color latent t and the decoder, scored by Gaussian-REML EVIDENCE
(reml_score, lower=better, the gamfit-native model-selection criterion). We compare:
  line (R^1) | circle (S^1) | plane (R^2) | sphere (S^2) | torus (T^2)
plus a EUCLIDEAN DIMENSION LADDER (d=1..4) whose reml elbow = intrinsic dim.
Hue is used ONLY to interpret the recovered coordinate (circular corr), never to fit.
Runs per checkpoint extra/ dir. gamfit-only. Read-only.
"""
from __future__ import annotations
import argparse, json, colorsys, csv
from pathlib import Path
import numpy as np
import gamfit


def color_block(extra: Path, layer: int, n_resp: int = 29):
    X = np.load(extra / "activations.npy")
    recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    by, fr, rgb = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i)
        rgb[r["color"]] = np.array(r["rgb"], float)
    Hd = H.copy()
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)
    cols = list(by); V = np.stack([Hd[by[c]].mean(0) for c in cols])
    hue = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255))[0] * 2 * np.pi for c in cols])
    Vc = V - V.mean(0); U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    return (U * S)[:, :n_resp].astype(float), hue          # lossless response block (n<=29 captures all variance)


def circ_corr(a, b):
    am = np.angle(np.mean(np.exp(1j * a))); bm = np.angle(np.mean(np.exp(1j * b)))
    sa = np.sin(a - am); sb = np.sin(b - bm)
    den = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / den) if den > 0 else float("nan")


def fit_manifold(Y, manifold, dim, hue, K=None):
    n = len(Y)
    if K is None:
        K = 12 if dim == 1 else 16
    rng = np.random.RandomState(0)
    import itertools
    if dim == 1:
        centers = np.linspace(0, 1, K).reshape(-1, 1)
    elif manifold == "sphere":
        c = rng.randn(K, 3); centers = c / np.linalg.norm(c, axis=1, keepdims=True)
    else:                                   # euclidean / torus, dim>=2: deterministic d-dim grid
        g = max(2, int(round(K ** (1.0 / dim))))
        centers = np.array(list(itertools.product(*([np.linspace(0, 1, g)] * dim))), float)
    kw = dict(y=Y, n_obs=n, latent_dim=(3 if manifold == "sphere" else dim),
              centers=centers.astype(float), penalty=np.eye(len(centers)),
              m=2, manifold=manifold, basis_kind="duchon", max_iter=100, seed=0)
    if manifold == "sphere":
        t0 = rng.randn(n, 3); t0 /= np.linalg.norm(t0, axis=1, keepdims=True)
        kw["t"] = t0.reshape(-1)
    res = gamfit.gaussian_reml_optimize_latent(**kw)
    t = np.asarray(res.get("t", res.get("latent"))).reshape(n, -1)
    ang = np.arctan2(t[:, 1], t[:, 0]) if t.shape[1] >= 2 else t[:, 0]
    hc = max(abs(circ_corr(ang, hue)), abs(circ_corr(-ang, hue)))
    return float(res.get("reml_score", res.get("score", np.nan))), hc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra_dirs", nargs="+")
    ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--csv", default="/tmp/color_gamfit_select.csv")
    a = ap.parse_args()
    CANDS = [("line", "euclidean", 1), ("circle", "circle", 1), ("plane", "euclidean", 2),
             ("sphere", "sphere", 2), ("torus", "torus", 2)]
    rows = []
    for k, d in enumerate(a.extra_dirs):
        lab = (a.label[k] if a.label and k < len(a.label) else Path(d).parent.name)
        Y, hue = color_block(Path(d), a.layer)
        print(f"\n######## {lab}  (n={len(Y)} colors, L{a.layer}, gamfit-only) ########")
        print("-- manifold model selection (gamfit REML evidence, LOWER=better) --")
        res = []
        for name, mani, dim in CANDS:
            try:
                reml, hc = fit_manifold(Y, mani, dim, hue)
                res.append((name, reml, hc))
                print(f"   {name:8s} reml={reml:9.1f}   hue_circ_corr={hc:+.3f}")
                rows.append({"label": lab, "kind": "manifold", "name": name, "reml": round(reml, 2), "hue_cc": round(hc, 4)})
            except Exception as e:
                print(f"   {name:8s} ERR {str(e)[:70]}")
        if res:
            win = min(res, key=lambda r: r[1])
            print(f"   --> gamfit prefers: {win[0]} (reml {win[1]:.1f})")
        print("-- euclidean DIMENSION LADDER (reml elbow = intrinsic dim) --")
        ladder = []
        for dim in [1, 2, 3, 4]:
            try:
                reml, _ = fit_manifold(Y, "euclidean", dim, hue, K=(12 if dim == 1 else 9 if dim == 2 else 8))
                ladder.append((dim, reml)); print(f"   d={dim}: reml={reml:9.1f}")
                rows.append({"label": lab, "kind": "ladder", "name": f"euclid_d{dim}", "reml": round(reml, 2), "hue_cc": ""})
            except Exception as e:
                print(f"   d={dim}: ERR {str(e)[:60]}")
    if rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("\nwrote", a.csv)


if __name__ == "__main__":
    main()
