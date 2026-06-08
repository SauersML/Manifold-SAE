"""Full UNSUPERVISED characterization of the color manifold — no hue presupposed.

Instead of testing for hue, discover what the color representation geometry actually is:
  * intrinsic dimension (participation ratio + TwoNN)
  * topology via persistent homology (H0 components, H1 loops, H2 voids) -- needs `ripser`
  * what each leading PC encodes (|corr| with hue/sat/val/light/R/G/B; hue circular)
  * Mantel: which perceptual attribute structures rep-distance (Spearman)

Finding (OLMo L44, final RL): a fat ~20-d cloud with ONE modest H1 loop (the hue ring,
persistence ~1.8 over noise); the leading PCs encode SATURATION (PC2/PC5) and RGB/VALUE
(PC3=red, PC1/PC4=green) more than hue (hue weak per-PC, max ~0.25); Mantel saturation ~
hue > lightness. So hue does not uniquely organize color — it 'wins' only in the supervised
topology CV because it was a hand-picked candidate. Read-only.
"""
from __future__ import annotations
import argparse, json, colorsys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")


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
    RGB = np.array([rgb[c] / 255 for c in cols]); hsv = np.array([colorsys.rgb_to_hsv(*r) for r in RGB])
    light = 0.3 * RGB[:, 0] + 0.59 * RGB[:, 1] + 0.11 * RGB[:, 2]
    A = {"hue": hsv[:, 0] * 2 * np.pi, "sat": hsv[:, 1], "val": hsv[:, 2], "light": light,
         "R": RGB[:, 0], "G": RGB[:, 1], "B": RGB[:, 2]}
    return V, A


def huecorr(vec, hue):
    a = 2 * np.pi * (vec - vec.min()) / (np.ptp(vec) + 1e-9)
    am = np.angle(np.mean(np.exp(1j * a))); bm = np.angle(np.mean(np.exp(1j * hue)))
    sa = np.sin(a - am); sb = np.sin(hue - bm)
    return abs(float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra_dirs", nargs="+"); ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--label", nargs="*", default=None)
    a = ap.parse_args()
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    for k, d in enumerate(a.extra_dirs):
        lab = (a.label[k] if a.label and k < len(a.label) else Path(d).parent.name)
        V, A = load(Path(d), a.layer)
        Vc = V - V.mean(0); U, S, _ = np.linalg.svd(Vc, full_matrices=False); Y = U * S
        pr = (S ** 2).sum() ** 2 / (S ** 4).sum()
        Dm = squareform(pdist(V)); np.fill_diagonal(Dm, np.inf)
        rr = np.sort(Dm, 1)[:, :2]; mu = rr[:, 1] / np.maximum(rr[:, 0], 1e-9); mu = mu[mu > 1]
        twonn = len(mu) / np.sum(np.log(mu))
        print(f"\n######## {lab} (L{a.layer}) ########")
        print(f"intrinsic dim: participation_ratio={pr:.1f} TwoNN={twonn:.1f}  var(PC1-6)={np.round((S**2/(S**2).sum())[:6],3)}")
        try:
            from ripser import ripser
            for h, dd in enumerate(ripser(V, maxdim=2)["dgms"]):
                fin = dd[np.isfinite(dd[:, 1])]; life = (fin[:, 1] - fin[:, 0]) if len(fin) else np.array([0.0])
                print(f"  H{h}: {len(dd)} features, top persistence={life.max():.2f}")
        except Exception:
            print("  (ripser not installed; skipping persistent homology)")
        print("what each PC encodes (|corr|; hue=circular):  hue sat val light R G B -> dominant")
        for i in range(6):
            e = {kk: (huecorr(Y[:, i], A["hue"]) if kk == "hue" else abs(np.corrcoef(Y[:, i], A[kk])[0, 1])) for kk in A}
            print(f"  PC{i+1}: " + " ".join(f"{e[kk]:.2f}" for kk in ["hue", "sat", "val", "light", "R", "G", "B"]) + f"  -> {max(e, key=e.get)}")
        print("Mantel (rep-dist vs attr-dist, Spearman):")
        repd = pdist(V)
        for kk, v in A.items():
            pd_ = pdist(np.c_[np.cos(v), np.sin(v)]) if kk == "hue" else pdist(v.reshape(-1, 1))
            print(f"  {kk:6s}: {spearmanr(repd, pd_).correlation:+.2f}")


if __name__ == "__main__":
    main()
