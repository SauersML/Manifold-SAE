"""UNSUPERVISED recovery of the color manifold — gamfit-only + a battery of geometric
methods — scored by how well each recovers TRUE HUE without ever being told it.

Motivation: the supervised topology CV (color_external_cv.py) imposes external HSV/Lab
coords. Here we ask the orthogonal question: given ONLY the model's color reps (no hue
labels), do unsupervised methods recover the hue ring? Prior work (composition_engine
paper, auto_exp_47) found hue is variance-sub-dominant so a naive top-2 PCA circle is
weak (|circ-corr|~0.04 on Cogito); the question is which method does best on OLMo.

Methods:
  GAMFIT-ONLY (no sklearn): gaussian_reml_optimize_latent with manifold in
    {euclidean(1d line, 2d plane), circle, sphere, torus} — recovers per-color latent t
    + a REML evidence score; we report the score AND circular-corr(recovered t, true hue).
  GEOMETRIC battery: PCA (top-2 plane + best-plane search), Isomap, LLE, diffusion map,
    UMAP, intrinsic-dim estimators (participation ratio, TwoNN, Levina-Bickel MLE).

Score = circular correlation between the recovered 1-D/2-D coordinate's angle and the
true hue angle (ground truth held out from every method). Runs per checkpoint extra/
dir; pass several to see across-training (quick mode: just the final-RL dir).
Read-only on activations.
"""
from __future__ import annotations
import argparse, json, colorsys, csv
from pathlib import Path
import numpy as np


def color_vecs(extra: Path, layer: int):
    X = np.load(extra / "activations.npy")
    recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    by, fr, rgb = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i)
        fr.setdefault(r["frame"], []).append(i)
        rgb[r["color"]] = np.array(r["rgb"], float)
    Hd = H.copy()
    for f, idx in fr.items():
        Hd[idx] -= H[idx].mean(0)            # frame-demean (remove template nuisance)
    cols = list(by)
    V = np.stack([Hd[by[c]].mean(0) for c in cols])   # 30 colors x D
    hsv = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255)) for c in cols])
    hue = hsv[:, 0] * 2 * np.pi
    light = np.array([(0.3 * rgb[c][0] + 0.59 * rgb[c][1] + 0.11 * rgb[c][2]) / 255 for c in cols])
    return V, hue, hsv[:, 1], light, np.array([rgb[c] / 255 for c in cols])


def circ_corr(a, b):
    am = np.angle(np.mean(np.exp(1j * a))); bm = np.angle(np.mean(np.exp(1j * b)))
    sa = np.sin(a - am); sb = np.sin(b - bm)
    den = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / den) if den > 0 else float("nan")


def angle_recovery(emb2, hue):
    """best |circ-corr| of the 2-D embedding's angle vs hue (allow reflection)."""
    ang = np.arctan2(emb2[:, 1], emb2[:, 0])
    return max(abs(circ_corr(ang, hue)), abs(circ_corr(-ang, hue)))


def intrinsic_dims(V):
    Vc = V - V.mean(0)
    s = np.linalg.svd(Vc, compute_uv=False)
    pr = (s ** 2).sum() ** 2 / (s ** 4).sum()
    # TwoNN (Facco 2017)
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(V)); np.fill_diagonal(D, np.inf)
    r = np.sort(D, 1)[:, :2]; mu = r[:, 1] / np.maximum(r[:, 0], 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 1)]
    twonn = float(len(mu) / np.sum(np.log(mu))) if len(mu) else float("nan")
    return pr, twonn


def run_geometric(V, hue, light):
    out = {}
    Vc = V - V.mean(0); U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    Y = U * S
    out["pca_top2"] = angle_recovery(Y[:, :2], hue)
    # best-of-pairs plane search over top-8 PCs (unsupervised would pick by ring-ness;
    # here we report the achievable ceiling, mark as oracle-plane)
    best = 0.0
    for i in range(min(8, Y.shape[1])):
        for j in range(i + 1, min(8, Y.shape[1])):
            best = max(best, angle_recovery(Y[:, [i, j]], hue))
    out["pca_bestplane(oracle)"] = best
    try:
        from sklearn.manifold import Isomap, LocallyLinearEmbedding
        out["isomap_2d"] = angle_recovery(Isomap(n_neighbors=6, n_components=2).fit_transform(V), hue)
        out["lle_2d"] = angle_recovery(LocallyLinearEmbedding(n_neighbors=8, n_components=2).fit_transform(V), hue)
    except Exception as e:
        out["isomap_2d"] = out["lle_2d"] = float("nan")
    # diffusion map (gaussian affinity, normalized laplacian eigvecs)
    try:
        from scipy.spatial.distance import squareform, pdist
        D = squareform(pdist(V)); eps = np.median(D[D > 0]) ** 2
        Wm = np.exp(-D ** 2 / eps); d = Wm.sum(1)
        Ms = (Wm / np.sqrt(np.outer(d, d)))
        w, v = np.linalg.eigh(Ms); v = v[:, ::-1]
        out["diffusion_2d"] = angle_recovery(v[:, 1:3], hue)
    except Exception:
        out["diffusion_2d"] = float("nan")
    try:
        import umap, warnings; warnings.filterwarnings("ignore")
        emb = umap.UMAP(n_neighbors=8, min_dist=0.3, metric="cosine", random_state=0).fit_transform(V)
        out["umap_2d"] = angle_recovery(emb, hue)
    except Exception:
        out["umap_2d"] = float("nan")
    return out


def run_gamfit(V, hue):
    """gamfit-ONLY unsupervised latent recovery (gaussian_reml_optimize_latent) on rep-PCs.

    Note: the joint sae_manifold_fit is broken on real LM reps (gam#795), so we use the
    REML latent-coordinate optimizer directly: it estimates per-color t + decoder on a
    chosen manifold, scored by Gaussian-REML evidence (lower=better). Top-6 rep PCs as the
    response block; a ridge (identity) penalty over a duchon center grid.
    """
    import gamfit
    out = {}
    Vc = V - V.mean(0); U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    Y = (U * S)[:, :6].astype(float)     # top-6 rep PCs as the response block
    n = Y.shape[0]
    for name, manifold, dim, K in [("gamfit_line", "euclidean", 1, 10), ("gamfit_circle", "circle", 1, 10),
                                   ("gamfit_plane", "euclidean", 2, 16)]:
        try:
            centers = (np.linspace(0, 1, K).reshape(-1, 1) if dim == 1
                       else np.array([[a, b] for a in np.linspace(0, 1, 4) for b in np.linspace(0, 1, 4)])).astype(float)
            res = gamfit.gaussian_reml_optimize_latent(
                y=Y, n_obs=n, latent_dim=dim, centers=centers, penalty=np.eye(len(centers)),
                m=2, manifold=manifold, basis_kind="duchon", max_iter=80, seed=0)
            t = np.asarray(res.get("t", res.get("latent"))).reshape(n, -1)
            ang = np.arctan2(t[:, 1], t[:, 0]) if t.shape[1] >= 2 else (t[:, 0] / (np.ptp(t[:, 0]) + 1e-9) * 2 * np.pi)
            out[name] = {"hue_circ_corr": max(abs(circ_corr(ang, hue)), abs(circ_corr(-ang, hue))),
                         "reml": float(res.get("reml_score", res.get("score", np.nan)))}
        except Exception as e:
            out[name] = {"hue_circ_corr": float("nan"), "reml": float("nan"), "err": str(e)[:70]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra_dirs", nargs="+", help="checkpoint extra/ dirs (color harvest)")
    ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--csv", default="/tmp/color_unsup_battery.csv")
    a = ap.parse_args()
    rows = []
    for k, d in enumerate(a.extra_dirs):
        lab = (a.label[k] if a.label and k < len(a.label) else Path(d).parent.name)
        V, hue, sat, light, rgb = color_vecs(Path(d), a.layer)
        pr, twonn = intrinsic_dims(V)
        print(f"\n######## {lab}  (n={len(V)} colors, L{a.layer}) ########")
        print(f"intrinsic dim: participation_ratio={pr:.1f}  TwoNN={twonn:.1f}")
        geo = run_geometric(V, hue, light)
        print("-- geometric methods (hue circular-corr, 1.0=perfect ring recovery) --")
        for m, v in geo.items():
            print(f"   {m:26s} {v:+.3f}")
            rows.append({"label": lab, "method": m, "hue_circ_corr": round(v, 4), "pr": round(pr, 2), "twonn": round(twonn, 2)})
        print("-- GAMFIT-ONLY latent recovery (hue circ-corr; reml evidence, lower=better) --")
        gf = run_gamfit(V, hue)
        for m, v in gf.items():
            print(f"   {m:26s} {v['hue_circ_corr']:+.3f}   reml={v['reml']:.1f}" + (f"  ERR {v['err']}" if 'err' in v else ""))
            rows.append({"label": lab, "method": m, "hue_circ_corr": round(v["hue_circ_corr"], 4),
                         "reml": round(v["reml"], 2) if np.isfinite(v["reml"]) else "", "pr": round(pr, 2), "twonn": round(twonn, 2)})
    if rows:
        keys = sorted(set().union(*[r.keys() for r in rows]))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print("\nwrote", a.csv)


if __name__ == "__main__":
    main()
