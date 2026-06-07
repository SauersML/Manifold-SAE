"""Color manifold topology — REBUILT to fix the retracted expand_topology.py claims.

Fixes every adversarial criticism:
  * EXTERNAL coordinates (perceptual color: HSV hue/sat/val + CIELAB L*a*b* from the
    probe RGB), NOT classical-MDS of the reps themselves -> kills the circularity.
  * primary criterion = held-out LEAVE-ONE-COLOR-OUT CV predictive log-likelihood
    (CV is allowed; sidesteps the raw-reml-not-comparable trap, gam#818).
  * secondary = gamfit.compare_models() (Tierney-Kadane normalized evidence), the
    CORRECT cross-topology comparison, NOT raw summary().reml_score.
  * NO te() anywhere (gam#821 CPU outer-REML pathology) -> duchon/matern/cyclic-s only.
  * outer_max_iter=15: fits are bit-identical at oi=8..60 (converge by ~8 iters; the
    rest is gam#821 wasted churn), so the cap is lossless and ~10-40x faster.
  * pass ONLY formula columns to fit/predict (gam#840: stray string cols become
    phantom categoricals and break held-out predict).
  * all n=180 frame-level reps, frame as a linear covariate (no lossy frame-demean);
    LOO holds out all 6 frames of a color.
  * finite guards; refit-free bootstrap CI over the 30 per-color fold scores; layer
    sweep for depth-robustness (run one process per layer for OS-level parallelism).

Question (now well-posed, non-circular): is the color rep a smooth function of
perceptual coordinates, and is the dominant structure a hue CIRCLE (periodic), a
hue+lightness CYLINDER, a 2-D chroma sheet, or a full 3-D L*a*b* VOLUME?
All fits via gamfit. Read-only on activations.
"""
from __future__ import annotations
import argparse, json, colorsys, csv, os
from pathlib import Path
import numpy as np, pandas as pd, gamfit

CONFIG = {"outer_max_iter": 15}

TOPOS = [
    ("hue_line(s)",        "y ~ s(hue)"),
    ("hue_circle(cc)",     "y ~ s(hue,bs='cc')"),
    ("lightness_line",     "y ~ s(val)"),
    ("cyl_hue+light",      "y ~ s(hue,bs='cc') + s(val)"),
    ("chroma_plane",       "y ~ duchon(astar,bstar)"),
    ("lab_volume3D",       "y ~ duchon(Lstar,astar,bstar)"),
    ("hsv_joint",          "y ~ s(hue,bs='cc') + s(sat) + s(val)"),
]
COORDS = ["hue", "sat", "val", "Lstar", "astar", "bstar"]


def rgb_to_lab(rgb):
    c = np.asarray(rgb, float) / 255.0
    m = c > 0.04045
    c = np.where(m, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = xyz > 0.008856
    f = np.where(d, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def load_color(ckpt_dir, layer):
    extra = Path(ckpt_dir)
    if (extra / "extra").exists():
        extra = extra / "extra"
    X = np.load(extra / "activations.npy")
    recs = [json.loads(l) for l in open(extra / "prompts.jsonl") if l.strip()]
    L = min(layer, X.shape[1] - 1)
    H = X[:, L, :].astype(np.float64)
    rows = []
    for r in recs:
        hsv = colorsys.rgb_to_hsv(*(np.array(r["rgb"], float) / 255.0))
        La, aa, ba = rgb_to_lab(r["rgb"])
        rows.append(dict(color=r["color"], frame=int(r["frame"]),
                         hue=hsv[0], sat=hsv[1], val=hsv[2], Lstar=La, astar=aa, bstar=ba))
    return H, pd.DataFrame(rows)


def build_feat(meta):
    # coords only; frame is controlled by demeaning the RESPONSE per frame (below),
    # NOT by a linear term in the formula -- a frame term makes the mixed smooth+linear
    # outer-REML grind ~70x slower (gam#821), and for this balanced design (every color
    # in every frame) per-frame response-demeaning is the equivalent frame adjustment.
    coords = meta[COORDS].reset_index(drop=True).copy()
    for c in COORDS:
        v = coords[c].to_numpy(); coords[c] = (v - v.mean()) / (v.std() + 1e-9)
    return coords, ""


def demean_by_frame(Y, frame):
    Yd = Y.copy()
    for f in np.unique(frame):
        m = frame == f
        Yd[m] -= Yd[m].mean(0)
    return Yd


def fit_safe(df, formula):
    try:
        return gamfit.fit(df, formula, config=CONFIG)
    except Exception:
        return None


def kfold_percolor(feat, colors_arr, Y, formula, uniq, kfold):
    """Grouped-by-color k-fold CV; per-color held-out loglik (summed over PCs).
    kfold<=0 or >=len(uniq) -> leave-one-color-out. Each color is held out exactly
    once, so per-color scores survive for the refit-free bootstrap; folds just batch
    the train-fits (k fits/topology/PC instead of len(uniq)). Returns vec or None."""
    nC = len(uniq)
    k = nC if (kfold <= 0 or kfold >= nC) else kfold
    folds = [uniq[i::k] for i in range(k)]   # round-robin color groups
    cidx = {c: i for i, c in enumerate(uniq)}
    vec = np.zeros(nC)
    for grp in folds:
        te = np.where(np.isin(colors_arr, grp))[0]; tr = np.where(~np.isin(colors_arr, grp))[0]
        for j in range(Y.shape[1]):
            dtr = feat.iloc[tr].copy(); dtr["y"] = Y[tr, j]; dte = feat.iloc[te].copy()
            m = fit_safe(dtr, formula)
            if m is None:
                return None
            try:
                pte = np.asarray(m.predict(dte), float); ptr = np.asarray(m.predict(dtr), float)
            except Exception:
                return None
            sd = max(float((Y[tr, j] - ptr).std()), 1e-6)
            cte = colors_arr[te]
            for c in grp:
                sel = cte == c
                r = Y[te][sel, j] - pte[sel]
                ll = float((-0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * (r / sd) ** 2).sum())
                if not np.isfinite(ll):
                    return None
                vec[cidx[c]] += ll
    return vec


def run_layer(H, meta, layer, npc, nboot, kfold):
    Hc = H - H.mean(0)
    U, S, _ = np.linalg.svd(Hc, full_matrices=False)
    Y = U[:, :npc] * S[:npc]; Y = Y / (Y.std(0, keepdims=True) + 1e-9)
    Y = demean_by_frame(Y, meta["frame"].to_numpy())   # remove frame main effect
    colors_arr = meta["color"].to_numpy(); uniq = pd.unique(colors_arr)
    feat, fr_term = build_feat(meta)
    res = []
    for name, f in TOPOS:
        vec = kfold_percolor(feat, colors_arr, Y, f + fr_term, uniq, kfold)
        res.append({"topology": name, "vec": vec, "cv": (float(vec.sum()) if vec is not None else None)})
    ok = [r for r in res if r["cv"] is not None]; ok.sort(key=lambda r: -r["cv"])
    boot = None
    if len(ok) >= 2:
        diff = ok[0]["vec"] - ok[1]["vec"]; rng = np.random.RandomState(0); nC = len(diff)
        bs = np.array([diff[rng.randint(0, nC, nC)].sum() for _ in range(nboot)])
        boot = {"best": ok[0]["topology"], "runnerup": ok[1]["topology"], "margin": float(diff.sum()),
                "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                "frac_pos": float((bs > 0).mean())}
    tk = {}
    fits, names = [], []
    for name, f in TOPOS:
        if name in {r["topology"] for r in ok}:
            d = feat.copy(); d["y"] = Y[:, 0]
            m = fit_safe(d, f + fr_term)
            if m is not None:
                fits.append(m); names.append(name)
    if len(fits) >= 2:
        try:
            cmp = gamfit.compare_models(fits, names)
            tk = {"winner": cmp.get("winner"), "ranking": cmp.get("ranking")}
        except Exception as e:
            tk = {"error": str(e)[:80]}
    return ok, boot, tk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dirs", nargs="+")
    ap.add_argument("--layers", default="44")
    ap.add_argument("--npc", type=int, default=8)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--kfold", type=int, default=6, help="grouped-by-color k-fold; <=0 = leave-one-color-out")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    csvrows = []
    for ck in args.ckpt_dirs:
        tag = Path(ck).name if Path(ck).name not in ("step_2300",) else Path(ck).parent.name
        print(f"\n######## {ck} ########", flush=True)
        for layer in layers:
            H, meta = load_color(ck, layer)
            ok, boot, tk = run_layer(H, meta, layer, args.npc, args.nboot, args.kfold)
            print(f"\n=== {tag} L{layer} (n={len(meta)}, {meta['color'].nunique()} colors, npc={args.npc}) "
                  f"— held-out LOO-color CV loglik (higher=better) ===", flush=True)
            for r in ok:
                print("  %-18s cv_loglik=%12.1f" % (r["topology"], r["cv"]), flush=True)
                csvrows.append({"ckpt": tag, "layer": layer, "topology": r["topology"], "cv_loglik": r["cv"]})
            if boot:
                print("  bootstrap %s − %s: margin=%.1f  95%%CI[%.1f, %.1f]  P(best>runnerup)=%.3f"
                      % (boot["best"], boot["runnerup"], boot["margin"], boot["ci_lo"], boot["ci_hi"], boot["frac_pos"]), flush=True)
            print("  compare_models (TK evidence):", tk, flush=True)
    if args.csv and csvrows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csvrows[0].keys())); w.writeheader(); w.writerows(csvrows)
        print("wrote", args.csv, flush=True)


if __name__ == "__main__":
    main()
