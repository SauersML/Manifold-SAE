"""Expanded gamfit topology battery for BOTH manifolds: tests whether a SPHERE (S2),
MATERN sheet, or 3-D (duchon/te) "hyperobject" beats the 2-D Duchon sheet, plus a
cyclic-spline CIRCLE and an additive JOINT for color. gamfit-native scores only
(reml_score=evidence, edf_total, deviance via m.summary()); capped outer iters.

Hang-proof: every single (topology, PC) fit runs in a child process with a hard
wall-clock timeout (the 3-D tensor fits on qualia n=508 are pathologically slow,
gam#813 — ~70s/PIRLS-iter — so they get marked TIMEOUT instead of wedging the whole
battery). Results stream to CSV incrementally and every print is flushed, so partial
progress always survives a kill/sleep. gamfit sphere() = exactly 2 coords (lat,lon ->
S^2); periodic via s(x,bs='cc'). Read-only on the activations.
"""
from __future__ import annotations
import glob, json, csv, os, sys, colorsys, time
import multiprocessing as mp
import numpy as np, pandas as pd, gamfit

CONFIG = {"outer_max_iter": 25}
FIT_TIMEOUT = float(os.environ.get("FIT_TIMEOUT", "240"))  # seconds per single fit


def classical_mds(D, k=3):
    n = len(D); J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); idx = np.argsort(-w)[:k]; w = np.clip(w[idx], 0, None)
    return V[:, idx] * np.sqrt(w + 1e-12)


def _fit_worker(df_dict, formula, n, q):
    """Runs in a child process; puts (reml, bic, edf, dev) or ('ERR', msg)."""
    try:
        df = pd.DataFrame(df_dict)
        s = gamfit.fit(df, formula, config=CONFIG).summary()
        dev, edf = float(s.deviance), float(s.edf_total)
        q.put((float(s.reml_score), dev + np.log(n) * edf, edf, dev))
    except Exception as e:  # noqa: BLE001
        q.put(("ERR", str(e)[:90]))


def fit_one(df, formula, n, timeout=FIT_TIMEOUT):
    """One fit in a child process with a hard timeout. Returns tuple or raises."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    df_dict = {c: df[c].to_numpy() for c in df.columns}
    p = ctx.Process(target=_fit_worker, args=(df_dict, formula, n, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        raise TimeoutError(f"TIMEOUT>{timeout:.0f}s")
    if q.empty():
        raise RuntimeError("worker died (no result)")
    r = q.get()
    if r and r[0] == "ERR":
        raise RuntimeError(r[1])
    return r


def battery(P, extra_cols, topos, npc=5, label=""):
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    mds = classical_mds(1 - Pn @ Pn.T, 3)
    Pc = P - P.mean(0); U, S, _ = np.linalg.svd(Pc, full_matrices=False); Y = U[:, :npc] * S[:npc]
    base = pd.DataFrame({"mds1": mds[:, 0], "mds2": mds[:, 1], "mds3": mds[:, 2], **extra_cols})
    out = []
    for name, tmpl in topos:
        t0 = time.time(); agg = [0.0, 0.0, 0.0, 0.0]; ok = True; err = ""
        for j in range(npc):
            df = base.copy(); df["y"] = Y[:, j]
            try:
                r = fit_one(df, tmpl, len(df)); agg = [a + b for a, b in zip(agg, r)]
            except Exception as e:  # noqa: BLE001
                ok = False; err = str(e)[:90]; break
        out.append((name, agg, ok, err, time.time() - t0))
    return out


COMMON = [
    ("line_s(mds1)",     "y ~ s(mds1)"),
    ("plane_te(mds)",    "y ~ te(mds1,mds2)"),
    ("sheet_duchon2D",   "y ~ duchon(mds1,mds2)"),
    ("sheet_matern2D",   "y ~ matern(mds1,mds2)"),
    ("sphere_S2",        "y ~ sphere(mds1,mds2)"),
    ("vol_duchon3D",     "y ~ duchon(mds1,mds2,mds3)"),
    ("vol_te3D",         "y ~ te(mds1,mds2,mds3)"),
]
COLOR_EXTRA = [
    ("circle_cc_hue",    "y ~ s(hue,bs='cc')"),
    ("joint_hue+sv",     "y ~ s(hue,bs='cc') + s(sat) + s(val)"),
]


def main():
    qroot = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/qdata")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/qdata/expand_topology.csv")
    fields = ["manifold", "checkpoint", "topology", "reml", "bic", "edf", "dev", "secs"]
    f = open(out, "w", newline=""); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); f.flush()

    def emit(manifold, ck, name, a, ok, err, secs):
        if ok:
            print("  %-16s reml=%9.1f bic=%9.1f edf=%6.1f  (%.0fs)" % (name, a[0], a[1], a[2], secs), flush=True)
            w.writerow({"manifold": manifold, "checkpoint": ck, "topology": name,
                        "reml": a[0], "bic": a[1], "edf": a[2], "dev": a[3], "secs": round(secs, 1)})
            f.flush()
        else:
            print("  %-16s %-12s (%.0fs)" % (name, err, secs), flush=True)

    for d in sorted(glob.glob(qroot + "/*")):
        if not os.path.isdir(d):
            continue
        ck = os.path.basename(d)
        # QUALIA manifold (pair reps, L25)
        if os.path.exists(d + "/activations.npy"):
            X = np.load(d + "/activations.npy"); recs = [json.loads(l) for l in open(d + "/prompts.jsonl") if l.strip()]
            role = np.array([r["role"] for r in recs]); P = X[:, 25, :][np.where(role == "pair")[0]].astype(np.float64)
            print(f"\n=== QUALIA {ck} (n={len(P)}) ===", flush=True)
            for name, a, ok, err, secs in battery(P, {}, COMMON, label="qualia"):
                emit("qualia", ck, name, a, ok, err, secs)
        # COLOR manifold (extra, L44)
        ed = d + "/extra"
        if os.path.exists(ed + "/activations.npy"):
            X = np.load(ed + "/activations.npy"); recs = [json.loads(l) for l in open(ed + "/prompts.jsonl") if l.strip()]
            H = X[:, 44, :].astype(np.float64); by, fr, rgb = {}, {}, {}
            for i, r in enumerate(recs):
                by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i); rgb[r["color"]] = np.array(r["rgb"], float)
            Hd = H.copy()
            for fkey, idx in fr.items():
                Hd[idx] -= H[idx].mean(0)
            cols = list(by); P = np.stack([Hd[by[c]].mean(0) for c in cols])
            hsv = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255.0)) for c in cols])
            print(f"=== COLOR {ck} (n={len(P)}) ===", flush=True)
            for name, a, ok, err, secs in battery(P, {"hue": hsv[:, 0], "sat": hsv[:, 1], "val": hsv[:, 2]},
                                                  COMMON + COLOR_EXTRA, label="color"):
                emit("color", ck, name, a, ok, err, secs)
    f.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
