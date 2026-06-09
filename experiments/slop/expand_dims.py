"""gamfit Duchon dimension LADDER: how many latent dims does each manifold need?

For each manifold (qualia pair reps L25 n=508; color frame-demeaned means L44 n=30),
build classical-MDS coords from the rep cosine-distance, then fit a Duchon spline of
increasing dimensionality d=1..DMAX (duchon(mds1..mds_d)) and read gamfit-native
reml_score (=evidence, lower=better) + bic. The dimension where reml stops dropping
is the intrinsic manifold dimension.

Circularity guard: MDS of (1-cosine) ~ PCA of the normalized reps, so predicting the
TOP-5 rep PCs from k MDS coords improves trivially as k->5. We therefore fit TWO
independent response blocks and require the elbow to agree:
  block A = top-5 rep PCs (the usual responses)
  block B = rep PCs 6..10 (held OUT of the top-5 MDS subspace; a genuine smoothness
            test -- if low-d latent coords still predict mid-spectrum directions and
            higher d adds nothing, that d is the real intrinsic dim, not an artifact).

Hang-proof: each fit runs in a spawn child with a hard timeout. Streamed/flushed CSV.
gamfit-only scores. Read-only on activations.
"""
from __future__ import annotations
import glob, json, csv, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd, gamfit

CONFIG = {"outer_max_iter": 25}
FIT_TIMEOUT = float(os.environ.get("FIT_TIMEOUT", "200"))
DMAX = int(os.environ.get("DMAX", "5"))


def classical_mds(D, k):
    n = len(D); J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); idx = np.argsort(-w)[:k]; w = np.clip(w[idx], 0, None)
    return V[:, idx] * np.sqrt(w + 1e-12)


def _fit_worker(df_dict, formula, n, q):
    try:
        df = pd.DataFrame(df_dict)
        s = gamfit.fit(df, formula, config=CONFIG).summary()
        dev, edf = float(s.deviance), float(s.edf_total)
        q.put((float(s.reml_score), dev + np.log(n) * edf, edf, dev))
    except Exception as e:  # noqa: BLE001
        q.put(("ERR", str(e)[:90]))


def fit_one(df, formula, n):
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    dd = {c: df[c].to_numpy() for c in df.columns}
    p = ctx.Process(target=_fit_worker, args=(dd, formula, n, q)); p.start(); p.join(FIT_TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join(); raise TimeoutError(f"TIMEOUT>{FIT_TIMEOUT:.0f}s")
    if q.empty():
        raise RuntimeError("worker died")
    r = q.get()
    if r and r[0] == "ERR":
        raise RuntimeError(r[1])
    return r


def ladder(P, manifold, ck, w, f):
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    mds = classical_mds(1 - Pn @ Pn.T, DMAX)
    Pc = P - P.mean(0); U, S, _ = np.linalg.svd(Pc, full_matrices=False)
    Yhi = U[:, :5] * S[:5]                                   # block A: top-5 PCs
    nlow = min(5, U.shape[1] - 5)
    Ylo = (U[:, 5:5 + nlow] * S[5:5 + nlow]) if nlow > 0 else None   # block B: PCs 6..10
    base = pd.DataFrame({f"mds{i+1}": mds[:, i] for i in range(DMAX)})
    for d in range(1, DMAX + 1):
        coords = ",".join(f"mds{i+1}" for i in range(d))
        formula = f"y ~ duchon({coords})" if d >= 2 else "y ~ s(mds1)"
        for blk, Y in (("topPC1-5", Yhi), ("PC6-10", Ylo)):
            if Y is None:
                continue
            t0 = time.time(); agg = [0.0, 0.0, 0.0, 0.0]; ok = True; err = ""
            for j in range(Y.shape[1]):
                df = base.copy(); df["y"] = Y[:, j]
                try:
                    r = fit_one(df, formula, len(df)); agg = [a + b for a, b in zip(agg, r)]
                except Exception as e:  # noqa: BLE001
                    ok = False; err = str(e)[:80]; break
            secs = time.time() - t0
            if ok:
                print("  %-8s d=%d %-22s reml=%9.1f bic=%9.1f edf=%6.1f (%.0fs)"
                      % (blk, d, formula, agg[0], agg[1], agg[2], secs), flush=True)
                w.writerow({"manifold": manifold, "checkpoint": ck, "block": blk, "dim": d,
                            "reml": agg[0], "bic": agg[1], "edf": agg[2], "dev": agg[3], "secs": round(secs, 1)})
                f.flush()
            else:
                print("  %-8s d=%d %-22s %s (%.0fs)" % (blk, d, formula, err, secs), flush=True)


def main():
    qroot = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/qdata")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/qdata/expand_dims.csv")
    f = open(out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["manifold", "checkpoint", "block", "dim", "reml", "bic", "edf", "dev", "secs"])
    w.writeheader(); f.flush()
    for d in sorted(glob.glob(qroot + "/*")):
        if not os.path.isdir(d):
            continue
        ck = os.path.basename(d)
        if os.path.exists(d + "/activations.npy"):
            X = np.load(d + "/activations.npy"); recs = [json.loads(l) for l in open(d + "/prompts.jsonl") if l.strip()]
            role = np.array([r["role"] for r in recs]); P = X[:, 25, :][np.where(role == "pair")[0]].astype(np.float64)
            print(f"\n=== QUALIA {ck} (n={len(P)}) ===", flush=True); ladder(P, "qualia", ck, w, f)
        ed = d + "/extra"
        if os.path.exists(ed + "/activations.npy"):
            X = np.load(ed + "/activations.npy"); recs = [json.loads(l) for l in open(ed + "/prompts.jsonl") if l.strip()]
            H = X[:, 44, :].astype(np.float64); by, fr = {}, {}
            for i, r in enumerate(recs):
                by.setdefault(r["color"], []).append(i); fr.setdefault(r["frame"], []).append(i)
            Hd = H.copy()
            for fk, idx in fr.items():
                Hd[idx] -= H[idx].mean(0)
            cols = list(by); P = np.stack([Hd[by[c]].mean(0) for c in cols])
            print(f"=== COLOR {ck} (n={len(P)}) ===", flush=True); ladder(P, "color", ck, w, f)
    f.close(); print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
