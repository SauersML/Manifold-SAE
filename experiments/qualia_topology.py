"""Qualia/self ENTITY-manifold topology selection via gamfit, across training.

For each checkpoint, take the entity pair reps (role==pair, 508 rows) at L25,
reduce to top rep-PCs as responses, build latent coords (t = qualia coordinate =
projection on the unit mean(exp)-mean(noexp) axis; mds1..3 = classical MDS of the
rep distance matrix), and fit candidate topologies with gamfit. Rank by gamfit-
native scores (reml_score primary; + deviance, edf; + bic/tk via gamfit helpers
when available). Question: is the entity manifold linear in t, a curved sheet, or
higher-dim? Uses config outer_max_iter cap (the 3D-te over-iteration 6x speedup).
Read-only; gamfit-only for all fits.
"""
from __future__ import annotations
import glob, json, csv, os, sys
import numpy as np
import gamfit
try:
    import gamfit._select_topology as _st
except Exception:
    _st = None

CONFIG = {"outer_max_iter": 30}
TOPOS = [
    ("line_s(t)",        "{y} ~ s(t)"),
    ("curved_s(mds1)",   "{y} ~ s(mds1)"),
    ("plane_te(t,mds2)", "{y} ~ te(t,mds2)"),
    ("sheet_te(mds)",    "{y} ~ te(mds1,mds2)"),
    ("sheet_duchon",     "{y} ~ duchon(mds1,mds2)"),
]


def classical_mds(D, k=3):
    n = len(D); J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(-w)[:k]; w = np.clip(w[idx], 0, None)
    return V[:, idx] * np.sqrt(w + 1e-12)


def fit_score(df, formula):
    m = gamfit.fit(df, formula, config=CONFIG)
    reml = float(getattr(m, "reml_score", np.nan))
    if np.isnan(reml):
        s = m.summary() if hasattr(m, "summary") else None
        reml = float(getattr(s, "reml_score", getattr(s, "reml", np.nan)) if s is not None else np.nan)
    edf = float(getattr(m, "edf_total", np.nan))
    dev = float(getattr(m, "deviance", np.nan))
    bic = np.nan
    if _st is not None:
        for fn in ("_bic_value", "bic_value"):
            if hasattr(_st, fn):
                try: bic = float(getattr(_st, fn)(m)); break
                except Exception: pass
    return reml, bic, edf, dev


def run_ckpt(d, layer=25, npc=5):
    X = np.load(os.path.join(d, "activations.npy"))
    recs = [json.loads(l) for l in open(os.path.join(d, "prompts.jsonl")) if l.strip()]
    L = min(layer, X.shape[1] - 1); H = X[:, L, :].astype(np.float64)
    role = np.array([r["role"] for r in recs]); side = np.array([r["side"] for r in recs])
    pair = np.where(role == "pair")[0]
    ie = np.where((role == "pair") & (side == "exp"))[0]; ino = np.where((role == "pair") & (side == "noexp"))[0]
    axis = H[ie].mean(0) - H[ino].mean(0); axis /= max(np.linalg.norm(axis), 1e-9)
    P = H[pair]; t = P @ axis
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    Dm = 1 - Pn @ Pn.T
    mds = classical_mds(Dm, 3)
    Pc = P - P.mean(0); U, S, _ = np.linalg.svd(Pc, full_matrices=False)
    Y = U[:, :npc] * S[:npc]
    import pandas as pd
    base = pd.DataFrame({"t": (t - t.mean()) / (t.std() + 1e-9),
                         "mds1": mds[:, 0], "mds2": mds[:, 1], "mds3": mds[:, 2]})
    rows = []
    for name, tmpl in TOPOS:
        agg = {"reml": 0.0, "bic": 0.0, "edf": 0.0, "dev": 0.0, "ok": True}
        for j in range(npc):
            df = base.copy(); df["y"] = Y[:, j]
            try:
                reml, bic, edf, dev = fit_score(df, tmpl.format(y="y"))
                agg["reml"] += reml; agg["bic"] += (bic if not np.isnan(bic) else 0)
                agg["edf"] += edf; agg["dev"] += dev
            except Exception as e:
                agg["ok"] = False; agg["err"] = str(e)[:80]; break
        rows.append((name, agg))
    return rows


def main():
    qroot = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/qdata"
    out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/nvme/qdata/qualia_topology.csv"
    dirs = sorted([d for d in glob.glob(qroot + "/*") if os.path.exists(os.path.join(d, "activations.npy"))])
    allrows = []
    for d in dirs:
        ck = os.path.basename(d)
        print(f"\n=== {ck} ===", flush=True)
        try:
            rows = run_ckpt(d)
        except Exception as e:
            print("  FAIL", e); continue
        best = min((r for r in rows if r[1]["ok"]), key=lambda r: r[1]["reml"], default=None)
        for name, a in rows:
            if a["ok"]:
                print("  %-18s reml=%9.1f bic=%9.1f edf=%6.2f dev=%9.1f" % (name, a["reml"], a["bic"], a["edf"], a["dev"]))
                allrows.append({"checkpoint": ck, "topology": name, "reml": a["reml"], "bic": a["bic"], "edf": a["edf"], "deviance": a["dev"]})
            else:
                print("  %-18s ERR %s" % (name, a.get("err")))
        if best: print("  -> BEST (reml):", best[0])
    if allrows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
