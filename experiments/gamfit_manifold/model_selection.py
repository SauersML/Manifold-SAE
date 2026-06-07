#!/usr/bin/env python
"""
Multi-criterion topology / dimensionality selection for the OLMo COLOR and
SELF/QUALIA manifolds, using ONLY gamfit-native model-selection scores.

What gamfit actually exposes on a fitted Model (verified empirically via dir(m),
m.summary().to_dict(), dir(gamfit)):
  - m.summary().reml_score   restricted marginal likelihood  (LOWER = better)   [PRIMARY]
  - m.evidence               Laplace model evidence -- numerically IDENTICAL to
                             reml_score on these Gaussian fits, so it is NOT a
                             separate criterion (reported once, as a check).
  - m.summary().deviance     residual deviance (RSS for Gaussian; LOWER = better)
  - m.summary().edf_total    effective degrees of freedom (model complexity used)
  - m.summary().lambdas      smoothing parameters

gamfit ALSO computes two further native scores inside gamfit.select_topology,
via gamfit._select_topology helpers (genuine gamfit code, not rolled-own):
  - BIC = deviance + log(n)*basis_size                (gamfit _bic_value;  LOWER = better)
  - TK  = reml_score + Tierney-Kadane null-space norm (gamfit _score_for_kind 'tk'; LOWER = better)
We call those gamfit helpers directly on each explicit-topology fit so every
candidate is scored by gamfit's own BIC and TK, not by a hand-written formula.

NOT exposed / deliberately NOT reported (they are not gamfit-native here):
  - LAML: no 'laml' field on the summary payload -> gamfit raises; skipped.
  - AIC / GCV / WAIC / LOO: gamfit does not compute them. Not reported.
  - Held-out CV log-lik / R^2: gamfit has no native held-out scorer (only
    predict()); per instruction we do NOT improvise one, so CV is skipped.

Aggregation over the top rep-PC responses: reml/bic/tk/deviance are extensive in
the number of responses, so SUMMED; edf is averaged (mean per-response complexity)
and also reported as a sum.
"""
import os, json, warnings
os.environ.setdefault("GAMFIT_LOG", "off")
os.environ.setdefault("RUST_LOG", "off")
import math
import numpy as np
import pandas as pd
import gamfit
import gamfit._select_topology as st

warnings.filterwarnings("ignore")
BASE = "runs/OLMO3_32B_TRAJ_RL31/step_2300"
OUT = "runs/ANALYSIS"
os.makedirs(OUT, exist_ok=True)
N_PC = 5


# --------------------------------------------------------------- gamfit-native scores
def score_topology(df, formula, n_obs):
    """Fit one (response, formula) and pull gamfit-native reml/bic/tk/dev/edf."""
    m = gamfit.fit(df, formula)
    s = m.summary()
    basis = st._basis_size(m)
    null_dim = st._extract_null_dim(m)
    if null_dim is None:
        null_dim = float(s.null_dim or 0.0)
    reml = st._score_for_kind(m, "reml", n_obs, basis, null_dim)
    bic = st._score_for_kind(m, "bic", n_obs, basis, null_dim)
    tk = st._score_for_kind(m, "tk", n_obs, basis, null_dim)
    return dict(reml=float(reml), bic=float(bic), tk=float(tk),
                deviance=float(s.deviance), edf=float(s.edf_total),
                basis=int(basis), evidence=float(m.evidence))


def run_battery(latents, responses, topologies, label):
    n, npc = responses.shape
    rows = []
    for name, tmpl in topologies:
        crit = {k: [] for k in ["reml", "bic", "tk", "deviance", "edf"]}
        evid = []
        ok, err = True, ""
        for j in range(npc):
            df = latents.copy()
            df["RESP"] = responses[:, j]
            try:
                c = score_topology(df, tmpl, n)
                for k in crit:
                    crit[k].append(c[k])
                evid.append(c["evidence"])
            except Exception as ex:
                ok, err = False, repr(ex)[:160]
                break
        if not ok:
            rows.append(dict(name=name, n=n, status="FAIL", error=err))
            print(f"  [{label}] {name:26s} FAIL {err}")
            continue
        row = dict(name=name, n=n, status="ok",
                   reml=float(np.sum(crit["reml"])),
                   bic=float(np.sum(crit["bic"])),
                   tk=float(np.sum(crit["tk"])),
                   deviance=float(np.sum(crit["deviance"])),
                   edf_mean=float(np.mean(crit["edf"])),
                   edf_sum=float(np.sum(crit["edf"])),
                   evidence=float(np.sum(evid)))
        rows.append(row)
        # sanity: evidence sum should equal reml sum (they are the same score)
        print(f"  [{label}] {name:26s} reml={row['reml']:11.3f} bic={row['bic']:11.3f} "
              f"tk={row['tk']:11.3f} dev={row['deviance']:10.4f} edf={row['edf_mean']:6.2f}")
    return pd.DataFrame(rows)


def winners(tbl):
    t = tbl[tbl["status"] == "ok"]
    out = {}
    for c in ["reml", "bic", "tk", "deviance"]:   # all lower-is-better
        out[c] = t.loc[t[c].idxmin(), "name"]
    return out


# ----------------------------------------------------------------------- shared utils
def reps_to_pcs(R, n_pc=N_PC):
    Rc = R - R.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    return U[:, :n_pc] * S[:n_pc], ((S**2) / (S**2).sum())[:n_pc]


def classical_mds(R, k=3):
    D = np.sqrt(np.maximum(((R[:, None] - R[None]) ** 2).sum(-1), 0))
    n = len(R)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    w, V = w[o][:k], V[:, o][:, :k]
    return V * np.sqrt(np.maximum(w, 0))


# ----------------------------------------------------------------------- COLOR
def load_color():
    A = np.load(f"{BASE}/extra/activations.npy")
    P = [json.loads(l) for l in open(f"{BASE}/extra/prompts.jsonl")]
    X = A[:, 44, :]
    frame = np.array([p["frame"] for p in P])
    color = np.array([p["color"] for p in P])
    rgb = {p["color"]: np.array(p["rgb"], float) for p in P}
    Xd = X.copy()
    for fr in np.unique(frame):
        msk = frame == fr
        Xd[msk] -= Xd[msk].mean(0, keepdims=True)
    cols = sorted(np.unique(color))
    R = np.stack([Xd[color == c].mean(0) for c in cols])
    RGB = np.stack([rgb[c] for c in cols]) / 255.0
    return cols, R, RGB


def rgb_hsv_lab(RGB):
    import colorsys
    H = np.array([colorsys.rgb_to_hsv(*r) for r in RGB])
    lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (lin(RGB) @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = lambda t: np.where(t > d**3, np.cbrt(t), t / (3 * d**2) + 4 / 29)
    fx, fy, fz = f(xyz[:, 0]), f(xyz[:, 1]), f(xyz[:, 2])
    Lab = np.c_[116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
    return H[:, 0], H[:, 1], H[:, 2], Lab


def color_manifold():
    print("\n=== COLOR manifold (L44, frame-demean, 30 colors) ===")
    cols, R, RGB = load_color()
    scores, var = reps_to_pcs(R)
    print(f"  top-{N_PC} rep-PC var explained: {np.round(var,3)} (cum {var.sum():.3f})")
    hue, sat, val, Lab = rgb_hsv_lab(RGB)
    mds = classical_mds(R, 3)
    lat = pd.DataFrame(dict(
        hue=hue, cos_hue=np.cos(2 * np.pi * hue), sin_hue=np.sin(2 * np.pi * hue),
        mds1=mds[:, 0], mds2=mds[:, 1], mds3=mds[:, 2],
        R=RGB[:, 0], G=RGB[:, 1], B=RGB[:, 2],
        L=Lab[:, 0], a=Lab[:, 1], bb=Lab[:, 2], sat=sat, val=val))
    topo = [
        ("line_s(mds1)",          "RESP ~ s(mds1)"),
        ("cyclic_hue_circle",     "RESP ~ s(hue,bs='cc')"),
        ("circle_cos_sin_hue",    "RESP ~ te(cos_hue,sin_hue)"),
        ("sheet_te(mds1,mds2)",   "RESP ~ te(mds1,mds2)"),
        ("sheet_duchon_mds",      "RESP ~ duchon(mds1,mds2)"),
        ("sheet_matern_mds",      "RESP ~ matern(mds1,mds2)"),
        # SKIP-slow-3Dte ("volume_te(mds1,2,3)",   "RESP ~ te(mds1,mds2,mds3)"),
        # SKIP-slow-3Dte ("rgb_volume_te",         "RESP ~ te(R,G,B)"),
        ("rgb_duchon",            "RESP ~ duchon(R,G,B)"),
        ("hsv_sheet_te(h,s)",     "RESP ~ te(hue,sat)"),
        ("lab_sheet_te(a,bb)",    "RESP ~ te(a,bb)"),
        # SKIP-slow-3Dte ("lab_volume_te(L,a,bb)", "RESP ~ te(L,a,bb)"),
    ]
    tbl = run_battery(lat, scores, topo, "COLOR")
    tbl.to_csv(f"{OUT}/model_selection_color.csv", index=False)
    return tbl


# ----------------------------------------------------------------------- QUALIA
def load_qualia():
    A = np.load(f"{BASE}/activations.npy")
    P = [json.loads(l) for l in open(f"{BASE}/prompts.jsonl")]
    X = A[:, 25, :]
    import collections
    pairs = collections.defaultdict(dict)
    for i, p in enumerate(P):
        if p["role"] == "pair" and p["side"] in ("exp", "noexp"):
            pairs[p["pair_id"]][p["side"]] = i
    pids = [pid for pid, d in pairs.items() if "exp" in d and "noexp" in d]
    exp = [pairs[pid]["exp"] for pid in pids]
    nox = [pairs[pid]["noexp"] for pid in pids]
    axis = X[exp].mean(0) - X[nox].mean(0)
    axis /= np.linalg.norm(axis)
    ent = 0.5 * (X[exp] + X[nox])
    t = ent @ axis
    kinds = [P[pairs[pid]["exp"]]["kind"] for pid in pids]
    return ent, t, kinds


def kind_axis(ent, kinds):
    mind = {"human", "mammal", "animal", "bird", "fish", "reptile", "self",
            "conscious_machine", "upload", "group_mind", "collective",
            "split_brain", "supernatural", "dead", "ai"}
    mech = {"rock", "tool", "robot", "vehicle", "artifact", "microbe", "virus",
            "fungus", "plant", "carnivorous_plant", "organoid", "chinese_room",
            "simulator", "simulated", "developmental", "neuro_edge", "fiction",
            "cnidarian", "mollusk", "insect"}
    ks = np.array(kinds)
    ax = ent[np.isin(ks, list(mind))].mean(0) - ent[np.isin(ks, list(mech))].mean(0)
    ax /= np.linalg.norm(ax)
    return ent @ ax


def qualia_manifold():
    print("\n=== SELF/QUALIA manifold (L25, 254 entity pairs) ===")
    ent, t, kinds = load_qualia()
    scores, var = reps_to_pcs(ent)
    print(f"  top-{N_PC} rep-PC var explained: {np.round(var,3)} (cum {var.sum():.3f})")
    kproj = kind_axis(ent, kinds)
    mds = classical_mds(ent, 3)
    z = lambda v: (v - v.mean()) / v.std()
    lat = pd.DataFrame(dict(t=z(t), kind=z(kproj),
                            mds1=mds[:, 0], mds2=mds[:, 1], mds3=mds[:, 2]))
    topo = [
        ("line_s(t)",             "RESP ~ s(t)"),
        ("line_s(kind)",          "RESP ~ s(kind)"),
        ("plane_te(t,kind)",      "RESP ~ te(t,kind)"),
        ("plane_te(t,mds2)",      "RESP ~ te(t,mds2)"),
        ("sheet_te(mds1,mds2)",   "RESP ~ te(mds1,mds2)"),
        ("sheet_duchon_mds",      "RESP ~ duchon(mds1,mds2)"),
        ("sheet_matern_mds",      "RESP ~ matern(mds1,mds2)"),
        ("additive_s(t)+s(kind)", "RESP ~ s(t)+s(kind)"),
        # SKIP-slow-3Dte ("volume_te(mds1,2,3)",   "RESP ~ te(mds1,mds2,mds3)"),
    ]
    tbl = run_battery(lat, scores, topo, "QUALIA")
    tbl.to_csv(f"{OUT}/model_selection_qualia.csv", index=False)
    return tbl


def report(tbl, name):
    print(f"\n----- {name}: per-criterion winners (all gamfit-native, lower=better) -----")
    w = winners(tbl)
    for c, win in w.items():
        print(f"  {c:9s} -> {win}")
    from collections import Counter
    v = Counter(w.values())
    print(f"  agreement: {dict(v)}  | plurality: {v.most_common(1)[0]}")
    # check evidence == reml
    t = tbl[tbl["status"] == "ok"]
    mism = (np.abs(t["evidence"] - t["reml"]) > 1e-6).sum()
    print(f"  check: m.evidence == reml_score for all rows? {'YES' if mism==0 else f'NO ({mism} differ)'}")


if __name__ == "__main__":
    ct = color_manifold()
    qt = qualia_manifold()
    report(ct, "COLOR")
    report(qt, "QUALIA")
    print(f"\nWrote {OUT}/model_selection_color.csv and {OUT}/model_selection_qualia.csv")
