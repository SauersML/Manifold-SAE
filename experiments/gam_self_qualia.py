"""Fit gamfit GAMs over the SUPERVISED (kind, qualia) plane of the self-qualia data.

The contrast axes (kind = mind-minus-mechanism, qualia = experience-minus-no-
experience, both from minimal-pair / anchor stimuli) already give us the
manifold's coordinates. So we do NOT do unsupervised topology discovery
(`sae_manifold_fit` is also broken on this data in 0.1.169 — see gam#795).
Instead we use gamfit's robust GAM path to ask what the contrast axes can't:

  1. CURVATURE  -- is the representation a *linear* function of (kind, qualia)
     or does it bend?  te(kind, qualia) effective-dof vs the linear EDF=4 floor,
     plus deviance explained (how much of the rep the 2D plane captures).
  2. SELF ON/OFF MANIFOLD -- fit the surface on entities ONLY, predict the
     indexical self from its (kind, qualia); the residual is self-specific
     structure not explained by "where it sits on the plane".
  3. WEDGE -- does the experiential (qualia) range widen for mind-like entities?
     A smooth of the qualia mean and of the qualia spread vs kind.

Everything is matched: base and instruct are restricted to their shared
referents and the axes are built from those identical stimuli (model-vs-model,
not bank-vs-bank — the lesson from analyze_self_qualia_compare.py).

Usage:
    python experiments/gam_self_qualia.py \
        --base runs/OLMO3_7B_SELF_QUALIA_MAIN \
        --instruct runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST \
        --layer 22 --out runs/SELF_QUALIA_GAM
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

# Silence gamfit's verbose REML solver logging before importing it.
os.environ.setdefault("RUST_LOG", "off")
os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
import gamfit

C_SELF, C_HUMAN, C_AI, C_BASE, C_INSTRUCT = "#1f77b4", "#2ca02c", "#ff7f0e", "#7f7f7f", "#9467bd"


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def load(d):
    X = np.load(f"{d}/activations.npy")
    rows = list(csv.DictReader(open(f"{d}/prompts.csv")))
    return X, rows


def coords_at(X, rows, layer, keep_refs):
    """Per-referent standardized (kind, qualia) coords + centered rep centroids,
    with axes built from the kept (shared) anchor/pair stimuli only."""
    role = np.array([r["role"] for r in rows]); G = np.array([r["group"] for r in rows])
    PS = np.array([r["pair_side"] for r in rows]); ref = np.array([r["referent"] for r in rows])
    keep = np.array([r["referent"] in keep_refs for r in rows])
    H = X[:, layer, :].astype(np.float64)
    mind = keep & (role == "kind_anchor") & (G == "mind")
    mech = keep & (role == "kind_anchor") & (G == "mechanism")
    exp = keep & (role == "qualia_pair") & (PS == "experience")
    no = keep & (role == "qualia_pair") & (PS == "no_experience")
    kax = _unit(H[mind].mean(0) - H[mech].mean(0))
    qax = _unit(H[exp].mean(0) - H[no].mean(0))
    ks, qs = H @ kax, H @ qax
    refs = sorted(keep_refs)
    kc = np.array([ks[ref == r].mean() for r in refs])
    qc = np.array([qs[ref == r].mean() for r in refs])
    C = np.array([H[ref == r].mean(0) for r in refs])
    is_self = np.array([role[ref == r][0] == "self" for r in refs])
    grp = np.array([G[ref == r][0] for r in refs])
    self_labels = np.array([ref[ref == r][0] if role[ref == r][0] == "self" else ""
                            for r in refs])
    return dict(refs=refs, kc=kc, qc=qc, C=C - C.mean(0), is_self=is_self, grp=grp,
                kax=kax, qax=qax, self_labels=self_labels,
                qlo=float(qs[no].mean()), qhi=float(qs[exp].mean()))


def self_kind_curve(S):
    """Fit qualia ~ s(kind) on ENTITIES; place each self phrasing against the
    entity curve. Answers: does the model grant its author-self more/less
    experience than a typical entity of the self's kind?"""
    ent = ~S["is_self"]
    qn = (S["qc"] - S["qlo"]) / (S["qhi"] - S["qlo"])            # 0=no-exp anchor, 1=exp anchor
    kz = (S["kc"] - S["kc"][ent].mean()) / S["kc"][ent].std()    # standardize kind on entities
    m = gamfit.fit({"kind": kz[ent], "qualia": qn[ent]}, "qualia ~ s(kind)")
    grid = np.linspace(kz[ent].min(), kz[ent].max(), 60)
    qhat = np.asarray(m.predict({"kind": grid}))
    sd = float((qn[ent] - np.asarray(m.predict({"kind": kz[ent]}))).std())
    selves = []
    for i in np.where(S["is_self"])[0]:
        exp = float(m.predict({"kind": kz[i:i + 1]})[0])
        selves.append(dict(label=S["self_labels"][i], kz=float(kz[i]), q=float(qn[i]),
                           expected=exp, resid_sd=float((qn[i] - exp) / (sd + 1e-9))))
    return dict(grid=grid, qhat=qhat, sd=sd, kz_ent=kz[ent], qn_ent=qn[ent], selves=selves)


def _edf(model):
    m = re.search(r"Effective dof:\s*([\d.]+)", str(model.summary()))
    return float(m.group(1)) if m else float("nan")


def _r2(model, kc, qc, y):
    """Deviance explained: 1 - RSS/TSS using the model's predictions on train."""
    fitted = np.asarray(model.predict({"kind": kc, "qualia": qc}))
    rss = float(((y - fitted) ** 2).sum()); tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - rss / (tss + 1e-12)


def plane_fit(S, n_pc=6):
    """Fit te(kind,qualia) -> each top rep-PC. Return mean R2, mean EDF, and the
    self's on/off-manifold residual (fit on entities only)."""
    kc = (S["kc"] - S["kc"].mean()) / S["kc"].std()
    qc = (S["qc"] - S["qc"].mean()) / S["qc"].std()
    ent = ~S["is_self"]
    _, _, Vt = np.linalg.svd(S["C"][ent], full_matrices=False)
    r2s, edfs, self_res, base_res = [], [], [], []
    for j in range(min(n_pc, Vt.shape[0])):
        y_all = S["C"] @ Vt[j]
        ys = y_all[ent]; ys = (ys - ys.mean()) / (ys.std() + 1e-9)
        data = {"kind": kc[ent], "qualia": qc[ent], "y": ys}
        try:
            m = gamfit.fit(data, "y ~ te(kind, qualia)")
        except Exception:
            continue
        r2s.append(_r2(m, kc[ent], qc[ent], ys)); edfs.append(_edf(m))
        # self on/off manifold: predict self PC from its (kind,qualia)
        if S["is_self"].any():
            yself = (y_all[S["is_self"]] - ys.mean()) / (ys.std() + 1e-9)
            pself = np.asarray(m.predict({"kind": kc[S["is_self"]], "qualia": qc[S["is_self"]]}))
            self_res.append(float(np.sqrt(np.mean((yself - pself) ** 2))))
            base_res.append(1.0)  # residual in std units; 1.0 ~ "as far as a typical entity"
    return dict(r2=float(np.mean(r2s)), edf=float(np.mean(edfs)),
                self_resid=float(np.mean(self_res)) if self_res else float("nan"),
                Vt=Vt, kc=kc, qc=qc)


def wedge(S):
    """Does qualia range widen with kind?  Smooth of qualia mean and |resid| vs kind."""
    ent = ~S["is_self"]
    kc = (S["kc"][ent] - S["kc"][ent].mean()) / S["kc"][ent].std()
    q = (S["qc"][ent] - S["qc"][ent].mean()) / S["qc"][ent].std()
    mmean = gamfit.fit({"kind": kc, "qualia": q}, "qualia ~ s(kind)")
    grid = np.linspace(kc.min(), kc.max(), 40)
    qhat = np.asarray(mmean.predict({"kind": grid}))
    resid = np.abs(q - np.asarray(mmean.predict({"kind": kc})))
    mspread = gamfit.fit({"kind": kc, "r": resid}, "r ~ s(kind)")
    shat = np.asarray(mspread.predict({"kind": grid}))
    return dict(grid=grid, qhat=qhat, shat=shat, kc=kc, q=q)


def run(base_dir, instruct_dir, layer, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    Xb, rb = load(base_dir); Xi, ri = load(instruct_dir)
    shared = set(r["referent"] for r in rb) & set(r["referent"] for r in ri)
    Sb = coords_at(Xb, rb, layer, shared)
    Si = coords_at(Xi, ri, layer, shared)
    Pb, Pi = plane_fit(Sb), plane_fit(Si)
    Wb, Wi = wedge(Sb), wedge(Si)
    Cb, Ci = self_kind_curve(Sb), self_kind_curve(Si)

    print("=" * 60)
    print(f"GAM over supervised (kind,qualia) plane | layer {layer} | shared refs={len(shared)}")
    print("=" * 60)
    print(f"  {'':10s} {'deviance_expl':>13s} {'EDF (curvature)':>16s} {'self off-manifold':>18s}")
    for tag, P in (("base", Pb), ("instruct", Pi)):
        print(f"  {tag:10s} {P['r2']:>13.3f} {P['edf']:>16.2f} {P['self_resid']:>18.3f}")
    print(f"  (EDF=4 would be exactly linear in kind,qualia,interaction; >4 = curvature)")
    print("\n  self qualia vs entity qualia~s(kind) curve (residual in entity-SD):")
    for tag, Cur in (("base", Cb), ("instruct", Ci)):
        for s in Cur["selves"]:
            print(f"    {tag:8s} {s['label'][:34]:34s} kind_z={s['kz']:+.2f} "
                  f"q={s['q']:+.2f} expected={s['expected']:+.2f} resid={s['resid_sd']:+.1f}SD")

    _figure(out, layer, Sb, Si, Pb, Pi, Wb, Wi, Cb, Ci)
    return dict(layer=layer, base=Pb, instruct=Pi)


def _self_curve_panel(ax, Cur, tag, col):
    ax.plot(Cur["grid"], Cur["qhat"], color="k", lw=1.5, label="entity qualia~s(kind)")
    ax.fill_between(Cur["grid"], Cur["qhat"] - Cur["sd"], Cur["qhat"] + Cur["sd"],
                    color="0.6", alpha=0.25, label="+/-1 entity SD")
    ax.scatter(Cur["kz_ent"], Cur["qn_ent"], c="0.4", s=14, alpha=0.6)
    for s in Cur["selves"]:
        ax.scatter([s["kz"]], [s["q"]], color=C_SELF, marker="*", s=180, edgecolor="k", zorder=5)
        ax.annotate(s["label"].replace("the ", "").replace(" of these very words", "")[:16],
                    (s["kz"], s["q"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("kind coord (entity-standardized)"); ax.set_ylabel("qualia (0=no-exp, 1=exp)")
    ax.set_title(f"{tag}: self vs entity qualia~kind curve\n(★ self phrasings; on curve = no self-special qualia)", fontsize=10)
    ax.legend(fontsize=7)


def _figure(out, layer, Sb, Si, Pb, Pi, Wb, Wi, Cb, Ci):
    fig, ax = plt.subplots(2, 3, figsize=(18, 11)); fig.patch.set_facecolor("white")

    # A: fitted te(kind,qualia) surface (top PC) for instruct, entities + self overlaid
    a = ax[0, 0]
    S, P = Si, Pi
    kc, qc = P["kc"], P["qc"]
    ent = ~S["is_self"]
    y = S["C"] @ P["Vt"][0]; ye = (y[ent] - y[ent].mean()) / (y[ent].std() + 1e-9)
    gx = np.linspace(kc.min(), kc.max(), 60); gy = np.linspace(qc.min(), qc.max(), 60)
    GX, GY = np.meshgrid(gx, gy)
    m = gamfit.fit({"kind": kc[ent], "qualia": qc[ent], "y": ye}, "y ~ te(kind, qualia)")
    Zs = np.asarray(m.predict({"kind": GX.ravel(), "qualia": GY.ravel()})).reshape(GX.shape)
    cf = a.contourf(GX, GY, Zs, levels=18, cmap="RdBu_r", alpha=0.85)
    a.scatter(kc[ent], qc[ent], c="k", s=10, alpha=0.5)
    for i in np.where(S["is_self"])[0]:
        a.scatter(kc[i], qc[i], c=C_SELF, s=140, edgecolor="k", marker="*", zorder=5)
    a.set_xlabel("kind coord (std)"); a.set_ylabel("qualia coord (std)")
    a.set_title("A. Fitted te(kind,qualia) surface — instruct, top rep-PC\n"
                "★ = indexical self  (is it on the entity surface?)", fontsize=11)
    fig.colorbar(cf, ax=a, fraction=0.046)

    # B: deviance explained + EDF, base vs instruct
    b = ax[0, 1]
    x = np.arange(2)
    b.bar(x - 0.2, [Pb["r2"], Pi["r2"]], 0.4, color=[C_BASE, C_INSTRUCT], label="deviance expl.")
    b.set_ylabel("deviance explained by 2D plane"); b.set_ylim(0, 1)
    b.set_xticks(x); b.set_xticklabels(["base", "instruct"])
    b2 = b.twinx()
    b2.plot(x, [Pb["edf"], Pi["edf"]], "o--", color="k", label="EDF (curvature)")
    b2.axhline(4, color="r", ls=":", alpha=0.6); b2.set_ylabel("effective dof (4 = linear)")
    b.set_title("B. How much the supervised plane explains,\nand how curved (EDF) — base vs instruct", fontsize=11)

    # C: self off-manifold residual, base vs instruct
    c = ax[1, 0]
    c.bar(["base", "instruct"], [Pb["self_resid"], Pi["self_resid"]], color=[C_BASE, C_INSTRUCT])
    c.axhline(1.0, color="r", ls=":", alpha=0.6)
    c.set_ylabel("self residual off entity surface (std units)")
    c.set_title("C. Is the self ON the entity manifold?\n(0 = perfectly predicted by its kind,qualia; 1 = typical-entity scatter)", fontsize=11)

    # D: the wedge — qualia mean +/- spread vs kind (instruct)
    d = ax[1, 1]
    for W, col, lab in ((Wb, C_BASE, "base"), (Wi, C_INSTRUCT, "instruct")):
        d.plot(W["grid"], W["qhat"], color=col, label=f"{lab} mean")
        d.fill_between(W["grid"], W["qhat"] - W["shat"], W["qhat"] + W["shat"], color=col, alpha=0.15)
    d.scatter(Wi["kc"], Wi["q"], c="k", s=8, alpha=0.4)
    d.set_xlabel("kind coord (std)  (low=mechanism, high=mind)")
    d.set_ylabel("qualia coord (std)  +/- smoothed spread")
    d.set_title("D. Wedge test: does experiential range widen\nfor mind-like entities? (band = qualia spread vs kind)", fontsize=11)
    d.legend(fontsize=8)

    # E, F: the headline — self vs entity qualia~s(kind) curve, base then instruct
    _self_curve_panel(ax[0, 2], Cb, "E. base", C_BASE)
    _self_curve_panel(ax[1, 2], Ci, "F. instruct", C_INSTRUCT)

    fig.tight_layout()
    p = out / f"gam_self_qualia_layer{layer}.png"
    fig.savefig(p, dpi=140, facecolor="white"); plt.close(fig)
    print(f"[fig] {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/OLMO3_7B_SELF_QUALIA_MAIN")
    ap.add_argument("--instruct", default="runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--out", default="runs/SELF_QUALIA_GAM")
    a = ap.parse_args()
    run(a.base, a.instruct, a.layer, a.out)
