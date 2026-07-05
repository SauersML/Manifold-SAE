"""Theorem I test: is LINEAR cross-layer transport of the weekday circle atom
FORCED to be a phase shift/reflection h(theta)=+/-theta+phi?

Superposed-Geometry Prediction P2 / Theorem I: for a circle (elliptical) atom
g_l(theta)=c_l+A_l e(theta), e=(cos,sin), any LINEAR transport W carrying
im g_l onto im g_{l+1} induces a coordinate map h with h(theta)=+/-theta+phi
up to noise, because enforcing ||e(h)||^2==1 forces the pulled-back operator
M = A'^+ W A into O(2) (conformal). Deviations should concentrate where the
atom's harmonic spectrum departs from a pure ellipse.

All VERDICT statistics come from gamfit (sae_manifold_fit circle charts +
layer_transport_fit: isometry_defect = departure of h from a phase shift,
transport_edf/residual_rms = free-spline h). numpy is used only for
orchestration, the induced-operator linear algebra, harmonic bookkeeping, and
cross-checks -- never for a verdict number.

Stages:
  fit    heavy: fit K=1 circle chart at each layer L11..L23, cache to charts.npz
  test   fast : per-hop layer_transport_fit + induced M conformal departure +
                harmonic-impurity correlation + shuffled-day null; write JSON.
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np

TWO_PI = 2.0 * math.pi
LAYERS = [f"L{l}" for l in range(11, 24)]


def demean_by_template(X, tmpl):
    Xc = X.copy()
    for t in np.unique(tmpl):
        m = tmpl == t
        Xc[m] -= X[m].mean(0, keepdims=True)
    return Xc


def fit_circle(X, seed=0, pca_rank=8, n_iter=30):
    """K=1 circle chart (DOSE recipe): returns theta, plane(amb,2), centered curve, r2."""
    import gamfit
    xm = X.mean(0, keepdims=True)
    xc = X - xm
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    lift = vt[:pca_rank]
    xf = xc @ lift.T
    res = gamfit.sae_manifold_fit(
        xf, K=1, d_atom=1, atom_topology="circle", assignment="ibp_map",
        random_state=seed, isometry_weight=0.0, n_iter=n_iter,
    )
    u = res.atom_angle_coordinate(0)
    if u is None:
        raise SystemExit("degenerate chart (no arc coordinate)")
    theta = np.mod(TWO_PI * np.asarray(u, float).ravel(), TWO_PI)
    fitted = np.asarray(res.fitted, float) @ lift + xm
    center = fitted.mean(0)
    centered = fitted - center
    return theta, centered, float(res.reconstruction_r2)


def harmonic_impurity(X_centered, theta, n_harm=5):
    """Energy in harmonics k>=2 relative to the fundamental k=1 of the REAL
    (demeaned) data reparameterized by the atom angle theta. A pure ellipse has
    all energy in k=1; impurity = ||coeff_{k>=2}||^2 / ||coeff_{k=1}||^2."""
    cols = [np.ones_like(theta)]
    idx = {}
    for k in range(1, n_harm + 1):
        idx[k] = (len(cols), len(cols) + 1)
        cols += [np.cos(k * theta), np.sin(k * theta)]
    B = np.stack(cols, 1)                       # (N, 1+2H)
    coef, *_ = np.linalg.lstsq(B, X_centered, rcond=None)  # (1+2H, ambient)
    e = {k: float((coef[i0:i1] ** 2).sum()) for k, (i0, i1) in idx.items()}
    fund = max(e[1], 1e-30)
    high = sum(e[k] for k in range(2, n_harm + 1))
    return high / fund, e


def induced_operator(theta_a, theta_b):
    """M: the 2x2 linear map on intrinsic circle coords, e(theta_b) ~ M e(theta_a).
    This is exactly the pulled-back transport A'^+ W A. Theorem I: M in O(2)."""
    Ea = np.stack([np.cos(theta_a), np.sin(theta_a)], 1)  # (N,2)
    Eb = np.stack([np.cos(theta_b), np.sin(theta_b)], 1)
    M, *_ = np.linalg.lstsq(Ea, Eb, rcond=None)   # Eb ~ Ea @ M  => rows map, M is (2,2), e_b = M^T e_a
    M = M.T                                        # so that e_b ~ M e_a
    # residual of the linear-in-e model
    pred = Ea @ M.T
    lin_resid_rms = float(np.sqrt(((Eb - pred) ** 2).sum(1).mean()))
    # conformal departure: MtM = lambda I ?  lambda = trace/2
    MtM = M.T @ M
    lam = float(np.trace(MtM) / 2.0)
    dep = MtM - lam * np.eye(2)
    conformal_departure = float(np.linalg.norm(dep) / max(lam, 1e-30))  # ||MtM-lam I||/lam
    # nearest O(2) via polar: M = Q S, S=sqrtm(MtM); how far S from lam^.5 I
    w, V = np.linalg.eigh(MtM)
    s = np.sqrt(np.clip(w, 0, None))
    anisotropy = float((s.max() - s.min()) / max(s.max(), 1e-30))  # 0 = conformal
    det = float(np.linalg.det(M))
    sign = "reflection" if det < 0 else "rotation"
    # induced angle map h(theta)=angle(M e(theta)); phase-model residual (circular)
    return dict(
        M=M.tolist(), lin_resid_rms=lin_resid_rms, conformal_departure=conformal_departure,
        anisotropy=anisotropy, det=det, sign=sign, singular_values=s.tolist(),
    )


def rigid_phase_residual(theta_a, theta_b):
    """Cross-check (numpy, NOT a verdict number): best +/-theta+phi circular RMS."""
    out = {}
    for s, name in [(1.0, "plus"), (-1.0, "minus")]:
        resid = np.angle(np.exp(1j * (theta_b - s * theta_a)))
        phi = math.atan2(np.sin(resid).mean(), np.cos(resid).mean())
        r = np.angle(np.exp(1j * (theta_b - (s * theta_a + phi))))
        out[name] = dict(phi=float(phi), circ_rms=float(np.sqrt((r ** 2).mean())))
    best = min(out.values(), key=lambda d: d["circ_rms"])
    return out, best["circ_rms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "test", "all"])
    ap.add_argument("--acts", required=True)
    ap.add_argument("--charts", default="charts.npz")
    ap.add_argument("--out", default="thmI_results.json")
    ap.add_argument("--pca-rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=30)
    args = ap.parse_args()

    d = np.load(args.acts)
    tmpl = d["template_ids"]

    if args.stage in ("fit", "all"):
        charts = {}
        for L in LAYERS:
            X = demean_by_template(d["acts_" + L], tmpl)
            th, cen, r2 = fit_circle(X, seed=args.seed, pca_rank=args.pca_rank)
            charts[f"theta_{L}"] = th
            charts[f"cen_{L}"] = cen
            charts[f"r2_{L}"] = np.array(r2)
            print(f"[fit] {L} r2={r2:.3f}", flush=True)
        np.savez_compressed(args.charts, **charts)
        print(f"[fit] cached {args.charts}", flush=True)

    if args.stage in ("test", "all"):
        import gamfit
        C = np.load(args.charts if args.stage == "test" else args.charts)
        theta = {L: C[f"theta_{L}"] for L in LAYERS}
        cen = {L: C[f"cen_{L}"] for L in LAYERS}
        r2 = {L: float(C[f"r2_{L}"]) for L in LAYERS}
        # per-atom harmonic impurity
        imp = {}
        for L in LAYERS:
            hi, e = harmonic_impurity(cen[L], theta[L])
            imp[L] = dict(impurity=hi, harmonic_energy=e)
        hops = []
        rng = np.random.default_rng(0)
        for a, b in zip(LAYERS[:-1], LAYERS[1:]):
            ta, tb = theta[a], theta[b]
            tr = gamfit.layer_transport_fit(ta, tb)   # VERDICT fit
            ind = induced_operator(ta, tb)
            rigid_all, rigid_rms = rigid_phase_residual(ta, tb)
            # NULL: shuffle target token correspondence
            null_defect, null_conc = [], []
            for _ in range(args.n_perm):
                perm = rng.permutation(len(tb))
                trn = gamfit.layer_transport_fit(ta, tb[perm])
                null_defect.append(trn["isometry_defect"])
                null_conc.append(trn["degree_concentration"])
            null_defect = np.array(null_defect); null_conc = np.array(null_conc)
            hop = dict(
                hop=f"{a}->{b}",
                degree=tr["degree"], degree_concentration=tr["degree_concentration"],
                isometry_defect=tr["isometry_defect"], isometry_defect_se=tr["isometry_defect_se"],
                transport_edf=tr["transport_edf"], residual_rms=tr["residual_rms"],
                rotation_offset=tr["rotation_offset"], topology_preserved=tr["topology_preserved"],
                conformal_departure=ind["conformal_departure"], anisotropy=ind["anisotropy"],
                lin_resid_rms=ind["lin_resid_rms"], det=ind["det"], sign=ind["sign"],
                rigid_circ_rms=rigid_rms,
                impurity_src=imp[a]["impurity"], impurity_dst=imp[b]["impurity"],
                impurity_mean=0.5 * (imp[a]["impurity"] + imp[b]["impurity"]),
                null_defect_mean=float(null_defect.mean()), null_defect_sd=float(null_defect.std()),
                null_conc_mean=float(null_conc.mean()), null_conc_sd=float(null_conc.std()),
                conc_z=float((tr["degree_concentration"] - null_conc.mean()) / max(null_conc.std(), 1e-9)),
            )
            hops.append(hop)
            print(f"[test] {hop['hop']} deg={hop['degree']} conc={hop['degree_concentration']:.3f} "
                  f"iso_def={hop['isometry_defect']:.3f} conf_dep={hop['conformal_departure']:.3f} "
                  f"imp={hop['impurity_mean']:.3f} nullconc={hop['null_conc_mean']:.3f} z={hop['conc_z']:.1f}",
                  flush=True)
        # correlation: deviation (isometry_defect) vs harmonic impurity across hops
        dev = np.array([h["isometry_defect"] for h in hops])
        conf = np.array([h["conformal_departure"] for h in hops])
        impm = np.array([h["impurity_mean"] for h in hops])
        def corr(x, y):
            if x.std() < 1e-12 or y.std() < 1e-12: return 0.0
            return float(np.corrcoef(x, y)[0, 1])
        summary = dict(
            r2=r2, impurity=imp, hops=hops,
            corr_isodefect_impurity=corr(dev, impm),
            corr_confdep_impurity=corr(conf, impm),
            corr_isodefect_confdep=corr(dev, conf),
            median_isometry_defect=float(np.median(dev)),
            median_conformal_departure=float(np.median(conf)),
            median_degree_concentration=float(np.median([h["degree_concentration"] for h in hops])),
            all_degree_one=all(h["degree"] == 1 for h in hops),
            median_null_conc=float(np.median([h["null_conc_mean"] for h in hops])),
            min_conc_z=float(np.min([h["conc_z"] for h in hops])),
        )
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print("[test] wrote", args.out, flush=True)
        print(json.dumps({k: summary[k] for k in
              ["corr_isodefect_impurity", "corr_confdep_impurity", "median_isometry_defect",
               "median_conformal_departure", "median_degree_concentration", "all_degree_one",
               "median_null_conc", "min_conc_z"]}, indent=2))


if __name__ == "__main__":
    main()
