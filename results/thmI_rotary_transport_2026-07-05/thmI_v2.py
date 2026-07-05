"""Theorem I test v2 -- stochastic-angle-robust.

v1 used gamfit's arc coordinate u(theta) as the circle angle. That coordinate
comes from a non-convergent outer BFGS and is NOT reproducible run-to-run (same
L17->L18 hop: isometry_defect 0.024 in a one-off vs 2.02 in the batch), so the
transport verdict was dominated by fit noise, not geometry.

v2 uses a DETERMINISTIC circle coordinate: phi_l = atan2 of the demeaned data
projected onto layer l's top-2 SVD plane. This is reproducible and gauge-fixed
per layer (arbitrary global phase/orientation only, which is exactly the +/-,phi
that Theorem I fits out). gamfit still supplies (i) the per-layer circle
CERTIFICATE (sae_manifold_fit r2 + planarity + honest arc coord, to prove a
circle atom is really there) and (ii) the transport VERDICT (layer_transport_fit
isometry_defect / degree / concentration) on the deterministic angles.

Also emits a u-vs-phi stability diagnostic: refit selected layers over several
seeds and report how much the arc-coordinate transport wobbles vs the
deterministic phi transport.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np

TWO_PI = 2.0 * math.pi
LAYERS = [f"L{l}" for l in range(11, 24)]


def demean_by_template(X, tmpl):
    Xc = X.copy()
    for t in np.unique(tmpl):
        m = tmpl == t
        Xc[m] -= X[m].mean(0, keepdims=True)
    return Xc


def data_plane_angle(Xc):
    """Deterministic circle coordinate: angle in the top-2 SVD plane of demeaned data."""
    _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    plane = vt[:2].T                    # (amb, 2)
    proj = Xc @ plane                   # (N, 2)
    phi = np.mod(np.arctan2(proj[:, 1], proj[:, 0]), TWO_PI)
    planarity = float((sv[:2] ** 2).sum() / max((sv ** 2).sum(), 1e-30))
    return phi, plane, proj, planarity


def gamfit_circle_cert(Xc, seed=0, pca_rank=8, n_iter=30):
    """gamfit circle certificate: r2, planarity(fitted), and arc coordinate u."""
    import gamfit
    xm = Xc.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc - xm, full_matrices=False)
    lift = vt[:pca_rank]
    xf = (Xc - xm) @ lift.T
    res = gamfit.sae_manifold_fit(
        xf, K=1, d_atom=1, atom_topology="circle", assignment="ibp_map",
        random_state=seed, isometry_weight=0.0, n_iter=n_iter)
    u = res.atom_angle_coordinate(0)
    u = None if u is None else np.mod(TWO_PI * np.asarray(u, float).ravel(), TWO_PI)
    fitted = np.asarray(res.fitted, float) @ lift + xm
    cen = fitted - fitted.mean(0)
    _, sv, _ = np.linalg.svd(cen, full_matrices=False)
    planarity = float((sv[:2] ** 2).sum() / max((sv ** 2).sum(), 1e-30))
    return dict(r2=float(res.reconstruction_r2), planarity=planarity, u=u)


def harmonic_impurity(Xc, phi, n_harm=5):
    cols = [np.ones_like(phi)]
    idx = {}
    for k in range(1, n_harm + 1):
        idx[k] = (len(cols), len(cols) + 1)
        cols += [np.cos(k * phi), np.sin(k * phi)]
    B = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(B, Xc, rcond=None)
    e = {k: float((coef[i0:i1] ** 2).sum()) for k, (i0, i1) in idx.items()}
    fund = max(e[1], 1e-30)
    return sum(e[k] for k in range(2, n_harm + 1)) / fund, e


def induced_operator(phi_a, phi_b):
    Ea = np.stack([np.cos(phi_a), np.sin(phi_a)], 1)
    Eb = np.stack([np.cos(phi_b), np.sin(phi_b)], 1)
    Mt, *_ = np.linalg.lstsq(Ea, Eb, rcond=None)   # Eb ~ Ea @ Mt ; e_b = Mt^T e_a
    M = Mt.T
    pred = Ea @ M.T
    lin_resid_rms = float(np.sqrt(((Eb - pred) ** 2).sum(1).mean()))
    MtM = M.T @ M
    lam = float(np.trace(MtM) / 2.0)
    conformal_departure = float(np.linalg.norm(MtM - lam * np.eye(2)) / max(lam, 1e-30))
    w, _ = np.linalg.eigh(MtM)
    s = np.sqrt(np.clip(w, 0, None))
    anisotropy = float((s.max() - s.min()) / max(s.max(), 1e-30))
    det = float(np.linalg.det(M))
    return dict(lin_resid_rms=lin_resid_rms, conformal_departure=conformal_departure,
                anisotropy=anisotropy, det=det, singular_values=s.tolist())


def rigid_phase_residual(phi_a, phi_b):
    out = {}
    for s in (1.0, -1.0):
        resid = np.angle(np.exp(1j * (phi_b - s * phi_a)))
        phi = math.atan2(np.sin(resid).mean(), np.cos(resid).mean())
        r = np.angle(np.exp(1j * (phi_b - (s * phi_a + phi))))
        out[s] = float(np.sqrt((r ** 2).mean()))
    s_best = 1.0 if out[1.0] <= out[-1.0] else -1.0
    return out[s_best], int(s_best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", default="thmI_v2_results.json")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--stability-layers", nargs="*", default=["L16", "L17", "L18", "L19"])
    ap.add_argument("--stability-seeds", type=int, default=3)
    args = ap.parse_args()
    import gamfit

    d = np.load(args.acts)
    tmpl = d["template_ids"]
    Xc = {L: demean_by_template(d["acts_" + L], tmpl) for L in LAYERS}

    # per-layer: deterministic phi + gamfit certificate
    phi, planarity, cert, imp = {}, {}, {}, {}
    for L in LAYERS:
        p, _, _, plan = data_plane_angle(Xc[L])
        phi[L] = p
        planarity[L] = plan
        c = gamfit_circle_cert(Xc[L])
        cert[L] = dict(r2=c["r2"], planarity=c["planarity"])
        # circle certificate: agreement of gamfit honest arc coord with plane angle
        if c["u"] is not None:
            corr = float(abs(np.exp(1j * (p - 2 * math.pi * (c["u"] / TWO_PI))).mean()))
            anti = float(abs(np.exp(1j * (p + c["u"])).mean()))
            cert[L]["arc_plane_agreement"] = max(corr, anti)
        hi, e = harmonic_impurity(Xc[L], p)
        imp[L] = dict(impurity=hi, energy=e)
        print(f"[cert] {L} r2={c['r2']:.3f} planarity(data)={plan:.3f} imp={hi:.3f}", flush=True)

    # per-hop transport verdict on DETERMINISTIC angles
    hops = []
    rng = np.random.default_rng(0)
    for a, b in zip(LAYERS[:-1], LAYERS[1:]):
        pa, pb = phi[a], phi[b]
        tr = gamfit.layer_transport_fit(pa, pb)      # VERDICT
        ind = induced_operator(pa, pb)
        rigid, sign = rigid_phase_residual(pa, pb)
        nd, nc = [], []
        for _ in range(args.n_perm):
            perm = rng.permutation(len(pb))
            trn = gamfit.layer_transport_fit(pa, pb[perm])
            nd.append(trn["isometry_defect"]); nc.append(trn["degree_concentration"])
        nd, nc = np.array(nd), np.array(nc)
        hop = dict(
            hop=f"{a}->{b}", degree=tr["degree"], degree_concentration=tr["degree_concentration"],
            isometry_defect=tr["isometry_defect"], isometry_defect_se=tr["isometry_defect_se"],
            transport_edf=tr["transport_edf"], residual_rms=tr["residual_rms"],
            rotation_offset=tr["rotation_offset"], topology_preserved=tr["topology_preserved"],
            conformal_departure=ind["conformal_departure"], anisotropy=ind["anisotropy"],
            lin_resid_rms=ind["lin_resid_rms"], det=ind["det"], rigid_circ_rms=rigid, sign=sign,
            impurity_src=imp[a]["impurity"], impurity_dst=imp[b]["impurity"],
            impurity_mean=0.5 * (imp[a]["impurity"] + imp[b]["impurity"]),
            null_defect_mean=float(nd.mean()), null_conc_mean=float(nc.mean()),
            null_conc_sd=float(nc.std()),
            conc_gap=float(tr["degree_concentration"] - nc.mean()),
        )
        hops.append(hop)
        print(f"[hop] {hop['hop']} deg={hop['degree']} conc={hop['degree_concentration']:.3f} "
              f"iso={hop['isometry_defect']:.3f} rigidRMS={rigid:.3f} confdep={hop['conformal_departure']:.3f} "
              f"aniso={hop['anisotropy']:.2f} imp={hop['impurity_mean']:.2f} nullconc={hop['null_conc_mean']:.3f}",
              flush=True)

    # u-vs-phi stability diagnostic: refit chosen layers over seeds
    stab = {}
    for L in args.stability_layers:
        us, phis = [], []
        for s in range(args.stability_seeds):
            c = gamfit_circle_cert(Xc[L], seed=s)
            if c["u"] is not None:
                us.append(c["u"])
        p, _, _, _ = data_plane_angle(Xc[L])
        # arc-coord seed-to-seed circular sd (aligned by best global phase+sign)
        def align(a, ref):
            best = None
            for sgn in (1, -1):
                r = np.angle(np.exp(1j * (a * sgn - ref)))
                off = math.atan2(np.sin(r).mean(), np.cos(r).mean())
                res = np.angle(np.exp(1j * (a * sgn - off - ref)))
                v = float(np.sqrt((res ** 2).mean()))
                if best is None or v < best: best = v
            return best
        if len(us) >= 2:
            u_wobble = float(np.mean([align(us[i], us[0]) for i in range(1, len(us))]))
        else:
            u_wobble = None
        # phi is deterministic -> wobble 0 by construction; record agreement to u
        phi_u_agree = align(us[0], p) if us else None
        stab[L] = dict(u_seed_wobble_rad=u_wobble, phi_deterministic=True,
                       phi_vs_u_resid_rad=phi_u_agree)
        print(f"[stab] {L} u_seed_wobble={u_wobble} phi_vs_u={phi_u_agree}", flush=True)

    dev = np.array([h["isometry_defect"] for h in hops])
    rig = np.array([h["rigid_circ_rms"] for h in hops])
    conf = np.array([h["conformal_departure"] for h in hops])
    impm = np.array([h["impurity_mean"] for h in hops])
    def corr(x, y):
        return 0.0 if x.std() < 1e-12 or y.std() < 1e-12 else float(np.corrcoef(x, y)[0, 1])
    summary = dict(
        cert=cert, planarity=planarity, impurity=imp, hops=hops, stability=stab,
        corr_rigid_impurity=corr(rig, impm),
        corr_confdep_impurity=corr(conf, impm),
        corr_rigid_confdep=corr(rig, conf),
        corr_isodefect_impurity=corr(dev, impm),
        median_rigid_circ_rms=float(np.median(rig)),
        median_conformal_departure=float(np.median(conf)),
        median_degree_concentration=float(np.median([h["degree_concentration"] for h in hops])),
        median_null_conc=float(np.median([h["null_conc_mean"] for h in hops])),
        median_conc_gap=float(np.median([h["conc_gap"] for h in hops])),
        n_hops_phaseshift=int((rig < 0.35).sum()),
        n_hops=len(hops),
    )
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("[done] wrote", args.out, flush=True)
    print(json.dumps({k: summary[k] for k in
          ["corr_rigid_impurity", "corr_confdep_impurity", "corr_rigid_confdep",
           "median_rigid_circ_rms", "median_conformal_departure", "median_degree_concentration",
           "median_null_conc", "median_conc_gap", "n_hops_phaseshift", "n_hops"]}, indent=2))


if __name__ == "__main__":
    main()
