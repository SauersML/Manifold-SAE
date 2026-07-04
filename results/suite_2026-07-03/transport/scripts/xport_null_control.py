"""Shuffled-day null for chart transport.

Falsification control for XPORT: if a hop *carries* the weekday circle
(degree-1, low isometry defect on the shared-token transport map), then
breaking the token correspondence between the two layers (a random row
permutation of the target angles) must destroy that signal -- degree
concentration should collapse toward chance and the isometry defect inflate.
Both circles stay intact at each layer; only the day-to-day pairing is broken.

Reuses the exact fit + gauge machinery of chart_transport_l11_l23.py so the
real number here is comparable to the sweep's hop for the same layer pair.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import gamfit
import chart_transport_l11_l23 as X


def demean_by_template(x, template_ids):
    x = x.copy()
    for t in np.unique(template_ids):
        m = template_ids == t
        x[m] -= x[m].mean(axis=0, keepdims=True)
    return x


def transport(theta_from, theta_to):
    return gamfit.layer_transport_fit(
        theta_from, theta_to, "circle", "circle", layer_from=0, layer_to=1
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-npz", type=Path, required=True)
    ap.add_argument("--from-layer", type=int, required=True)
    ap.add_argument("--to-layer", type=int, required=True)
    ap.add_argument("--pca-rank", type=int, default=48)
    ap.add_argument("--fit-timeout", type=int, default=1200)
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    with np.load(a.probe_npz) as z:
        kf, kt = f"acts_L{a.from_layer}", f"acts_L{a.to_layer}"
        xf = np.asarray(z[kf], dtype=np.float64)
        xt = np.asarray(z[kt], dtype=np.float64)
        tids = np.asarray(z["template_ids"], dtype=np.int64)
    xf = demean_by_template(xf, tids)
    xt = demean_by_template(xt, tids)

    cf = X.fit_layer_circle(xf, kf, a.seed, a.fit_timeout, pca_rank=a.pca_rank)
    ct = X.fit_layer_circle(xt, kt, a.seed, a.fit_timeout, pca_rank=a.pca_rank)
    X.anchor_gauges_to_first_layer([cf, ct])

    real = transport(cf.theta, ct.theta)
    rng = np.random.default_rng(a.seed)
    perm_iso, perm_deg_conc, perm_deg = [], [], []
    n = ct.theta.shape[0]
    for _ in range(a.n_perm):
        p = rng.permutation(n)
        r = transport(cf.theta, ct.theta[p])
        perm_iso.append(float(r["isometry_defect"]))
        perm_deg_conc.append(float(r["degree_concentration"]))
        perm_deg.append(int(r["degree"]))

    def stats(v):
        v = np.asarray(v, dtype=np.float64)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    out = {
        "hop": f"L{a.from_layer}->L{a.to_layer}",
        "n_obs": int(n), "n_perm": a.n_perm,
        "real": {
            "isometry_defect": float(real["isometry_defect"]),
            "isometry_defect_se": float(real["isometry_defect_se"]),
            "degree": int(real["degree"]),
            "degree_concentration": float(real["degree_concentration"]),
            "rotation_offset": float(real["rotation_offset"]),
            "topology_preserved": bool(real["topology_preserved"]),
        },
        "shuffled_null": {
            "isometry_defect": stats(perm_iso),
            "degree_concentration": stats(perm_deg_conc),
            "degree_is_1_fraction": float(np.mean(np.asarray(perm_deg) == 1)),
        },
    }
    idc = out["shuffled_null"]["degree_concentration"]
    out["degree_conc_z_vs_null"] = (
        (out["real"]["degree_concentration"] - idc["mean"]) / idc["sd"]
        if idc["sd"] > 0 else None
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
