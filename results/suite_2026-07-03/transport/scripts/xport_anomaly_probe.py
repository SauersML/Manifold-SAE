"""Characterize the L20->L21 transport anomaly (degree-2, defect-4.56).

Three questions:
  1. Fit artifact? -> refit L20,L21 at ranks 8/16/24, re-transport, watch degree.
  2. Genuine double-winding or glued half-circles? -> circular winding
     concentration c_k = |mean exp(i(th21 - k*th20))| for k in {-2,-1,1,2};
     plus per-day correspondence scatter (th20 vs th21).
  3. Localized? -> skip-hops L19->L21, L20->L22, L19->L22 at rank 8.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import gamfit
import chart_transport_l11_l23 as X

TWO_PI = 2 * np.pi


def demean(x, t):
    x = x.copy()
    for tt in np.unique(t):
        m = t == tt; x[m] -= x[m].mean(0, keepdims=True)
    return x


def transport(a, b):
    r = gamfit.layer_transport_fit(a, b, "circle", "circle", layer_from=0, layer_to=1)
    return {"degree": int(r["degree"]), "degree_concentration": float(r["degree_concentration"]),
            "isometry_defect": float(r["isometry_defect"]), "isometry_defect_se": float(r["isometry_defect_se"]),
            "topology_preserved": bool(r["topology_preserved"]), "rotation_offset": float(r["rotation_offset"])}


def wind(th_from, th_to):
    out = {}
    for k in (-2, -1, 1, 2):
        out[str(k)] = float(abs(np.mean(np.exp(1j * (th_to - k * th_from)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-timeout", type=int, default=1200)
    a = ap.parse_args()

    with np.load(a.probe_npz) as z:
        tids = np.asarray(z["template_ids"], dtype=np.int64)
        labels = np.asarray(z["labels"], dtype=np.int64)
        acts = {L: demean(np.asarray(z[f"acts_L{L}"], dtype=np.float64), tids) for L in (19, 20, 21, 22)}

    result = {"rank_sensitivity": {}, "winding": {}, "skip_hops": {}, "n_obs": int(len(labels))}

    # (1) rank sensitivity on the anomaly hop
    charts_r8 = {}
    for r in (8, 16, 24):
        c20 = X.fit_layer_circle(acts[20], "acts_L20", a.seed, a.fit_timeout, pca_rank=r)
        c21 = X.fit_layer_circle(acts[21], "acts_L21", a.seed, a.fit_timeout, pca_rank=r)
        X.anchor_gauges_to_first_layer([c20, c21])
        result["rank_sensitivity"][f"rank{r}"] = transport(c20.theta, c21.theta)
        result["rank_sensitivity"][f"rank{r}"]["r2_L20"] = c20.reconstruction_r2
        result["rank_sensitivity"][f"rank{r}"]["r2_L21"] = c21.reconstruction_r2
        if r == 8:
            charts_r8[20], charts_r8[21] = c20, c21

    # (2) winding diagnosis on rank-8 fitted arc coords
    th20, th21 = charts_r8[20].theta, charts_r8[21].theta
    result["winding"]["concentration_by_k"] = wind(th20, th21)
    result["winding"]["scatter_theta20"] = th20.tolist()
    result["winding"]["scatter_theta21"] = th21.tolist()
    result["winding"]["day"] = labels.tolist()
    # per-day circular mean of each layer's angle
    dm = {}
    for d in range(7):
        m = labels == d
        dm[d] = {"L20": float(np.angle(np.mean(np.exp(1j * th20[m])))),
                 "L21": float(np.angle(np.mean(np.exp(1j * th21[m]))))}
    result["winding"]["per_day_mean_angle"] = dm

    # (3) skip-hops at rank 8 to localize the disruption.
    # degree/concentration/isometry from layer_transport_fit are invariant to the
    # per-layer phase gauge, so transport the raw fitted arc coords directly.
    c = {L: X.fit_layer_circle(acts[L], f"acts_L{L}", a.seed, a.fit_timeout, pca_rank=8) for L in (19, 22)}
    c[20], c[21] = charts_r8[20], charts_r8[21]
    for (f, t) in [(19, 20), (20, 21), (19, 21), (20, 22), (19, 22)]:
        result["skip_hops"][f"L{f}->L{t}"] = transport(c[f].theta, c[t].theta)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    # compact console summary
    print("RANK SENSITIVITY (anomaly hop L20->L21):")
    for r in (8, 16, 24):
        s = result["rank_sensitivity"][f"rank{r}"]
        print(f"  rank{r}: degree={s['degree']} conc={s['degree_concentration']:.3f} "
              f"iso={s['isometry_defect']:.3g} r2=({s['r2_L20']:.3f},{s['r2_L21']:.3f})")
    print("WINDING concentration by k:", {k: round(v, 3) for k, v in result["winding"]["concentration_by_k"].items()})
    print("SKIP HOPS:")
    for k, s in result["skip_hops"].items():
        print(f"  {k}: degree={s['degree']} conc={s['degree_concentration']:.3f} iso={s['isometry_defect']:.3g}")


if __name__ == "__main__":
    main()
