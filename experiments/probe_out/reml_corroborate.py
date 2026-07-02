"""Standalone REML corroboration for the curved-feature probes.

The main probe fits gamfit's manifold SAE via the torch backend (robust, exposes
per-sample chart coords). This script corroborates the reconstruction EV with the
REML solver gamfit.sae_manifold_fit (K=1, d_atom=1, periodic 'circle' topology) on the
SAME per-template-demeaned, PCA-reduced activations. REML can fail to converge on thin
data, so we retry with more iterations and record the outcome honestly.

Run:  .venv/bin/python experiments/probe_out/reml_corroborate.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent.parent))  # repo root for manifold_sae


def demean(X, tidx):
    Xd = X.copy()
    for t in np.unique(tidx):
        m = tidx == t
        Xd[m] = X[m] - X[m].mean(0, keepdims=True)
    return Xd


def reduce(X, r):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[: min(r, Vt.shape[0])].T


def main():
    from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check
    bypass_gamfit_cuda_check()
    import torch
    torch.set_num_threads(4)
    import gamfit

    layer_of = {}
    jf = OUT / "curved_feature_probes.json"
    if jf.exists():
        for name, r in json.loads(jf.read_text())["results"].items():
            layer_of[name] = r["layer"]

    rj = OUT / "reml_corroboration.json"
    out = json.loads(rj.read_text()) if rj.exists() else {}  # resume
    rdim = int(os.environ.get("CURVED_PROBE_RDIM", "12"))
    for name in ("weekday", "month", "year"):
        if out.get(name, {}).get("reml_ev_insample") is not None:
            continue  # already done
        npz = OUT / f"harvest_{name}.npz"
        if not npz.exists():
            out[name] = {"skipped": "no cached harvest"}
            continue
        z = np.load(npz, allow_pickle=False)
        L = layer_of.get(name, int(z["layers"][0]))
        cyclic = bool(z["cyclic"])
        X = demean(z[f"L{L}"], z["template_idx"])
        red = reduce(X, min(rdim, X.shape[0] - 2))
        err = None
        rec = None
        for n_iter in (60, 120, 200):
            try:
                kw = dict(K=1, d_atom=1, n_iter=n_iter)
                if cyclic:
                    kw["atom_topology"] = "circle"
                fit = gamfit.sae_manifold_fit(red, **kw)
                rec = {"reml_ev_insample": float(fit.reconstruction_r2),
                       "n_iter": n_iter, "layer": L, "cyclic": cyclic,
                       "topology": "circle" if cyclic else "default_open"}
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"
        out[name] = rec or {"reml_ev_insample": None, "error": err, "layer": L}
        print(f"[reml] {name}: {out[name]}", flush=True)
        rj.write_text(json.dumps(out, indent=2))  # checkpoint each set

    rj.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {rj}")


if __name__ == "__main__":
    main()
