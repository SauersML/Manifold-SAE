"""Full sweep driver for Theorem A falsification + Theorem B curvature-margin."""
import json, sys, time
import numpy as np
import search


def sweep_p_transition(seeds, n_z, n_starts):
    """Fraction of z with a second parse vs ambient p (codimension slack)."""
    out = []
    for p in [4, 5, 6, 8]:
        fracs = []
        for s in seeds:
            r = search.run_trial(seed=1000 + s, p=p, curvature=1.0,
                                 center_scale=1.0, n_z=n_z, n_starts=n_starts)
            fracs.append(r["frac"])
        out.append(dict(p=p, slack=p - 1 - 4, fracs=fracs,
                        mean_frac=float(np.mean(fracs)),
                        max_frac=float(np.max(fracs))))
        print("P-TRANS", json.dumps(out[-1]), flush=True)
    return out


def sweep_resolution(seeds, p=5):
    """Does the double-parse fraction shrink as sampling density grows?
    Theorem A predicts double parses live on a measure-zero set -> fraction
    should NOT be a stable positive number; increasing n_z probes more of Sigma
    and increasing n_starts probes more of parse space. We report both."""
    out = []
    for n_z in [50, 100, 200]:
        for n_starts in [50, 100]:
            fracs = []
            for s in seeds:
                r = search.run_trial(seed=2000 + s, p=p, curvature=1.0,
                                     center_scale=1.0, n_z=n_z, n_starts=n_starts)
                fracs.append(r["frac"])
            out.append(dict(n_z=n_z, n_starts=n_starts,
                            mean_frac=float(np.mean(fracs)),
                            max_frac=float(np.max(fracs))))
            print("RES", json.dumps(out[-1]), flush=True)
    return out


def sweep_center(seeds, p=5, n_z=100, n_starts=80):
    """Off-center vs centered (c=0). Prop 1: centered single circle's cone is a
    flat plane. Test whether centering induces ambiguity for the 2-atom sum."""
    out = []
    for cs in [0.0, 0.5, 1.0, 2.0]:
        fracs = []
        for s in seeds:
            r = search.run_trial(seed=3000 + s, p=p, curvature=1.0,
                                 center_scale=cs, n_z=n_z, n_starts=n_starts)
            fracs.append(r["frac"])
        out.append(dict(center_scale=cs, mean_frac=float(np.mean(fracs)),
                        max_frac=float(np.max(fracs))))
        print("CENTER", json.dumps(out[-1]), flush=True)
    return out


def sweep_curvature(seeds, p=5, n_z=100, n_starts=80):
    """Theorem B story: sweep curvature near-flat -> strongly curved.
    We report double-parse fraction AND the identifiability margin. Prediction:
    near-flat -> flat co-collapse ambiguity approached continuously."""
    out = []
    for curv in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        fracs, margins = [], []
        for s in seeds:
            r = search.run_trial(seed=4000 + s, p=p, curvature=curv,
                                 center_scale=1.0, n_z=n_z, n_starts=n_starts)
            fracs.append(r["frac"])
            if r["median_margin"] is not None:
                margins.append(r["median_margin"])
        out.append(dict(curvature=curv, mean_frac=float(np.mean(fracs)),
                        max_frac=float(np.max(fracs)),
                        n_double_trials=len(margins),
                        median_margin=float(np.median(margins)) if margins else None))
        print("CURV", json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = list(range(int(sys.argv[2]))) if len(sys.argv) > 2 else list(range(5))
    t0 = time.time()
    results = {}
    if which in ("all", "ptrans"):
        results["p_transition"] = sweep_p_transition(seeds, n_z=120, n_starts=80)
    if which in ("all", "res"):
        results["resolution"] = sweep_resolution(seeds, p=5)
    if which in ("all", "center"):
        results["center"] = sweep_center(seeds)
    if which in ("all", "curv"):
        results["curvature"] = sweep_curvature(seeds)
    results["elapsed_sec"] = time.time() - t0
    results["n_seeds"] = len(seeds)
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DONE", results["elapsed_sec"], "sec", flush=True)
