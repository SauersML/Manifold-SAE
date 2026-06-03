"""RUN 4 calibration: does LOW trust predict WRONGNESS?

SYNTHETIC leg (known truth): plant single-atom circle regions spanning the
conditioning / coverage / type axes, score trust (solver-independent), and
measure a nonparametric coordinate-recovery error against the PLANTED truth.
Claim to falsify: trust correlates NEGATIVELY with coordinate error (low trust
=> high error), and the low-trust group has materially higher error than the
high-trust group.

REAL leg (LLM activations): harvest Qwen2.5-1.5B layer-14 last-token activations
for cyclic categories (days, months, directions, notes -> known ring) and linear
ones (letters, numbers). For each category, bootstrap the nonparametric circle
coordinate across resamples and measure cross-seed DISAGREEMENT (mean pairwise
1 - circ_procrustes_r2 between bootstrap coordinate estimates on the shared
tokens). Claim to falsify: trust correlates NEGATIVELY with cross-seed
disagreement (low trust => seeds disagree). Everything is solver-independent
because the typed gam fit is BLOCKED (RemlConvergenceError, gamfit 0.1.151).

Outputs (written to --out): trust_synth.csv, trust_real.csv, calibration JSON,
and two PNG calibration scatter/curves.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from experiments.trust_score import (
    trust_score, nonparam_circle_coord, circ_procrustes_r2, local_pca_plane,
)


# ---------------------------------------------------------------------------
# SYNTHETIC ground truth: single circle regions spanning the trust axes
# ---------------------------------------------------------------------------

def plant_region(kind, *, n, D, noise, seed, coverage=1.0, ecc=0.0):
    """Plant a single labelled region with a KNOWN coordinate.
      circle  : full ring (well specified)
      arc     : ring seen over `coverage` of the period (coverage axis)
      flat    : highly eccentric ellipse -> near-1D, ill-conditioned (sigma axis)
      blob    : genuine 2D disk -> WRONG type (level-0 / misspecification axis)
      line    : 1D segment -> wrong topology (topo-margin axis)
    Returns X (n,D) and the planted t in [0,1) (NaN where no circle truth)."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((D, 2)))[0]
    if kind == "circle":
        t = rng.uniform(0, 1, n)
        p = np.c_[np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]
    elif kind == "arc":
        t = rng.uniform(0, coverage, n)
        p = np.c_[np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]
    elif kind == "flat":
        t = rng.uniform(0, 1, n)
        p = np.c_[np.cos(2 * np.pi * t), (1.0 - ecc) * np.sin(2 * np.pi * t)]
    elif kind == "blob":
        r = np.sqrt(rng.uniform(0, 1, n)); a = rng.uniform(0, 2 * np.pi, n)
        p = np.c_[r * np.cos(a), r * np.sin(a)]
        t = (a / (2 * np.pi)) % 1.0  # angle is the only meaningful coord
    elif kind == "line":
        t = rng.uniform(0, 1, n)
        p = np.c_[2 * t - 1, np.zeros(n)]
    else:
        raise ValueError(kind)
    X = p @ Q.T + noise * rng.standard_normal((n, D))
    return X, t


SYNTH_CASES = [
    # (label, kind, kwargs) spanning the axes; multiple seeds each.
    ("circle_clean", "circle", dict(noise=0.02)),
    ("circle_noisy", "circle", dict(noise=0.20)),
    ("arc_90", "arc", dict(noise=0.02, coverage=0.25)),
    ("arc_180", "arc", dict(noise=0.02, coverage=0.5)),
    ("flat_e6", "flat", dict(noise=0.02, ecc=0.6)),
    ("flat_e9", "flat", dict(noise=0.02, ecc=0.9)),
    ("blob", "blob", dict(noise=0.02)),
    ("line", "line", dict(noise=0.02)),
    ("sparse", "circle", dict(noise=0.02)),  # small n, set below
]


def run_synthetic(out, D=8, n=200, seeds=8):
    rows = []
    for label, kind, kw in SYNTH_CASES:
        nn = 25 if label == "sparse" else n
        for s in range(seeds):
            X, t_true = plant_region(kind, n=nn, D=D, seed=1000 + s, **kw)
            rep = trust_score(X)
            t_hat = nonparam_circle_coord(X)
            r2 = circ_procrustes_r2(t_hat, t_true)
            coord_err = 1.0 - r2  # 0 = perfect recovery, 1 = no recovery
            rows.append(dict(label=label, kind=kind, seed=s,
                             trust=rep.trust, sigma_min=rep.sigma_min,
                             coherence=rep.coherence, topo_margin=rep.topo_margin,
                             coverage=rep.coverage, level0=rep.level0,
                             untyped=rep.untyped, coord_err=float(coord_err),
                             coord_r2=float(r2)))
    return rows


# ---------------------------------------------------------------------------
# REAL LLM activations: cyclic vs linear categories, cross-seed disagreement
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen2.5-1.5B"
LAYER = 14


def harvest_real(prompts_path):
    """Return {category: (X (n,D), values list)} from real layer-14 last-token
    activations. Cached to --out/real_acts.npz on first run."""
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    data = _json.load(open(prompts_path))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(dev).eval()
    by_cat = {}
    with torch.no_grad():
        for item in data:
            ids = tok(item["prompt"], return_tensors="pt").input_ids.to(dev)
            hs = model(ids, output_hidden_states=True).hidden_states[LAYER]
            v = hs[0, -1].float().cpu().numpy()
            by_cat.setdefault(item["category"], ([], []))
            by_cat[item["category"]][0].append(v)
            by_cat[item["category"]][1].append(item["value"])
    return {c: (np.array(xs), vals) for c, (xs, vals) in by_cat.items()}


def bootstrap_disagreement(X, n_boot=8, frac=0.8, seed=0):
    """Cross-seed coordinate disagreement: bootstrap-resample tokens, estimate the
    nonparametric circle coordinate on each resample, evaluate every resample's
    coordinate on a FIXED held probe set, and report mean pairwise (1 - circ R2)
    between resamples. High = the coordinate is unstable across seeds (untrustworthy)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    # fixed probe = all tokens; each bootstrap fits a plane on a subsample then
    # projects ALL tokens, so coordinates are comparable token-for-token.
    coords = []
    for b in range(n_boot):
        idx = rng.choice(n, int(frac * n), replace=True)
        plane = local_pca_plane(X[idx], 2)
        proj = (X - X.mean(0)) @ plane
        ang = np.arctan2(proj[:, 1], proj[:, 0])
        coords.append((ang / (2 * np.pi)) % 1.0)
    diss = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            r2 = circ_procrustes_r2(coords[i], coords[j])
            diss.append(1.0 - r2)
    return float(np.mean(diss))


CYCLIC = {"days", "months", "directions", "notes"}


def run_real(out, prompts_path):
    cache = os.path.join(out, "real_acts.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        by_cat = {c: (d[f"X_{c}"], list(d[f"v_{c}"])) for c in d["cats"]}
    else:
        by_cat = harvest_real(prompts_path)
        save = {"cats": np.array(list(by_cat.keys()), dtype=object)}
        for c, (X, v) in by_cat.items():
            save[f"X_{c}"] = X
            save[f"v_{c}"] = np.array(v, dtype=object)
        np.savez(cache, **save)
    rows = []
    for cat, (X, vals) in by_cat.items():
        rep = trust_score(X)
        disag = bootstrap_disagreement(X)
        rows.append(dict(category=cat, n=len(X), is_cyclic=cat in CYCLIC,
                         trust=rep.trust, sigma_min=rep.sigma_min,
                         coherence=rep.coherence, topo_margin=rep.topo_margin,
                         coverage=rep.coverage, level0=rep.level0,
                         untyped=rep.untyped, disagreement=disag))
    return rows


# ---------------------------------------------------------------------------
# correlation + plotting
# ---------------------------------------------------------------------------

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def write_csv(path, rows):
    if not rows:
        open(path, "w").write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=os.path.join(
        os.path.dirname(__file__), "prompts_cyclic.json"))
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=== SYNTHETIC calibration (low trust -> high coordinate error) ===")
    synth = run_synthetic(args.out, seeds=args.seeds)
    write_csv(os.path.join(args.out, "trust_synth.csv"), synth)
    tr = [r["trust"] for r in synth]
    er = [r["coord_err"] for r in synth]
    rho_s = spearman(tr, er)
    lo = [r["coord_err"] for r in synth if r["trust"] < np.median(tr)]
    hi = [r["coord_err"] for r in synth if r["trust"] >= np.median(tr)]
    print(f"  n={len(synth)} synthetic atoms across {len(SYNTH_CASES)} cases")
    print(f"  Spearman(trust, coord_err) = {rho_s:.3f}  (want strongly NEGATIVE)")
    print(f"  mean coord_err  low-trust={np.mean(lo):.3f}  high-trust={np.mean(hi):.3f}")
    print("  per-case means:")
    cases = {}
    for r in synth:
        cases.setdefault(r["label"], []).append(r)
    print(f"    {'case':14s} {'trust':>6s} {'cErr':>6s} {'sigm':>5s} {'topo':>5s} "
          f"{'cov':>5s} {'lvl0':>5s} {'untyped%':>8s}")
    for lab, rs in cases.items():
        print(f"    {lab:14s} {np.mean([x['trust'] for x in rs]):6.3f} "
              f"{np.mean([x['coord_err'] for x in rs]):6.3f} "
              f"{np.mean([x['sigma_min'] for x in rs]):5.2f} "
              f"{np.mean([x['topo_margin'] for x in rs]):5.2f} "
              f"{np.mean([x['coverage'] for x in rs]):5.2f} "
              f"{np.mean([x['level0'] for x in rs]):5.2f} "
              f"{100*np.mean([x['untyped'] for x in rs]):7.0f}%")

    real = []
    rho_r = float("nan")
    if not args.skip_real:
        print("\n=== REAL calibration (low trust -> cross-seed disagreement) ===")
        try:
            real = run_real(args.out, args.prompts)
            write_csv(os.path.join(args.out, "trust_real.csv"), real)
            tr2 = [r["trust"] for r in real]
            ds = [r["disagreement"] for r in real]
            rho_r = spearman(tr2, ds)
            print(f"  n={len(real)} real categories")
            print(f"  Spearman(trust, disagreement) = {rho_r:.3f}  (want NEGATIVE)")
            print(f"    {'category':12s} {'cyc':>4s} {'n':>4s} {'trust':>6s} "
                  f"{'disagr':>7s} {'sigm':>5s} {'topo':>5s} {'cov':>5s}")
            for r in sorted(real, key=lambda x: -x["trust"]):
                print(f"    {r['category']:12s} {str(r['is_cyclic']):>4s} "
                      f"{r['n']:4d} {r['trust']:6.3f} {r['disagreement']:7.3f} "
                      f"{r['sigma_min']:5.2f} {r['topo_margin']:5.2f} {r['coverage']:5.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"  REAL leg unavailable ({type(e).__name__}: {e}); synthetic stands alone.")

    summary = dict(
        gamfit_solver="BLOCKED (RemlConvergenceError on K=1 circle smoke, gamfit 0.1.151)",
        synthetic=dict(n=len(synth), spearman_trust_vs_coord_err=rho_s,
                       low_trust_mean_err=float(np.mean(lo)),
                       high_trust_mean_err=float(np.mean(hi))),
        real=dict(n=len(real), spearman_trust_vs_disagreement=rho_r,
                  rows=real),
    )
    json.dump(summary, open(os.path.join(args.out, "calibration.json"), "w"), indent=2)

    try:
        _plot(args.out, synth, real)
    except Exception as e:  # noqa: BLE001
        print(f"  (plot skipped: {type(e).__name__}: {e})")
    print(f"\n  wrote {args.out}/trust_synth.csv, calibration.json"
          + (", trust_real.csv" if real else ""))


def _plot(out, synth, real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tr = np.array([r["trust"] for r in synth])
    er = np.array([r["coord_err"] for r in synth])
    labels = [r["label"] for r in synth]
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(uniq):
        m = [j for j, l in enumerate(labels) if l == lab]
        axes[0].scatter(tr[m], er[m], s=28, color=cmap(i % 10), label=lab, alpha=0.8)
    axes[0].set_xlabel("trust score"); axes[0].set_ylabel("coord error (1 - circ R2)")
    axes[0].set_title("SYNTHETIC: low trust -> high coord error")
    axes[0].legend(fontsize=6, ncol=2); axes[0].grid(alpha=0.3)
    if real:
        tr2 = np.array([r["trust"] for r in real])
        ds = np.array([r["disagreement"] for r in real])
        cyc = np.array([r["is_cyclic"] for r in real])
        axes[1].scatter(tr2[cyc], ds[cyc], s=60, c="C0", label="cyclic")
        axes[1].scatter(tr2[~cyc], ds[~cyc], s=60, c="C3", label="linear")
        for r in real:
            axes[1].annotate(r["category"], (r["trust"], r["disagreement"]),
                             fontsize=7)
        axes[1].set_xlabel("trust score"); axes[1].set_ylabel("cross-seed disagreement")
        axes[1].set_title("REAL: low trust -> cross-seed disagreement")
        axes[1].legend(); axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "REAL leg not run", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration.png"), dpi=130)
    print(f"  wrote {out}/calibration.png")


if __name__ == "__main__":
    main()
