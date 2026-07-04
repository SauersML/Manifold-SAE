#!/usr/bin/env python3
"""Aggregate the per-feature premise results into one premise_deviance.json, recomputing ROBUST
statistics from the saved per-row deltas (deltas_<name>.npz). No model fitting here — the
deltas are gamfit reconstruction residuals; this is pure resampling/bookkeeping.

Why robust: the paired dividend Δ = D(line) − D(circle) is a difference of squared held-out
deviances. A held-out point that projects catastrophically onto the CLOSED circle (the circle
cannot extrapolate the way the line can) produces a huge single-row deviance that dominates the
MEAN. So the mean-based sign-flip test is outlier-driven; the robust headline is the
distribution-free SIGN TEST on each row's contrast plus the MEDIAN dividend. We report all of:
median, sign-test p, mean + sign-flip p, and a 10%-winsorized mean + its sign-flip p."""
import glob
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA = os.path.join(BASE, "data")
B_PERM = 20000

FEATURE_ORDER = ["weekday_8b_L18", "month_8b_L18", "color_35b_L17", "weekday_35b_L17",
                 "month_35b_L17", "sycophancy_8b_L18", "hedging_8b_L18", "dayofmonth_8b_L18"]


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return float("nan")
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    p0 = pmf[k]
    return float(min(1.0, sum(pm for pm in pmf if pm <= p0 + 1e-15)))


def robust_stats(delta, seed=0, B=B_PERM):
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(B, n)) * 2 - 1
    T = float(d.mean())
    null = (signs * d[None, :]).mean(1)
    p_mean = (1 + int(np.sum(np.abs(null) >= abs(T)))) / (B + 1)
    lo, hi = np.percentile(d, [10, 90])
    dw = np.clip(d, lo, hi); Tw = float(dw.mean())
    nullw = (signs * dw[None, :]).mean(1)
    p_wins = (1 + int(np.sum(np.abs(nullw) >= abs(Tw)))) / (B + 1)
    nz = d[d != 0.0]; k = int(np.sum(nz > 0))
    return dict(n=int(n), mean=T, median=float(np.median(d)),
                frac_positive=float(np.mean(d > 0)), n_positive=k, n_nonzero=int(len(nz)),
                sign_test_p=binom_two_sided(k, len(nz)),
                winsorized_mean=Tw, p_winsorized=float(p_wins),
                p_two_sided=float(p_mean), sd=float(d.std(ddof=1)) if n > 1 else float("nan"))


def verdict(beh):
    """Robust verdict from behavioral dividend: sign test + median direction."""
    if not beh or beh.get("n", 0) == 0:
        return "fit_fragile_no_verdict"
    sp = beh.get("sign_test_p", 1.0)
    med = beh.get("median", 0.0)
    if sp < 0.05 and med > 0:
        return "curvature_pays"
    if sp < 0.05 and med < 0:
        return "curvature_costs"
    return "honest_negative"


def main():
    results = []
    for npz in sorted(glob.glob(os.path.join(DATA, "deltas_*.npz"))):
        name = os.path.basename(npz)[len("deltas_"):-len(".npz")]
        z = np.load(npz)
        rj = os.path.join(DATA, f"result_{name}.json")
        meta = json.load(open(rj)) if os.path.exists(rj) else {}
        beh = robust_stats(z["beh_delta"], seed=1)
        raw = robust_stats(z["raw_delta"], seed=2)
        beh_sur = robust_stats(z["beh_delta_surrogate"], seed=3) if "beh_delta_surrogate" in z.files else {}
        rec = dict(
            name=name, model=meta.get("model"), layer=meta.get("layer"),
            expect_cyclic=meta.get("expect_cyclic"), n=meta.get("n"), p=meta.get("p"),
            n_templates=meta.get("n_templates"), rdim=meta.get("rdim"),
            held_out_mean_deviance=meta.get("held_out_mean_deviance"),
            fold_info=meta.get("fit_info") or meta.get("fold_info"),
            paired_deviance_behavioral=beh, paired_deviance_raw=raw,
            surrogate_gaussian_behavioral=beh_sur,
            surrogate_gaussian_raw=meta.get("surrogate_gaussian_raw"),
            verdict=verdict(beh),
        )
        results.append(rec)
    results.sort(key=lambda r: FEATURE_ORDER.index(r["name"]) if r["name"] in FEATURE_ORDER else 99)
    json.dump(dict(instrument="held_out_paired_deviance_robust", n_perm=B_PERM, results=results),
              open(os.path.join(DATA, "premise_deviance.json"), "w"), indent=2)
    for r in results:
        b = r["paired_deviance_behavioral"]
        print(f"{r['name']:20s} {r['verdict']:16s} med={b.get('median',0):+.4g} "
              f"signp={b.get('sign_test_p',1):.3g} meanp={b.get('p_two_sided',1):.3g} "
              f"frac+={b.get('frac_positive',0):.2f} surr_med={r['surrogate_gaussian_behavioral'].get('median',0):+.4g}")
    print("wrote", os.path.join(DATA, "premise_deviance.json"))


if __name__ == "__main__":
    main()
