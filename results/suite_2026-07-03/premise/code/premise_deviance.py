"""Held-out paired deviance premise instrument.

For a candidate feature (weekday, month, color, sycophancy, hedging, ...) we ask the ONE
question the dose-calibration crown cannot answer on its own: does adding *curvature* to
the 1-D chart reduce reconstruction deviance on rows the fit never saw — independent of any
dose calibration or metric normalization?

Design (see DESIGN.md):
  * two nested unsupervised charts on the demeaned residual stream H = X_last - tmpl_mean,
    both via the identical `gamfit.sae_manifold_fit`, K=1, d_atom=1, rank-`rdim` working
    frame, differing ONLY in `atom_topology`: "linear" (a straight 1-D line) vs "circle"
    (the same dimension, allowed to bend/close).
  * held-out by a 2-fold COMPLEMENTARY split over templates (templates recovered directly
    from the unique rows of tmpl_mean — assumption-free): fit on fold A's templates, score
    fold B's rows, then swap. Every row is scored exactly ONCE, on a fit that never saw its
    template. The PCA working frame + centering mean are fit on the fit rows only (no leak).
  * per held-out row i, for each topology, full-space residual
        r_i = H_i - (mu + Vt^T reconstruct(H_red)_i)
    -> raw deviance  D_i = ||r_i||^2   (Gaussian / activation units^2)
    -> behavioral deviance  D_i^F = sum_k (u_ik . r_i)^2 = ||r_i^T U_i||^2   (nats, ~2*KL;
       the part of the residual that still moves the model's output distribution).
    The frame-truncation (out-of-frame) part of r_i is identical for both topologies (same
    mu, Vt) and cancels in the paired difference.
  * curvature dividend per row:  Delta_i = D_i(linear) - D_i(circle)  (raw and behavioral).
    Delta_i > 0  <=>  curvature pays on row i.
  * PAIRED SIGN-FLIP permutation test. H0: neither topology is favored on any given row
    (line/circle exchangeable *conditional on the row*). The randomization distribution
    induced by H0 is independent per-row sign flips of Delta_i. Statistic T = mean_i Delta_i;
    null T*_b = mean_i eps_i Delta_i, eps_i in {+1,-1}. Two-sided
        p = (1 + #{ |T*_b| >= |T| }) / (B+1).
    This is the exact paired randomization test — correct *because* both deviances share a
    row, so the only freedom H0 leaves is the +/- orientation of each within-row contrast.

Falsification (run before trusting anything):
  * the sign-flip null is centered at 0 by construction — we report its mean/sd to confirm.
  * GAUSSIAN-MATCHED surrogate negative control: replace H with a multivariate normal matched
    to H's covariance (structureless, same 2nd moments), run the entire 2-fold pipeline.
    Curvature must NOT pay (Delta^F ~ 0, p not significant) — else the instrument is biased
    toward the circle by its extra geometric freedom and the real p-values are worthless.

A feature where curvature does not pay on the REAL data (Delta^F ~ 0, p n.s.) is reported as
a headline HONEST NEGATIVE, not buried.

CPU job. gamfit chart fits are fragile/slow: each fit is wrapped in a wall-clock alarm with
seed retries; a grind or guard-abort is RECORDED as a fold failure, never hung.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

ROOT = os.environ.get("ROOT", "/projects/standard/hsiehph/sauer354")
OUT = os.environ.get("PREMISE_OUT", os.path.join(ROOT, "premise_out"))
os.makedirs(OUT, exist_ok=True)
RDIM = int(os.environ.get("PREMISE_RDIM", "8"))
NITER = int(os.environ.get("PREMISE_NITER", "50"))
FIT_ALARM = int(os.environ.get("PREMISE_FIT_ALARM", "300"))
B_PERM = int(os.environ.get("PREMISE_BPERM", "20000"))
SEED = int(os.environ.get("PREMISE_SEED", "0"))

CACHES = [
    # name, npz, model, layer, expect_cyclic
    ("weekday_8b_L18", f"{ROOT}/dose_qwen8b_out/harvest_cache_weekday_L18_n70.npz", "qwen3-8b", 18, True),
    ("month_8b_L18", f"{ROOT}/dose_month_out/harvest_cache_month_L18_n120.npz", "qwen3-8b", 18, True),
    ("color_35b_L17", f"{ROOT}/dose_qwen36b_out/harvest_cache_color_L17_n48.npz", "qwen3-35b", 17, True),
    ("weekday_35b_L17", f"{ROOT}/dose_qwen36b_out/harvest_cache_weekday_L17_n42.npz", "qwen3-35b", 17, True),
    ("month_35b_L17", f"{ROOT}/dose_qwen36b_out/harvest_cache_month_L17_n72.npz", "qwen3-35b", 17, True),
    # safety caches produced by harvest_safety_acts.py (graded, NOT cyclic)
    ("sycophancy_8b_L18", f"{OUT}/harvest_cache_sycophancy_L18.npz", "qwen3-8b", 18, False),
    ("hedging_8b_L18", f"{OUT}/harvest_cache_hedging_L18.npz", "qwen3-8b", 18, False),
]


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def robust_fit(Hr, topo, seconds=FIT_ALARM):
    """Fit one K=1 d_atom=1 chart of the given topology on reduced-frame Hr, with a hard
    wall-clock alarm and seed retries. Return (sae, info) or (None, info-with-error)."""
    import gamfit
    attempts = [dict(n_iter=NITER, random_state=SEED),
                dict(n_iter=NITER + 20, random_state=SEED + 101),
                dict(n_iter=NITER + 40, random_state=SEED + 202)]
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for kw in attempts:
            signal.alarm(int(seconds))
            t0 = time.time()
            try:
                sae = gamfit.sae_manifold_fit(Hr, K=1, d_atom=1, atom_topology=topo, **kw)
                signal.alarm(0)
                r2 = float(sae.reconstruction_r2)
                if np.isfinite(r2) and len(sae.atom_topologies) == 1:
                    return sae, dict(ok=True, r2=r2, seconds=time.time() - t0, kw=kw,
                                     topo_fitted=list(sae.atom_topologies))
                last = dict(ok=False, reason=f"r2={r2} atoms={len(sae.atom_topologies)}")
            except _Timeout:
                signal.alarm(0)
                last = dict(ok=False, reason=f"timeout>{seconds}s", kw=kw)
            except Exception as exc:  # noqa: BLE001
                signal.alarm(0)
                last = dict(ok=False, reason=f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}", kw=kw)
        return None, last
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def templates_from_tmpl_mean(tm):
    """Recover template id per row from the per-template mean (rows of a template are
    identical in tmpl_mean). Assumption-free grouping."""
    key = np.round(tm, 5)
    uniq, tid = np.unique(key, axis=0, return_inverse=True)
    return tid, len(uniq)


def deviances_for_fold(H, U, fit_idx, test_idx):
    """Fit linear+circle on fit_idx (reduced frame from fit rows), return per-test-row raw &
    behavioral deviance for each topology, plus fit info. Residuals in FULL space."""
    mu = H[fit_idx].mean(0)
    Hc = H[fit_idx] - mu
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    rdim = min(RDIM, Vt.shape[0])
    Vt = np.ascontiguousarray(Vt[:rdim])
    Hr_all = np.ascontiguousarray((H - mu) @ Vt.T)
    out = {}
    info = {}
    for topo in ("linear", "circle"):
        sae, fi = robust_fit(Hr_all[fit_idx], topo)
        info[topo] = fi
        if sae is None:
            out[topo] = None
            continue
        rec_red = np.asarray(sae.reconstruct(Hr_all[test_idx]), dtype=np.float64)  # (nt, rdim)
        resid = H[test_idx] - (mu + rec_red @ Vt)                                  # (nt, p) full
        raw = np.einsum("np,np->n", resid, resid)
        if U is not None:
            proj = np.einsum("np,nps->ns", resid, U[test_idx])                     # (nt, 8)
            behav = np.einsum("ns,ns->n", proj, proj)
        else:
            behav = np.full(len(test_idx), np.nan)
        out[topo] = dict(raw=raw, behav=behav)
    return out, info


def run_pipeline(H, U, tid):
    """2-fold complementary split over templates -> per-row (once) paired deviances."""
    tids = np.unique(tid)
    foldA = tids[::2]
    foldB = tids[1::2]
    n = len(H)
    raw_lin = np.full(n, np.nan); raw_cir = np.full(n, np.nan)
    beh_lin = np.full(n, np.nan); beh_cir = np.full(n, np.nan)
    fold_info = []
    for fit_tpls, name in ((foldA, "A"), (foldB, "B")):
        fit_idx = np.where(np.isin(tid, fit_tpls))[0]
        test_idx = np.where(~np.isin(tid, fit_tpls))[0]
        if len(fit_idx) < 4 or len(test_idx) == 0:
            fold_info.append(dict(fold=name, skipped="too few rows"))
            continue
        out, info = deviances_for_fold(H, U, fit_idx, test_idx)
        fold_info.append(dict(fold=name, n_fit=int(len(fit_idx)), n_test=int(len(test_idx)),
                              fit_info=info))
        if out.get("linear") is None or out.get("circle") is None:
            continue
        raw_lin[test_idx] = out["linear"]["raw"]; raw_cir[test_idx] = out["circle"]["raw"]
        beh_lin[test_idx] = out["linear"]["behav"]; beh_cir[test_idx] = out["circle"]["behav"]
    return dict(raw_lin=raw_lin, raw_cir=raw_cir, beh_lin=beh_lin, beh_cir=beh_cir,
                fold_info=fold_info)


def signflip_test(delta, B=B_PERM, seed=0):
    """Paired sign-flip randomization test. Returns dict with observed mean, p, null summary."""
    delta = np.asarray(delta, float)
    delta = delta[np.isfinite(delta)]
    n = len(delta)
    if n == 0:
        return dict(n=0, mean=float("nan"), p=float("nan"))
    T = float(delta.mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(B, n)) * 2 - 1
    null = (signs * delta[None, :]).mean(1)
    p = (1 + int(np.sum(np.abs(null) >= abs(T)))) / (B + 1)
    sd = float(delta.std(ddof=1)) if n > 1 else float("nan")
    return dict(n=int(n), mean=T, sd=sd, median=float(np.median(delta)),
                frac_positive=float(np.mean(delta > 0)),
                effect_size=(T / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")),
                p_two_sided=float(p),
                null_mean=float(null.mean()), null_sd=float(null.std()))


def gaussian_surrogate(H, tid, seed=0):
    """Multivariate-normal surrogate matched to H's covariance (structureless, same 2nd
    moments), preserving n and the template block sizes (tid reused)."""
    rng = np.random.default_rng(seed)
    mu = H.mean(0)
    Hc = H - mu
    # low-rank matched covariance via SVD (p >> n), sample in row space
    U_, S_, Vt_ = np.linalg.svd(Hc, full_matrices=False)
    # each surrogate row = mu + (randn scaled by singular values) @ Vt_
    Z = rng.standard_normal((len(H), len(S_)))
    Hg = mu + (Z * (S_ / np.sqrt(max(len(H) - 1, 1)))) @ Vt_
    return Hg


def analyze_cache(name, npz, model, layer, cyclic):
    if not os.path.exists(npz):
        return dict(name=name, error="cache missing", npz=npz)
    z = np.load(npz)
    X = z["X_last"].astype(np.float64)
    tm = z["tmpl_mean"].astype(np.float64)
    U = z["U_last"].astype(np.float64) if "U_last" in z.files else None
    H = X - tm
    n, p = H.shape
    tid, T = templates_from_tmpl_mean(tm)
    print(f"[{name}] n={n} p={p} templates={T} rows/tpl={n/T:.1f} U={'yes' if U is not None else 'NO'}",
          flush=True)
    res = run_pipeline(H, U, tid)
    raw_delta = res["raw_lin"] - res["raw_cir"]
    beh_delta = res["beh_lin"] - res["beh_cir"]
    stat_raw = signflip_test(raw_delta, seed=SEED)
    stat_beh = signflip_test(beh_delta, seed=SEED + 1)

    # held-out mean deviance per topology (context for the effect size)
    def _m(a):
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else float("nan")
    heldout = dict(
        raw_linear=_m(res["raw_lin"]), raw_circle=_m(res["raw_cir"]),
        behav_linear=_m(res["beh_lin"]), behav_circle=_m(res["beh_cir"]),
    )

    # negative control: gaussian-matched surrogate, identical pipeline
    Hg = gaussian_surrogate(H, tid, seed=SEED + 7)
    resg = run_pipeline(Hg, U, tid)
    beh_delta_g = resg["beh_lin"] - resg["beh_cir"]
    raw_delta_g = resg["raw_lin"] - resg["raw_cir"]
    stat_beh_g = signflip_test(beh_delta_g, seed=SEED + 2)
    stat_raw_g = signflip_test(raw_delta_g, seed=SEED + 3)

    verdict = "curvature_pays" if (stat_beh["p_two_sided"] < 0.05 and stat_beh["mean"] > 0) \
        else ("honest_negative" if np.isfinite(stat_beh["mean"]) else "fit_fragile_no_verdict")

    rec = dict(
        name=name, model=model, layer=layer, expect_cyclic=bool(cyclic),
        n=int(n), p=int(p), n_templates=int(T), rdim=RDIM, n_perm=B_PERM,
        held_out_mean_deviance=heldout,
        paired_deviance_raw=stat_raw, paired_deviance_behavioral=stat_beh,
        surrogate_gaussian_behavioral=stat_beh_g, surrogate_gaussian_raw=stat_raw_g,
        fold_info=res["fold_info"], verdict=verdict,
    )
    # persist per-row deltas for figures + per-feature result (independent runs merge cleanly)
    np.savez(os.path.join(OUT, f"deltas_{name}.npz"),
             raw_lin=res["raw_lin"], raw_cir=res["raw_cir"],
             beh_lin=res["beh_lin"], beh_cir=res["beh_cir"],
             raw_delta=raw_delta, beh_delta=beh_delta,
             beh_delta_surrogate=beh_delta_g, tid=tid)
    with open(os.path.join(OUT, f"result_{name}.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[{name}] VERDICT={verdict} behav Delta mean={stat_beh['mean']:.4g} "
          f"p={stat_beh['p_two_sided']:.4g} frac+={stat_beh.get('frac_positive')} | "
          f"surrogate p={stat_beh_g['p_two_sided']:.4g} mean={stat_beh_g['mean']:.4g}", flush=True)
    return rec


def main():
    only = os.environ.get("PREMISE_ONLY")
    results = []
    for name, npz, model, layer, cyclic in CACHES:
        if only and name not in only.split(","):
            continue
        try:
            results.append(analyze_cache(name, npz, model, layer, cyclic))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append(dict(name=name, error=f"{type(exc).__name__}: {exc}"))
    payload = dict(
        instrument="held_out_paired_deviance", rdim=RDIM, n_iter=NITER, n_perm=B_PERM,
        gamfit_note="linear vs circle, K=1 d_atom=1, 2-fold complementary template split",
        results=results,
    )
    with open(os.path.join(OUT, "premise_deviance.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print("[done] wrote", os.path.join(OUT, "premise_deviance.json"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
