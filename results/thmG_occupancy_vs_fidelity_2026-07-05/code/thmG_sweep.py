"""Theorem G / Prediction P1 — occupancy sets topology, fidelity does not.

The claim under test: a manifold atom's TOPOLOGY verdict (circle vs interval; closed
loop vs open arc) is set by OCCUPANCY (number of firing rows n_eff), NOT by FIDELITY
(reconstruction SNR / 1/sigma). Fidelity cannot buy topology; only occupancy can.

We reuse the PREMISE held-out paired-deviance instrument verbatim (a straight `linear`
chart vs a `circle` chart, K=1 d_atom=1, fit via gamfit.sae_manifold_fit on the demeaned
residual stream, scored on held-out rows by a leave-template-out 2-fold complementary
split, paired sign-flip / sign test, with a Gaussian-matched surrogate null). ALL fitting
is gamfit; numpy is orchestration/scoring only (no hand-rolled manifold fits).

We then run that instrument on a 2D grid:

  OCCUPANCY axis  — subsample the number of active rows n_eff by keeping a subset of
                    templates (all C categories retained, so the C angular positions of a
                    cyclic feature survive). n_eff = C * n_templates_kept.
  FIDELITY  axis  — hold n_eff fixed (all templates) and DEGRADE reconstruction SNR by
                    adding controlled isotropic Gaussian noise sigma*I to the full-space
                    activations before the identical pipeline. Larger sigma = lower fidelity.
                    (A model-side fidelity knob, working-frame rank rdim, is available via
                    THMG_FIDELITY_KNOB=rdim as a cross-check.)

For every (occupancy, fidelity) cell we record the PREMISE dividend Delta = D(linear) -
D(circle) on held-out rows, in behavioral (output-Fisher, nats) and raw activation units:
median Delta, sign-test p, frac rows circle wins, plus the Gaussian-surrogate median in the
SAME cell, and the MEASURED fidelity (held-out reconstruction R2 for each topology) so the
fidelity axis is calibrated in achieved SNR, not just nominal sigma.

PREDICTION (to falsify): verdict strength (|median Delta|, significance) SHARPENS along the
OCCUPANCY axis but stays ~FLAT along the FIDELITY axis. If it sharpens with fidelity at fixed
occupancy, Theorem G is FALSE.

CPU job (acn116). Each gamfit fit is wrapped in a wall-clock alarm with a seed retry; a
grind/guard-abort is recorded as a cell failure, never hung.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

ROOT = os.environ.get("ROOT", "/projects/standard/hsiehph/sauer354")
OUT = os.environ.get("THMG_OUT", os.path.join(ROOT, "thmG_out"))
os.makedirs(OUT, exist_ok=True)

RDIM = int(os.environ.get("THMG_RDIM", "8"))
NITER = int(os.environ.get("THMG_NITER", "25"))
FIT_ALARM = int(os.environ.get("THMG_FIT_ALARM", "240"))
B_PERM = int(os.environ.get("THMG_BPERM", "5000"))
RHO_SEARCH = os.environ.get("THMG_RHO_SEARCH", "0") != "0"  # off: rho-on hangs the linear fit (gam#2138)
_sw = os.environ.get("THMG_SMOOTH", "")
SMOOTH = float(_sw) if _sw not in ("", "none", "None") else None  # fixed smoothing penalty
SEED = int(os.environ.get("THMG_SEED", "0"))
FIDELITY_KNOB = os.environ.get("THMG_FIDELITY_KNOB", "noise")  # noise | rdim

CACHES = {
    "weekday_8b_L18": (f"{ROOT}/dose_qwen8b_out/harvest_cache_weekday_L18_n70.npz", 7, True),
    "month_8b_L18": (f"{ROOT}/dose_month_out/harvest_cache_month_L18_n120.npz", 12, True),
    "sycophancy_8b_L18": (f"{OUT}/../premise_out/harvest_cache_sycophancy_L18.npz", 7, False),
    "hedging_8b_L18": (f"{OUT}/../premise_out/harvest_cache_hedging_L18.npz", 7, False),
}


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def robust_fit(Hr, topo, seconds=FIT_ALARM):
    """One K=1 d_atom=1 chart of the given topology, hard wall-clock alarm, one seed retry.
    Structure search off (K,topology fixed); outer-rho REML on (regularizes the circle so its
    extra freedom does not overfit held-out rows — the surrogate gate depends on this)."""
    import gamfit
    fixedkw = dict(_run_structure_search=False, _run_outer_rho_search=RHO_SEARCH)
    if SMOOTH is not None:
        fixedkw["smoothness_weight"] = SMOOTH  # fixed penalty replaces the hanging outer-rho search
    attempts = [dict(n_iter=NITER, random_state=SEED),
                dict(n_iter=NITER, random_state=SEED + 101),
                dict(n_iter=NITER, random_state=SEED + 202),
                dict(n_iter=NITER, random_state=SEED + 303)]
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    last = dict(ok=False, reason="no attempt")
    try:
        for kw in attempts:
            signal.alarm(int(seconds))
            t0 = time.time()
            try:
                try:
                    sae = gamfit.sae_manifold_fit(Hr, K=1, d_atom=1, atom_topology=topo,
                                                  **kw, **fixedkw)
                except TypeError:
                    sae = gamfit.sae_manifold_fit(Hr, K=1, d_atom=1, atom_topology=topo, **kw)
                signal.alarm(0)
                r2 = float(sae.reconstruction_r2)
                if np.isfinite(r2) and len(sae.atom_topologies) == 1:
                    return sae, dict(ok=True, r2=r2, seconds=time.time() - t0)
                last = dict(ok=False, reason=f"r2={r2} atoms={len(sae.atom_topologies)}")
            except _Timeout:
                signal.alarm(0)
                last = dict(ok=False, reason=f"timeout>{seconds}s")
            except Exception as exc:  # noqa: BLE001
                signal.alarm(0)
                last = dict(ok=False, reason=f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        return None, last
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def template_ids(z, C):
    if "template" in z.files:
        tid = np.asarray(z["template"]).astype(int)
        _, tid = np.unique(tid, return_inverse=True)
        return tid
    kept = np.asarray(z["kept"]).astype(int) if "kept" in z.files else np.arange(len(z["X_last"]))
    tid = kept // int(C)
    _, tid = np.unique(tid, return_inverse=True)
    return tid


def deviances_for_fold(H, U, fit_idx, test_idx, rdim):
    """Fit linear+circle on fit_idx (reduced frame from fit rows only), per-test-row raw &
    behavioral deviance for each topology + measured held-out reconstruction R2. Full-space
    residuals; frame-truncation part cancels in the paired difference."""
    mu = H[fit_idx].mean(0)
    Hc = H[fit_idx] - mu
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    rdim = min(rdim, Vt.shape[0])
    Vt = np.ascontiguousarray(Vt[:rdim])
    Hr_all = np.ascontiguousarray((H - mu) @ Vt.T)
    # total held-out variance in-frame (for a measured-fidelity R2)
    Xt = Hr_all[test_idx]
    tss = float(np.sum((Xt - Xt.mean(0)) ** 2)) + 1e-12
    out, info = {}, {}
    for topo in ("linear", "circle"):
        sae, fi = robust_fit(Hr_all[fit_idx], topo)
        info[topo] = fi
        if sae is None:
            out[topo] = None
            continue
        try:
            rec_red = np.asarray(sae.reconstruct(Hr_all[test_idx]), dtype=np.float64)
            if rec_red.shape != (len(test_idx), rdim):
                raise ValueError("reconstruct shape")
        except Exception as exc:  # noqa: BLE001
            info[topo]["oos"] = f"reconstruct_fail: {type(exc).__name__}: {str(exc)[:60]}"
            out[topo] = None
            continue
        # measured fidelity: in-frame held-out reconstruction R2
        rss_frame = float(np.sum((Xt - rec_red) ** 2))
        r2_heldout = 1.0 - rss_frame / tss
        resid = H[test_idx] - (mu + rec_red @ Vt)
        raw = np.einsum("np,np->n", resid, resid)
        if U is not None:
            proj = np.einsum("np,nps->ns", resid, U[test_idx])
            behav = np.einsum("ns,ns->n", proj, proj)
        else:
            behav = np.full(len(test_idx), np.nan)
        out[topo] = dict(raw=raw, behav=behav, r2_heldout=r2_heldout)
    return out, info


def run_pipeline(H, U, tid, rdim=RDIM):
    tids = np.unique(tid)
    foldA, foldB = tids[::2], tids[1::2]
    n = len(H)
    raw_lin = np.full(n, np.nan); raw_cir = np.full(n, np.nan)
    beh_lin = np.full(n, np.nan); beh_cir = np.full(n, np.nan)
    r2_lin, r2_cir, folds = [], [], []
    for fit_tpls, name in ((foldA, "A"), (foldB, "B")):
        fit_idx = np.where(np.isin(tid, fit_tpls))[0]
        test_idx = np.where(~np.isin(tid, fit_tpls))[0]
        if len(fit_idx) < 4 or len(test_idx) == 0:
            folds.append(dict(fold=name, skipped="too few rows")); continue
        out, info = deviances_for_fold(H, U, fit_idx, test_idx, rdim)
        folds.append(dict(fold=name, n_fit=int(len(fit_idx)), n_test=int(len(test_idx)), fit_info=info))
        if out.get("linear") is None or out.get("circle") is None:
            continue
        raw_lin[test_idx] = out["linear"]["raw"]; raw_cir[test_idx] = out["circle"]["raw"]
        beh_lin[test_idx] = out["linear"]["behav"]; beh_cir[test_idx] = out["circle"]["behav"]
        r2_lin.append(out["linear"]["r2_heldout"]); r2_cir.append(out["circle"]["r2_heldout"])
    return dict(raw_lin=raw_lin, raw_cir=raw_cir, beh_lin=beh_lin, beh_cir=beh_cir,
                r2_lin=float(np.mean(r2_lin)) if r2_lin else float("nan"),
                r2_cir=float(np.mean(r2_cir)) if r2_cir else float("nan"),
                fold_info=folds)


def _binom_two_sided(k, n, p=0.5):
    from math import comb
    if n == 0:
        return float("nan")
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    p0 = pmf[k]
    return float(min(1.0, sum(pm for pm in pmf if pm <= p0 + 1e-15)))


def signflip_test(delta, B=B_PERM, seed=0):
    delta = np.asarray(delta, float)
    delta = delta[np.isfinite(delta)]
    n = len(delta)
    if n == 0:
        return dict(n=0, median=float("nan"), mean=float("nan"), sign_test_p=float("nan"),
                    p_two_sided=float("nan"), frac_circle_wins=float("nan"))
    T = float(delta.mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(B, n)) * 2 - 1
    null = (signs * delta[None, :]).mean(1)
    p_mean = (1 + int(np.sum(np.abs(null) >= abs(T)))) / (B + 1)
    nz = delta[delta != 0.0]
    k = int(np.sum(nz > 0))  # delta>0 <=> circle wins that row (D_line > D_circle)
    p_sign = _binom_two_sided(k, len(nz))
    return dict(n=int(n), median=float(np.median(delta)), mean=T,
                frac_circle_wins=float(np.mean(delta > 0)),
                sign_test_p=p_sign, n_circle_wins=k, n_nonzero=int(len(nz)),
                p_two_sided=float(p_mean))


def gaussian_surrogate(H, seed=0):
    rng = np.random.default_rng(seed)
    mu = H.mean(0); Hc = H - mu
    _, S_, Vt_ = np.linalg.svd(Hc, full_matrices=False)
    Z = rng.standard_normal((len(H), len(S_)))
    return mu + (Z * (S_ / np.sqrt(max(len(H) - 1, 1)))) @ Vt_


def subsample_templates(tid, n_keep, seed):
    """Keep n_keep templates (all categories under each survive). Returns row mask."""
    tids = np.unique(tid)
    rng = np.random.default_rng(seed)
    keep = rng.choice(tids, size=min(n_keep, len(tids)), replace=False)
    return np.isin(tid, keep)


def signal_scale(H):
    """Per-coordinate RMS of the demeaned residual stream — the unit for isotropic noise."""
    Hc = H - H.mean(0)
    return float(np.sqrt(np.mean(Hc ** 2)))


def _fits_ok(fold_info):
    """Count (fold, topology) fits that converged and were scored. A cell is CONVERGED only if
    all 2 folds x 2 topologies returned a scored held-out reconstruction (4/4). This is the gate
    the confound demands: a non-converged cell is EXCLUDED, never scored as a weak verdict."""
    n_ok = 0
    reasons = []
    for f in fold_info:
        if f.get("skipped"):
            reasons.append(f"{f.get('fold')}:skipped")
            continue
        fi = f.get("fit_info", {})
        for topo in ("linear", "circle"):
            t = fi.get(topo, {})
            oos = t.get("oos", "")
            if t.get("ok") and not oos.startswith("reconstruct_fail"):
                n_ok += 1
            else:
                why = "reconstruct_fail" if oos.startswith("reconstruct_fail") else t.get("reason", "skip")
                reasons.append(f"{f.get('fold')}:{topo}:{str(why)[:40]}")
    return n_ok, reasons


def cell(H, U, tid, rdim, sigma_frac, noise_seed):
    """One grid cell: add isotropic noise (fidelity), run the pipeline on real + surrogate.
    Records explicit convergence (n_fits_ok / 4) so non-converged cells can be excluded."""
    Hc = H
    if sigma_frac > 0:
        rng = np.random.default_rng(1000 + noise_seed)
        Hc = H + rng.standard_normal(H.shape) * (sigma_frac * signal_scale(H))
    res = run_pipeline(Hc, U, tid, rdim=rdim)
    beh_delta = res["beh_lin"] - res["beh_cir"]
    raw_delta = res["raw_lin"] - res["raw_cir"]
    Hg = gaussian_surrogate(Hc, seed=SEED + 7 + noise_seed)
    resg = run_pipeline(Hg, U, tid, rdim=rdim)
    beh_g = resg["beh_lin"] - resg["beh_cir"]
    raw_g = resg["raw_lin"] - resg["raw_cir"]
    n_ok, reasons = _fits_ok(res["fold_info"])
    ng_ok, _ = _fits_ok(resg["fold_info"])
    return dict(
        rdim=int(rdim),
        n_eff=int(np.sum(np.isfinite(raw_delta))),
        n_fits_ok=int(n_ok), n_fits_total=4, converged=bool(n_ok == 4),
        surrogate_fits_ok=int(ng_ok),
        nonconv_reasons=reasons,
        behav=signflip_test(beh_delta, seed=SEED + 1),
        raw=signflip_test(raw_delta, seed=SEED + 2),
        surrogate_behav=signflip_test(beh_g, seed=SEED + 3),
        surrogate_raw=signflip_test(raw_g, seed=SEED + 4),
        measured_fidelity=dict(r2_line=res["r2_lin"], r2_circle=res["r2_cir"]),
        fold_info=res["fold_info"],
    )


_CACHE_MEM = {}


def _load_feature(npz, C):
    if npz not in _CACHE_MEM:
        z = np.load(npz)
        X = z["X_last"].astype(np.float64); tm = z["tmpl_mean"].astype(np.float64)
        U = z["U_last"].astype(np.float64) if "U_last" in z.files else None
        _CACHE_MEM[npz] = (X - tm, U, template_ids(z, C))
    return _CACHE_MEM[npz]


def cell_worker(spec):
    """Top-level worker for a single grid cell (picklable spec). Reloads the cache per-process
    (memoized), applies the occupancy subsample or the fidelity noise at the cell's working rank,
    runs the real+surrogate pipeline, returns the cell result dict tagged with its coordinates."""
    H_full, U, tid_full = _load_feature(spec["npz"], spec["C"])
    n_tpl_total = len(np.unique(tid_full))
    if spec["n_tpl"] < n_tpl_total:
        mask = subsample_templates(tid_full, spec["n_tpl"], seed=200 + spec["sub_seed"])
        H, U2, tid = H_full[mask], (U[mask] if U is not None else None), tid_full[mask]
    else:
        H, U2, tid = H_full, U, tid_full
    c = cell(H, U2, tid, int(spec["rank"]), sigma_frac=spec["sigma_frac"], noise_seed=spec["sub_seed"])
    c.update(axis=spec["axis"], n_tpl=int(spec["n_tpl"]), sigma_frac=float(spec["sigma_frac"]),
             sub_seed=int(spec["sub_seed"]))
    return spec["name"], c


def build_specs(name, npz, C, occ_levels, rank_levels, noise_levels, base_rank, noise_seeds):
    """Three axes, all at rho=off, all sharing the SAME base config where they overlap so cells
    are comparable (the confound guard: config must not covary with occupancy):
      * OCCUPANCY  — vary n_tpl at fixed base_rank, sigma=0 (best fidelity). n_eff = C*n_tpl.
      * FIDELITY(rank)  — vary working rank at FULL occupancy, sigma=0 (model-complexity knob).
      * FIDELITY(noise) — vary additive isotropic sigma at FULL occupancy, base_rank (data-SNR knob).
    The full-occupancy base_rank sigma=0 cell is shared across all three axes."""
    H_full, U, tid_full = _load_feature(npz, C)
    n_tpl_total = len(np.unique(tid_full))
    specs = []
    for n_tpl in occ_levels:
        if n_tpl > n_tpl_total:
            continue
        subs = [0] if n_tpl == n_tpl_total else list(range(max(1, len(noise_seeds))))
        for ss in subs:
            specs.append(dict(name=name, npz=npz, C=C, axis="occupancy", n_tpl=int(n_tpl),
                              rank=int(base_rank), sigma_frac=0.0, sub_seed=int(ss)))
    for rank in rank_levels:
        specs.append(dict(name=name, npz=npz, C=C, axis="fidelity_rank", n_tpl=int(n_tpl_total),
                          rank=int(rank), sigma_frac=0.0, sub_seed=0))
    for sigma in noise_levels:
        for ns in noise_seeds:
            specs.append(dict(name=name, npz=npz, C=C, axis="fidelity_noise", n_tpl=int(n_tpl_total),
                              rank=int(base_rank), sigma_frac=float(sigma), sub_seed=int(ns)))
    return specs, n_tpl_total


def _cell_to_queue(spec, q):
    try:
        nm, c = cell_worker(spec)
        q.put(("ok", nm, c))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", spec["name"], f"{type(exc).__name__}: {str(exc)[:160]}"))


def main():
    import multiprocessing as mp
    # Confound-guarded design (rho OFF everywhere; see the fit-config probe): occupancy varies at
    # a FIXED base_rank that converges at the SMALLEST occupancy cell; fidelity is swept two ways
    # at FULL occupancy — working rank (model complexity) and additive noise sigma (data SNR).
    occ = json.loads(os.environ.get("THMG_OCC", "[4,6,8,10]"))
    ranks = json.loads(os.environ.get("THMG_RANKS", "[3,4,5,6]"))
    noise = json.loads(os.environ.get("THMG_NOISE", "[0.0,0.25,0.5,1.0,2.0]"))
    base_rank = int(os.environ.get("THMG_BASE_RANK", "4"))
    nseeds = json.loads(os.environ.get("THMG_NSEEDS", "[0,1,2]"))
    only = os.environ.get("THMG_ONLY", "weekday_8b_L18,sycophancy_8b_L18").split(",")
    nworkers = int(os.environ.get("THMG_WORKERS", "12"))
    cell_timeout = int(os.environ.get("THMG_CELL_TIMEOUT", "300"))

    all_specs, meta = [], {}
    for name in only:
        npz, C, cyclic = CACHES[name]
        specs, ntpl = build_specs(name, npz, C, occ, ranks, noise, base_rank, nseeds)
        all_specs += specs
        meta[name] = dict(name=name, C=C, cyclic=bool(cyclic), n_templates=int(ntpl),
                          base_rank=base_rank, n_iter=NITER, n_perm=B_PERM, rho_search=RHO_SEARCH,
                          occ_levels=occ, rank_levels=ranks, noise_levels=noise, noise_seeds=nseeds,
                          grid=[])
    print(f"[plan] {len(all_specs)} cells over {only} with {nworkers} workers, "
          f"base_rank={base_rank} rho={RHO_SEARCH} cell_timeout={cell_timeout}s", flush=True)

    # Kill-based scheduler: gamfit's Rust fit holds the GIL, so an in-process signal alarm
    # cannot interrupt a grinding fit. Each cell runs in its own multiprocessing.Process; a
    # cell that exceeds cell_timeout is TERMINATED (SIGTERM) and recorded as a failure (NaN),
    # so one pathological fit cannot hang the whole sweep.
    ctx = mp.get_context("spawn")
    pending = list(all_specs)
    inflight = []  # (proc, spec, queue, t0)
    done = fails = 0
    total = len(all_specs)
    while pending or inflight:
        while pending and len(inflight) < nworkers:
            spec = pending.pop()
            q = ctx.Queue()
            p = ctx.Process(target=_cell_to_queue, args=(spec, q))
            p.start()
            inflight.append([p, spec, q, time.time()])
        still = []
        for item in inflight:
            p, spec, q, t0 = item
            got = None
            if not q.empty():
                got = q.get()
            if got is not None:
                p.join(timeout=5)
                if got[0] == "ok":
                    _, nm, c = got
                    meta[nm]["grid"].append(c)
                    done += 1
                    cv = "CONV" if c["converged"] else f"NONCONV({c['n_fits_ok']}/4)"
                    print(f"[{done+fails}/{total}] OK {nm} {c['axis']} n_eff={c['n_eff']} "
                          f"rank={c['rdim']} sig={c['sigma_frac']} {cv} | beh med={c['behav']['median']:.4g} "
                          f"p={c['behav']['sign_test_p']:.3g} | raw med={c['raw']['median']:.4g} "
                          f"p={c['raw']['sign_test_p']:.3g} | r2c={c['measured_fidelity']['r2_circle']:.3g} "
                          f"surR p={c['surrogate_raw']['sign_test_p']:.3g}", flush=True)
                    with open(os.path.join(OUT, f"thmG_{nm}.json"), "w") as fh:
                        json.dump(meta[nm], fh, indent=2)
                else:
                    fails += 1
                    print(f"[{done+fails}/{total}] ERR {got[1]} {spec['axis']} "
                          f"n_tpl={spec['n_tpl']} sig={spec['sigma_frac']} :: {got[2]}", flush=True)
            elif not p.is_alive():
                p.join()
                fails += 1
                print(f"[{done+fails}/{total}] DIED {spec['name']} {spec['axis']} "
                      f"n_tpl={spec['n_tpl']} sig={spec['sigma_frac']} (exit {p.exitcode})", flush=True)
            elif time.time() - t0 > cell_timeout:
                p.terminate(); p.join(timeout=5)
                if p.is_alive():
                    p.kill(); p.join()
                fails += 1
                print(f"[{done+fails}/{total}] TIMEOUT {spec['name']} {spec['axis']} "
                      f"n_tpl={spec['n_tpl']} sig={spec['sigma_frac']} >{cell_timeout}s (killed)", flush=True)
            else:
                still.append(item)
        inflight = still
        time.sleep(1.0)

    for nm in only:
        print(f"[{nm}] wrote thmG_{nm}.json ({len(meta[nm]['grid'])} cells)", flush=True)
    print(f"[done] {done} ok, {fails} failed of {total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
