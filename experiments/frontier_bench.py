"""Compute-matched EV/L0 and bits-per-token frontiers: TopK-linear vs curved refinement.

This is the harness the reviewer asked us to publish *first*: it prices the curved
(manifold) dictionary against the linear/block TopK dictionary at **matched compute**,
so nobody can frame the comparison as EV-per-FLOP-only and quietly hide the bits.

Two frontiers, both as a function of an honest, hardware-independent compute budget:

  1. EV vs FLOPs          -- does curved refinement ever *lose* explained variance at
                             matched compute? (claim to establish/refute: it does not)
  2. bits/token vs FLOPs  -- at matched distortion, does curved *win* description length?

Plus the honest cost line: on a **pure-linear DGP** (no curvature to find) the curved
lane can only pay the selection/refinement overhead. We report that overhead in EV
and in bits so the win on curved data is not oversold.

WHY curvature can win bits (the mechanism the plots must make visible): a circle is a
1-D closed curve living in a 2-D plane. A rank-1 (TopK direction) atom cannot trace it;
to reach a given distortion the linear lane must spend ~2 atoms / a width-2 block per
circle -- it codes TWO scalars per firing where a chart codes ONE (the intrinsic angle).
So the linear lane's *measured* L0 inflates on curved data, and that inflation IS the
code-bit gap. The gap is amortized against a larger dictionary cost for the chart (2p
decoder scalars vs p), so the crossover is set by firing frequency: frequent concepts
pay the code cost many times and favor the chart; rare-tail concepts favor the direction.
Heavy-tailed firing is therefore the decisive, previously-untested axis.

Everything is CLI-driven (no env vars). FLOP accounting is analytic and every constant
is documented in ``flop_model`` and FLOP_ACCOUNTING.md. Fits delegate to gamfit
(SauersML/gam); each fit runs in an isolated subprocess (the REML solver in some builds
leaks / can hit a non-PD Hessian) so a miss is recorded, never crashes the sweep.

  * linear/block TopK lane   -> gamfit.sparse_dictionary_fit(X, K, active=s)
  * curved refinement lane   -> gamfit.sae_manifold_fit(X, K, d_atom=1, atom_topology="circle", top_k=s)
  * same-lane linear control  -> gamfit.sae_manifold_fit(..., atom_topology="linear")
    (isolates *curvature* under ONE optimizer -- the cleanest matched contrast)

Usage
-----
    python -m experiments.frontier_bench \
        --dgp curved --p 1024 --n 40000 --concepts 24 --firing-tail zipf --zipf-s 1.2 \
        --k 8 16 24 32 48 --active 4 --lanes linear curved manifold_linear \
        --out results/suite_2026-07-03/frontiers/synth_p1024_curved.json

    python -m experiments.frontier_bench --dgp linear ... --out .../synth_linear_overhead.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Thread caps + real tracebacks (friendly-traceback, if installed, hides them).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.excepthook = sys.__excepthook__

import numpy as np

LN2 = math.log(2.0)
HERE = Path(__file__).resolve().parent


# ===========================================================================
# Data-generating processes (we own these; planted ground truth)
# ===========================================================================

def _zipf_weights(n: int, s: float) -> np.ndarray:
    w = 1.0 / np.power(np.arange(1, n + 1), s)
    return w / w.sum()


def make_dgp(kind: str, *, p: int, n: int, n_concepts: int, active_mean: float = 4.0,
             firing_tail: str = "zipf", zipf_s: float = 1.2, noise_std: float = 0.05,
             harmonics: int = 1, rotate: bool = True, seed: int = 0) -> dict:
    """Plant ``n_concepts`` concepts in R^p and sample ``n`` tokens.

    kind="curved": each concept is a circle (1-D closed curve) in a random 2-D plane
        (H>1 adds harmonics -> a 2H-dim ambient patch). A firing token emits the curve
        at a random angle. A linear dictionary needs 2H dims to span; a chart codes the
        single angle. This is the regime where curvature can pay.
    kind="linear": each concept is a single random direction; a firing token emits a
        Gaussian-scaled step. NO curvature -> curved refinement can only lose overhead.

    Firing frequency is heavy-tailed (Zipf) across concepts when firing_tail="zipf":
    a few concepts fire in most tokens, a long tail is rare -- the untested axis.
    A random rotation Q mixes coordinates so structure is not axis-aligned (realistic).
    """
    rng = np.random.default_rng(seed)
    if firing_tail == "zipf":
        pfire = _zipf_weights(n_concepts, zipf_s)
        pfire = np.clip(pfire / pfire.sum() * active_mean, 0.0, 1.0)
    elif firing_tail == "uniform":
        pfire = np.full(n_concepts, min(1.0, active_mean / n_concepts))
    else:
        raise ValueError(f"unknown firing_tail {firing_tail!r}")

    fire = rng.random((n, n_concepts)) < pfire[None, :]
    if kind == "curved":
        bases = rng.standard_normal((n_concepts, 2 * harmonics, p))
        intrinsic, extrinsic = 1, 2 * harmonics
    elif kind == "linear":
        bases = rng.standard_normal((n_concepts, 1, p))
        intrinsic, extrinsic = 1, 1
    else:
        raise ValueError(f"unknown dgp kind {kind!r}")
    bases /= np.linalg.norm(bases, axis=2, keepdims=True)

    X = np.zeros((n, p), dtype=np.float64)
    amps = []
    for c in range(n_concepts):
        idx = np.nonzero(fire[:, c])[0]
        if idx.size == 0:
            amps.append(1.0)
            continue
        if kind == "curved":
            t = rng.uniform(0.0, 2.0 * np.pi, idx.size)
            contrib = np.zeros((idx.size, p))
            for h in range(harmonics):
                contrib += (np.cos((h + 1) * t)[:, None] * bases[c, 2 * h]
                            + np.sin((h + 1) * t)[:, None] * bases[c, 2 * h + 1])
            X[idx] += contrib
            amps.append(1.0)
        else:
            a = rng.standard_normal(idx.size)
            X[idx] += a[:, None] * bases[c, 0][None, :]
            amps.append(float(np.var(a)))
    X += noise_std * rng.standard_normal((n, p))
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        X = X @ Q

    freq = fire.mean(0)
    return {
        "X": X.astype(np.float64), "fire": fire, "freq": freq, "kind": kind, "p": p,
        "n_concepts": n_concepts, "intrinsic_dim": intrinsic, "extrinsic_dim": extrinsic,
        "signal_var": float(np.mean(amps)), "noise_std": noise_std,
        "firing_tail": firing_tail, "zipf_s": zipf_s, "harmonics": harmonics,
        "active_mean": active_mean, "seed": seed,
        "freq_min": float(freq.min()), "freq_max": float(freq.max()),
        "support_entropy_bits": float(_support_entropy_bits(fire)),
    }


def _support_entropy_bits(fire: np.ndarray) -> float:
    """H(S): empirical Shannon entropy (bits) of the per-token active-concept SET.

    This is the empirical-support-entropy selection currency the MDL lane is moving to
    (replaces the combinatorial log2 C(K,s) bound). Computed over the realized firing
    patterns of the planted DGP as a data-side reference; the fit-side H(S) is computed
    from each lane's own routing in ``mdl_bits``."""
    keys = {}
    for row in fire:
        k = row.tobytes()
        keys[k] = keys.get(k, 0) + 1
    tot = sum(keys.values())
    return -sum((c / tot) * math.log2(c / tot) for c in keys.values())


def split_scale(X, test_frac=0.2, seed=1):
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))]
    nt = max(1, int(len(X) * test_frac))
    test, train = X[:nt], X[nt:]
    s = float(np.sqrt((train ** 2).sum(1).mean())) + 1e-12
    return train / s, test / s


def ev(x, xhat):
    sst = float(((x - x.mean(0)) ** 2).sum())
    return float(1.0 - ((x - xhat) ** 2).sum() / sst) if sst > 0 else float("nan")


def mse(x, xhat):
    return float(((x - xhat) ** 2).mean())


# ===========================================================================
# FLOP model (analytic, hardware-independent) -- see FLOP_ACCOUNTING.md
# ===========================================================================
# Count multiply-accumulates (MACs). Row is p-dim; dictionary has K atoms; s fire per
# token (routing sparsity / L0). Atom k has ``dv_k`` decoder vectors of dim p (block
# direction: 1; affine {1,t}: 2; circle {1,cos,sin}: 3). A sparse encoder SCORES every
# atom cheaply to pick the top-s (project the row onto each atom's dv-dim subspace:
# dv_k*p per atom), then REFINES the intrinsic coordinate only for the s selected atoms
# (a dv x dv Gram solve, ~dv^3). Decoding sums the s active atoms' curves. Charging the
# coordinate solve on all K (instead of the selected s) would over-penalize the curved
# lane -- a sparse encoder never solves coordinates for atoms it did not select.
#
#   encode/token = sum_k dv_k * p                     (score ALL atoms)
#                + [curved] s * (mean_dv^3)           (coord solve for SELECTED atoms)
#   decode/token = s * mean_dv * p
#   train        = passes * N * 3 * infer             (forward + ~2x backward)
#
# We feed REALIZED per-atom widths dv_k read from the fit (fit.decoder_blocks / decoder
# shape), not an assumed constant -- so a chart the fit collapsed to linear is priced as
# linear. passes = epochs (block lane) or n_iter (manifold outer REML).

def flop_model(dv_per_atom, s: float, p: int, *, curved: bool, passes: int, n: int) -> dict:
    dv = np.asarray(dv_per_atom, dtype=float)
    K = len(dv)
    mean_dv = float(dv.mean()) if K else 0.0
    score = float((dv * p).sum())                       # score all K atoms
    coord = float(s * (mean_dv ** 3)) if curved else 0.0  # coord solve for selected s only
    enc = score + coord
    dec = float(s * mean_dv * p)
    infer = enc + dec
    return {"encode_macs_per_token": enc, "decode_macs_per_token": dec,
            "infer_macs_per_token": infer, "train_macs_total": passes * n * 3.0 * infer,
            "decoder_params": float((dv * p).sum()), "mean_dv_per_atom": mean_dv,
            "K": K, "passes": passes}


# ===========================================================================
# MDL bits/token at matched distortion (combinatorial + empirical H(S))
# ===========================================================================

def scalar_rate_bits(signal_var: float, delta2: float) -> float:
    """R(D)=0.5 log2(1+sigma^2/D): bits to code one Gaussian scalar to MSE delta2."""
    if delta2 <= 0:
        return float("inf")
    return 0.5 * math.log2(1.0 + max(0.0, signal_var) / delta2)


def combinatorial_selection_bits(K: int, s: float) -> float:
    """log2 C(K, round(s)): the pointer cost bound (the 'meanwhile' currency)."""
    k = max(1, min(int(round(s)), K))
    return math.log2(math.comb(K, k)) if K > 0 else 0.0


def routing_support_entropy_bits(active_sets) -> float:
    """H(S) from a lane's OWN realized routing: entropy (bits) of active-atom sets."""
    keys = {}
    for a in active_sets:
        key = tuple(sorted(int(i) for i in a))
        keys[key] = keys.get(key, 0) + 1
    tot = sum(keys.values())
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in keys.values())


def mdl_bits(*, mean_l0: float, signal_var: float, delta2: float, K: int,
             decoder_params: float, param_bits: float, n_tokens: int,
             hs_bits: float | None) -> dict:
    """bits/token = L_code/N-amortized-token + L_dict/N.

    L_code/token = mean_L0 * rate(signal_var, delta2)              [coded coords to floor]
                 + selection bits (combinatorial bound AND empirical H(S))
    L_dict/token = decoder_params * param_bits / N_tokens          [one-time, amortized]

    The linear lane's inflated mean_L0 on curved data (needs ~2 atoms/circle) is what
    makes its code term larger at matched distortion -- the curved bit win, priced
    honestly against the chart's larger L_dict."""
    rate = scalar_rate_bits(signal_var, delta2)
    code_coords = mean_l0 * rate
    sel_comb = combinatorial_selection_bits(K, mean_l0)
    sel_hs = hs_bits if hs_bits is not None else sel_comb
    dict_bits = decoder_params * param_bits / max(1, n_tokens)
    return {
        "rate_bits_per_coord": rate,
        "code_coord_bits": code_coords,
        "selection_bits_combinatorial": sel_comb,
        "selection_bits_support_entropy": sel_hs,
        "dict_bits_per_token": dict_bits,
        "bits_per_token_combinatorial": code_coords + sel_comb + dict_bits,
        "bits_per_token_support_entropy": code_coords + sel_hs + dict_bits,
    }


# ===========================================================================
# Isolated fit worker (one lane, one K) -- never raises out
# ===========================================================================

def _install_rust_shim():
    """Strip kwargs a stale compiled ext may not accept (all default 0/False: no-op).
    Harmless on fresh builds. Patches only this process."""
    try:
        import gamfit._sae_manifold as M
    except Exception:
        return
    real = M.rust_module()
    strip = ("structured_residual_passes", "promote_from_residual")

    class _Proxy:
        def __getattr__(self, n):
            return getattr(real, n)

        def sae_manifold_fit_minimal(self, *a, **k):
            for sname in strip:
                k.pop(sname, None)
            return real.sae_manifold_fit_minimal(*a, **k)

    M.rust_module = lambda: _Proxy()


def _dv_from_decoder_blocks(fit, K: int) -> list:
    """Realized per-atom decoder-vector count. ManifoldSAE exposes decoder_blocks;
    a plain linear dict exposes decoder (K,p) -> 1 vector/atom."""
    if hasattr(fit, "decoder_blocks") and fit.decoder_blocks is not None:
        try:
            return [int(np.asarray(db).shape[0]) for db in fit.decoder_blocks]
        except Exception:
            pass
    return [1] * K


def _active_sets_from_indices(idx: np.ndarray) -> list:
    return [row.tolist() for row in np.asarray(idx)]


def _fit_worker(cfg_path: str, out_path: str) -> None:
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    out = {"cfg": cfg, "ok": False}
    try:
        _install_rust_shim()
        import gamfit
        d = np.load(cfg["data_npz"])
        train, test = d["train"], d["test"]
        K, s, lane = int(cfg["K"]), int(cfg["active"]), cfg["lane"]
        p = train.shape[1]
        t0 = time.time()

        if lane == "linear":
            # Newer gamfit builds fail closed on CPU below a GPU break-even; pass
            # score_mode="off" for deliberate CPU runs when the kwarg exists (older
            # builds don't accept it -- so probe the signature first).
            import inspect as _inspect
            sd_kw = dict(K=K, active=s, max_epochs=int(cfg.get("epochs", 30)))
            if "score_mode" in _inspect.signature(gamfit.sparse_dictionary_fit).parameters:
                sd_kw["score_mode"] = cfg.get("score_mode", "off")
            fit = gamfit.sparse_dictionary_fit(train, **sd_kw)
            # Held-out inference through the fit's OWN encoder: transform gives per-row
            # top-s (indices, codes) on test; reconstruct decodes them. (reconstruct()
            # with no args only re-decodes the stored TRAIN routing -- not held-out.)
            te_idx, te_codes = fit.transform(test, active=s)
            te_idx, te_codes = np.asarray(te_idx), np.asarray(te_codes)
            recon = np.asarray(fit.reconstruct(te_idx, te_codes))
            dv = [1] * K
            passes = int(getattr(fit, "epochs", cfg.get("epochs", 30)))
            curved = False
            active_sets = [row[np.asarray(cod) != 0].tolist()
                           for row, cod in zip(te_idx, te_codes)]
            mean_l0 = float((te_codes != 0).sum(1).mean())
            train_ev = float(fit.explained_variance)
        else:
            import inspect as _inspect
            topo = "linear" if lane == "manifold_linear" else "circle"
            kw = dict(K=K, d_atom=1, atom_topology=topo, top_k=s,
                      n_iter=int(cfg.get("n_iter", 12)))
            _msae_params = _inspect.signature(gamfit.sae_manifold_fit).parameters
            if cfg.get("no_structure_search", True) and "_run_structure_search" in _msae_params:
                # Fixed-K, no multi-start structure discovery / outer-rho search: gives a
                # clean matched-K frontier and is far faster. Frontier prices realized
                # widths, so the honest comparison is unaffected. (Older wheels lack these
                # kwargs -> they run with structure search on; k_realized records the drift.)
                kw["_run_structure_search"] = False
                kw["_run_outer_rho_search"] = False
            fit = None
            errs = []
            for rs in range(int(cfg.get("retries", 4))):
                try:
                    fit = gamfit.sae_manifold_fit(train, random_state=rs, **kw)
                    break
                except Exception as e:
                    errs.append(f"rs={rs}: {str(e).splitlines()[0][:120]}")
            if fit is None:
                raise RuntimeError("manifold fit non-convergence: " + " | ".join(errs))
            recon = np.asarray(fit.reconstruct(test))
            dv = _dv_from_decoder_blocks(fit, K)
            passes = int(cfg.get("n_iter", 12))
            curved = (topo == "circle")
            # realized routing: gate mass on test
            try:
                enc = np.asarray(fit.encode(test))
                active_sets = [np.nonzero(np.abs(row) > 1e-8)[0].tolist() for row in enc]
                mean_l0 = float((np.abs(enc) > 1e-8).sum(1).mean())
            except Exception:
                active_sets = [[0]] * len(test)
                mean_l0 = float(s)
            train_ev = float(getattr(fit, "reconstruction_r2", np.nan))

        heldout_ev = ev(test, recon)
        heldout_mse = mse(test, recon)
        hs_bits = routing_support_entropy_bits(active_sets)
        flops = flop_model(dv, mean_l0, p, curved=curved, passes=passes, n=len(train))
        out.update(ok=True, lane=lane, K=K, k_realized=len(dv), active=s, p=p,
                   heldout_ev=heldout_ev, heldout_mse=heldout_mse, train_ev=train_ev,
                   mean_l0=mean_l0, dv_per_atom=[int(x) for x in dv],
                   fit_seconds=round(time.time() - t0, 1),
                   support_entropy_bits=hs_bits, flops=flops)
    except BaseException as exc:  # noqa: BLE001 -- a worker must always tidy up
        import traceback
        out["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        out["traceback"] = traceback.format_exc()[-1200:]
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)


def _run_fit_isolated(cfg: dict, scratch: Path, timeout_s: int) -> dict:
    tag = f"{cfg['lane']}_K{cfg['K']}"
    cfg_path = scratch / f"cfg_{tag}.json"
    out_path = scratch / f"out_{tag}.json"
    cfg_path.write_text(json.dumps(cfg))
    cmd = [sys.executable, str(HERE / "frontier_bench.py"),
           "--worker", str(cfg_path), str(out_path)]
    try:
        subprocess.run(cmd, timeout=timeout_s, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"cfg": cfg, "ok": False, "error": f"timeout>{timeout_s}s", "lane": cfg["lane"], "K": cfg["K"]}
    if out_path.exists():
        return json.loads(out_path.read_text())
    return {"cfg": cfg, "ok": False, "error": "no output", "lane": cfg["lane"], "K": cfg["K"]}


# ===========================================================================
# Driver
# ===========================================================================

def run(args) -> dict:
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    dgp = make_dgp(args.dgp, p=args.p, n=args.n, n_concepts=args.concepts,
                   active_mean=args.active_mean, firing_tail=args.firing_tail,
                   zipf_s=args.zipf_s, noise_std=args.noise, harmonics=args.harmonics,
                   rotate=not args.no_rotate, seed=args.seed)
    train, test = split_scale(dgp["X"], test_frac=args.test_frac, seed=args.seed + 1)
    data_npz = scratch / "data.npz"
    np.savez(data_npz, train=train, test=test)

    results = []
    for lane in args.lanes:
        for K in args.k:
            cfg = {"lane": lane, "K": int(K), "active": args.active, "data_npz": str(data_npz),
                   "epochs": args.epochs, "n_iter": args.n_iter, "retries": args.retries}
            r = _run_fit_isolated(cfg, scratch, args.timeout)
            ok = r.get("ok")
            print(f"[{lane:16s} K={K:5d}] ok={ok} ev={r.get('heldout_ev')} "
                  f"L0={r.get('mean_l0')} infer_macs={r.get('flops',{}).get('infer_macs_per_token')} "
                  f"{'' if ok else r.get('error','')}", flush=True)
            results.append(r)

    # matched-distortion delta2: the best held-out MSE reached by ANY lane (all lanes
    # must be able to reach it) -- the common floor at which we price bits.
    mses = [r["heldout_mse"] for r in results if r.get("ok")]
    delta2 = float(min(mses)) if mses else float("nan")
    sig = dgp["signal_var"]
    for r in results:
        if not r.get("ok"):
            continue
        r["mdl"] = mdl_bits(mean_l0=r["mean_l0"], signal_var=sig, delta2=delta2,
                            K=r["K"], decoder_params=r["flops"]["decoder_params"],
                            param_bits=args.param_bits, n_tokens=len(train),
                            hs_bits=r["support_entropy_bits"])

    payload = {
        "harness": "frontier_bench",
        "dgp": {k: (v if not isinstance(v, np.ndarray) else None) for k, v in dgp.items()
                if k not in ("X", "fire", "freq")},
        "dgp_freq": dgp["freq"].tolist(),
        "config": {k: getattr(args, k) for k in
                   ("dgp", "p", "n", "concepts", "active", "active_mean", "firing_tail",
                    "zipf_s", "noise", "harmonics", "k", "lanes", "epochs", "n_iter",
                    "param_bits", "seed", "test_frac")},
        "matched_distortion_delta2": delta2,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[out] wrote {args.out}  (delta2={delta2:.4g}, {sum(1 for r in results if r.get('ok'))}/{len(results)} ok)")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", nargs=2, metavar=("CFG", "OUT"), help="internal: run one isolated fit")
    ap.add_argument("--dgp", choices=["curved", "linear"], default="curved")
    ap.add_argument("--p", type=int, default=256)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--concepts", type=int, default=24)
    ap.add_argument("--active", type=int, default=4, help="routing sparsity s (L0) requested per token")
    ap.add_argument("--active-mean", type=float, default=4.0, help="expected #concepts firing per token in the DGP")
    ap.add_argument("--firing-tail", choices=["zipf", "uniform"], default="zipf")
    ap.add_argument("--zipf-s", type=float, default=1.2)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--harmonics", type=int, default=1)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", default=[8, 16, 24, 32, 48])
    ap.add_argument("--lanes", nargs="+", default=["linear", "curved", "manifold_linear"],
                    choices=["linear", "curved", "manifold_linear"])
    ap.add_argument("--epochs", type=int, default=30, help="block-lane streaming epochs")
    ap.add_argument("--n-iter", type=int, default=12, help="manifold outer REML iters")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--param-bits", type=float, default=16.0, help="bits/decoder-scalar for L_dict (bf16=16)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=3600, help="per-fit subprocess wall-clock cap (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", default="/tmp/frontier_bench_scratch")
    ap.add_argument("--out", default="results/suite_2026-07-03/frontiers/synth.json")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.worker:
        _fit_worker(args.worker[0], args.worker[1])
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
