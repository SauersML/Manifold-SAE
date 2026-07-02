"""#1784 whitened-fit convergence probe at full residual-stream width.

Question: does the WHITENED (isometry-gauged) sae_manifold fit CONVERGE on
FULL-WIDTH (p=4096) real-shaped activations at K_curved ~= 32?  #1784 is the
large-K non-convergence class: the IBP-MAP ordered stick-breaking prior mean
pi_k = (alpha/(alpha+1))^(k+1) decays geometrically in the atom index, so the
historical fixed alpha=1 -> (0.5)^(k+1) masks every atom past ~3.  Late atoms
then carry ~0 gate mass, leaving the per-row joint Hessian RANK-DEFICIENT, and
the outer REML/Laplace criterion refuses to rank an off-optimum value ->
RemlConvergenceError.  The installed fix defaults the concentration to the
K-aware alpha = 1/(exp(1/K)-1) ~= K-1/2 so pi_{K-1} ~= 1/e (the prior SPANS the
whole dictionary, every atom stays alive, the joint solve stays well-posed).

This probe FALSIFIES-OR-CONFIRMS that fix at full width by:
  (1) using REAL OLMo-3-7B-Instruct residual-stream activations at full width
      p=4096 (a narrow mid-depth layer slab; genuinely anisotropic / heavy-
      tailed, so the whitening / isometry gauge actually bites) -- falls back to
      real-SHAPED synthetic ONLY if the harvest is missing (provenance is
      recorded in the JSON so the reader always knows which was used);
  (2) sweeping K_curved in {8,16,32,48} with the isometry gauge ON
      (isometry_weight=1.0, the whitened-fit default);
  (3) running an ADVERSARIAL arm at each K that FORCES the old broken
      ibp_alpha=1.0 to reproduce the non-convergence and pin the mechanism.

CLASSIFICATION (the load-bearing hardening).  A caught exception is NOT
automatically "#1784 non-convergence": on a memory-starved box the SAE term's
streaming_plan admission guard raises
  "SaeManifoldTerm::streaming_plan: predicted working set N bytes exceeds
   budget M bytes"
WRAPPED INSIDE a "REML smoothing optimization failed to converge" string -- the
fit never even starts.  Reporting that as a #1784 failure would be a false
negative on the fix.  So each arm is classified into:
  * CONVERGED               -- finite fit returned
  * NONCONVERGENCE_1784     -- REML/inner-solve convergence failure that is NOT
                               the memory-budget guard (the true #1784 signal)
  * REFUSED_MEMORY_BUDGET   -- the streaming_plan admission guard fired
                               (INFRASTRUCTURE: this box too small, not #1784)
  * OOM_KILLED              -- the worker process was killed (SIGKILL/jetsam)
                               before it could even report (also infrastructure)
  * OTHER_ERROR            -- anything else (surfaced loudly; likely a bug)

ISOLATION.  This box (8 GiB, shared with a churning fleet) OOM-kills full-width
fits, and the kill takes down the whole process.  So each (K, arm) fit runs in
its OWN worker subprocess; the parent records a SIGKILL as OOM_KILLED for that
one arm and CONTINUES.  Results are written incrementally (K ascending) so the
scientifically load-bearing K=32 arm lands before the riskier K=48.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# ---- MEMORY/ARCHITECTURE guards (INFRASTRUCTURE, NOT the #1784 prior fix).
# Both surface WRAPPED in "REML ... failed to converge" as a RemlConvergenceError,
# so they must be detected BEFORE the convergence markers or they masquerade as a
# #1784 non-convergence -> false negative on the fix.
#   (a) streaming_plan admission guard  -- live-host-memory in-core budget.
#   (b) dense evidence cache guard      -- fixed ~2 GiB budget; the dense REML
#       evidence cache is O((K.M.p)^2) and refuses at large K.p, demanding the
#       "cost-only streaming route".  This is the massive-K DENSE-route ceiling
#       (issue #1026/#1405 lineage), a DIFFERENT limit from #1784's stick-breaking
#       prior-mean fix.  Keeping (K.M.p)^2 under budget is what lets the REML
#       criterion actually RUN so the #1784 convergence question can be answered.
_STREAMING_PLAN_GUARD = "streaming_plan: predicted working set"
_DENSE_CACHE_GUARD = "dense evidence cache"
_COST_ONLY_STREAMING = "cost-only streaming route is required"
# a GENUINE convergence failure surfaces one of these and is NONE of the guards.
_CONVERGENCE_MARKERS = (
    "failed to converge",
    "RemlConvergence",
    "did not converge",
    "inner solve",
    "non-PD",
    "non-pd",
    "rank-deficient",
    "rank deficient",
    "singular",
    "indefinite",
)

CACHE = Path(os.environ.get(
    "PROBE_CACHE",
    os.path.join(tempfile.gettempdir(), "whitened_probe_X.npy"),
))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
REAL_HARVEST = (
    "runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST/activations.npy"  # (760,32,4096)
)
# a narrow mid-depth slab: coherent residual-stream depth, still full width 4096.
REAL_LAYER_SLAB = (15, 16, 17)


def load_real(repo: Path, n_rows: int, width: int, seed: int = 7):
    """Real OLMo-3-7B-Instruct residual activations at width `width`.

    Pools a NARROW mid-depth layer slab (adjacent layers ~ one coherent
    residual-stream distribution), subsamples `n_rows` with a fixed seed, and --
    when `width < 4096` -- projects onto the top-`width` PRINCIPAL COMPONENTS of
    that real slab.  The projection is a genuine reduced-width residual
    representation: it keeps the true anisotropic (power-law) spectrum and heavy
    tails, only dropping the lowest-variance directions.  Reduced width is used
    because the #1784 mechanism (ordered stick-breaking prior mass vanishing in
    the ATOM INDEX -> rank-deficient cross-row IBP joint Hessian) is essentially
    INDEPENDENT of the output width p -- it is a property of K and the prior,
    not of p -- while this 8 GiB box's jetsam killer will not let a full-width
    p=4096 fit run to completion (documented in the report).
    Returns (X, provenance) or (None, reason) if the harvest is absent.
    """
    # PROBE_HARVEST overrides with an ABSOLUTE path (e.g. node2 /dev/shm copy).
    override = os.environ.get("PROBE_HARVEST")
    path = Path(override) if override else (repo / REAL_HARVEST)
    if not path.exists():
        return None, f"harvest missing: {path}"
    A = np.load(path, mmap_mode="r")  # (prompts, layers, p_full)
    _, n_layers, p_full = A.shape
    layers = [l for l in REAL_LAYER_SLAB if l < n_layers]
    slab = np.asarray(A[:, layers, :], dtype=np.float64)  # (prompts,|layers|,p_full)
    X = np.ascontiguousarray(slab.reshape(-1, p_full))    # (prompts*|layers|, p_full)
    rng = np.random.default_rng(seed)
    if X.shape[0] > n_rows:
        idx = rng.choice(X.shape[0], size=n_rows, replace=False)
        X = np.ascontiguousarray(X[idx])
    projected = False
    if width < X.shape[1]:
        Xc = X - X.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        X = np.ascontiguousarray((U[:, :width] * S[:width]))  # top-width PC scores
        projected = True
    prov = {
        "source": "real",
        "harvest": str(path),
        "model": "OLMo-3-7B-Instruct (residual stream, last-token)",
        "layer_slab": list(layers),
        "p_full_harvest": int(p_full),
        "width_projection": ("top-%d PCA scores" % width) if projected else "none (full width)",
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
    }
    return X, prov


def make_synthetic(n: int, p: int, n_planted: int, seed: int = 7):
    """Real-SHAPED fallback: anisotropic power-law spectrum + planted circles +
    heavy-tailed (Student-t df=3) noise.  Used ONLY if the real harvest is gone.
    """
    rng = np.random.default_rng(seed)
    rank = 64
    V = rng.standard_normal((p, rank))
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    scales = 1.0 / (np.arange(1, rank + 1) ** 0.9)
    scores = rng.standard_normal((n, rank)) * scales[None, :]
    Z = scores @ V.T
    for a in range(n_planted):
        u = rng.standard_normal(p)
        v = rng.standard_normal(p)
        u /= np.linalg.norm(u)
        v -= (v @ u) * u
        v /= np.linalg.norm(v)
        c0 = 0.5 * rng.standard_normal(p)
        rows = np.arange(a, n, n_planted)
        t = rng.uniform(0.0, 2.0 * np.pi, size=rows.size)
        Z[rows] += c0[None, :] + 2.0 * (
            np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :]
        )
    Z += rng.standard_t(3.0, size=(n, p)) * 0.05
    prov = {"source": "synthetic", "n": int(n), "p": int(p), "planted": int(n_planted)}
    return np.ascontiguousarray(Z, dtype=np.float64), prov


def anisotropy(X: np.ndarray) -> float:
    s = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False)
    return float(s[0] / s[min(63, len(s) - 1)])


# --------------------------------------------------------------------------- #
# Worker: one fit, one process.  Prints exactly one JSON line to stdout.
# --------------------------------------------------------------------------- #
def gate_mass(model) -> np.ndarray:
    A = np.asarray(model.assignments, dtype=float)
    return np.abs(A).sum(axis=0)


def classify(exc: BaseException) -> tuple[str, str]:
    msg = str(exc)
    # budget/architecture guards FIRST (they are wrapped in "failed to converge").
    if _STREAMING_PLAN_GUARD in msg:
        return "REFUSED_STREAMING_PLAN", msg
    if _DENSE_CACHE_GUARD in msg or _COST_ONLY_STREAMING in msg:
        return "REFUSED_DENSE_CACHE", msg
    if any(m in msg for m in _CONVERGENCE_MARKERS) or "Convergence" in type(exc).__name__:
        return "NONCONVERGENCE_1784", msg
    return "OTHER_ERROR", msg


def worker(K: int, ibp_alpha, n_iter: int) -> dict:
    import gamfit
    from gamfit import sae_manifold_fit

    X = np.load(CACHE, mmap_mode="r")
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, p = X.shape
    rec = {
        "K": int(K),
        "n": int(n),
        "p": int(p),
        "ibp_alpha": ("K-aware-default" if ibp_alpha is None else float(ibp_alpha)),
        "isometry_weight": 1.0,
        "n_iter": int(n_iter),
        "gamfit": gamfit.__file__,
    }
    t0 = time.time()
    try:
        kwargs = dict(
            X=X, K=K, d_atom=1, atom_topology="circle",
            assignment="ibp_map", isometry_weight=1.0,
            n_iter=n_iter, random_state=0,
        )
        if ibp_alpha is not None:
            kwargs["ibp_alpha"] = float(ibp_alpha)
        model = sae_manifold_fit(**kwargs)
        fitted = np.asarray(model.fitted, dtype=float)
        gm = gate_mass(model)
        share = gm / max(gm.sum(), 1e-300)
        rec.update(
            status="CONVERGED",
            reconstruction_r2=float(getattr(model, "reconstruction_r2", np.nan)),
            penalized_loss_score=float(getattr(model, "penalized_loss_score", np.nan)),
            fitted_has_naninf=bool(not np.all(np.isfinite(fitted))),
            gate_mass_min_share=float(share.min()),
            dead_atoms=int((share < (0.1 / K)).sum()),
            n_atoms_returned=int(len(model.atoms)),
        )
    except BaseException as exc:  # noqa: BLE001 -- classify, never blanket-blame #1784
        status, msg = classify(exc)
        rec.update(
            status=status,
            error_type=type(exc).__name__,
            error_msg=msg[:600],
            traceback_tail="".join(traceback.format_exc().splitlines(keepends=True)[-6:]),
        )
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _run_arm_once(K: int, ibp_alpha, n_iter: int, timeout_s: int) -> dict:
    """Run one fit in an isolated child; a SIGKILL becomes OOM_KILLED."""
    alpha_tok = "none" if ibp_alpha is None else repr(float(ibp_alpha))
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           str(K), alpha_tok, str(n_iter)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"K": int(K), "ibp_alpha": alpha_tok, "status": "TIMEOUT",
                "wall_s": round(time.time() - t0, 2), "timeout_s": timeout_s}
    # The worker prints its JSON on the LAST non-empty stdout line.
    line = ""
    for ln in proc.stdout.splitlines():
        if ln.strip().startswith("{"):
            line = ln
    if proc.returncode != 0 and not line:
        # killed (SIGKILL=-9 => returncode -9) or crashed before printing.
        killed = proc.returncode < 0
        return {
            "K": int(K), "ibp_alpha": alpha_tok,
            "status": "OOM_KILLED" if killed else "OTHER_ERROR",
            "returncode": proc.returncode,
            "stderr_tail": "".join(proc.stderr.splitlines(keepends=True)[-6:]),
            "wall_s": round(time.time() - t0, 2),
        }
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        rec = {"K": int(K), "ibp_alpha": alpha_tok, "status": "OTHER_ERROR",
               "stdout_tail": proc.stdout[-600:], "stderr_tail": proc.stderr[-600:],
               "wall_s": round(time.time() - t0, 2)}
    return rec


# Statuses that are INFRASTRUCTURE (fleet jetsam storm / OS kill / a memory
# budget that fluctuates with live free RAM), NOT science.  All are retried: the
# budget guards here are driven by the box's fluctuating available memory (the
# in-core / dense-cache budget is min(2 GiB, live-derived)), so the SAME shape
# admits in a calm window and refuses under fleet pressure -- a retry that lands
# in a calm window yields the real convergence verdict.  A genuine
# NONCONVERGENCE_1784 or OTHER_ERROR is a science outcome and is NOT retried.
_RETRYABLE = {"OOM_KILLED", "TIMEOUT", "REFUSED_DENSE_CACHE", "REFUSED_STREAMING_PLAN"}


def wait_for_calm(max_load: float, max_wait_s: float = 120.0) -> float:
    """Block until the 1-min load average drops below `max_load` (this box's
    jetsam kills correlate with the fleet's rustc build-storm load spikes, not
    average free RAM), or until `max_wait_s` elapses.  Returns the load it
    proceeded at.  A no-op when `max_load <= 0`.
    """
    if max_load <= 0:
        return -1.0
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        load1 = os.getloadavg()[0]
        if load1 < max_load:
            return load1
        time.sleep(4.0)
    return os.getloadavg()[0]


def run_arm(K: int, ibp_alpha, n_iter: int, timeout_s: int, retries: int,
            max_load: float) -> dict:
    """Run an arm, retrying only INFRASTRUCTURE kills (not science outcomes).
    Each attempt is gated on the box being calm (load below `max_load`)."""
    attempts = []
    for attempt in range(1, retries + 2):
        load = wait_for_calm(max_load)
        rec = _run_arm_once(K, ibp_alpha, n_iter, timeout_s)
        attempts.append({"attempt": attempt, "status": rec.get("status"),
                         "wall_s": rec.get("wall_s"), "load1_at_start": round(load, 1)})
        if rec.get("status") not in _RETRYABLE:
            rec["attempts_log"] = attempts
            return rec
        print(f"    [retry] attempt {attempt} @load={load:.1f} -> {rec.get('status')} "
              f"({rec.get('wall_s')}s); retrying...", flush=True)
        time.sleep(3.0)  # let the jetsam storm pass
    rec["attempts_log"] = attempts
    rec["exhausted_retries"] = True
    return rec


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    repo = out_dir.parent
    N_ROWS = int(os.environ.get("PROBE_N", "600"))
    WIDTH = int(os.environ.get("PROBE_P", "4096"))
    N_ITER = int(os.environ.get("PROBE_ITER", "30"))
    K_SWEEP = [int(k) for k in os.environ.get("PROBE_KS", "8,16,32,48").split(",")]
    TIMEOUT_S = int(os.environ.get("PROBE_TIMEOUT", "900"))
    RETRIES = int(os.environ.get("PROBE_RETRIES", "6"))
    MAX_LOAD = float(os.environ.get("PROBE_MAX_LOAD", "0"))  # 0 = no load gating
    OUT_NAME = os.environ.get("PROBE_OUT", "whitened_convergence_results.json")
    # PROBE_EXPLICIT_KAWARE=1: pass the K-aware alpha EXPLICITLY for the fix arm
    # instead of ibp_alpha=None.  Needed when the gamfit binary predates the #1784
    # K-aware DEFAULT (e.g. node2's #1782-era build): the fix is purely the alpha
    # VALUE (α = max(1, 1/expm1(1/K)) ≈ K−½) and the Rust prior π_k=(α/(α+1))^{k+1}
    # is unchanged, so supplying that value explicitly reproduces the fix faithfully.
    EXPLICIT_KAWARE = os.environ.get("PROBE_EXPLICIT_KAWARE", "0") == "1"

    # ---- prepare data ONCE, cache for the workers.
    X, prov = load_real(repo, N_ROWS, WIDTH)
    if X is None:
        print(f"[data] real harvest unavailable ({prov}); using synthetic", flush=True)
        X, prov = make_synthetic(N_ROWS, WIDTH, n_planted=24)
    prov["anisotropy_top_over_64th_sv"] = round(anisotropy(X), 1)
    np.save(CACHE, X)
    print(f"[data] provenance: {json.dumps(prov)}", flush=True)
    print(f"[data] cached X {X.shape} -> {CACHE}  (n_iter={N_ITER}, retries={RETRIES})", flush=True)

    results = []
    results_path = out_dir / OUT_NAME
    for K in K_SWEEP:
        fix_alpha = max(1.0, 1.0 / math.expm1(1.0 / K)) if EXPLICIT_KAWARE else None
        for ibp_alpha, tag in ((fix_alpha, "kaware_fix"), (1.0, "alpha1_adversarial")):
            print(f"\n=== K={K}  arm={tag}  (ibp_alpha={ibp_alpha}) ===", flush=True)
            rec = run_arm(K, ibp_alpha, N_ITER, TIMEOUT_S, RETRIES, MAX_LOAD)
            rec["tag"] = tag
            results.append(rec)
            print(json.dumps(rec, indent=2), flush=True)
            results_path.write_text(json.dumps(
                {"provenance": prov, "n_iter": N_ITER, "results": results}, indent=2))

    print("\n\n================ SUMMARY ================", flush=True)
    print(f"data: {prov.get('source')}  n={prov.get('n')}  p={prov.get('p')}  "
          f"aniso={prov.get('anisotropy_top_over_64th_sv')}", flush=True)
    hdr = f"{'arm':<20}{'K':>4}{'status':>24}{'R2':>9}{'dead':>6}{'wall_s':>9}"
    print(hdr)
    for r in results:
        print(f"{r.get('tag',''):<20}{r.get('K',-1):>4}{r.get('status',''):>24}"
              f"{r.get('reconstruction_r2', float('nan')):>9.3f}"
              f"{r.get('dead_atoms', -1):>6}{r.get('wall_s', float('nan')):>9.1f}")
    print("========================================", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        K = int(sys.argv[2])
        ibp_alpha = None if sys.argv[3] == "none" else float(sys.argv[3])
        n_iter = int(sys.argv[4])
        print(json.dumps(worker(K, ibp_alpha, n_iter)), flush=True)
    else:
        main()
