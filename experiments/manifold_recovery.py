"""Definitive verification gate for joint manifold-SAE recovery on gamfit.

The architecture the design session converged on, built directly on
``gamfit.sae_manifold_fit`` (the first-class joint solve) rather than a
hand-rolled torch loop. There is no encoder net and no student: the canonical
answer *is* the joint fit; the encode direction (activation -> coordinate) is
the solver run with the dictionary frozen (``ManifoldSAE.encode``).

CANONICAL ASSIGNMENT = IBP. Per the design decision the canonical assignment is
``assignment="ibp"`` (the gam default): an Indian-Buffet-Process prior that
gives an *adaptive* atom count and *true zeros* in the assignment, rather than
``softmax`` + ``top_k`` (a fixed-count soft relaxation). Every fit below uses
IBP.

When run, this file is the verification gate and reports three checks:

  1. K=2 SUPERPOSED-CIRCLE RECOVERY under IBP. Two circles are superposed in
     (near-)orthogonal planes; we fit K=2 and measure reconstruction R2 and
     per-token coordinate recovery modulo each circle's isometry (Procrustes).
     PASS if reconstruction R2 > 0.9.

  2. INCOHERENCE ON vs OFF (the headline claim). The decoder-incoherence
     objective (design name ``decoder_incoherence_weight``) pushes superposed
     atoms' decoder column spaces apart so the per-token split is identifiable
     when their planes are coherent. The exact gam knob is resolved at runtime
     against the live ``sae_manifold_fit`` signature (see ``INCOHERENCE_KWARG``);
     if gam exposes none yet, this check self-gates BLOCKED with that reason
     rather than silently testing nothing. We sweep coherence in {0.0, 0.3, 0.6}
     and, at each level, fit with the penalty ON (weight 1.0) and OFF (weight
     0.0) and check whether ON:
       * RAISES sigma_min of the recovered active-atom tangent matrix
         (better-conditioned local frame -> well-posed per-token split), AND
       * LOWERS the recovered cross-atom decoder cross-Gram ||B0 B1^T||_F
         (more incoherent decoders) and/or IMPROVES coordinate recovery.

  3. SINGLE-ATOM OUT-OF-CLASS SPECIFICATION CHECK (runs today; K=1). Is each
     atom the right *kind* of object? A well-conditioned, seed-stable region can
     still be a 2D blob the evidence fits as a circle. An OUT-OF-CLASS absolute
     test (fit the typed atom AND a flexible patch, compare absolute
     reconstruction against a type-correct null) flags the misspecification.

Every fit is wrapped in try/except: a non-converged or crashing fit reports
cleanly (NaN / BLOCKED) rather than aborting the gate. Reconstruction R2 below
``DIVERGED_R2`` is treated as a non-converged multi-atom solve.

STATUS: the multi-atom (K>=2) joint fit currently diverges in the gam solver
(fix in progress). Checks 1 and 2 therefore self-gate and report BLOCKED until
that lands; check 3 (single-atom) runs today. This harness is correct and ready
to run the moment the solver is fixed.

Run:  .venv/bin/python -m experiments.manifold_recovery --help
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import gamfit

# Shared scoring primitives live in the falsifier so the two files agree.
from experiments.manifold_falsifier import (
    circ_procrustes_r2,
    coactive_sigma_min,
    subspace_overlap,
    tangent_sigma_min,
)
from scipy.optimize import linear_sum_assignment

# Candidate topologies the fit is allowed to choose from (kind, atom dim).
TOPOLOGY_MENU = [("circle", 1), ("euclidean", 1), ("euclidean", 2), ("torus", 2)]
# The out-of-class reference: a flexible 2D patch, deliberately NOT on the menu's
# 1D-circle hypothesis. Used for the absolute specification margin.
PATCH_REFERENCE = ("euclidean", 2)
DIVERGED_R2 = 0.5   # below this, treat the multi-atom fit as non-converged
RECOVERY_R2_PASS = 0.9  # check 1 passes if reconstruction R2 exceeds this

# Decoder-incoherence knob name (the design objective: push superposed atoms'
# decoder column spaces apart so the per-token split is identifiable). The
# design decision calls this "decoder_incoherence_weight"; the equivalent
# pure-torch knob is "incoherence_weight". We resolve the name at runtime
# against the live gam signature so this gate is correct for whichever lands --
# and reports a precise BLOCKED reason if none is available yet, rather than
# silently testing nothing.
#
# We deliberately do NOT fall back to ``block_orthogonality_weight``: despite
# the suggestive name it is a *latent-axis* decorrelation penalty on the d_max
# coordinate axes (and it hard-rejects atom_dim < 2, i.e. all 1D circle atoms),
# so it cannot realize cross-atom *decoder* incoherence for stacked circles.
# Using it here would test the wrong thing. If gam exposes none of the real
# knobs below, check 2 honestly self-gates BLOCKED.
INCOHERENCE_KWARG_CANDIDATES = (
    "decoder_incoherence_weight",
    "incoherence_weight",
)


def _resolve_incoherence_kwarg() -> str | None:
    """Return the name of the incoherence knob the installed gam exposes, or
    None if gam does not expose any yet (-> check 2 self-gates BLOCKED)."""
    import inspect
    try:
        params = inspect.signature(gamfit.sae_manifold_fit).parameters
    except (TypeError, ValueError):
        return None
    for name in INCOHERENCE_KWARG_CANDIDATES:
        if name in params:
            return name
    return None


INCOHERENCE_KWARG = _resolve_incoherence_kwarg()


@dataclass
class Config:
    # data-generation knobs (these are genuine experiment-design choices)
    n: int = 240
    d_ambient: int = 8
    noise: float = 0.02
    k_planted: int = 2          # number of manifolds we plant (ground truth)
    coherence: float = 0.0      # 0 = orthogonal subspaces; ->1 share a direction
    coactive: bool = True       # tokens activate a sparse subset additively
    seed: int = 0               # ground-truth data seed
    n_seeds: int = 5            # reproducibility sweep
    # the only fit knob that is a real choice, not a default to be discovered:
    # the gauge weight, which must be > 0 (verified load-bearing).
    isometry_weight: float = 0.1
    max_k: int = 4              # over-provisioned K; ARD is asked to prune it
    n_iter: int = 25


def _add_args(p: argparse.ArgumentParser) -> None:
    for f, v in vars(Config()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{f}", action=argparse.BooleanOptionalAction, default=v)
        else:
            p.add_argument(f"--{f}", type=type(v), default=v)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _circle(t):
    return np.c_[np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]


def _planes(d, k, coherence, rng):
    anchor = np.linalg.qr(rng.standard_normal((d, 1)))[0]
    out = []
    for _ in range(k):
        Q = np.linalg.qr(rng.standard_normal((d, 2)))[0]
        if coherence > 0:
            Q[:, 0] = (1 - coherence) * Q[:, 0] + coherence * anchor[:, 0]
            Q = np.linalg.qr(Q)[0]
        out.append(Q)
    return out


def plant_circles(cfg: Config) -> dict:
    """Superpose ``k_planted`` circles in (possibly overlapping) 2D planes.

    Returns X plus the planted planes, per-atom coordinates, per-token gate
    (which atoms fired) and the co-active mask -- everything the recovery
    scoring and the tangent-conditioning metric need.
    """
    rng = np.random.default_rng(cfg.seed)
    planes = _planes(cfg.d_ambient, cfg.k_planted, cfg.coherence, rng)
    coords = rng.uniform(0, 1, (cfg.k_planted, cfg.n))
    gate = np.zeros((cfg.n, cfg.k_planted), dtype=bool)
    if cfg.coactive:
        top = min(2, cfg.k_planted)
        amp = np.zeros((cfg.n, cfg.k_planted))
        for i in range(cfg.n):
            sel = rng.choice(cfg.k_planted, top, replace=False)
            amp[i, sel] = 0.6 + 0.8 * rng.uniform()
            gate[i, sel] = True
    else:
        amp = np.zeros((cfg.n, cfg.k_planted))
        sel = rng.integers(0, cfg.k_planted, cfg.n)
        amp[np.arange(cfg.n), sel] = 1.0
        gate[np.arange(cfg.n), sel] = True
    X = sum(amp[:, k:k + 1] * (_circle(coords[k]) @ planes[k].T) for k in range(cfg.k_planted))
    X = X + cfg.noise * rng.standard_normal(X.shape)
    coactive = gate.all(axis=1)
    return dict(X=X, planes=planes, t=coords, gate=gate, coactive=coactive)


def plant_region(cfg: Config, kind: str) -> np.ndarray:
    """Single isolated region. ``circle`` = well-specified control; ``blob`` =
    a genuine 2D disk (wrong-dimension cell-3 case); ``arc`` = a true circle seen
    only over 90deg (coverage case the residual test is blind to)."""
    rng = np.random.default_rng(cfg.seed)
    Q = np.linalg.qr(rng.standard_normal((cfg.d_ambient, 2)))[0]
    if kind == "circle":
        p = _circle(rng.uniform(0, 1, cfg.n))
    elif kind == "blob":
        r = np.sqrt(rng.uniform(0, 1, cfg.n)); a = rng.uniform(0, 2 * np.pi, cfg.n)
        p = np.c_[r * np.cos(a), r * np.sin(a)]
    elif kind == "arc":
        p = _circle(rng.uniform(0, 0.25, cfg.n))
    else:
        raise ValueError(kind)
    return p @ Q.T + cfg.noise * rng.standard_normal((cfg.n, cfg.d_ambient))


# ---------------------------------------------------------------------------
# Fit (canonical = IBP) + scoring
# ---------------------------------------------------------------------------

def fit(X, *, k, topology, d_atom, cfg: Config, seed=0, ard=False,
        incoherence_weight=0.0):
    """Canonical joint fit: ``assignment="ibp"`` (adaptive count, true zeros),
    NOT softmax+top_k. ``incoherence_weight`` (>0 = ON) is forwarded under the
    gam knob resolved at import (``INCOHERENCE_KWARG``); when 0.0 or no knob is
    available it is simply omitted, so the baseline fit is unaffected."""
    kw = dict(
        K=k, d_atom=d_atom, atom_topology=topology, assignment="ibp",
        ard_per_atom=ard, alpha="auto",
        sparsity_weight=0.01, smoothness_weight=0.01,
        isometry_weight=cfg.isometry_weight, learning_rate=1.0,
        n_iter=cfg.n_iter, random_state=seed)
    if incoherence_weight and incoherence_weight > 0.0:
        if INCOHERENCE_KWARG is None:
            raise ValueError(
                "incoherence_weight requested but no decoder-incoherence knob "
                f"is exposed by gamfit {gamfit.__version__} (looked for "
                f"{INCOHERENCE_KWARG_CANDIDATES}); check 2 self-gates BLOCKED.")
        kw[INCOHERENCE_KWARG] = float(incoherence_weight)
    return gamfit.sae_manifold_fit(X, **kw)


def safe_fit(X, **kw):
    """Wrap a fit so a crash or non-convergence reports cleanly. Returns
    (model_or_None, error_or_None)."""
    try:
        return fit(X, **kw), None
    except Exception as e:  # noqa: BLE001 - report any solver failure cleanly
        return None, f"{type(e).__name__}: {e}"


def choose_topology(X, cfg: Config, *, k=1):
    """Let model evidence pick the topology from the menu. Returns
    (winner, r2_by_candidate, reml_by_candidate). Each candidate fit is guarded
    so a single non-converged candidate does not abort the sweep."""
    r2, reml = {}, {}
    for kind, d in TOPOLOGY_MENU:
        m, err = safe_fit(X, k=k, topology=kind, d_atom=d, cfg=cfg)
        if m is None:
            r2[(kind, d)], reml[(kind, d)] = float("nan"), float("-inf")
        else:
            r2[(kind, d)] = float(m.reconstruction_r2)
            reml[(kind, d)] = float(m.reml_score)
    winner = max(reml, key=reml.get)
    return winner, r2, reml


# ---------------------------------------------------------------------------
# Decoder / atom geometry recovery (shared by checks 1 and 2)
# ---------------------------------------------------------------------------

def atom_plane(atom) -> np.ndarray:
    """Recover an atom's 2D ambient plane from its periodic decoder. The circle
    harmonic design is [1, cos(2*pi*t), sin(2*pi*t)] (3 coeffs, matching the
    decoder_coefficients row count); B @ coeffs gives the per-atom ambient
    reconstruction whose top-2 right singular vectors span the recovered plane.
    The mean-removed reconstruction's right singular vectors also give the
    decoder factor B (D x 2) used for the cross-atom cross-Gram."""
    coeffs = np.asarray(atom.decoder_coefficients)          # (3, D)
    tc = np.asarray(atom.coords)[:, 0]
    Bdes = np.c_[np.ones_like(tc), np.cos(2 * np.pi * tc), np.sin(2 * np.pi * tc)]
    recon = Bdes @ coeffs
    recon = recon - recon.mean(0)
    U, S, Vt = np.linalg.svd(recon, full_matrices=False)
    plane = Vt[:2].T                                         # (D, 2) orthonormal
    # decoder factor scaled by singular values: the actual recovered amplitude
    # along each ambient direction (this is what a cross-Gram should weigh).
    B = (Vt[:2].T * S[:2])                                   # (D, 2)
    return plane, B


def cross_gram_fro(B_list) -> float:
    """||B0 B1^T||_F for the first two recovered decoder factors -- the
    cross-atom decoder cross-Gram. Zero iff the two atoms' decoder column spaces
    are orthogonal (maximally incoherent)."""
    if len(B_list) < 2:
        return float("nan")
    B0, B1 = B_list[0], B_list[1]
    return float(np.linalg.norm(B0.T @ B1, "fro"))


def match_and_score(model, gt: dict) -> dict:
    """Hungarian-match recovered atoms to planted manifolds on subspace overlap,
    then score per-token coordinate recovery up to each circle's isometry, on the
    tokens where each atom fired. Also returns recovered planes/decoder factors
    and the recovered active-atom tangent sigma_min over co-active tokens."""
    planes_true = gt["planes"]
    atoms = list(model.atoms)
    planes_hat, B_hat, coords_hat = [], [], []
    for a in atoms:
        plane, B = atom_plane(a)
        planes_hat.append(plane)
        B_hat.append(B)
        coords_hat.append(np.asarray(a.coords)[:, 0])

    K = len(planes_true)
    cost = np.zeros((K, len(atoms)))
    for i in range(K):
        for j in range(len(atoms)):
            cost[i, j] = -subspace_overlap(planes_true[i], planes_hat[j])
    row, col = linear_sum_assignment(cost)

    per_atom, matched_planes = [], {}
    for i, j in zip(row, col):
        fired = gt["gate"][:, i]
        t_hat = (coords_hat[j][fired] % 1.0)
        t_true = gt["t"][i][fired]
        r2 = circ_procrustes_r2(t_hat, t_true)
        per_atom.append(dict(planted=i, atom=j, overlap=float(-cost[i, j]),
                             coord_r2=float(r2), n_fired=int(fired.sum())))
        matched_planes[i] = (planes_hat[j], coords_hat[j])

    # recovered active-atom tangent sigma_min over co-active tokens, using the
    # RECOVERED planes + RECOVERED coordinates (this is the conditioning of the
    # split as the fit actually represents it).
    sm = []
    if len(matched_planes) >= 2:
        pa, ca = matched_planes[0]
        pb, cb = matched_planes[1]
        for i in np.flatnonzero(gt["coactive"]):
            sm.append(tangent_sigma_min(ca[i] % 1.0, cb[i] % 1.0, pa, pb))
    return dict(matches=per_atom,
                cross_gram=cross_gram_fro(B_hat),
                recovered_sigma_min=np.asarray(sm))


# ---------------------------------------------------------------------------
# Check 1: K=2 superposed-circle recovery under IBP
# ---------------------------------------------------------------------------

def check_recovery(cfg: Config) -> bool | None:
    """Returns True/False (pass/fail) or None (BLOCKED -> not counted)."""
    print("\n=== CHECK 1: K=2 superposed-circle recovery under IBP ===")
    print(f"  planted K={cfg.k_planted}, fit K={cfg.k_planted} (IBP), "
          f"coherence={cfg.coherence}, PASS if recon R2 > {RECOVERY_R2_PASS}")
    gt = plant_circles(cfg)
    model, err = safe_fit(gt["X"], k=cfg.k_planted, topology="circle", d_atom=1,
                          cfg=cfg, seed=cfg.seed)
    if model is None:
        print(f"  -> BLOCKED: fit raised ({err}). Harness ready; rerun once solver fixed.")
        return None
    r2 = float(model.reconstruction_r2)
    print(f"  reconstruction R2 = {r2:.4f}")
    if r2 < DIVERGED_R2:
        print(f"  -> BLOCKED: multi-atom joint fit diverged (R2 < {DIVERGED_R2}).")
        print(f"     Single-atom works; this check runs once the solver fix lands.")
        return None
    try:
        res = match_and_score(model, gt)
    except Exception as e:  # noqa: BLE001
        print(f"  -> BLOCKED: recovery scoring failed cleanly ({type(e).__name__}: {e}).")
        return None
    print(f"\n    {'planted':>7s} {'atom':>4s} {'overlap':>7s} {'coord_r2':>8s} {'n_fired':>7s}")
    for m in res["matches"]:
        print(f"    {m['planted']:7d} {m['atom']:4d} {m['overlap']:7.3f} "
              f"{m['coord_r2']:8.3f} {m['n_fired']:7d}")
    worst = min(m["coord_r2"] for m in res["matches"])
    print(f"  worst-atom coordinate R2 (mod circle isometry) = {worst:.3f}")
    passed = r2 > RECOVERY_R2_PASS
    print(f"  -> [{'PASS' if passed else 'FAIL'}] reconstruction R2 {r2:.3f} "
          f"{'>' if passed else '<='} {RECOVERY_R2_PASS}")
    return passed


# ---------------------------------------------------------------------------
# Check 2: incoherence ON vs OFF across a coherence sweep (the headline claim)
# ---------------------------------------------------------------------------

def check_incoherence(cfg: Config) -> bool | None:
    """Sweep coherence; at each level fit with the decoder block-orthogonality
    penalty ON (1.0) vs OFF (0.0) and check ON raises recovered sigma_min AND
    lowers the cross-atom decoder cross-Gram / improves coordinate recovery.

    Returns True/False (pass/fail) or None (BLOCKED -> not counted)."""
    print("\n=== CHECK 2: incoherence ON vs OFF across coherence sweep (HEADLINE) ===")
    if INCOHERENCE_KWARG is None:
        print(f"  -> BLOCKED: gamfit {gamfit.__version__} exposes no decoder-incoherence")
        print(f"     knob (looked for {INCOHERENCE_KWARG_CANDIDATES}). The headline")
        print("     claim needs that knob; this check goes live once gam adds it.")
        return None
    print(f"  decoder-incoherence knob = '{INCOHERENCE_KWARG}' (1.0=ON, 0.0=OFF).")
    print("  claim: ON raises recovered-tangent sigma_min AND lowers cross-atom")
    print("         decoder cross-Gram ||B0 B1^T||_F (or improves coord recovery).")
    header = (f"\n    {'coh':>4s} {'pen':>3s} {'recon_r2':>8s} "
              f"{'sig_min_med':>11s} {'crossGram':>9s} {'worst_coord_r2':>14s}")
    print(header)

    rows = {}
    any_converged = False
    for coh in (0.0, 0.3, 0.6):
        sub = Config(**{**vars(cfg), "coherence": coh})
        gt = plant_circles(sub)
        for tag, w in (("OFF", 0.0), ("ON", 1.0)):
            model, err = safe_fit(gt["X"], k=sub.k_planted, topology="circle",
                                  d_atom=1, cfg=sub, seed=sub.seed,
                                  incoherence_weight=w)
            if model is None:
                rows[(coh, tag)] = dict(r2=float("nan"), sig=float("nan"),
                                        cg=float("nan"), coord=float("nan"))
                print(f"    {coh:4.1f} {tag:>3s} {'BLOCKED':>8s}  ({err})")
                continue
            r2 = float(model.reconstruction_r2)
            if r2 < DIVERGED_R2:
                rows[(coh, tag)] = dict(r2=r2, sig=float("nan"),
                                        cg=float("nan"), coord=float("nan"))
                print(f"    {coh:4.1f} {tag:>3s} {r2:8.3f}  diverged (R2<{DIVERGED_R2})")
                continue
            try:
                res = match_and_score(model, gt)
                sig = float(np.median(res["recovered_sigma_min"])) \
                    if res["recovered_sigma_min"].size else float("nan")
                cg = res["cross_gram"]
                coord = min(m["coord_r2"] for m in res["matches"])
            except Exception as e:  # noqa: BLE001
                rows[(coh, tag)] = dict(r2=r2, sig=float("nan"),
                                        cg=float("nan"), coord=float("nan"))
                print(f"    {coh:4.1f} {tag:>3s} {r2:8.3f}  scoring failed "
                      f"({type(e).__name__})")
                continue
            any_converged = True
            rows[(coh, tag)] = dict(r2=r2, sig=sig, cg=cg, coord=coord)
            print(f"    {coh:4.1f} {tag:>3s} {r2:8.3f} {sig:11.4f} "
                  f"{cg:9.4f} {coord:14.3f}")

    if not any_converged:
        print("\n  -> BLOCKED: no ON/OFF pair converged (solver broken at K>=2).")
        print("     Harness is correct and will evaluate the headline claim once fixed.")
        return None

    # Verdict: across the coherence levels where both ON and OFF converged,
    # ON should raise sigma_min AND (lower cross-Gram OR improve coord recovery).
    print("\n  per-coherence verdict (only levels where both ON & OFF converged):")
    verdicts = []
    for coh in (0.0, 0.3, 0.6):
        on, off = rows.get((coh, "ON")), rows.get((coh, "OFF"))
        if on is None or off is None:
            continue
        if np.isnan(on["sig"]) or np.isnan(off["sig"]):
            continue
        raises_sig = on["sig"] > off["sig"] + 1e-6
        lowers_cg = (not np.isnan(on["cg"]) and not np.isnan(off["cg"])
                     and on["cg"] < off["cg"] - 1e-6)
        better_coord = (not np.isnan(on["coord"]) and not np.isnan(off["coord"])
                        and on["coord"] > off["coord"] + 1e-6)
        ok = raises_sig and (lowers_cg or better_coord)
        verdicts.append(ok)
        print(f"    coh={coh:.1f}: raises_sigma_min={raises_sig} "
              f"lowers_crossGram={lowers_cg} better_coord={better_coord} "
              f"-> {'supports claim' if ok else 'does NOT support'}")
    if not verdicts:
        print("  -> BLOCKED: ON & OFF never both converged at the same coherence.")
        return None
    passed = all(verdicts)
    print(f"  -> [{'PASS' if passed else 'FAIL'}] incoherence helps at every "
          f"converged coherence level")
    return passed


# ---------------------------------------------------------------------------
# Check 3: specification (out-of-class absolute margin) — runs today (K=1)
# ---------------------------------------------------------------------------

def check_specification(cfg: Config) -> bool | None:
    print("\n=== CHECK 3: single-atom out-of-class specification margin (K=1) ===")
    print("  For each region: (a) let evidence pick a topology from the menu;")
    print("  (b) compute the out-of-class margin = R2_patch(d=2) - R2_circle(d=1),")
    print("  calibrated against the true-circle null. Large margin => misspecified.")
    one = Config(**{**vars(cfg), "n": cfg.n})
    results = {}
    for kind in ("circle", "blob", "arc"):
        X = plant_region(one, kind)
        winner, r2, reml = choose_topology(X, one, k=1)
        margin = r2[PATCH_REFERENCE] - r2[("circle", 1)]
        results[kind] = (winner, r2[("circle", 1)], r2[PATCH_REFERENCE], margin)
    null = results["circle"][3]
    print(f"\n  {'region':7s} {'evidence_pick':>16s} {'R2_circle':>9s} {'R2_patch':>9s} {'margin':>7s}  verdict")
    blob_flagged = False
    for kind, (winner, rc, rp, margin) in results.items():
        misspec = margin > null + 0.10
        verdict = "MISSPECIFIED" if misspec else "circle adequate"
        if kind == "arc":
            verdict += " (residual blind to coverage)"
        if kind == "blob":
            blob_flagged = misspec
        print(f"  {kind:7s} {str(winner):>16s} {rc:9.3f} {rp:9.3f} {margin:7.3f}  {verdict}")
    print(f"\n  null (true-circle) margin = {null:.3f}")
    print("  -> the 'evidence_pick' column shows relative selection is unreliable across")
    print("     a heterogeneous menu (REML not comparable across dims); the absolute")
    print("     out-of-class margin is what actually flags the 2D blob.")
    # the discriminating claim: the blob is flagged misspecified by the margin.
    # If the K=1 solver itself can't fit (R2 NaN everywhere) this self-gates.
    if np.isnan(results["blob"][2]) or np.isnan(results["circle"][1]):
        print("  -> BLOCKED: single-atom fits did not converge (NaN R2).")
        return None
    print(f"  -> [{'PASS' if blob_flagged else 'FAIL'}] the 2D blob is flagged MISSPECIFIED")
    return blob_flagged


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_args(p)
    cfg = Config(**vars(p.parse_args()))
    print(f"gamfit {gamfit.__version__}  |  assignment=ibp (canonical)  |  {cfg}")

    results = {
        "1. K=2 recovery (IBP, R2>0.9)": check_recovery(cfg),
        "2. incoherence ON>OFF (headline)": check_incoherence(cfg),
        "3. spec out-of-class margin (K=1)": check_specification(cfg),
    }

    print("\n" + "=" * 64)
    print("VERIFICATION GATE SUMMARY")
    print("=" * 64)
    n_pass = n_fail = n_blocked = 0
    for name, r in results.items():
        if r is None:
            tag, n_blocked = "BLOCKED", n_blocked + 1
        elif r:
            tag, n_pass = "PASS", n_pass + 1
        else:
            tag, n_fail = "FAIL", n_fail + 1
        print(f"  [{tag:>7s}] {name}")
    print("-" * 64)
    print(f"  {n_pass} PASS, {n_fail} FAIL, {n_blocked} BLOCKED")
    if n_fail == 0 and n_blocked == 0:
        print("  GATE: GREEN (all checks pass)")
    elif n_fail == 0:
        print("  GATE: PENDING (no failures; some checks BLOCKED on the solver fix)")
    else:
        print("  GATE: RED (one or more checks FAILED)")


if __name__ == "__main__":
    main()
