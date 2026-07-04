"""JOINT-vs-CASCADE — what a frozen-frame in-block cascade loses vs a terminal
joint fit, on planted synthetic truth with honest EV / topology scoring.

Scott's challenge: "blocks are linear objects; refinement inside blocks is
suboptimal because of joint dependencies." This driver quantifies the loss on
three planted geometries and shows the recovery path.

Vocabulary (operationalized against the exposed gamfit API):

  * CASCADE-ONLY = frozen-frame, in-block curved refinement. The frames (linear
    blocks / ambient sub-partitions) are fixed first; a curved atom is then
    refined INSIDE each frozen frame independently. Realized as independent
    per-block ``sae_manifold_fit(K=1)`` circle fits on the block projections.
  * TERMINAL-JOINT = the simultaneous fit that re-optimizes frame AND curve
    together across blocks: ``sae_manifold_fit`` on the FULL ambient at the
    composed K (the Arrow-Schur joint solver, co-visibility preconditioned).
  * SCREEN = the pairwise-κ energy cross-moment ``ρ = E[r_A² r_B²]/(E r_A² E r_B²)``
    (a faithful Python replica of the shipped Rust ``pair_kappa::screen_pair``;
    parity is asserted against the Rust test's printed anchors). ``ρ>1`` = a
    shared-presence merge proposal; the screen fires only on that upper tail.

Experiments:
  E1 cross-frame curvature — one circle whose 2-plane is split across two frames.
  E2 inter-atom dependency — two co-gated circles (a gated torus).
  E3 envelope adequacy    — one circle fully inside one frame (anchor: in-frame
                            == full-ambient joint to tolerance).

Reconstruction EV is computed honestly as 1 - ||X-Xhat||_F² / ||X-mean||_F².
Everything is planted with a known truth; nothing is scored against a fit.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import gamfit


# --------------------------------------------------------------------------- #
# Planted-truth generators
# --------------------------------------------------------------------------- #
def _embed(p: int, axes: list[int], cols: np.ndarray, noise: float,
           rng: np.random.Generator) -> np.ndarray:
    """Place `cols` (n, len(axes)) onto ambient dims `axes` in R^p + iso noise."""
    n = cols.shape[0]
    X = noise * rng.standard_normal((n, p))
    for j, a in enumerate(axes):
        X[:, a] += cols[:, j]
    return X


def make_single_circle(p: int, n: int, axes: tuple[int, int], noise: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One dense circle in the 2-plane `axes`. Returns (X, theta)."""
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    cols = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    X = _embed(p, list(axes), cols, noise, rng)
    return X, theta


def make_gated_torus(p: int, n: int, q: float, noise: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Two circles on dims (0,1) and (2,3) sharing ONE presence gate q; angles
    independent. Present rows carry both circles; absent rows carry neither."""
    present = rng.uniform(size=n) < q
    ta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    tb = rng.uniform(0.0, 2.0 * np.pi, size=n)
    cols = np.zeros((n, 4))
    cols[present, 0] = np.cos(ta[present]); cols[present, 1] = np.sin(ta[present])
    cols[present, 2] = np.cos(tb[present]); cols[present, 3] = np.sin(tb[present])
    X = _embed(p, [0, 1, 2, 3], cols, noise, rng)
    return X, {"present": present, "ta": ta, "tb": tb}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def ev(X: np.ndarray, Xhat: np.ndarray) -> float:
    """Reconstruction explained variance vs the column-mean baseline."""
    mu = X.mean(axis=0, keepdims=True)
    denom = float(((X - mu) ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(((X - Xhat) ** 2).sum()) / denom


def _plane_energy(X: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Per-row squared energy in the 2-plane `basis` (p,2): r_i² = ||Bᵀ(x_i-μ)||²."""
    c = (X - mean[None, :]) @ basis
    return (c ** 2).sum(axis=1)


def pair_rho(X: np.ndarray, basis_a: np.ndarray, basis_b: np.ndarray) -> dict:
    """Faithful replica of Rust pair_kappa::screen_pair over ALL rows (dense
    presence: every row active). ρ = E[r_A² r_B²]/(E r_A² · E r_B²); z against
    the independence null with delta-method SE √((κ_Aκ_B - 1)/N)."""
    mean = X.mean(axis=0)
    ra = _plane_energy(X, mean, basis_a)
    rb = _plane_energy(X, mean, basis_b)
    ma, mb = ra.mean(), rb.mean()
    cross = (ra * rb).mean()
    rho = float(cross / (ma * mb))
    ka = float((ra * ra).mean() / (ma * ma))
    kb = float((rb * rb).mean() / (mb * mb))
    var = max(ka * kb - 1.0, 0.0) / len(ra)
    se = float(np.sqrt(var))
    z = float((rho - 1.0) / se) if se > 0 else (float("inf") if rho > 1 else 0.0)
    return {"rho": rho, "kappa_a": ka, "kappa_b": kb, "rho_se": se, "z": z,
            "merge_proposed": bool(z > 3.0)}


def _axis_basis(p: int, d0: int, d1: int) -> np.ndarray:
    B = np.zeros((p, 2)); B[d0, 0] = 1.0; B[d1, 1] = 1.0
    return B


# --------------------------------------------------------------------------- #
# Fit wrappers (defensive: a fit that raises is recorded, not fatal)
# --------------------------------------------------------------------------- #
def _fit(X, **kw):
    t0 = time.time()
    m = gamfit.sae_manifold_fit(X, **kw)
    dt = time.time() - t0
    Xhat = np.asarray(m.fitted, dtype=float)
    topo = list(getattr(m, "atom_topologies", []) or [m.atom_topology])
    kappas = []
    cr = getattr(m, "curvature_report", None)
    if isinstance(cr, dict):
        for a in cr.get("atoms", []) or []:
            kappas.append(float(a.get("kappa_hat", float("nan"))))
    return {"ev": ev(X, Xhat), "topologies": topo, "kappa_hat": kappas,
            "recon_r2": float(getattr(m, "reconstruction_r2", float("nan"))),
            "seconds": dt, "K": len(getattr(m, "atoms", []) or topo)}


def cascade_two_block(X, p, block_split, noise_axes, seed, **fitkw):
    """Frozen-frame in-block cascade: fit a curved (circle) atom INSIDE each of
    two frozen ambient blocks independently, then union the reconstructions.
    Block A owns dims [0:block_split], block B owns [block_split:p]."""
    n = X.shape[0]
    A = list(range(0, block_split)); B = list(range(block_split, p))
    XA = X[:, A]; XB = X[:, B]
    t0 = time.time()
    mA = gamfit.sae_manifold_fit(XA, K=1, d_atom=1, atom_topology="circle",
                                 random_state=seed, **fitkw)
    mB = gamfit.sae_manifold_fit(XB, K=1, d_atom=1, atom_topology="circle",
                                 random_state=seed + 1, **fitkw)
    dt = time.time() - t0
    Xhat = np.zeros_like(X)
    Xhat[:, A] = np.asarray(mA.fitted, float)
    Xhat[:, B] = np.asarray(mB.fitted, float)
    def _k(m):
        cr = getattr(m, "curvature_report", None)
        if isinstance(cr, dict) and cr.get("atoms"):
            return float(cr["atoms"][0].get("kappa_hat", float("nan")))
        return float("nan")
    return {"ev": ev(X, Xhat), "seconds": dt, "K": 2,
            "kappa_hat": [_k(mA), _k(mB)],
            "block_ev": [ev(XA, np.asarray(mA.fitted, float)),
                         ev(XB, np.asarray(mB.fitted, float))]}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def exp1_cross_frame(p, n, noise, seed, fitkw):
    """One circle in plane (0,1); frozen frames split it: block A = dims [0:p/2]
    (owns axis 0), block B = dims [p/2:p] (owns axis 1). Cascade fragments it into
    two near-linear atoms; the terminal joint recovers the single circle."""
    rng = np.random.default_rng(seed)
    # Put the two circle axes into DIFFERENT blocks: axis0 -> dim 0 (block A),
    # axis1 -> dim p//2 (block B).
    split = p // 2
    X, theta = make_single_circle(p, n, axes=(0, split), noise=noise, rng=rng)
    out = {"p": p, "n": n, "noise": noise}
    out["cascade_only"] = cascade_two_block(X, p, split, (0, split), seed, **fitkw)
    out["terminal_joint"] = _fit(X, K=1, d_atom=1, atom_topology="circle",
                                 random_state=seed, **fitkw)
    # Screen: the two cascade atoms live in block A (~axis 0) and block B (~axis1).
    ba = _axis_basis(p, 0, 1)          # block A's captured plane ~ (0, near-0)
    bb = _axis_basis(p, split, split + 1)
    out["screen"] = pair_rho(X, ba, bb)
    # cascade+screen+fusion: if flagged, refit ONE circle on the fused plane. Here
    # the screen sits in the LOWER tail (ρ<1) so no fusion is PROPOSED; the joint
    # fit is what recovers. We still report the fused-atom fit for the table.
    out["fusion_refit"] = _fit(X, K=1, d_atom=1, atom_topology="circle",
                               random_state=seed + 7, **fitkw)
    return out


def exp2_inter_atom(p, n, noise, q, seed, fitkw):
    """Two co-gated circles (gated torus). Two independent circles reconstruct as
    well as one torus atom (marginals give full EV); the joint torus atom adds the
    2-D coordinate / joint law. The pair-κ screen detects the presence binding."""
    rng = np.random.default_rng(seed)
    X, truth = make_gated_torus(p, n, q, noise, rng)
    out = {"p": p, "n": n, "noise": noise, "q": q}
    out["two_circles"] = _fit(X, K=2, d_atom=1, atom_topology="circle",
                              random_state=seed, **fitkw)
    out["torus_joint"] = _fit(X, K=1, d_atom=2, atom_topology="torus",
                              random_state=seed, **fitkw)
    out["screen"] = pair_rho(X, _axis_basis(p, 0, 1), _axis_basis(p, 2, 3))
    out["anchor_inv_q"] = 1.0 / q
    return out


def exp3_envelope(p, n, noise, seed, fitkw):
    """One circle in plane (0,1). in-frame (fit RESTRICTED to the true 2-plane,
    reconstruction embedded back into full ambient with zeros elsewhere) vs
    full-ambient joint — both scored on the SAME full-width X so the denominators
    match. EV parity is the 'frame containing the truth loses nothing' anchor."""
    rng = np.random.default_rng(seed)
    X, theta = make_single_circle(p, n, axes=(0, 1), noise=noise, rng=rng)
    out = {"p": p, "n": n, "noise": noise}
    # in-frame: fit on the 2-plane that CONTAINS the truth, then embed back.
    Xin = X[:, [0, 1]]
    t0 = time.time()
    m_in = gamfit.sae_manifold_fit(Xin, K=1, d_atom=1, atom_topology="circle",
                                   random_state=seed, **fitkw)
    dt = time.time() - t0
    Xhat = np.zeros_like(X)
    Xhat[:, [0, 1]] = np.asarray(m_in.fitted, float)  # noise dims predicted 0
    cr = getattr(m_in, "curvature_report", None)
    k_in = float(cr["atoms"][0]["kappa_hat"]) if isinstance(cr, dict) and cr.get("atoms") else float("nan")
    out["in_frame"] = {"ev": ev(X, Xhat), "seconds": dt, "K": 1,
                       "kappa_hat": [k_in],
                       "topologies": list(getattr(m_in, "atom_topologies", []) or [m_in.atom_topology])}
    out["full_ambient_joint"] = _fit(X, K=1, d_atom=1, atom_topology="circle",
                                     random_state=seed, **fitkw)
    out["ev_gap"] = out["full_ambient_joint"]["ev"] - out["in_frame"]["ev"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--q", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=60)
    args = ap.parse_args()

    fitkw = {"n_iter": args.n_iter}
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    results = {"gamfit_version": gamfit.__version__, "config": vars(args), "runs": []}
    print(f"[env] gamfit {gamfit.__version__}", flush=True)

    for p in args.p:
        for name, fn in [
            ("E3_envelope", lambda: exp3_envelope(p, args.n, args.noise, args.seed, fitkw)),
            ("E1_cross_frame", lambda: exp1_cross_frame(p, args.n, args.noise, args.seed, fitkw)),
            ("E2_inter_atom", lambda: exp2_inter_atom(p, args.n, args.noise, args.q, args.seed, fitkw)),
        ]:
            t0 = time.time()
            try:
                rec = fn()
                rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001 - record, don't abort the sweep
                import traceback
                rec = {"p": p, "status": "error", "error": repr(e),
                       "traceback": traceback.format_exc()}
            rec["experiment"] = name
            rec["wall_seconds"] = time.time() - t0
            results["runs"].append(rec)
            print(f"[{name} p={p}] status={rec.get('status')} "
                  f"wall={rec['wall_seconds']:.1f}s", flush=True)
            # checkpoint after every run
            (outdir / "results.json").write_text(json.dumps(results, indent=2))

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {outdir/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
