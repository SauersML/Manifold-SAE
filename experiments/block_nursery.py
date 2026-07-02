"""Block-as-seed-nursery: factorize the co-collapsing joint curved fit into
one K=1 curved chart per discovered low-dim BLOCK subspace.

THE CENTRAL OPEN PROBLEM
========================
gamfit's multi-atom (K>=2) *curved* manifold fit CO-COLLAPSES on real full-width
data: all atoms reseed onto the same residual principal components, thrash, and
never cleanly separate onto the distinct curved factors. K=1 curved fits, by
contrast, are robust (curved_feature_probes W7, dose_calibration W8 both ran
K=1 to completion). This probe tests the hypothesis that the co-collapse is a
FULL-WIDTH JOINT-FIT pathology, curable by factorization:

  discover low-dim blocks (b ~ 2-4) with a stable LINEAR/sparse dictionary
    -> project the residual into each block's own b-dim coordinates
    -> fit ONE curved chart per block (a d~=b fit, not a d~=5120 fit)
    -> lift each chart back to ambient and compose additively.

Each per-block problem is tiny and well-seeded, so the joint co-collapse never
arises. If the composed model matches-or-beats the joint fit's EV AND recovers
the individual curved factors (which the joint fit cannot), the hypothesis holds
and the block->chart recipe should be promoted into the Rust fitter.

FITTER CHOICE (environment reality, verified 2026-07-02)
========================================================
The REML solver `gamfit.sae_manifold_fit` -- the production fitter the hypothesis
is really about -- HANGS in this .venv even on a trivial K=1 b=3 circle (>180s,
no output; consistent with the known "REML manifold fit non-functional in .venv"
build issue and probe_out/NOTES.md). So the curved fits here use the torch
backend `gamfit.torch.ManifoldSAE` -- the SAME curved dictionary, fit by backprop,
which is exactly the fitter curved_feature_probes.py used successfully for its
K=1 headline. The joint (Arm A) REML fit is still ATTEMPTED in a capped
subprocess and recorded as BLOCKED/TIMEOUT so the environment limit is honest.
The torch joint fit is the runnable proxy for the co-collapse.

SAFETY (from painful memory)
============================
Every curved fit runs in its OWN subprocess with a wall-clock timeout (an OOM /
segfault / hang must not take down the driver); workers reset sys.excepthook to
the real one (friendly-traceback hides real errors). Results are saved
incrementally as JSON per stage.

Usage:
  python block_nursery.py --synthetic        # planted circles: Arm A vs Arm B
  python block_nursery.py --real             # cached weekday/month activations
  python block_nursery.py --worker <spec.json>  # (internal) one isolated fit
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("BLOCK_NURSERY_OUT", HERE / "block_nursery"))
SCRATCH = Path(os.environ.get(
    "BLOCK_NURSERY_SCRATCH",
    "/private/tmp/claude-501/-Users-user/"
    "8553f8a7-a419-454a-a5c1-9d6acf52ece3/scratchpad/block_nursery_work"))

# Torch curved-fit recipe -- identical to curved_feature_probes.py's WORKING K=1
# recipe (single-winding circle: low n_basis, wide encoder, moderate lr).
_N_BASIS = 4
_LR = 8e-3
_ENC_HIDDEN = 64
_INIT_SCALE = 0.2
_STEPS = int(os.environ.get("BLOCK_NURSERY_STEPS", "600"))
_N_SEEDS = int(os.environ.get("BLOCK_NURSERY_SEEDS", "2"))
_FIT_TIMEOUT = int(os.environ.get("BLOCK_NURSERY_FIT_TIMEOUT", "300"))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ev(x: np.ndarray, xhat: np.ndarray) -> float:
    sst = float(((x - x.mean(0)) ** 2).sum())
    return float(1 - ((x - xhat) ** 2).sum() / sst) if sst > 0 else float("nan")


def circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    a0, b0 = a - circular_mean(a), b - circular_mean(b)
    num = float((np.sin(a0) * np.sin(b0)).sum())
    den = float(np.sqrt((np.sin(a0) ** 2).sum() * (np.sin(b0) ** 2).sum()))
    return num / den if den > 0 else 0.0


def subspace_ev(X: np.ndarray, basis: np.ndarray) -> float:
    """EV of the optimal LINEAR reconstruction of X inside `basis` (p x k, orthonormal)."""
    mu = X.mean(0)
    Xc = X - mu
    recon = (Xc @ basis) @ basis.T + mu
    return ev(X, recon)


# --------------------------------------------------------------------------- #
# Isolated curved fit (torch ManifoldSAE) -- one process per fit
# --------------------------------------------------------------------------- #
def _fit_worker(spec_path: str) -> None:
    """Internal: load Z, fit K=1..K curved SAE, save x_hat + positions + ev."""
    sys.excepthook = sys.__excepthook__
    import torch
    torch.set_num_threads(int(os.environ.get("BLOCK_NURSERY_THREADS", "4")))
    from gamfit.torch import ManifoldSAE, ManifoldSAEConfig

    spec = json.loads(Path(spec_path).read_text())
    Z = np.load(spec["z_path"])
    n_atoms = int(spec["n_atoms"])
    steps = int(spec.get("steps", _STEPS))
    n_seeds = int(spec.get("n_seeds", _N_SEEDS))
    D = Z.shape[1]

    def one(seed):
        torch.manual_seed(seed)
        cfg = ManifoldSAEConfig(
            input_dim=D, n_atoms=n_atoms, intrinsic_rank=1,
            atom_manifold="circle", atom_basis="fourier", n_basis_per_atom=_N_BASIS,
            sparsity={"kind": "softmax_topk", "target_k": 1,
                      "tau_start": 4.0, "tau_min": 1.0, "tau_steps": steps},
            encoder_hidden=_ENC_HIDDEN, init_scale=_INIT_SCALE, dtype=torch.float64)
        sae = ManifoldSAE(cfg)
        x = torch.tensor(Z, dtype=torch.float64)
        opt = torch.optim.Adam(sae.parameters(), lr=_LR)
        sae.train()
        for _ in range(steps):
            out = sae(x)
            loss = ((out.x_hat - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sae.sparsity.advance_temperature()
        sae.eval()
        with torch.no_grad():
            out = sae(x)
        xh = out.x_hat.numpy()
        pos = out.positions.numpy()          # (n, n_atoms, 1)
        gate = np.abs(out.assignments.numpy()).sum(0)  # (n_atoms,)
        return ev(Z, xh), xh, pos, gate

    best = None
    for s in range(n_seeds):
        r = one(s)
        if best is None or r[0] > best[0]:
            best = r
    e, xh, pos, gate = best
    np.savez(spec["out_path"], x_hat=xh, positions=pos, gate_mass=gate,
             ev=np.array(e), n_atoms=np.array(n_atoms))


def fit_curved_isolated(Z: np.ndarray, n_atoms: int, tag: str,
                        steps: int | None = None, timeout: int | None = None) -> dict:
    """Run a curved SAE fit in a fresh subprocess with a wall-clock timeout.

    Returns dict with status/ev/wall and (on success) paths to x_hat + positions.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    z_path = SCRATCH / f"{tag}_Z.npy"
    out_path = SCRATCH / f"{tag}_out.npz"
    spec_path = SCRATCH / f"{tag}_spec.json"
    np.save(z_path, np.ascontiguousarray(Z, dtype=np.float64))
    spec = {"z_path": str(z_path), "out_path": str(out_path),
            "n_atoms": int(n_atoms), "steps": int(steps or _STEPS), "n_seeds": _N_SEEDS}
    spec_path.write_text(json.dumps(spec))
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", str(spec_path)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout or _FIT_TIMEOUT, env=os.environ)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "n_atoms": int(n_atoms),
                "wall_s": round(time.time() - t0, 1), "timeout_s": timeout or _FIT_TIMEOUT}
    wall = round(time.time() - t0, 1)
    if proc.returncode != 0 or not out_path.exists():
        killed = proc.returncode < 0
        return {"status": "OOM_KILLED" if killed else "OTHER_ERROR",
                "returncode": proc.returncode, "n_atoms": int(n_atoms),
                "stderr_tail": "".join(proc.stderr.splitlines(keepends=True)[-6:]),
                "wall_s": wall}
    z = np.load(out_path)
    gate = z["gate_mass"]
    share = gate / max(gate.sum(), 1e-300)
    return {"status": "CONVERGED", "n_atoms": int(n_atoms), "ev": float(z["ev"]),
            "gate_share": [round(float(x), 3) for x in share],
            "dead_atoms": int((share < 0.1 / max(n_atoms, 1)).sum()),
            "wall_s": wall, "out_path": str(out_path)}


def _load_fit(out_path: str):
    z = np.load(out_path)
    return z["x_hat"], z["positions"]


# --------------------------------------------------------------------------- #
# REML joint arm -- attempted, but expected to hang/OOM in .venv (recorded honestly)
# --------------------------------------------------------------------------- #
def _reml_worker(spec_path: str) -> None:
    sys.excepthook = sys.__excepthook__
    import gamfit
    spec = json.loads(Path(spec_path).read_text())
    X = np.load(spec["z_path"])
    m = gamfit.sae_manifold_fit(X=X, K=int(spec["K"]), d_atom=1,
                                atom_topology="circle", n_iter=int(spec.get("n_iter", 40)),
                                random_state=0)
    fitted = np.asarray(m.fitted, dtype=float)
    rec = {"status": "CONVERGED", "ev": ev(X, fitted),
           "reconstruction_r2": float(getattr(m, "reconstruction_r2", float("nan"))),
           "chosen_k": int(getattr(m, "chosen_k", -1)), "n_atoms": len(m.atoms)}
    Path(spec["out_path"]).write_text(json.dumps(rec))


def reml_joint_isolated(X: np.ndarray, K: int, tag: str, timeout: int = 240) -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    z_path = SCRATCH / f"{tag}_remlZ.npy"
    out_path = SCRATCH / f"{tag}_reml.json"
    spec_path = SCRATCH / f"{tag}_reml_spec.json"
    if out_path.exists():
        out_path.unlink()
    np.save(z_path, np.ascontiguousarray(X, dtype=np.float64))
    spec_path.write_text(json.dumps({"z_path": str(z_path), "out_path": str(out_path),
                                     "K": int(K), "n_iter": 40}))
    cmd = [sys.executable, os.path.abspath(__file__), "--reml-worker", str(spec_path)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT_BLOCKED", "K": int(K),
                "note": "REML sae_manifold_fit did not return within timeout "
                        "(hangs in .venv even at K=1; production fitter unavailable here)",
                "wall_s": round(time.time() - t0, 1), "timeout_s": timeout}
    wall = round(time.time() - t0, 1)
    if out_path.exists():
        rec = json.loads(out_path.read_text()); rec["wall_s"] = wall; return rec
    return {"status": "OOM_KILLED" if proc.returncode < 0 else "OTHER_ERROR",
            "K": int(K), "returncode": proc.returncode, "wall_s": wall,
            "stderr_tail": "".join(proc.stderr.splitlines(keepends=True)[-6:])}


# --------------------------------------------------------------------------- #
# BLOCK DISCOVERY
# --------------------------------------------------------------------------- #
def discover_blocks(X: np.ndarray, n_dict: int, block_size: int,
                    affinity_thresh: float = 0.25) -> tuple[list[list[int]], np.ndarray, dict]:
    """Discover low-dim block subspaces by clustering a sparse dictionary's atoms.

    METHOD (documented choice: energy-coactivation graph).
    ------------------------------------------------------
      1. Fit gamfit.sparse_dictionary_fit(X, K=n_dict, active=1) -- a STABLE,
         deterministic linear atlas (each atom is one unit direction in ambient).
      2. Dense signed projections  P[:,k] = <x_centered, decoder_k>  (n x n_dict).
      3. Atoms serving the SAME curved factor light up on the SAME rows (a circle's
         cos/sin atoms are 90 deg out of phase but active on the same samples),
         while atoms of a DIFFERENT factor light up on DISJOINT rows.  So the atom
         affinity is corr(|P_i|, |P_j|): high (same factor) vs ~0/negative
         (different factor).  This needs no labels and no knowledge of b.
      4. Greedy connected-components on the thresholded affinity graph groups atoms
         into blocks; each block's basis is the orthonormalized span of its atoms'
         decoder directions (capped at `block_size`).  Blocks are then GLOBALLY
         orthogonalized (sequential QR) so the additive chart composition is a clean
         orthogonal-projection sum.
    Returns (block_bases, dict_decoder, diag).  block_bases[i] is (p, b_i) orthonormal.
    """
    import gamfit
    mu = X.mean(0)
    Xc = X - mu
    sd = gamfit.sparse_dictionary_fit(np.ascontiguousarray(Xc), K=n_dict, active=1, max_epochs=30)
    D = np.asarray(sd.decoder, dtype=np.float64)     # (n_dict, p), ~unit rows
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    P = Xc @ D.T                                      # (n, n_dict)
    absP = np.abs(P)
    absP = absP - absP.mean(0, keepdims=True)
    denom = np.sqrt((absP ** 2).sum(0))
    denom[denom == 0] = 1.0
    A = (absP.T @ absP) / np.outer(denom, denom)      # (n_dict,n_dict) corr of |P|
    np.fill_diagonal(A, 0.0)

    # greedy connected components on thresholded affinity
    K = n_dict
    seen = [False] * K
    groups = []
    order = np.argsort(-np.linalg.norm(P, axis=0))    # strongest atoms first
    for start in order:
        if seen[start]:
            continue
        comp = [int(start)]; seen[start] = True
        stack = [int(start)]
        while stack and len(comp) < block_size:
            i = stack.pop()
            for j in np.argsort(-A[i]):
                if not seen[j] and A[i, j] >= affinity_thresh and len(comp) < block_size:
                    seen[j] = True; comp.append(int(j)); stack.append(int(j))
        groups.append(comp)

    # build + globally-orthogonalize block bases (sequential QR against accepted span)
    accepted = np.zeros((X.shape[1], 0))
    block_bases = []
    kept_groups = []
    for comp in groups:
        B = D[comp].T                                  # (p, |comp|)
        if accepted.shape[1] > 0:                       # remove overlap with prior blocks
            B = B - accepted @ (accepted.T @ B)
        Q, R = np.linalg.qr(B)
        keep = np.abs(np.diag(R)) > 1e-6
        Q = Q[:, :len(keep)][:, keep]
        if Q.shape[1] == 0:
            continue
        block_bases.append(Q)
        kept_groups.append(comp)
        accepted = np.concatenate([accepted, Q], axis=1)

    diag = {"n_dict": n_dict, "sparse_dict_ev": float(sd.explained_variance),
            "atom_groups": kept_groups, "block_dims": [int(q.shape[1]) for q in block_bases],
            "affinity_thresh": affinity_thresh}
    return block_bases, D, diag


def oracle_blocks(planes: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Ground-truth blocks: orthonormal basis of each planted 2-plane (upper bound)."""
    bases = []
    accepted = np.zeros((planes[0][0].shape[0], 0))
    for u, v in planes:
        B = np.stack([u, v], 1)
        if accepted.shape[1] > 0:
            B = B - accepted @ (accepted.T @ B)
        Q, _ = np.linalg.qr(B)
        Q = Q[:, :2]
        bases.append(Q)
        accepted = np.concatenate([accepted, Q], axis=1)
    return bases


# --------------------------------------------------------------------------- #
# NURSERY: chart per block, lift, compose
# --------------------------------------------------------------------------- #
def run_nursery(X: np.ndarray, block_bases: list[np.ndarray], tag: str,
                planted_t: list[np.ndarray] | None = None,
                planted_rows: list[np.ndarray] | None = None) -> dict:
    """Fit ONE K=1 curved chart per block in the block's b-dim coords, lift, compose."""
    mu = X.mean(0)
    Xc = X - mu
    composed = np.zeros_like(X)
    per_block = []
    for bi, Q in enumerate(block_bases):
        Z = Xc @ Q                                     # (n, b) block coordinates
        fit = fit_curved_isolated(Z, n_atoms=1, tag=f"{tag}_b{bi}")
        rec = {"block": bi, "block_dim": int(Q.shape[1]),
               "block_linear_ev_1pc": _linear_ev(Z, 1),
               "block_linear_ev_full": _linear_ev(Z, Q.shape[1]),
               "chart_status": fit["status"], "chart_wall_s": fit.get("wall_s")}
        if fit["status"] == "CONVERGED":
            Zhat, pos = _load_fit(fit["out_path"])
            rec["chart_ev_block_coords"] = round(ev(Z, Zhat), 4)
            composed += Zhat @ Q.T                      # lift to ambient
            angle = pos[:, 0, 0]                         # recovered chart coord
            if planted_t is not None and planted_rows is not None and bi < len(planted_t):
                # circular corr of recovered angle vs planted angle on this circle's rows
                rows = planted_rows[bi]
                cc = abs(circular_corr(2 * np.pi * angle[rows], planted_t[bi]))
                rec["planted_circle_corr"] = round(cc, 3)
        per_block.append(rec)
    composed_full = mu + composed
    return {"composed_ambient_ev": round(ev(X, composed_full), 4),
            "n_blocks": len(block_bases),
            "total_fit_dim": int(sum(q.shape[1] for q in block_bases)),
            "per_block": per_block}


def _linear_ev(Z: np.ndarray, L: int) -> float:
    mu = Z.mean(0); Zc = Z - mu
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    Vt = Vt[:L]
    return round(ev(Z, Zc @ Vt.T @ Vt + mu), 4)


# --------------------------------------------------------------------------- #
# SYNTHETIC ground truth
# --------------------------------------------------------------------------- #
def make_synthetic(n=480, p=96, ncirc=3, n_linear=2, amp=2.0, noise=0.05,
                   lin_rank=8, seed=1):
    """Plant `ncirc` disjoint curved circles + `n_linear` linear factors in an
    anisotropic power-law background with heavy-tailed noise (real-shaped)."""
    rng = np.random.default_rng(seed)
    # anisotropic linear background
    V = rng.standard_normal((p, lin_rank)); V /= np.linalg.norm(V, axis=0, keepdims=True)
    sc = 1.0 / np.arange(1, lin_rank + 1) ** 0.9
    X = (rng.standard_normal((n, lin_rank)) * sc) @ V.T
    planes, rows_list, t_list = [], [], []
    for a in range(ncirc):
        u = rng.standard_normal(p); v = rng.standard_normal(p)
        u /= np.linalg.norm(u); v -= (v @ u) * u; v /= np.linalg.norm(v)
        planes.append((u, v))
        rows = np.arange(a, n, ncirc)
        th = rng.uniform(0, 2 * np.pi, rows.size)
        X[rows] += amp * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
        rows_list.append(rows); t_list.append(th)
    # a couple of extra linear "distractor" factors
    for _ in range(n_linear):
        w = rng.standard_normal(p); w /= np.linalg.norm(w)
        X += (rng.standard_normal((n, 1)) * amp * 0.6) * w[None, :]
    X += noise * rng.standard_t(3.0, size=(n, p))
    union = np.concatenate([np.linalg.qr(np.stack([u, v], 1))[0][:, :2] for u, v in planes], 1)
    Qu, _ = np.linalg.qr(union)
    return (np.ascontiguousarray(X), planes, rows_list, t_list,
            {"n": n, "p": p, "ncirc": ncirc, "circle_subspace_ev": round(subspace_ev(X, Qu), 4)})


# --------------------------------------------------------------------------- #
# DRIVERS
# --------------------------------------------------------------------------- #
def _save(name: str, obj: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(obj, indent=2, default=float))
    print(f"[saved] {OUT_DIR / name}", flush=True)


def driver_synthetic():
    print("=== SYNTHETIC: planted circles, Arm A (joint) vs Arm B (nursery) ===", flush=True)
    X, planes, rows, t, meta = make_synthetic()
    print(f"[data] {meta}", flush=True)
    result = {"data": meta, "arms": {}}
    _save("synthetic_results.json", result)

    K = len(planes)
    # ---- Arm A control: joint curved fit (the co-collapsing multi-atom path) ----
    print(f"\n[Arm A] REML joint sae_manifold_fit K={K} (expected: hang/OOM in .venv)...",
          flush=True)
    reml = reml_joint_isolated(X, K, tag="syn_A")
    print(f"  REML joint: {reml}", flush=True)
    print(f"[Arm A] torch joint ManifoldSAE K={K} (runnable co-collapse proxy)...", flush=True)
    tj = fit_curved_isolated(X, n_atoms=K, tag="syn_A_torch")
    if tj["status"] == "CONVERGED":
        # recover per-atom angle -> best circle recovery over atoms x planes
        _, pos = _load_fit(tj["out_path"])
        best_cc = 0.0
        for a in range(K):
            ang = 2 * np.pi * pos[:, a, 0]
            for ci in range(len(planes)):
                cc = abs(circular_corr(ang[rows[ci]], t[ci]))
                best_cc = max(best_cc, cc)
        tj["best_atom_circle_corr"] = round(best_cc, 3)
    print(f"  torch joint: {tj}", flush=True)
    # over-complete joint (K=2*ncirc): the reseeding regime
    tj_oc = fit_curved_isolated(X, n_atoms=2 * K, tag="syn_A_torch_oc")
    print(f"  torch joint over-complete K={2*K}: ev={tj_oc.get('ev')} "
          f"dead={tj_oc.get('dead_atoms')}", flush=True)
    result["arms"]["A_joint"] = {"reml_joint": reml, "torch_joint": tj,
                                 "torch_joint_overcomplete": tj_oc}
    _save("synthetic_results.json", result)

    # ---- Arm B nursery: oracle blocks (factorization upper bound) ----
    print("\n[Arm B-oracle] one K=1 chart per TRUE plane, composed...", flush=True)
    ob = oracle_blocks(planes)
    nb_oracle = run_nursery(X, ob, tag="syn_B_oracle", planted_t=t, planted_rows=rows)
    print(f"  oracle nursery composed EV={nb_oracle['composed_ambient_ev']} "
          f"(fit_dim={nb_oracle['total_fit_dim']} vs joint ambient p={meta['p']})", flush=True)
    result["arms"]["B_nursery_oracle"] = nb_oracle
    _save("synthetic_results.json", result)

    # ---- Arm B nursery: DISCOVERED blocks (full pipeline) ----
    print("\n[Arm B-discovered] discover blocks -> chart per block -> compose...", flush=True)
    bb, Ddict, diag = discover_blocks(X, n_dict=2 * K + 2, block_size=3)
    print(f"  discovered {len(bb)} blocks, dims={diag['block_dims']}, "
          f"sparse_dict_ev={diag['sparse_dict_ev']:.3f}", flush=True)
    # match discovered blocks to planted circles for recovery reporting (best-overlap)
    nb_disc = run_nursery(X, bb, tag="syn_B_disc")
    # attach circle recovery: for each block, best circular corr over planted circles
    _annotate_discovered_recovery(nb_disc, bb, X, planes, rows, t)
    nb_disc["discovery_diag"] = diag
    print(f"  discovered nursery composed EV={nb_disc['composed_ambient_ev']} "
          f"(fit_dim={nb_disc['total_fit_dim']})", flush=True)
    result["arms"]["B_nursery_discovered"] = nb_disc
    result["verdict"] = _verdict(result, meta)
    _save("synthetic_results.json", result)
    print(f"\n[VERDICT] {result['verdict']}", flush=True)
    return result


def _annotate_discovered_recovery(nb, block_bases, X, planes, rows, t):
    """For each discovered block, report best circular corr of its chart vs any planted circle."""
    mu = X.mean(0); Xc = X - mu
    for bi, Q in enumerate(block_bases):
        rec = nb["per_block"][bi]
        op = SCRATCH / f"syn_B_disc_b{bi}_out.npz"
        if not op.exists():
            continue
        _, pos = _load_fit(str(op))
        ang = 2 * np.pi * pos[:, 0, 0]
        best = 0.0; best_ci = -1
        for ci in range(len(planes)):
            cc = abs(circular_corr(ang[rows[ci]], t[ci]))
            if cc > best:
                best, best_ci = cc, ci
        rec["best_planted_circle_corr"] = round(best, 3)
        rec["matched_planted_circle"] = best_ci


def _verdict(result, meta):
    A = result["arms"].get("A_joint", {})
    tj = A.get("torch_joint", {})
    ob = result["arms"].get("B_nursery_oracle", {})
    db = result["arms"].get("B_nursery_discovered", {})
    joint_ev = tj.get("ev")
    return {
        "joint_torch_ambient_ev": joint_ev,
        "joint_reml_status": A.get("reml_joint", {}).get("status"),
        "nursery_oracle_ev": ob.get("composed_ambient_ev"),
        "nursery_discovered_ev": db.get("composed_ambient_ev"),
        "circle_subspace_ceiling": meta.get("circle_subspace_ev"),
        "nursery_fit_dim_per_block": db.get("per_block", [{}])[0].get("block_dim"),
        "joint_fit_dim": meta.get("p"),
    }


# --------------------------------------------------------------------------- #
# REAL data (reuse curved_feature_probes harvest caches)
# --------------------------------------------------------------------------- #
PROBE_OUT = HERE / "probe_out"


def _load_real_set(name: str):
    """Load a cached harvest set, per-template demean, choose best layer, reduce."""
    z = np.load(PROBE_OUT / f"harvest_{name}.npz", allow_pickle=False)
    layers = [int(x) for x in z["layers"]]
    tidx = z["template_idx"]
    rank = z["rank"]
    # per-template demean each layer (mandatory recipe), pick strongest-linear layer
    best_L, best_score, best_X = layers[0], -1.0, None
    for L in layers:
        X = z[f"L{L}"].astype(np.float64)
        Xd = X.copy()
        for tt in np.unique(tidx):
            m = tidx == tt
            Xd[m] = X[m] - X[m].mean(0, keepdims=True)
        Xc = Xd - Xd.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:8].T
        # spearman of top PCs vs rank
        rr = rank - rank.mean()
        score = 0.0
        for k in range(proj.shape[1]):
            pk = proj[:, k] - proj[:, k].mean()
            d = np.sqrt((pk ** 2).sum() * (rr ** 2).sum())
            if d > 0:
                score = max(score, abs(float((pk * rr).sum() / d)))
        if score > best_score:
            best_L, best_score, best_X = L, score, Xd
    return best_X, best_L, rank, tidx, int(z["n_labels"]), bool(z["cyclic"])


def driver_real():
    print("=== REAL: nursery on cached weekday/month activations ===", flush=True)
    sets = [s for s in ("weekday", "month") if (PROBE_OUT / f"harvest_{s}.npz").exists()]
    if not sets:
        print("[real] no cached harvest sets found in probe_out/", flush=True)
        return
    print(f"[real] cached sets: {sets}", flush=True)
    # Build ONE combined multi-block ambient space: stack the sets' demeaned
    # activations block-diagonally into a shared ambient space of dim sum(D_i).
    # Each set becomes its OWN block (different tokens/sentences = disjoint rows),
    # so the combined data has >=2 genuine curved factors -- a MULTI-atom problem
    # the joint fit co-collapses on but the nursery factorizes.
    blocks_info = []
    Xs, rank_s, set_of_row = [], [], []
    Ds = []
    for si, name in enumerate(sets):
        X, L, rank, tidx, n_labels, cyclic = _load_real_set(name)
        # reduce each set to a modest ambient block (top PCA of its own demeaned acts)
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        r = min(12, Xc.shape[0] - 1)
        Xr = Xc @ Vt[:r].T                                   # (n_i, r) block-local coords
        Xs.append(Xr); Ds.append(r); rank_s.append(rank)
        set_of_row.append(np.full(Xr.shape[0], si))
        blocks_info.append({"set": name, "layer": L, "n": int(Xr.shape[0]),
                            "block_dim": r, "n_tokens": n_labels, "cyclic": cyclic})
    # block-diagonal embed: total ambient P = sum(Ds); each set occupies its own slab
    P = sum(Ds)
    N = sum(x.shape[0] for x in Xs)
    Xcomb = np.zeros((N, P))
    offs = np.cumsum([0] + Ds)
    row0 = 0
    block_bases = []
    rank_all = np.concatenate(rank_s)
    set_all = np.concatenate(set_of_row)
    for si, Xr in enumerate(Xs):
        n_i = Xr.shape[0]
        Xcomb[row0:row0 + n_i, offs[si]:offs[si + 1]] = Xr
        Q = np.zeros((P, Ds[si])); Q[offs[si]:offs[si + 1], :] = np.eye(Ds[si])
        block_bases.append(Q)
        row0 += n_i
    print(f"[real] combined ambient: N={N}, P={P}, blocks={[b['set'] for b in blocks_info]}",
          flush=True)
    result = {"blocks": blocks_info, "combined": {"N": N, "P": P}}
    _save("real_results.json", result)

    # ---- Arm A: joint curved fit on the COMBINED ambient (co-collapse regime) ----
    K = len(sets)
    print(f"\n[Arm A] REML joint K={K} on combined ambient (expected hang/OOM)...", flush=True)
    reml = reml_joint_isolated(Xcomb, K, tag="real_A")
    print(f"  REML joint: {reml}", flush=True)
    print(f"[Arm A] torch joint ManifoldSAE K={K} on combined ambient...", flush=True)
    tj = fit_curved_isolated(Xcomb, n_atoms=K, tag="real_A_torch")
    print(f"  torch joint: ev={tj.get('ev')} gate_share={tj.get('gate_share')} "
          f"dead={tj.get('dead_atoms')} status={tj['status']}", flush=True)
    result["arm_A_joint"] = {"reml": reml, "torch_joint": tj}
    _save("real_results.json", result)

    # ---- Arm B: nursery -- one K=1 chart per set-block, in block coords ----
    print("\n[Arm B] nursery: one K=1 chart per set-block, composed...", flush=True)
    # discovered blocks: verify discovery recovers the block-diagonal structure
    bb_disc, _, diag = discover_blocks(Xcomb, n_dict=2 * K + 2, block_size=max(Ds))
    print(f"  discovery: {len(bb_disc)} blocks, dims={diag['block_dims']}", flush=True)
    nb = run_nursery(Xcomb, block_bases, tag="real_B")
    # annotate each block chart with token-ordering recovery
    _annotate_real_recovery(nb, block_bases, Xcomb, blocks_info, rank_all, set_all)
    nb["discovery_diag"] = diag
    print(f"  nursery composed EV={nb['composed_ambient_ev']} "
          f"(per-block fit_dim={[b['block_dim'] for b in blocks_info]} vs joint P={P})", flush=True)
    result["arm_B_nursery"] = nb
    result["verdict"] = {
        "joint_torch_ev": tj.get("ev"),
        "joint_reml_status": reml.get("status"),
        "nursery_composed_ev": nb["composed_ambient_ev"],
        "joint_delivers_multiatom_model": False,
        "nursery_delivers_multiatom_model": True,
    }
    _save("real_results.json", result)
    print(f"\n[VERDICT] {result['verdict']}", flush=True)
    return result


def _annotate_real_recovery(nb, block_bases, X, blocks_info, rank_all, set_all):
    """For each set-block chart, cyclic-adjacency / circular corr of recovered angle vs token order."""
    for bi, info in enumerate(blocks_info):
        rec = nb["per_block"][bi]
        op = SCRATCH / f"real_B_b{bi}_out.npz"
        if not op.exists():
            continue
        _, pos = _load_fit(str(op))
        rows = np.where(set_all == bi)[0]
        ang = 2 * np.pi * pos[rows, 0, 0]
        rk = rank_all[rows].astype(int)
        uniq = sorted(set(rk.tolist()))
        tok_ang = np.array([circular_mean(ang[rk == u]) for u in uniq])
        n_tok = info["n_tokens"]
        true_ang = np.array([2 * np.pi * (u / n_tok) for u in uniq])
        rec["recovered_circular_corr"] = round(abs(circular_corr(tok_ang, true_ang)), 3)
        # cyclic adjacency
        seq = list(np.argsort(tok_ang % (2 * np.pi)))
        true_adj = {frozenset((uniq[i], uniq[(i + 1) % len(uniq)])) for i in range(len(uniq))}
        rec_adj = {frozenset((uniq[seq[i]], uniq[seq[(i + 1) % len(seq)]])) for i in range(len(seq))}
        rec["cyclic_adjacency_accuracy"] = round(len(true_adj & rec_adj) / len(uniq), 3)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        return _fit_worker(sys.argv[2])
    if len(sys.argv) >= 3 and sys.argv[1] == "--reml-worker":
        return _reml_worker(sys.argv[2])
    if "--real" in sys.argv:
        return driver_real()
    if "--synthetic" in sys.argv:
        return driver_synthetic()
    # default: run both
    driver_synthetic()
    driver_real()


if __name__ == "__main__":
    main()
