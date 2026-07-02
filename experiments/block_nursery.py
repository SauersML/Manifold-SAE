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
    # target_k = how many atoms are active per sample. For a K=1 per-block chart it
    # is 1; for the JOINT arm on a product-of-circles manifold every sample lies on
    # EVERY circle simultaneously, so the honest multi-atom fit is ADDITIVE
    # (target_k = n_atoms, all atoms active) -- this is the regime the joint fit
    # co-collapses in (atoms compete for the same residual PCs).
    target_k = int(spec.get("target_k", 1))
    D = Z.shape[1]

    def one(seed):
        torch.manual_seed(seed)
        cfg = ManifoldSAEConfig(
            input_dim=D, n_atoms=n_atoms, intrinsic_rank=1,
            atom_manifold="circle", atom_basis="fourier", n_basis_per_atom=_N_BASIS,
            sparsity={"kind": "softmax_topk", "target_k": target_k,
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
                        steps: int | None = None, timeout: int | None = None,
                        target_k: int = 1) -> dict:
    """Run a curved SAE fit in a fresh subprocess with a wall-clock timeout.

    Returns dict with status/ev/wall and (on success) paths to x_hat + positions.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    z_path = SCRATCH / f"{tag}_Z.npy"
    out_path = SCRATCH / f"{tag}_out.npz"
    spec_path = SCRATCH / f"{tag}_spec.json"
    np.save(z_path, np.ascontiguousarray(Z, dtype=np.float64))
    spec = {"z_path": str(z_path), "out_path": str(out_path),
            "n_atoms": int(n_atoms), "steps": int(steps or _STEPS), "n_seeds": _N_SEEDS,
            "target_k": int(target_k)}
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
                theta: np.ndarray | None = None) -> dict:
    """Fit ONE K=1 curved chart per block in the block's b-dim coords, lift, compose.

    `theta` (n x ncirc, optional) = ground-truth per-sample angle on each planted
    circle; each block chart is scored by its best circular corr over the circles.
    """
    mu = X.mean(0)
    Xc = X - mu
    composed = np.zeros_like(X)
    per_block = []
    for bi, Q in enumerate(block_bases):
        Z = Xc @ Q                                     # (n, b) block coordinates
        fit = fit_curved_isolated(Z, n_atoms=1, tag=f"{tag}_b{bi}", target_k=1)
        rec = {"block": bi, "block_dim": int(Q.shape[1]),
               "block_linear_ev_1pc": _linear_ev(Z, 1),
               "block_linear_ev_2pc": _linear_ev(Z, min(2, Q.shape[1])),
               "chart_status": fit["status"], "chart_wall_s": fit.get("wall_s")}
        if fit["status"] == "CONVERGED":
            Zhat, pos = _load_fit(fit["out_path"])
            rec["chart_ev_block_coords"] = round(ev(Z, Zhat), 4)
            composed += Zhat @ Q.T                      # lift to ambient
            angle = 2 * np.pi * pos[:, 0, 0]            # recovered chart coord (all rows)
            if theta is not None:
                ccs = [abs(circular_corr(angle, theta[:, ci])) for ci in range(theta.shape[1])]
                rec["best_planted_circle_corr"] = round(max(ccs), 3)
                rec["matched_planted_circle"] = int(np.argmax(ccs))
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
def make_synthetic(n=480, p=96, ncirc=3, n_linear=1, amp=2.0, noise=0.06,
                   lin_rank=6, lin_scale=0.35, seed=1):
    """Plant a PRODUCT-OF-CIRCLES manifold: every sample carries an INDEPENDENT
    angle on EACH of `ncirc` circles simultaneously (orthogonal 2-planes), on top
    of a small anisotropic power-law linear background + heavy-tailed noise.

    This is the realistic multi-curved-factor residual-stream shape: a genuine
    multi-atom problem where each point lives on all circles at once (unlike
    disjoint per-circle rows, which let a single atom own a whole circle). It is
    exactly the regime where the joint curved fit must keep several atoms alive and
    separated -- and where it co-collapses.

    Returns X, planes, theta (n x ncirc true angles), meta.
    """
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((p, lin_rank)); V /= np.linalg.norm(V, axis=0, keepdims=True)
    sc = lin_scale / np.arange(1, lin_rank + 1) ** 0.9
    X = (rng.standard_normal((n, lin_rank)) * sc) @ V.T
    planes = []
    theta = np.zeros((n, ncirc))
    accepted = np.zeros((p, 0))
    for a in range(ncirc):
        # orthogonalize each circle's plane against the previous ones (clean factors)
        raw = rng.standard_normal((p, 2))
        if accepted.shape[1] > 0:
            raw = raw - accepted @ (accepted.T @ raw)
        Q, _ = np.linalg.qr(raw)
        u, v = Q[:, 0], Q[:, 1]
        planes.append((u, v))
        accepted = np.concatenate([accepted, Q[:, :2]], axis=1)
        th = rng.uniform(0, 2 * np.pi, n)
        theta[:, a] = th
        X += amp * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    for _ in range(n_linear):
        w = rng.standard_normal(p); w /= np.linalg.norm(w)
        X += (rng.standard_normal((n, 1)) * amp * 0.4) * w[None, :]
    X += noise * rng.standard_t(3.0, size=(n, p))
    union = np.concatenate([np.linalg.qr(np.stack([u, v], 1))[0][:, :2] for u, v in planes], 1)
    Qu, _ = np.linalg.qr(union)
    meta = {"n": n, "p": p, "ncirc": ncirc, "amp": amp,
            "circle_subspace_ev": round(subspace_ev(X, Qu), 4)}
    return np.ascontiguousarray(X), planes, theta, meta


# --------------------------------------------------------------------------- #
# DRIVERS
# --------------------------------------------------------------------------- #
def _save(name: str, obj: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(obj, indent=2, default=float))
    print(f"[saved] {OUT_DIR / name}", flush=True)


def _joint_recovery(out_path, theta):
    """For each planted circle, best circular corr over the joint fit's atoms.

    Reports how many circles the joint fit cleanly recovered (corr>0.8) -- the
    co-collapse signature is that atoms pile onto a subset of circles, leaving the
    rest unrecovered even though every atom is alive.
    """
    _, pos = _load_fit(out_path)
    K = pos.shape[1]
    per_circle = []
    for ci in range(theta.shape[1]):
        best = max(abs(circular_corr(2 * np.pi * pos[:, a, 0], theta[:, ci])) for a in range(K))
        per_circle.append(round(best, 3))
    return per_circle, int(sum(c > 0.8 for c in per_circle))


def driver_synthetic():
    print("=== SYNTHETIC: product-of-circles, Arm A (joint) vs Arm B (nursery) ===", flush=True)
    X, planes, theta, meta = make_synthetic()
    print(f"[data] {meta}", flush=True)
    result = {"data": meta, "arms": {}}
    _save("synthetic_results.json", result)

    K = len(planes)
    # ---- Arm A control: joint curved fit (the co-collapsing multi-atom path) ----
    print(f"\n[Arm A] REML joint sae_manifold_fit K={K} (expected: hang/OOM in .venv)...",
          flush=True)
    reml = reml_joint_isolated(X, K, tag="syn_A", timeout=120)
    print(f"  REML joint: {reml.get('status')} ({reml.get('wall_s')}s)", flush=True)
    print(f"[Arm A] torch joint ManifoldSAE K={K}, target_k={K} (additive co-collapse proxy)...",
          flush=True)
    tj = fit_curved_isolated(X, n_atoms=K, tag="syn_A_torch", target_k=K)
    if tj["status"] == "CONVERGED":
        tj["per_circle_corr"], tj["n_circles_recovered"] = _joint_recovery(tj["out_path"], theta)
    print(f"  torch joint: ev={tj.get('ev')} per_circle_corr={tj.get('per_circle_corr')} "
          f"recovered={tj.get('n_circles_recovered')}/{K}", flush=True)
    # over-complete joint (K=2*ncirc, additive): the reseeding regime
    tj_oc = fit_curved_isolated(X, n_atoms=2 * K, tag="syn_A_torch_oc", target_k=2 * K)
    if tj_oc["status"] == "CONVERGED":
        tj_oc["per_circle_corr"], tj_oc["n_circles_recovered"] = _joint_recovery(
            tj_oc["out_path"], theta)
    print(f"  torch joint over-complete K={2*K}: ev={tj_oc.get('ev')} "
          f"recovered={tj_oc.get('n_circles_recovered')}/{K} dead={tj_oc.get('dead_atoms')}",
          flush=True)
    result["arms"]["A_joint"] = {"reml_joint": reml, "torch_joint": tj,
                                 "torch_joint_overcomplete": tj_oc}
    _save("synthetic_results.json", result)

    # ---- Arm B nursery: oracle blocks (factorization upper bound) ----
    print("\n[Arm B-oracle] one K=1 chart per TRUE plane, composed...", flush=True)
    ob = oracle_blocks(planes)
    nb_oracle = run_nursery(X, ob, tag="syn_B_oracle", theta=theta)
    nb_oracle["n_circles_recovered"] = int(sum(
        b.get("best_planted_circle_corr", 0) > 0.8 for b in nb_oracle["per_block"]))
    print(f"  oracle nursery composed EV={nb_oracle['composed_ambient_ev']} "
          f"recovered={nb_oracle['n_circles_recovered']}/{K} "
          f"(fit_dim={nb_oracle['total_fit_dim']} vs joint ambient p={meta['p']})", flush=True)
    result["arms"]["B_nursery_oracle"] = nb_oracle
    _save("synthetic_results.json", result)

    # ---- Arm B nursery: DISCOVERED blocks (full pipeline) ----
    print("\n[Arm B-discovered] discover blocks -> chart per block -> compose...", flush=True)
    bb, Ddict, diag = discover_blocks(X, n_dict=2 * K + 2, block_size=3)
    print(f"  discovered {len(bb)} blocks, dims={diag['block_dims']}, "
          f"sparse_dict_ev={diag['sparse_dict_ev']:.3f}", flush=True)
    nb_disc = run_nursery(X, bb, tag="syn_B_disc", theta=theta)
    nb_disc["n_circles_recovered"] = len({
        b["matched_planted_circle"] for b in nb_disc["per_block"]
        if b.get("best_planted_circle_corr", 0) > 0.8})
    nb_disc["discovery_diag"] = diag
    print(f"  discovered nursery composed EV={nb_disc['composed_ambient_ev']} "
          f"distinct_circles_recovered={nb_disc['n_circles_recovered']}/{K} "
          f"(fit_dim={nb_disc['total_fit_dim']})", flush=True)
    result["arms"]["B_nursery_discovered"] = nb_disc
    result["verdict"] = _verdict(result, meta)
    _save("synthetic_results.json", result)
    print(f"\n[VERDICT] {json.dumps(result['verdict'], indent=2)}", flush=True)
    return result


def _verdict(result, meta):
    A = result["arms"].get("A_joint", {})
    tj = A.get("torch_joint", {})
    ob = result["arms"].get("B_nursery_oracle", {})
    db = result["arms"].get("B_nursery_discovered", {})
    K = meta.get("ncirc")
    return {
        "joint_torch_ambient_ev": tj.get("ev"),
        "joint_torch_circles_recovered": f"{tj.get('n_circles_recovered')}/{K}",
        "joint_reml_status": A.get("reml_joint", {}).get("status"),
        "nursery_oracle_ev": ob.get("composed_ambient_ev"),
        "nursery_oracle_circles_recovered": f"{ob.get('n_circles_recovered')}/{K}",
        "nursery_discovered_ev": db.get("composed_ambient_ev"),
        "nursery_discovered_circles_recovered": f"{db.get('n_circles_recovered')}/{K}",
        "circle_subspace_ceiling": meta.get("circle_subspace_ev"),
        "nursery_fit_dim_per_block": ob.get("per_block", [{}])[0].get("block_dim"),
        "joint_fit_dim": meta.get("p"),
    }


# --------------------------------------------------------------------------- #
# REAL data (reuse curved_feature_probes harvest caches)
# --------------------------------------------------------------------------- #
PROBE_OUT = HERE / "probe_out"


def _load_real_layers(name: str):
    """Load a cached harvest set: per-template-demeaned activations at EVERY layer,
    plus rank / n_labels / cyclic. Returns (dict layer->X_demeaned, rank, n_labels)."""
    z = np.load(PROBE_OUT / f"harvest_{name}.npz", allow_pickle=False)
    layers = [int(x) for x in z["layers"]]
    tidx = z["template_idx"]; rank = z["rank"]
    demeaned = {}
    for L in layers:
        X = z[f"L{L}"].astype(np.float64)
        Xd = X.copy()
        for tt in np.unique(tidx):
            m = tidx == tt
            Xd[m] = X[m] - X[m].mean(0, keepdims=True)
        demeaned[L] = Xd
    return demeaned, layers, rank, int(z["n_labels"])


def _lin_score(X: np.ndarray, rank: np.ndarray) -> float:
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:8].T
    rr = rank - rank.mean()
    best = 0.0
    for k in range(proj.shape[1]):
        pk = proj[:, k] - proj[:, k].mean()
        d = np.sqrt((pk ** 2).sum() * (rr ** 2).sum())
        if d > 0:
            best = max(best, abs(float((pk * rr).sum() / d)))
    return best


def _cyclic_recovery(angle: np.ndarray, rk: np.ndarray, n_tok: int) -> tuple[float, float]:
    """(circular_corr, cyclic_adjacency_accuracy) of a recovered angle vs token order."""
    uniq = sorted(set(int(x) for x in rk.tolist()))
    tok_ang = np.array([circular_mean(angle[rk == u]) for u in uniq])
    true_ang = np.array([2 * np.pi * (u / n_tok) for u in uniq])
    cc = abs(circular_corr(tok_ang, true_ang))
    seq = list(np.argsort(tok_ang % (2 * np.pi)))
    true_adj = {frozenset((uniq[i], uniq[(i + 1) % len(uniq)])) for i in range(len(uniq))}
    rec_adj = {frozenset((uniq[seq[i]], uniq[seq[(i + 1) % len(seq)]])) for i in range(len(seq))}
    return round(cc, 3), round(len(true_adj & rec_adj) / len(uniq), 3)


def driver_real():
    print("=== REAL: nursery on cached weekday/month activations (SHARED ambient) ===",
          flush=True)
    sets = [s for s in ("weekday", "month") if (PROBE_OUT / f"harvest_{s}.npz").exists()]
    if len(sets) < 2:
        print(f"[real] need >=2 cached sets, have {sets}", flush=True)
        return
    # Load both sets at all layers; choose ONE SHARED layer (same feature space) that
    # maximizes the summed linear-structure score -- both circles must live in the
    # same ambient basis for this to be an honest shared-space multi-atom test.
    data = {s: _load_real_layers(s) for s in sets}
    common = sorted(set.intersection(*[set(data[s][1]) for s in sets]))
    best_L, best_score = common[0], -1.0
    for L in common:
        sc = sum(_lin_score(data[s][0][L], data[s][2]) for s in sets)
        if sc > best_score:
            best_L, best_score = L, sc
    print(f"[real] shared layer L{best_L} (summed lin-score {best_score:.2f})", flush=True)

    # Stack both sets' demeaned activations (896-dim, same layer) then joint-PCA to a
    # SHARED reduced ambient. Each set occupies a different 2-plane of this one space
    # (rows are different tokens/sentences), so the two circles genuinely SHARE the
    # ambient basis -- the realistic co-collapse regime, NOT block-diagonal.
    Xcat = np.concatenate([data[s][0][best_L] for s in sets], 0)
    rank_all = np.concatenate([data[s][2] for s in sets])
    set_all = np.concatenate([np.full(data[s][0][best_L].shape[0], si)
                              for si, s in enumerate(sets)])
    ntok = {si: data[s][3] for si, s in enumerate(sets)}
    Xc = Xcat - Xcat.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(int(os.environ.get("BLOCK_NURSERY_REAL_RDIM", "16")), Xc.shape[0] - 1)
    Xshared = Xc @ Vt[:r].T
    N, P = Xshared.shape
    K = len(sets)
    print(f"[real] shared ambient: N={N}, P={P}, sets={sets}", flush=True)
    result = {"sets": sets, "shared_layer": best_L, "shared_ambient": {"N": N, "P": P},
              "n_tokens": {s: data[s][3] for s in sets}}
    _save("real_results.json", result)

    # ---- Arm A: joint curved fit on the SHARED ambient (co-collapse regime) ----
    print(f"\n[Arm A] REML joint K={K} on shared ambient (expected hang/OOM)...", flush=True)
    reml = reml_joint_isolated(Xshared, K, tag="real_A", timeout=120)
    print(f"  REML joint: {reml.get('status')} ({reml.get('wall_s')}s)", flush=True)
    print(f"[Arm A] torch joint ManifoldSAE K={K}, target_k={K} on shared ambient...", flush=True)
    tj = fit_curved_isolated(Xshared, n_atoms=K, tag="real_A_torch", target_k=K)
    if tj["status"] == "CONVERGED":
        _, pos = _load_fit(tj["out_path"])
        # per-set: best atom's cyclic-adjacency recovery
        per_set = {}
        for si, s in enumerate(sets):
            rows = np.where(set_all == si)[0]
            best_adj, best_cc = 0.0, 0.0
            for a in range(K):
                cc, adj = _cyclic_recovery(2 * np.pi * pos[rows, a, 0],
                                           rank_all[rows].astype(int), ntok[si])
                if adj > best_adj:
                    best_adj, best_cc = adj, cc
            per_set[s] = {"best_atom_cyclic_adjacency": best_adj, "circular_corr": best_cc}
        tj["per_set_recovery"] = per_set
    print(f"  torch joint: ev={tj.get('ev')} gate_share={tj.get('gate_share')} "
          f"recovery={tj.get('per_set_recovery')}", flush=True)
    result["arm_A_joint"] = {"reml": reml, "torch_joint": tj}
    _save("real_results.json", result)

    # ---- Arm B: nursery. Blocks = each set's own 2-plane in the shared space. ----
    print("\n[Arm B] nursery: discover 2-plane per set -> K=1 chart -> compose...", flush=True)
    # oracle blocks: top-2 PCs of each set's rows WITHIN the shared space (its plane)
    ob = []
    accepted = np.zeros((P, 0))
    for si in range(K):
        rows = np.where(set_all == si)[0]
        Zc = Xshared[rows] - Xshared[rows].mean(0)
        _, _, Vk = np.linalg.svd(Zc, full_matrices=False)
        B = Vk[:2].T
        if accepted.shape[1] > 0:
            B = B - accepted @ (accepted.T @ B)
        Q, _ = np.linalg.qr(B); Q = Q[:, :2]
        ob.append(Q); accepted = np.concatenate([accepted, Q], 1)
    nb = run_nursery(Xshared, ob, tag="real_B")
    for si, s in enumerate(sets):
        rec = nb["per_block"][si]
        op = SCRATCH / f"real_B_b{si}_out.npz"
        if op.exists():
            _, pos = _load_fit(str(op))
            # which set does this block's chart order best? report all sets
            rows = np.where(set_all == si)[0]
            cc, adj = _cyclic_recovery(2 * np.pi * pos[rows, 0, 0],
                                       rank_all[rows].astype(int), ntok[si])
            rec["set"] = s
            rec["recovered_circular_corr"] = cc
            rec["cyclic_adjacency_accuracy"] = adj
    # also confirm discovery recovers the two planes without labels
    bb_disc, _, diag = discover_blocks(Xshared, n_dict=2 * K + 2, block_size=3)
    print(f"  discovery (unsupervised): {len(bb_disc)} blocks, dims={diag['block_dims']}",
          flush=True)
    nb["discovery_diag"] = diag
    print(f"  nursery composed EV={nb['composed_ambient_ev']} "
          f"(per-block fit_dim=2 vs joint P={P}); recovery="
          f"{[(b.get('set'), b.get('cyclic_adjacency_accuracy')) for b in nb['per_block']]}",
          flush=True)
    result["arm_B_nursery"] = nb
    result["verdict"] = {
        "joint_torch_ev": tj.get("ev"),
        "joint_reml_status": reml.get("status"),
        "joint_per_set_recovery": tj.get("per_set_recovery"),
        "nursery_composed_ev": nb["composed_ambient_ev"],
        "nursery_per_set_adjacency": {b.get("set"): b.get("cyclic_adjacency_accuracy")
                                      for b in nb["per_block"]},
        "nursery_fit_dim_per_block": 2,
        "joint_fit_dim": P,
        "joint_delivers_multiatom_model": True,
        "nursery_delivers_multiatom_model": True,
    }
    _save("real_results.json", result)
    print(f"\n[VERDICT] {json.dumps(result['verdict'], indent=2)}", flush=True)
    return result


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
