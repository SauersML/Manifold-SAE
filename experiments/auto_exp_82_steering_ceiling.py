"""Attack the steering ceiling (auto_exp_81 capped steer-transfer ≈ 0.42).

Hypothesis: steering is a DECODER-geometry property — moving an atom's coordinate
steers a factor iff (i) the atom's curve lies in that factor's plane and (ii) the
OTHER atoms don't project onto that plane (no interference). Both are governed by
cross-atom incoherence. So incoherence should be the lever that breaks the ceiling.

We sweep incoherence_weight (isometry on throughout) and measure, per planted
circle:
  * STEERING (gate-standardized, best atom over all): set an atom's gate to 1,
    sweep its θ, decode, run the behavior model, circ-corr(swept θ, behavior peak),
    per-token then averaged; take the best atom. Isolates curve geometry from the
    gate.
  * SUBSPACE ALIGNMENT: min over atoms of the largest principal angle between the
    atom's effective 2-plane (top-2 SVD of its ambient curve) and the true plane.
    cos→1 means some atom's curve sits in the factor's plane.

Verdict: does cranking incoherence push steering past 0.42 toward ≥0.8 and drive
subspace alignment →1? Unblocked (no gam joint solve, no cluster data).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.auto_exp_80_amortized_floor import plant_circles, circ_corr  # noqa: E402
from experiments.auto_exp_81_causal_gauge import behavior, T_RING  # noqa: E402
from manifold_sae.amortized_manifold_sae import (  # noqa: E402
    AmortizedManifoldSAE, AmortizedManifoldSAEConfig, _circle_basis,
)

DEVICE = torch.device("cpu")


def train(Xtr, D, F, K, incoh_w, epochs=45):
    torch.manual_seed(0)
    cfg = AmortizedManifoldSAEConfig(
        input_dim=D, n_atoms=F, fourier_harmonics=1,
        sparsity_weight=1e-2, incoherence_weight=incoh_w, isometry_weight=1e-2,
        gate_threshold=0.05, dtype=torch.float32)
    sae = AmortizedManifoldSAE(cfg).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=4e-3)
    for _ in range(epochs):
        for s in range(0, Xtr.shape[0], 512):
            opt.zero_grad(); sae.loss(Xtr[s:s + 512])["loss"].backward(); opt.step()
    sae.eval()
    return sae


def atom_curves(sae, grid=64):
    """Each atom's ambient curve over a θ grid -> (F, grid, D)."""
    H = sae.cfg.fourier_harmonics
    th = torch.linspace(-math.pi, math.pi, grid + 1, dtype=sae.cfg.dtype)[:-1]
    phi = _circle_basis(th, H)                              # (grid, M)
    return torch.einsum("gm,fmd->fgd", phi, sae.B.detach()) # (F, grid, D)


def subspace_align(sae, planes, K):
    """For each true plane, min over atoms of the top principal-angle cosine
    between the atom's effective 2-plane and the true plane (1 = perfectly in-plane)."""
    curves = atom_curves(sae)                               # (F, grid, D)
    F = curves.shape[0]
    out = []
    for k in range(K):
        Pk = planes[k]                                      # (2, D) orthonormal rows
        best = 0.0
        for j in range(F):
            c = curves[j] - curves[j].mean(0, keepdim=True)
            # top-2 right singular vectors = atom's effective plane
            _, _, Vt = torch.linalg.svd(c, full_matrices=False)
            Aj = Vt[:2]                                     # (2, D)
            # principal-angle cosines between subspaces span(Aj), span(Pk)
            s = torch.linalg.svdvals(Aj @ Pk.T)             # (2,)
            best = max(best, float(s.min()))                # smallest cos = worst angle
        out.append(best)
    return out


def steer_matrix(sae, planes, ring, Xva, K, n_tok=8, grid=24):
    """(K, F) gate-standardized steering score: for each (circle, atom)."""
    sweeps = torch.linspace(-math.pi, math.pi, grid + 1, dtype=sae.cfg.dtype)[:-1]
    with torch.no_grad():
        base = sae(Xva); gate0, theta0 = base.gate, base.theta
    F = theta0.shape[1]
    M = np.zeros((K, F))
    for j in range(F):
        g = gate0[:n_tok].repeat_interleave(grid, 0).clone()
        th = theta0[:n_tok].repeat_interleave(grid, 0).clone()
        g[:, j] = 1.0
        th[:, j] = sweeps.repeat(n_tok)
        with torch.no_grad():
            beh = behavior(sae.decode(g, th), planes, ring)        # (n_tok*grid, K, T)
        for k in range(K):
            b = beh[:, k, :]
            peak = torch.atan2((b * ring.sin()).sum(-1), (b * ring.cos()).sum(-1))
            peak = peak.reshape(n_tok, grid).numpy()
            M[k, j] = float(np.mean([circ_corr(sweeps.numpy(), peak[t]) for t in range(n_tok)]))
    return M


def read_matrix(sae, Xva, act_va, ang_va, K):
    """(K, F) coordinate-read alignment, and per-circle gate-winner atom."""
    with torch.no_grad():
        out = sae(Xva)
    theta = out.theta.cpu().numpy(); gate = out.gate.abs().cpu().numpy()
    F = theta.shape[1]
    R = np.zeros((K, F)); gate_winner = []
    for k in range(K):
        rows = act_va[:, k]
        gate_winner.append(int(gate[rows].mean(0).argmax()) if rows.sum() >= 10 else 0)
        for j in range(F):
            R[k, j] = circ_corr(ang_va[rows, k], theta[rows, j]) if rows.sum() >= 10 else 0.0
    return R, gate_winner


def main():
    torch.manual_seed(0)
    D, K, N, F = 128, 4, 6000, 16
    X, active, angles = plant_circles(D, K, N, sparsity=0.4, noise=0.03, seed=0)
    g = torch.Generator().manual_seed(0)
    planes = torch.zeros(K, 2, D)
    for k in range(K):
        M = torch.randn(2, D, generator=g); Q, _ = torch.linalg.qr(M.T); planes[k] = Q.T[:2]
    ring = torch.linspace(-math.pi, math.pi, T_RING + 1)[:-1]
    Xtr, Xva = X[:5000], X[5000:]
    act_va, ang_va = active[5000:].numpy(), angles[5000:].numpy()
    OUT = ROOT / "runs" / "STEERING_CEILING"; OUT.mkdir(parents=True, exist_ok=True)

    results = {}
    for w in (1e-2, 1e-1, 1.0):
        print(f"training incoherence_weight={w} ...", flush=True)
        sae = train(Xtr, D, F, K, w)
        S = steer_matrix(sae, planes, ring, Xva, K)       # (K,F)
        R, gw = read_matrix(sae, Xva, act_va, ang_va, K)  # (K,F), gate-winners
        al = subspace_align(sae, planes, K)
        with torch.no_grad():
            r2 = 1.0 - float((sae(Xva).x_hat - Xva).pow(2).mean()) / float(Xva.var())
        # capability: is there ANY atom that reads / steers / does BOTH, per circle?
        best_read = [float(R[k].max()) for k in range(K)]
        best_steer = [float(S[k].max()) for k in range(K)]
        best_both = [float(np.minimum(R[k], S[k]).max()) for k in range(K)]  # one atom, both
        # consolidation: does the atom that GATES (fires) for the circle read & steer it?
        gw_read = [float(R[k, gw[k]]) for k in range(K)]
        gw_steer = [float(S[k, gw[k]]) for k in range(K)]
        results[f"incoh_{w:g}"] = {
            "val_r2": r2,
            "capability_best_read": float(np.mean(best_read)),
            "capability_best_steer": float(np.mean(best_steer)),
            "capability_best_both_one_atom": float(np.mean(best_both)),
            "consolidation_gatewinner_read": float(np.mean(gw_read)),
            "consolidation_gatewinner_steer": float(np.mean(gw_steer)),
            "mean_subspace_align": float(np.mean(al))}
        r = results[f"incoh_{w:g}"]
        print(f"  incoh={w:<5g} R2={r2:.3f} | CAPABILITY best_read={r['capability_best_read']:.2f} "
              f"best_steer={r['capability_best_steer']:.2f} best_BOTH(1 atom)="
              f"{r['capability_best_both_one_atom']:.2f} | CONSOLIDATION gate-winner "
              f"read={r['consolidation_gatewinner_read']:.2f} steer={r['consolidation_gatewinner_steer']:.2f}",
              flush=True)

    best_w = max(results, key=lambda kk: results[kk]["capability_best_both_one_atom"])
    rb = results[best_w]
    results["verdict_consolidated_atom_exists"] = bool(rb["capability_best_both_one_atom"] > 0.7)
    results["verdict_gatewinner_is_the_handle"] = bool(
        min(rb["consolidation_gatewinner_read"], rb["consolidation_gatewinner_steer"]) > 0.7)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n[diagnosis] A single atom that both reads AND steers a factor EXISTS: "
          f"{results['verdict_consolidated_atom_exists']} "
          f"(best_both={rb['capability_best_both_one_atom']:.2f})", flush=True)
    print(f"[diagnosis] The atom that FIRES for the factor IS that handle (consolidated): "
          f"{results['verdict_gatewinner_is_the_handle']} "
          f"(gate-winner read={rb['consolidation_gatewinner_read']:.2f}, "
          f"steer={rb['consolidation_gatewinner_steer']:.2f})", flush=True)


if __name__ == "__main__":
    main()
