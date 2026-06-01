"""Consolidation experiment: tie gate + coordinate + curve to the SAME atom.

auto_exp_82 converged: the dictionary CONTAINS a read+steer handle for each factor
(0.72), but the atom that GATES isn't it (gate-winner reads only 0.44). The factor's
read / write / gate live in three different atoms. Fix = a TIED architecture: don't
learn a free gate-head and a free coordinate-head; instead read BOTH the coordinate
and the presence off the atom's OWN curve B_j. Then the firing atom is, by
construction, the reading atom and the steering atom.

Tied atom (H=1 circle, B_j = [center, v1, v2] in R^{3xD}):
  frame: orthonormal (e1,e2)=GS(v1,v2), center c0.
  read:  project (x - b_dec - c0) onto (e1,e2) -> (a,b); theta_j=atan2(b,a);
         presence gate_j = sigmoid(s*(|(a,b)| - thr)).  Both from B_j.
  write: decode = sum_j gate_j * (B_j evaluated at theta_j).

Compare gate-winner read & steer: free-head baseline (auto_exp_82) vs tied.
Hypothesis: tying lifts gate-winner read 0.44 -> ~0.8 (consolidated handle).
Unblocked. CPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.auto_exp_80_amortized_floor import plant_circles, circ_corr  # noqa: E402
from experiments.auto_exp_81_causal_gauge import behavior, T_RING  # noqa: E402
from experiments.auto_exp_82_steering_ceiling import (  # noqa: E402
    train as train_freehead, steer_matrix, read_matrix,
)
from manifold_sae.amortized_manifold_sae import _circle_basis  # noqa: E402

DEVICE = torch.device("cpu")


class ConsolidatedManifoldSAE(nn.Module):
    """Gate + coordinate both read off the atom's own curve B_j (tied)."""

    def __init__(self, D, F, gate_thr=0.5, sharp=8.0,
                 sparsity_w=1e-2, incoh_w=1e-2, iso_w=1e-2):
        super().__init__()
        self.B = nn.Parameter(torch.randn(F, 3, D) * (1.0 / math.sqrt(D)))
        self.b_dec = nn.Parameter(torch.zeros(D))
        self.gate_thr, self.sharp = gate_thr, sharp
        self.sparsity_w, self.incoh_w, self.iso_w = sparsity_w, incoh_w, iso_w
        self.cfg = SimpleNamespace(fourier_harmonics=1, dtype=torch.float32,
                                   n_atoms=F, input_dim=D)

    def _frame(self):
        c0, v1, v2 = self.B[:, 0], self.B[:, 1], self.B[:, 2]
        e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v2p = v2 - (v2 * e1).sum(-1, keepdim=True) * e1
        e2 = v2p / v2p.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return c0, e1, e2

    def encode(self, x):
        c0, e1, e2 = self._frame()
        xc = x.unsqueeze(1) - self.b_dec - c0.unsqueeze(0)   # (N,F,D)
        a = (xc * e1).sum(-1); b = (xc * e2).sum(-1)         # (N,F)
        theta = torch.atan2(b, a)
        mag = torch.sqrt(a * a + b * b + 1e-9)
        gate = torch.sigmoid(self.sharp * (mag - self.gate_thr))
        return gate, theta, mag

    def decode(self, gate, theta):
        phi = _circle_basis(theta, 1)                        # (N,F,3)
        curve = torch.einsum("nfm,fmd->nfd", phi, self.B)
        return torch.einsum("nf,nfd->nd", gate, curve) + self.b_dec

    def forward(self, x):
        gate, theta, _ = self.encode(x)
        return SimpleNamespace(x_hat=self.decode(gate, theta), gate=gate, theta=theta)

    def _incoherence(self):
        F = self.B.shape[0]
        rows = self.B.reshape(F * 3, -1)
        rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        g2 = (rows @ rows.t()).pow(2)
        aid = torch.arange(F * 3) // 3
        cross = aid.unsqueeze(0) != aid.unsqueeze(1)
        return (g2 * cross).sum() / cross.sum().clamp_min(1)

    def _isometry(self, G=32):
        th = torch.linspace(-math.pi, math.pi, G + 1)[:-1]
        dphi = torch.stack([torch.zeros_like(th), -torch.sin(th), torch.cos(th)], -1)
        dg = torch.einsum("gm,fmd->fgd", dphi, self.B)
        speed = dg.norm(dim=-1)
        mean = speed.mean(-1, keepdim=True).clamp_min(1e-8)
        return (speed / mean).var(-1).mean()

    def loss(self, x):
        gate, theta, mag = self.encode(x)
        recon = (self.decode(gate, theta) - x).pow(2).mean()
        sparsity = gate.mean()
        total = (recon + self.sparsity_w * sparsity
                 + self.incoh_w * self._incoherence() + self.iso_w * self._isometry())
        return {"loss": total, "recon": recon}


def train_consolidated(Xtr, D, F, epochs=45):
    torch.manual_seed(0)
    sae = ConsolidatedManifoldSAE(D, F).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=4e-3)
    for _ in range(epochs):
        for s in range(0, Xtr.shape[0], 512):
            opt.zero_grad(); sae.loss(Xtr[s:s + 512])["loss"].backward(); opt.step()
    sae.eval()
    return sae


def diagnose(sae, planes, ring, Xva, act_va, ang_va, K):
    S = steer_matrix(sae, planes, ring, Xva, K)        # (K,F)
    R, gw = read_matrix(sae, Xva, act_va, ang_va, K)   # (K,F), gate-winners
    with torch.no_grad():
        r2 = 1.0 - float((sae(Xva).x_hat - Xva).pow(2).mean()) / float(Xva.var())
    best_both = float(np.mean([np.minimum(R[k], S[k]).max() for k in range(K)]))
    gw_read = float(np.mean([R[k, gw[k]] for k in range(K)]))
    gw_steer = float(np.mean([S[k, gw[k]] for k in range(K)]))
    return {"val_r2": r2, "capability_best_both_one_atom": best_both,
            "consolidation_gatewinner_read": gw_read,
            "consolidation_gatewinner_steer": gw_steer}


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
    OUT = ROOT / "runs" / "CONSOLIDATION"; OUT.mkdir(parents=True, exist_ok=True)

    print("training free-head baseline ...", flush=True)
    free = train_freehead(Xtr, D, F, K, incoh_w=1e-2)
    print("training consolidated (tied) ...", flush=True)
    cons = train_consolidated(Xtr, D, F)

    results = {
        "free_head": diagnose(free, planes, ring, Xva, act_va, ang_va, K),
        "consolidated_tied": diagnose(cons, planes, ring, Xva, act_va, ang_va, K),
    }
    rf, rc = results["free_head"], results["consolidated_tied"]
    results["verdict_consolidation_fixes_gatewinner_read"] = bool(
        rc["consolidation_gatewinner_read"] > rf["consolidation_gatewinner_read"] + 0.25)
    results["verdict_gatewinner_is_handle"] = bool(
        min(rc["consolidation_gatewinner_read"], rc["consolidation_gatewinner_steer"]) > 0.7)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    for name in ("free_head", "consolidated_tied"):
        r = results[name]
        print(f"[{name:18s}] R2={r['val_r2']:.3f}  best_both(1 atom)="
              f"{r['capability_best_both_one_atom']:.2f}  GATE-WINNER read="
              f"{r['consolidation_gatewinner_read']:.2f} steer="
              f"{r['consolidation_gatewinner_steer']:.2f}", flush=True)
    print(f"[verdict] tying lifts gate-winner read (+0.25): "
          f"{results['verdict_consolidation_fixes_gatewinner_read']}  | "
          f"gate-winner IS the handle (read&steer>0.7): "
          f"{results['verdict_gatewinner_is_handle']}", flush=True)


if __name__ == "__main__":
    main()
