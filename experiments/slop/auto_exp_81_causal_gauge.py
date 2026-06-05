"""Gauge-pinning experiment: does a CAUSAL/behavior signal recover the coordinate
where reconstruction (auto_exp_80) could not?

auto_exp_80 showed: recon R²≈0.93 but coordinate alignment ≈0.3 even at one atom
per circle — reconstruction does not pin the coordinate gauge. Thesis: the gauge
is pinned by causality — the coordinate must be the variable whose value drives
the model's downstream BEHAVIOR along that factor's behavior manifold (the
representation-manifold ↔ behavior-manifold correspondence of the steering
papers). Here we ground that with a known behavior model so it's fully
falsifiable and needs nothing blocked (no gam joint solve, no cluster data).

Setup: plant K circles in R^D. A fixed "behavior model" reads each circle's angle
from any activation and emits a ring-softmax over T behavior tokens, peaked at the
true angle (rep-circle → behavior-circle). Three SAEs, identical except:
  (1) recon-only      : recon + IBP sparsity + incoherence            (= floor)
  (2) + isometry      : (1) + arc-length gauge on the metric
  (3) + causal pin    : (2) + behavior-prediction through a rotation-only readout
                        (the coordinates must explain behavior using only a
                        per-circle atom-selection + phase — no reparametrization,
                        so θ is forced toward the true angle).
Metrics: recon R²; coordinate recovery (circ-corr best-atom θ vs true angle);
held-out STEERING TRANSFER (sweep the atom's θ, decode, run the behavior model,
check the behavior peak follows) — the causal prize. Plus emergence: pin only
circle 0, check the other circles' coordinates recover unpinned.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.slop.auto_exp_80_amortized_floor import plant_circles, circ_corr  # noqa: E402
from manifold_sae.amortized_manifold_sae import (  # noqa: E402
    AmortizedManifoldSAE,
    AmortizedManifoldSAEConfig,
)

DEVICE = torch.device("cpu")  # gam penalties are CPU/float64-only (gam#362)
T_RING = 12
KAPPA = 4.0


def behavior(v: torch.Tensor, planes: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """Fixed 'model': read each circle's angle from activation v, emit a ring
    softmax (N,K,T) peaked at that angle. Stands in for the LLM's factor-dependent
    output distribution; rep-circle drives behavior-circle."""
    proj = torch.einsum("nd,kcd->nkc", v, planes)              # (N,K,2)
    ang = torch.atan2(proj[..., 1], proj[..., 0])              # (N,K)
    diff = ring.view(1, 1, -1) - ang.unsqueeze(-1)             # (N,K,T)
    return torch.softmax(KAPPA * torch.cos(diff), dim=-1)


class PinReadout(nn.Module):
    """Predict behavior from the SAE coordinates using ONLY a per-circle soft
    atom-selection + learned phase — a rotation-class map that cannot
    nonlinearly reparametrize θ. So driving the pin loss down forces some atom's
    θ to BE the true angle (up to rotation/reflection), which is the gauge pin."""

    def __init__(self, F: int, K: int):
        super().__init__()
        self.sel = nn.Parameter(torch.zeros(K, F))   # atom-selection logits per circle
        self.phi = nn.Parameter(torch.zeros(K))      # per-circle phase

    def forward(self, theta: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
        # Straight-through ONE-HOT atom selection per circle: behavior must be
        # predicted from a SINGLE atom's angle (+ phase), never an average — so
        # the only way to fit it is for that atom's θ to BE the true angle.
        w_soft = torch.softmax(self.sel, dim=-1)               # (K,F)
        hard = torch.zeros_like(w_soft).scatter_(-1, w_soft.argmax(-1, keepdim=True), 1.0)
        w = hard + w_soft - w_soft.detach()                    # STE one-hot
        u = torch.stack([torch.cos(theta), torch.sin(theta)], -1)  # (N,F,2)
        d = torch.einsum("kf,nfc->nkc", w, u)                  # (N,K,2) = selected atom's unit vec
        peak = torch.atan2(d[..., 1], d[..., 0]) + self.phi.view(1, -1)  # (N,K)
        diff = ring.view(1, 1, -1) - peak.unsqueeze(-1)
        return torch.softmax(KAPPA * torch.cos(diff), dim=-1)  # (N,K,T)


def make_cfg(D, F, use_iso):
    return AmortizedManifoldSAEConfig(
        input_dim=D, n_atoms=F, fourier_harmonics=1,
        sparsity_weight=1e-2, incoherence_weight=1e-2,
        isometry_weight=(1e-2 if use_iso else 0.0),
        gate_threshold=0.05, dtype=torch.float32)


def _peak(beh, ring):
    """(n,K,T) behavior dists -> (n,K) circular-mean peak angles."""
    return torch.atan2((beh * ring.sin()).sum(-1), (beh * ring.cos()).sum(-1))


def train(X, beh, ring, planes, D, F, K, use_iso, use_pin, pin_circles,
          intervene=False, epochs=45):
    torch.manual_seed(0)
    sae = AmortizedManifoldSAE(make_cfg(D, F, use_iso)).to(DEVICE)
    readout = PinReadout(F, K).to(DEVICE) if use_pin else None
    params = list(sae.parameters()) + (list(readout.parameters()) if readout else [])
    opt = torch.optim.Adam(params, lr=4e-3)
    N = X.shape[0]
    for ep in range(epochs):
        do_intervene = intervene and ep >= epochs // 2  # warmup: reconstruct first
        for s in range(0, N, 512):
            xb, bb = X[s:s + 512], beh[s:s + 512]
            opt.zero_grad()
            loss = sae.loss(xb)["loss"]
            if use_pin:
                gate, theta, _ = sae.encode(xb)
                pred = readout(theta, ring)                    # (n,K,T)
                kk = pin_circles if pin_circles is not None else list(range(K))
                ce = -(bb[:, kk] * (pred[:, kk] + 1e-9).log()).sum(-1).mean()
                loss = loss + 1.0 * ce
                if do_intervene:
                    # INTERVENTION pin: steering the circle's atom by Δ must move
                    # the behavior peak by Δ (pins the decoder's WRITE to the
                    # factor's plane → steerability, not just read-alignment).
                    sel = readout.sel.argmax(-1)               # (K,) atom per circle
                    base_pk = _peak(behavior(sae.decode(gate, theta), planes, ring), ring)
                    delta = float((torch.rand(1).item() * 2 - 1) * math.pi)
                    iloss = 0.0
                    for k in kk:
                        th = theta.clone(); th[:, sel[k]] = th[:, sel[k]] + delta
                        pk = _peak(behavior(sae.decode(gate, th), planes, ring), ring)[:, k]
                        moved = pk - base_pk[:, k]             # observed behavior shift
                        iloss = iloss + (1.0 - torch.cos(moved - delta)).mean()
                    loss = loss + 0.1 * (iloss / max(len(kk), 1))  # gentle, post-warmup
            loss.backward(); opt.step()
    sae.eval()
    return sae, readout


def coord_recovery(sae, Xva, act_va, ang_va, K):
    with torch.no_grad():
        out = sae(Xva)
    theta = out.theta.cpu().numpy()
    F = theta.shape[1]
    aligns, best_atoms = [], []
    for k in range(K):
        rows = act_va[:, k]
        if rows.sum() < 10:
            aligns.append(0.0); best_atoms.append(0); continue
        # Assignment-agnostic: best over ALL atoms (does ANY atom track circle k?).
        cc = [circ_corr(ang_va[rows, k], theta[rows, j]) for j in range(F)]
        j = int(np.argmax(cc))
        best_atoms.append(j)
        aligns.append(cc[j])
    return aligns, best_atoms


def steering_transfer(sae, planes, ring, Xva, act_va, best_atoms, K, n_tok=12, grid=24):
    """Causal test: for each circle k, sweep its atom's θ on held-out tokens,
    decode the steered activation, run the behavior model, and check the
    behavior peak for circle k follows the swept angle (circ-corr)."""
    sweeps = torch.linspace(-math.pi, math.pi, grid + 1)[:-1]
    scores = []
    with torch.no_grad():
        base = sae(Xva)
        gate0, theta0 = base.gate, base.theta
        for k in range(K):
            rows = np.where(act_va[:, k])[0][:n_tok]
            if len(rows) < 3:
                scores.append(0.0); continue
            j = best_atoms[k]
            peaks_all, swept_all = [], []
            for r in rows:
                g = gate0[r:r + 1].repeat(grid, 1)
                th = theta0[r:r + 1].repeat(grid, 1).clone()
                th[:, j] = sweeps
                steered = sae.decode(g, th)                    # (grid, D)
                b = behavior(steered, planes, ring)[:, k, :]   # (grid, T)
                peak = torch.atan2((b * ring.sin()).sum(-1), (b * ring.cos()).sum(-1))
                peaks_all.append(peak.numpy()); swept_all.append(sweeps.numpy())
            scores.append(circ_corr(np.concatenate(swept_all), np.concatenate(peaks_all)))
    return scores


def main():
    torch.manual_seed(0)
    D, K, N, F = 128, 4, 6000, 16
    X, active, angles = plant_circles(D, K, N, sparsity=0.4, noise=0.03, seed=0)
    g = torch.Generator().manual_seed(0)
    planes = torch.zeros(K, 2, D)
    for k in range(K):  # SAME planting RNG as plant_circles so planes match the data
        M = torch.randn(2, D, generator=g); Q, _ = torch.linalg.qr(M.T); planes[k] = Q.T[:2]
    ring = torch.linspace(-math.pi, math.pi, T_RING + 1)[:-1]
    beh = behavior(X, planes, ring)                            # (N,K,T)
    ntr = 5000
    Xtr, Xva, btr = X[:ntr], X[ntr:], beh[:ntr]
    act_va, ang_va = active[ntr:].numpy(), angles[ntr:].numpy()
    var = float(Xva.var())
    OUT = ROOT / "runs" / "CAUSAL_GAUGE"; OUT.mkdir(parents=True, exist_ok=True)

    def evaluate(sae):
        with torch.no_grad():
            r2 = 1.0 - float((sae(Xva).x_hat - Xva).pow(2).mean()) / var
        al, ba = coord_recovery(sae, Xva, act_va, ang_va, K)
        st = steering_transfer(sae, planes, ring, Xva, act_va, ba, K)
        return {"val_r2": r2, "coord_align": al, "mean_align": float(np.mean(al)),
                "steer_transfer": st, "mean_steer": float(np.mean(st))}

    results = {}
    print("training recon-only ...", flush=True)
    sae1, _ = train(Xtr, btr, ring, planes, D, F, K, use_iso=False, use_pin=False, pin_circles=None)
    results["recon_only"] = evaluate(sae1)
    print("training +isometry ...", flush=True)
    sae2, _ = train(Xtr, btr, ring, planes, D, F, K, use_iso=True, use_pin=False, pin_circles=None)
    results["plus_isometry"] = evaluate(sae2)
    print("training +predictive_pin ...", flush=True)
    sae3, _ = train(Xtr, btr, ring, planes, D, F, K, use_iso=True, use_pin=True, pin_circles=None)
    results["plus_predictive_pin"] = evaluate(sae3)
    print("training +interventional_pin ...", flush=True)
    sae3b, _ = train(Xtr, btr, ring, planes, D, F, K, use_iso=True, use_pin=True,
                     pin_circles=None, intervene=True)
    results["plus_interventional_pin"] = evaluate(sae3b)
    print("training emergence (interventional, pin circle 0 only) ...", flush=True)
    sae4, _ = train(Xtr, btr, ring, planes, D, F, K, use_iso=True, use_pin=True,
                    pin_circles=[0], intervene=True)
    em = evaluate(sae4)
    results["emergence_pin_circle0"] = {
        **em, "pinned_circle_align": em["coord_align"][0],
        "free_circles_mean_align": float(np.mean(em["coord_align"][1:]))}

    rpred = results["plus_predictive_pin"]
    rint = results["plus_interventional_pin"]
    results["verdict_coordinate_recovered"] = bool(rint["mean_align"] > 0.8)
    results["verdict_steerable"] = bool(rint["mean_steer"] > 0.8)
    results["verdict_intervention_beats_prediction_on_steer"] = bool(
        rint["mean_steer"] > rpred["mean_steer"] + 0.3)
    results["verdict_free_emerges"] = bool(
        results["emergence_pin_circle0"]["free_circles_mean_align"] > 0.6)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    for name in ("recon_only", "plus_isometry", "plus_predictive_pin",
                 "plus_interventional_pin"):
        r = results[name]
        print(f"[{name:24s}] R2={r['val_r2']:.3f}  coord_align={r['mean_align']:.3f}  "
              f"steer_transfer={r['mean_steer']:.3f}", flush=True)
    e = results["emergence_pin_circle0"]
    print(f"[emergence] pinned circle align={e['pinned_circle_align']:.3f}  "
          f"free circles align={e['free_circles_mean_align']:.3f}", flush=True)
    print(f"[verdict] coord recovered (>0.8): {results['verdict_coordinate_recovered']}  "
          f"| steerable (>0.8): {results['verdict_steerable']}  "
          f"| intervention>prediction on steer (+0.3): "
          f"{results['verdict_intervention_beats_prediction_on_steer']}  "
          f"| free emerges (>0.6): {results['verdict_free_emerges']}", flush=True)


if __name__ == "__main__":
    main()
