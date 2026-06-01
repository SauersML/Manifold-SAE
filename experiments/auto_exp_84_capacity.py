"""Capacity sweep: is the consolidation gap an OVERCOMPLETENESS problem?

auto_exp_82/83: a clean read+steer handle exists per factor (0.72), but the atom
that GATES isn't it (gate-winner read 0.44), and tying didn't fix it. Hypothesis:
with F >> K, factors split across atoms and the most-firing atom is a promiscuous
overlap atom, not the clean handle. Test: sweep F around K=4 and watch the
gate-winner read/steer.

Stakes: if consolidation only holds at F≈K, then in the massively-overcomplete
regime real LLM SAEs require, single-atom interpretation is impossible and the
unit must be a GROUP of atoms (the Goodfire conclusion) — even with manifold
atoms. If it holds at F>>K too, capacity-matching (ARD) is the lever.
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
from experiments.auto_exp_82_steering_ceiling import (  # noqa: E402
    train as train_freehead, steer_matrix, read_matrix,
)

DEVICE = torch.device("cpu")


def diagnose(sae, planes, ring, Xva, act_va, ang_va, K):
    S = steer_matrix(sae, planes, ring, Xva, K)
    R, gw = read_matrix(sae, Xva, act_va, ang_va, K)
    with torch.no_grad():
        r2 = 1.0 - float((sae(Xva).x_hat - Xva).pow(2).mean()) / float(Xva.var())
    best_both = float(np.mean([np.minimum(R[k], S[k]).max() for k in range(K)]))
    gw_read = float(np.mean([R[k, gw[k]] for k in range(K)]))
    gw_steer = float(np.mean([S[k, gw[k]] for k in range(K)]))
    return r2, best_both, gw_read, gw_steer


def main():
    torch.manual_seed(0)
    D, K, N = 128, 4, 6000
    X, active, angles = plant_circles(D, K, N, sparsity=0.4, noise=0.03, seed=0)
    g = torch.Generator().manual_seed(0)
    planes = torch.zeros(K, 2, D)
    for k in range(K):
        M = torch.randn(2, D, generator=g); Q, _ = torch.linalg.qr(M.T); planes[k] = Q.T[:2]
    ring = torch.linspace(-math.pi, math.pi, T_RING + 1)[:-1]
    Xtr, Xva = X[:5000], X[5000:]
    act_va, ang_va = active[5000:].numpy(), angles[5000:].numpy()
    OUT = ROOT / "runs" / "CAPACITY"; OUT.mkdir(parents=True, exist_ok=True)

    results = {}
    for F in (4, 6, 8, 16, 32):
        print(f"training F={F} (K={K}) ...", flush=True)
        sae = train_freehead(Xtr, D, F, K, incoh_w=1e-2)
        r2, bb, gwr, gws = diagnose(sae, planes, ring, Xva, act_va, ang_va, K)
        results[f"F_{F}"] = {"overcomplete_ratio": F / K, "val_r2": r2,
                             "capability_best_both": bb,
                             "gatewinner_read": gwr, "gatewinner_steer": gws,
                             "consolidated": bool(min(gwr, gws) > 0.7)}
        print(f"  F={F:<3d} (x{F/K:.0f})  R2={r2:.3f}  best_both={bb:.2f}  "
              f"GATE-WINNER read={gwr:.2f} steer={gws:.2f}  "
              f"consolidated={results[f'F_{F}']['consolidated']}", flush=True)

    # Does the gate-winner consolidate at matched capacity but dilute when overcomplete?
    gwr_matched = results[f"F_{K}"]["gatewinner_read"]
    gwr_over = results["F_32"]["gatewinner_read"]
    results["verdict_capacity_is_the_lever"] = bool(gwr_matched > gwr_over + 0.2)
    results["verdict_consolidates_at_matched_F"] = results[f"F_{K}"]["consolidated"]
    results["verdict_holds_when_overcomplete"] = results["F_32"]["consolidated"]
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n[verdict] gate-winner consolidates at F=K but dilutes at F=8K: "
          f"{results['verdict_capacity_is_the_lever']} "
          f"(read {gwr_matched:.2f} @F={K} vs {gwr_over:.2f} @F=32)", flush=True)
    print(f"[verdict] single-atom interpretable at matched F={K}: "
          f"{results['verdict_consolidates_at_matched_F']}  | "
          f"holds when overcomplete (F=32): {results['verdict_holds_when_overcomplete']}",
          flush=True)
    if not results["verdict_holds_when_overcomplete"]:
        print("[implication] overcomplete regime -> single-atom interp fails -> "
              "unit of interpretation must be a GROUP of atoms (Goodfire).", flush=True)


if __name__ == "__main__":
    main()
