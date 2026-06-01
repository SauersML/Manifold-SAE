"""Plot the session's findings (toy/synthetic — illustrative, not truth)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.auto_exp_80_amortized_floor import plant_circles, circ_corr  # noqa: E402
from experiments.auto_exp_82_steering_ceiling import train as train_freehead  # noqa: E402


def load(name):
    return json.loads((ROOT / "runs" / name / "metrics.json").read_text())


fig, ax = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Manifold-SAE session findings  (synthetic K=4 circles in R$^{128}$ — illustrative, not truth)",
             fontsize=13, fontweight="bold")

# ---- Panel A: capability vs consolidation (the core finding) ----
cap = load("CAPACITY")["F_16"]
a = ax[0, 0]
labels = ["best-read\n(any atom)", "best-steer\n(any atom)", "read+steer\n(one atom)",
          "GATE-WINNER\nread", "GATE-WINNER\nsteer"]
vals = [cap["gatewinner_read"], cap["gatewinner_steer"], cap["capability_best_both"],
        cap["gatewinner_read"], cap["gatewinner_steer"]]
# use STEERING_CEILING incoh_0.01 for the capability trio (richer)
sc = load("STEERING_CEILING")["incoh_0.01"]
vals = [sc["capability_best_read"], sc["capability_best_steer"], sc["capability_best_both_one_atom"],
        sc["consolidation_gatewinner_read"], sc["consolidation_gatewinner_steer"]]
colors = ["#2a7", "#2a7", "#2a7", "#c33", "#c33"]
a.bar(labels, vals, color=colors)
a.axhline(0.8, ls="--", c="gray", lw=1)
a.set_ylim(0, 1); a.set_ylabel("score (circular corr / capability)")
a.set_title("A. The handle EXISTS (green) but the FIRING atom isn't it (red)\n"
            "→ read/steer/gate live in different atoms = dilution", fontsize=10)
for i, v in enumerate(vals):
    a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

# ---- Panel B: capacity sweep (overcompleteness hypothesis falsified) ----
capr = load("CAPACITY")
Fs = sorted([int(k.split("_")[1]) for k in capr if k.startswith("F_")])
gwr = [capr[f"F_{f}"]["gatewinner_read"] for f in Fs]
gws = [capr[f"F_{f}"]["gatewinner_steer"] for f in Fs]
bb = [capr[f"F_{f}"]["capability_best_both"] for f in Fs]
b = ax[0, 1]
b.plot(Fs, gwr, "o-", c="#c33", label="gate-winner READ")
b.plot(Fs, gws, "s-", c="#e8a", label="gate-winner steer")
b.plot(Fs, bb, "^-", c="#2a7", label="best read+steer (any atom)")
b.axvline(4, ls=":", c="gray"); b.text(4.2, 0.05, "F = K", fontsize=8, color="gray")
b.axhline(0.8, ls="--", c="gray", lw=1)
b.set_xlabel("dictionary size F  (K=4 true factors)"); b.set_ylabel("score")
b.set_ylim(0, 1); b.legend(fontsize=8, loc="center right")
b.set_title("B. Capacity isn't the lever: gate-winner read is flat-bad\n"
            "at every F, even matched F=K", fontsize=10)

# ---- Panel C: causal-gauge — pins raise READ, not STEER ----
cg = load("CAUSAL_GAUGE")
models = ["recon_only", "plus_isometry", "plus_predictive_pin", "plus_interventional_pin"]
short = ["recon", "+isom", "+pred\npin", "+interv\npin"]
r2 = [cg[m]["val_r2"] for m in models]
al = [cg[m]["mean_align"] for m in models]
st = [cg[m]["mean_steer"] for m in models]
c = ax[1, 0]
x = np.arange(len(models)); w = 0.27
c.bar(x - w, r2, w, label="reconstruction R²", color="#88a")
c.bar(x, al, w, label="coordinate READ", color="#2a7")
c.bar(x + w, st, w, label="STEER transfer", color="#c33")
c.set_xticks(x); c.set_xticklabels(short, fontsize=9); c.set_ylim(0, 1)
c.legend(fontsize=8); c.set_ylabel("score")
c.set_title("C. Behavior pins lift READ (green) but not STEER (red);\n"
            "isometry is what moved steer; interv. pin hurt recon", fontsize=10)

# ---- Panel D: what the read divergence looks like (retrain one model) ----
torch.manual_seed(0)
D, K, N, F = 128, 4, 6000, 16
X, active, angles = plant_circles(D, K, N, sparsity=0.4, noise=0.03, seed=0)
Xva = X[5000:]; act = active[5000:].numpy(); ang = angles[5000:].numpy()
sae = train_freehead(X[:5000], D, F, K, incoh_w=1e-2)
with torch.no_grad():
    o = sae(Xva)
th = o.theta.numpy(); g = o.gate.abs().numpy()
k = 0
rows = act[:, k]
cc = [circ_corr(ang[rows, k], th[rows, j]) for j in range(F)]
best_atom = int(np.argmax(cc))
gate_winner = int(g[rows].mean(0).argmax())
d = ax[1, 1]
d.scatter(ang[rows, k], th[rows, best_atom], s=6, alpha=0.4, c="#2a7",
          label=f"best-aligned atom #{best_atom}  (corr {cc[best_atom]:.2f})")
d.scatter(ang[rows, k], th[rows, gate_winner], s=6, alpha=0.4, c="#c33",
          label=f"GATE-WINNER atom #{gate_winner}  (corr {cc[gate_winner]:.2f})")
d.set_xlabel("true factor angle (rad)"); d.set_ylabel("atom's read coordinate θ (rad)")
d.legend(fontsize=8, loc="upper left")
d.set_title("D. Circle-0: the best-aligned atom tracks the factor (green),\n"
            "the atom that FIRES for it does not (red)", fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = ROOT / "runs" / "FINDINGS_PLOT.png"
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")
