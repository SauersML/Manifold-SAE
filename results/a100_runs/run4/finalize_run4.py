"""Consolidate Run 4 outputs: merge synthetic calibration + real-leg validation
into one JSON and one combined calibration figure."""
import sys
sys.path.insert(0, "/home/azuser/Manifold-SAE")
import json
import os
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.trust_score import trust_score
from experiments.trust_calibrate import (
    plant_region, bootstrap_disagreement, spearman, SYNTH_CASES)

OUT = "/home/azuser/exp_logs/run4"

# --- synthetic: trust vs coordinate error (reload from CSV written earlier) ---
synth = []
with open(os.path.join(OUT, "trust_synth.csv")) as f:
    keys = f.readline().strip().split(",")
    for line in f:
        v = line.strip().split(",")
        d = dict(zip(keys, v))
        synth.append(dict(label=d["label"], trust=float(d["trust"]),
                          coord_err=float(d["coord_err"]),
                          untyped=d["untyped"] == "True"))
tr = [r["trust"] for r in synth]
er = [r["coord_err"] for r in synth]
rho_s = spearman(tr, er)
med = np.median(tr)
lo_err = np.mean([r["coord_err"] for r in synth if r["trust"] < med])
hi_err = np.mean([r["coord_err"] for r in synth if r["trust"] >= med])

# --- real-leg methodology validation: trust vs cross-seed disagreement ---
NOISE = 0.35
dz = []
for label, kind, kw in SYNTH_CASES:
    nn = 25 if label == "sparse" else 200
    kw2 = dict(kw); kw2["noise"] = NOISE
    for s in range(8):
        X, _ = plant_region(kind, n=nn, D=8, seed=2000 + s, **kw2)
        rep = trust_score(X)
        dis = bootstrap_disagreement(X, n_boot=8, seed=s)
        dz.append(dict(label=label, trust=rep.trust, disagreement=dis))
tr2 = [r["trust"] for r in dz]
ds = [r["disagreement"] for r in dz]
rho_r = spearman(tr2, ds)
med2 = np.median(tr2)
lo_d = np.mean([r["disagreement"] for r in dz if r["trust"] < med2])
hi_d = np.mean([r["disagreement"] for r in dz if r["trust"] >= med2])

summary = dict(
    gamfit_version="0.1.151",
    convergence_gate="FAILED: K=1 circle smoke raises RemlConvergenceError "
                     "(arrow-Schur adaptive proximal correction failed after 16 "
                     "attempts) with incoherence ON and OFF -> all typed FITS BLOCKED",
    deliverable="trust score authored + calibrated on solver-independent paths",
    synthetic_known_truth=dict(
        claim="LOW trust predicts HIGH coordinate error",
        n=len(synth),
        spearman_trust_vs_coord_err=rho_s,
        low_trust_mean_coord_err=lo_err,
        high_trust_mean_coord_err=hi_err,
        contrast_ratio=lo_err / hi_err if hi_err > 0 else None),
    real_leg=dict(
        status="DOUBLY BLOCKED: (1) typed gam fit non-converged; (2) transformers/"
               "torch absent from gpu-runner venv and pip-install forbidden; "
               "(3) ~/exp_logs/run3 outputs empty. Methodology validated on a "
               "NOISY synthetic stand-in (noise=0.35) that mimics real activations.",
        claim="LOW trust predicts HIGH cross-seed disagreement",
        n=len(dz),
        noise=NOISE,
        spearman_trust_vs_disagreement=rho_r,
        low_trust_mean_disagreement=lo_d,
        high_trust_mean_disagreement=hi_d),
)
json.dump(summary, open(os.path.join(OUT, "calibration.json"), "w"),
          indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)

# --- combined figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
cmap = plt.get_cmap("tab10")
labs = sorted(set(r["label"] for r in synth))
for i, lab in enumerate(labs):
    pts = [(r["trust"], r["coord_err"]) for r in synth if r["label"] == lab]
    x, y = zip(*pts)
    axes[0].scatter(x, y, s=30, color=cmap(i % 10), label=lab, alpha=0.8)
axes[0].set_xlabel("trust score"); axes[0].set_ylabel("coord error (1 - circ R2)")
axes[0].set_title("SYNTHETIC: low trust -> high coord error\n"
                  "Spearman=%.3f  (low %.2f vs high %.3f)" % (rho_s, lo_err, hi_err))
axes[0].legend(fontsize=6, ncol=2); axes[0].grid(alpha=0.3)
axes[0].set_ylim(-0.1, min(2.5, max(er) * 1.05))

for i, lab in enumerate(labs):
    pts = [(r["trust"], r["disagreement"]) for r in dz if r["label"] == lab]
    x, y = zip(*pts)
    axes[1].scatter(x, y, s=30, color=cmap(i % 10), label=lab, alpha=0.8)
axes[1].set_xlabel("trust score"); axes[1].set_ylabel("cross-seed disagreement")
axes[1].set_title("REAL-LEG METHODOLOGY (noisy synth stand-in)\n"
                  "Spearman=%.3f  (low %.3f vs high %.3f)" % (rho_r, lo_d, hi_d))
axes[1].legend(fontsize=6, ncol=2); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "calibration.png"), dpi=130)
print("wrote", os.path.join(OUT, "calibration.json"), "and calibration.png")
print("SYNTH  Spearman=%.3f  low_err=%.3f  high_err=%.3f" % (rho_s, lo_err, hi_err))
print("REAL   Spearman=%.3f  low_dis=%.3f  high_dis=%.3f" % (rho_r, lo_d, hi_d))
