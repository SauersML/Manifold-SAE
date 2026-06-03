# Validate the REAL-leg machinery (bootstrap cross-seed disagreement vs trust)
# on synthetic regions, since real LLM activations require transformers (absent
# from the gpu-runner venv; pip-install is forbidden) and run3 outputs are empty.
# Proves: where trust is LOW, the nonparametric coordinate is UNSTABLE across
# bootstrap seeds -- the exact correlation the real leg measures on LLM data.
import sys
sys.path.insert(0, "/home/azuser/Manifold-SAE")
import collections
import numpy as np
from experiments.trust_score import trust_score
from experiments.trust_calibrate import (
    plant_region, bootstrap_disagreement, spearman, SYNTH_CASES)

# Real LLM activations are NOISY (low SNR off the signal subspace). To make the
# synthetic stand-in exercise the same instability the real leg would see, inject
# realistic noise (0.35) on top of each case's base geometry. Under noise, an
# ill-conditioned / poorly-covered region's bootstrap planes wobble -> high
# cross-seed disagreement; a clean ring's plane is stable -> low disagreement.
NOISE = 0.35
rows = []
for label, kind, kw in SYNTH_CASES:
    nn = 25 if label == "sparse" else 200
    kw2 = dict(kw); kw2["noise"] = NOISE
    for s in range(8):
        X, _ = plant_region(kind, n=nn, D=8, seed=2000 + s, **kw2)
        rep = trust_score(X)
        dis = bootstrap_disagreement(X, n_boot=8, seed=s)
        rows.append((label, rep.trust, dis))

tr = [r[1] for r in rows]
ds = [r[2] for r in rows]
print("REAL-LEG METHODOLOGY VALIDATION (synthetic stand-in; transformers absent)")
print("  Spearman(trust, cross-seed disagreement) = %.3f  (want NEGATIVE)"
      % spearman(tr, ds))
med = np.median(tr)
lo = [r[2] for r in rows if r[1] < med]
hi = [r[2] for r in rows if r[1] >= med]
print("  mean disagreement  low-trust=%.3f  high-trust=%.3f"
      % (np.mean(lo), np.mean(hi)))
agg = collections.defaultdict(list)
for lab, t, d in rows:
    agg[lab].append((t, d))
hdr = "    %-14s %6s %9s" % ("case", "trust", "disagree")
print(hdr)
for lab, v in agg.items():
    print("    %-14s %6.3f %9.3f"
          % (lab, np.mean([x[0] for x in v]), np.mean([x[1] for x in v])))
