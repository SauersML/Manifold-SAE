"""R-review reproducibility spot-check for N-nursery synthetic (reduced n, fast).
Confirms the qualitative verdict: discovered nursery EV ≈ oracle, recovers circles,
joint torch lower; held-out throughout. Not a re-derivation — a smaller-n sanity run."""
import os, sys
os.environ.setdefault("BLOCK_NURSERY_STEPS", "250")
sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
import numpy as np
import block_nursery as B

# reduced synthetic: n=210 (vs 480), same p/ncirc structure, different seed
X, planes, theta, meta = B.make_synthetic(n=210, p=96, ncirc=3, seed=7)
tr, te = B.train_test_split(X.shape[0], frac=0.7, seed=0)
K = len(planes)
print(f"[repro] n={X.shape[0]} p={meta['p']} ncirc={K} circle_ceiling={meta['circle_subspace_ev']}", flush=True)

# joint torch (held-out)
tj = B.fit_curved_isolated(X, n_atoms=K, tag="repro_joint", train_idx=tr, test_idx=te, target_k=K)
print(f"[repro] joint torch ev_test={tj.get('ev')}", flush=True)

# oracle nursery
ob = B.oracle_blocks(planes)
nbo = B.run_nursery(X, ob, tag="repro_oracle", train_idx=tr, test_idx=te, theta=theta)
oc = sum(b.get("best_planted_circle_corr", 0) > 0.8 for b in nbo["per_block"])
print(f"[repro] oracle nursery ev_test={nbo['composed_ambient_ev_test']} circles>0.8={oc}/{K}", flush=True)

# discovered nursery (discover on TRAIN only)
bb, _, diag = B.discover_blocks(X[tr], n_dict=2 * K + 2, block_size=3)
nbd = B.run_nursery(X, bb, tag="repro_disc", train_idx=tr, test_idx=te, theta=theta)
dc = len({b["matched_planted_circle"] for b in nbd["per_block"]
          if b.get("best_planted_circle_corr", 0) > 0.8})
corrs = [b.get("best_planted_circle_corr") for b in nbd["per_block"]]
print(f"[repro] discovered nursery ev_test={nbd['composed_ambient_ev_test']} "
      f"blocks={diag['block_dims']} distinct_circles>0.8={dc}/{K} per_block_corr={corrs}", flush=True)

print("\n[repro VERDICT] discovered≈oracle:",
      abs(nbd['composed_ambient_ev_test'] - nbo['composed_ambient_ev_test']) < 0.08,
      "| nursery>joint:", nbd['composed_ambient_ev_test'] > tj.get('ev', 0),
      "| recovers>=2 circles:", dc >= 2, flush=True)
