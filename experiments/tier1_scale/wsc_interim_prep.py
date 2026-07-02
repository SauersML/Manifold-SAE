"""Build an interim T1 harvest + manifest from existing REAL OLMo-7B activations.

Reshapes /dev/shm/w6/olmo7b_rich_last.npy (760 x 32 x 4096) to (24320 x 4096)
rows, shuffles (seeded), splits train/heldout, writes 2D fp32 npy shards and a
MANIFEST.json (with T0 = train mean/scale baked) that tier1_scale_run.py reads.
This is an INTERIM slice on real frontier activations while WS-D's big harvest
lands; scale K up on the real manifest when it publishes.
"""
import json
from pathlib import Path
import numpy as np

SRC = "/dev/shm/w6/olmo7b_rich_last.npy"
OUT = Path("/dev/shm/sauers_gpu/tier1/interim")
OUT.mkdir(parents=True, exist_ok=True)

a = np.load(SRC)                       # (760, 32, 4096)
X = a.reshape(-1, a.shape[-1]).astype(np.float32)   # (24320, 4096)
rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0])
X = np.ascontiguousarray(X[perm])
n = X.shape[0]
n_held = max(1, int(0.15 * n))
held = X[:n_held]
train = X[n_held:]

# T0 on TRAIN only (mean + scale), baked into the manifest.
mean = train.mean(0).astype(np.float32)
scale = train.std(0).astype(np.float32)
scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)

# Split train into a few shards to exercise the streaming path.
n_shards = 4
shard_paths = []
for i, chunk in enumerate(np.array_split(train, n_shards)):
    p = OUT / f"train_{i:03d}.npy"
    np.save(p, np.ascontiguousarray(chunk))
    shard_paths.append((str(p), int(chunk.shape[0]), "train"))
hp = OUT / "heldout_000.npy"
np.save(hp, np.ascontiguousarray(held))

manifest = {
    "source": SRC,
    "model": "OLMo-7B",
    "note": "INTERIM real-activation slice (token-limited); scale K on WS-D harvest",
    "P": int(X.shape[1]),
    "dtype": "float32",
    "layer": "last",
    "tokens_total": int(n),
    "t0": {"mean": mean.tolist(), "scale": scale.tolist()},
    "shards": (
        [{"path": p, "tokens": t, "split": s} for p, t, s in shard_paths]
        + [{"path": str(hp), "tokens": int(held.shape[0]), "split": "heldout"}]
    ),
}
mpath = OUT / "MANIFEST.json"
mpath.write_text(json.dumps(manifest))
print("INTERIM_MANIFEST", mpath, "train_rows", train.shape[0], "held", held.shape[0],
      "P", X.shape[1])
