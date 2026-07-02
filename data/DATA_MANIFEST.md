# DATA_MANIFEST — real frontier activations for the manifold-SAE run (Lane 1)

Owner: DATA. Built 2026-07-02 on MSI (`$ROOT = /projects/standard/hsiehph/sauer354`).
Loader: `data/build_matrix.py` (this repo) — pushed to `$ROOT/msae_l17/build_matrix.py`.

Two datasets, both processed by the identical pipeline: concatenate the target
layer across cache files, split at **whole-file granularity** (never by row),
compute Tier-0 on TRAIN files only, write raw fp32 matrices + a Tier-0 transform
description + a split manifest.

## Split policy (pre-registered, non-negotiable)

**Whole-file split, never by row.** Adjacent tokens inside a rollout/chunk are
correlated; a row-level split leaks correlated neighbours into held-out and
inflates held-out EV for every downstream lane. Each L17 shard is one SuperGPQA
question (5 rollouts, ~10.2k tokens) — the natural contiguous unit — so a
whole-shard split cannot straddle a rollout boundary. Selection is a
deterministic seeded shuffle (`seed=0`): fill held-out to its token target
first, then fill train to its target; remaining files are held in reserve.

## Tier-0 (computed on TRAIN only, stored as a transform — NOT applied)

The written matrices are **raw** (un-normalised) so every lane applies the same
transform and the artifacts stay auditable. `tier0.json` contains:
- `per_dim_mean` — [d] fp32, per-dim mean over train tokens.
- `rogue_dims` — massive-activation channels: per-dim RMS robust-z (median/MAD)
  > 6, up to 3. These are kept, not removed.
- `global_rms_scale` — sqrt(mean per-dim variance over NON-rogue dims), so the
  scale is not dominated by the massive channels.

Apply: `x' = (x - per_dim_mean) / global_rms_scale`.

## Dataset 1 — L17 (headline target)

- Model: Qwen3.6-35B-A3B. Data: SuperGPQA reasoning rollouts. Layer: `acts_L17`.
- Source: `$ROOT/msae_l17/data/shards/q*.safetensors` (200 shards; each has
  `acts_L11`/`acts_L17`/`acts_L23` = [tok, 2048] fp16, + `outcomes`, `response_lens`).
- Total available: 2,005,719 tokens. Used: 120 train shards + 20 held-out
  shards (disjoint); 60 shards held in reserve.

| artifact | path | shape | tokens |
|---|---|---|---|
| train | `$ROOT/msae_l17/L17_train.f32.npy` | [1204602, 2048] fp32 | 1,204,602 |
| held-out | `$ROOT/msae_l17/L17_heldout.f32.npy` | [201169, 2048] fp32 | 201,169 |
| tier0 | `$ROOT/msae_l17/tier0.json` | — | — |
| manifest | `$ROOT/msae_l17/split_manifest.json` | — | — |

Tier-0: `global_rms_scale = 0.0493`; rogue dims `[1269, 491, 863]` (robust-z
`[418.5, 82.3, 77.2]`, RMS `[1.72, 0.37, 0.35]`). Verified: 0 non-finite.

## Dataset 2 — creditscope L30 (generality replicate)

- Model: Qwen3.5-35B-A3B. Data: `sarel/creditscope-fino1-activations`.
  Layer 30 residual **post**.
- Source: `sarel/creditscope-activations-v2` (HF dataset), downloaded to
  `$ROOT/creditscope_acts/activations/layer_30_residual_post/chunk_*.npy`
  (8 chunks, [~50k, 2048] fp16). Only L30-post was pulled (the SAE target); the
  other layers (0/10/39, pre/post) remain undownloaded to save space.
- Total: 360,002 tokens. Split: 7 train chunks + 1 held-out chunk. (A 200k
  held-out would starve a 360k dataset, so held-out is one ~50k chunk.)

| artifact | path | shape | tokens |
|---|---|---|---|
| train | `$ROOT/creditscope/L30_train.f32.npy` | [309921, 2048] fp32 | 309,921 |
| held-out | `$ROOT/creditscope/L30_heldout.f32.npy` | [50081, 2048] fp32 | 50,081 |
| tier0 | `$ROOT/creditscope/tier0.json` | — | — |
| manifest | `$ROOT/creditscope/split_manifest.json` | — | — |

Tier-0: `global_rms_scale = 0.1172`; rogue dims `[1108, 1269, 2045]` (robust-z
`[236.0, 188.8, 113.8]`). Verified: 0 non-finite.

Cross-model note: massive-activation channel **1269** is a rogue dim in BOTH
Qwen3.6 L17 and Qwen3.5 L30.

## Consuming this data

```python
import numpy as np, json
X = np.load("$ROOT/msae_l17/L17_train.f32.npy", mmap_mode="r")   # raw fp32
t0 = json.load(open("$ROOT/msae_l17/tier0.json"))
mean = np.asarray(t0["per_dim_mean"], dtype=np.float32)
Xn = (X - mean) / t0["global_rms_scale"]                         # apply tier-0
```

Held-out EV must be measured on the held-out matrix only. Never refit or
recompute Tier-0 on held-out.

## Reproduce

```
python build_matrix.py --files "$ROOT/msae_l17/data/shards/*.safetensors" \
  --format safetensors --layer acts_L17 --out $ROOT/msae_l17 \
  --heldout-target 200000 --train-target 1200000 --seed 0 --prefix L17
python build_matrix.py --files "$ROOT/creditscope_acts/activations/layer_30_residual_post/chunk_*.npy" \
  --format npy --out $ROOT/creditscope --heldout-target 50000 --seed 0 --prefix L30
```
