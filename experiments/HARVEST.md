# WS-D harvest — frontier activation shards for SAC (2026-07-02)

The big activation harvest that feeds T1 scale (WS-C), the encoder/corpus sweep
(WS-E), and the science battery (WS-F/WS-I). All shards live on **node2** under
`/dev/shm/sauers_gpu/harvest/` and follow the `residual_shard_io` bf16-memmap
contract (`/Users/user/gam/examples/residual_shard_io.py`, present on-node at
`/models/sauers_build/gam_fable/examples/residual_shard_io.py`).

## Model / layers / corpus

- **Model:** `Qwen3-32B` (base), d_model **5120**, 64 layers, native bf16.
  OLMo-3-7B/32B and Qwen3-8B are **not** on node2; Qwen3-32B is the frontier
  member of the requested Qwen3 family present on disk, and is the stronger
  science target (WS-C T1 at K=32k, WS-F calendar/dose).
- **Layers (mid-stack):** 24, 32, 40. One forward pass hooks all three
  (`model.model.layers[L]` output = residual stream after block L), BOS/position-0
  dropped per sequence.
- **Corpus:** `HuggingFaceFW/fineweb` (`sample-10BT`), streamed, `text` field,
  seq_len 256. (node2 has working outbound internet; HF streams fine.)

## Storage discipline (why not a flat 50–100M × all layers)

Each token-row is `5120 × 2 B ≈ 10 KB`, so 50M tokens is ~0.5 TB **per layer**.
node2 root is 98% full and `/models` is 99% full; the only viable scratch is
`/dev/shm` (RAM-backed, ~660 GB free on a shared co-tenant box). To stay well
inside that with size discipline, the token budget is **asymmetric** — the
primary mid-stack layer gets the large T1/encoder corpus, the flanking layers
get a smaller cross-layer set, all from one shared forward pass:

| set              | layer | token cap | ~size   |
|------------------|-------|-----------|---------|
| primary (T1)     | 32    | 30M       | ~307 GB |
| flank (x-layer)  | 24    | 6M        | ~61 GB  |
| flank (x-layer)  | 40    | 6M        | ~61 GB  |

Total ≈ 430 GB. The shards are append-friendly bf16, so the harvest can be
extended later if a real disk volume is freed.

## T0 stats (baked into every manifest, key `t0`)

Computed from the per-dim mean + RMS that `ShardWriter` already accumulates over
the true pre-quantization activations — no shard re-read:

- `mean`, `std` (= `sqrt(rms^2 - mean^2)`), `rms` — per-dim, length d_model.
- `scale_median_std`, `scale_median_rms` — robust whitening reference.
- `rogue_dims` — the massive-activation coordinate dims (RMS a robust outlier:
  `rms > 5× median_rms` OR MAD-z > 8), reported by index + rms + ratio + z.
  (Qwen3-32B L32 shows ~376 rogue dims, dominated by dim 731 at ~50× median.)

## Probe harvests (calendar + color, same model/layers)

Small dedicated sets — target-token residual (last sub-token of the word) across
natural template sentences — for `weekday` (7-circle), `month` (12-circle),
`year` (ordered, non-cyclic), `color` (hue circle). One ShardWriter dir per
(set, layer), tag `qwen3_32b_probe_<set>_l<L>`; each manifest carries per-row
`labels`, `order`/continuous coord, and `cyclic`, so WS-F curved-feature probes
align rows to concepts directly.

## How to consume

```python
from residual_shard_io import load_shards, stratified_subsample
r = load_shards("/dev/shm/sauers_gpu/harvest/qwen3_32b_fineweb_l32")
for batch in r.batches(4096):        # float32 (<=4096, 5120)
    model.partial_fit(batch)
t0 = r.manifest["t0"]                # mean/std/rms/rogue_dims/scale
sub = stratified_subsample(r, 1_000_000)   # curved-fit subsample
```

Index of every set with paths/token-counts/T0 summary:
`/dev/shm/sauers_gpu/harvest/MANIFEST.json` (built by
`experiments/harvest_publish_manifest.py`).

## Reproduce

```
CUDA_VISIBLE_DEVICES=6,7 python experiments/harvest_frontier_multilayer.py \
  --model /models/Qwen3-32B --layers 24,32,40 --caps 6000000,30000000,6000000 \
  --out-root /dev/shm/sauers_gpu/harvest --tag qwen3_32b_fineweb
CUDA_VISIBLE_DEVICES=4,5 python experiments/harvest_frontier_probes.py \
  --model /models/Qwen3-32B --layers 24,32,40 \
  --out-root /dev/shm/sauers_gpu/harvest --tag qwen3_32b
```
(run via Heimdall, wrapped to exit 0 with rc+log under `/dev/shm/sauers_gpu/harvest/logs/`.)
