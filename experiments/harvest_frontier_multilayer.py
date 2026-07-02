#!/usr/bin/env python
"""WS-D big harvest: multi-layer residual-stream activations from a frontier LM.

One forward pass hooks several decoder layers at once and streams each layer's
token activations into its own bf16 sharded harvest directory (the
``residual_shard_io`` contract). Per-layer token budgets may differ (the primary
mid-stack layer gets the large T1/encoder corpus; the flanking layers get a
smaller cross-layer set), so we run ONE forward pass and append to each layer's
writer only until that layer's cap is reached.

On finalize we augment every manifest with a ``t0`` block computed from the
per-dim running stats already accumulated by ShardWriter (mean + norm), adding
the per-dim standard deviation, a robust overall scale, and the rogue
(massive-activation) dimensions. No shard re-read is needed for T0.

The harvest job runs server-side under Heimdall; a provisional manifest is
flushed periodically so a crash leaves a readable partial harvest.

Example (node2, 2 GPUs via device_map):
  CUDA_VISIBLE_DEVICES=6,7 python harvest_frontier_multilayer.py \
      --model /models/Qwen3-32B \
      --layers 24,32,40 --caps 6000000,30000000,6000000 \
      --out-root /dev/shm/sauers_gpu/harvest --tag qwen3_32b_fineweb
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# residual_shard_io lives beside the reference harvester in the gam checkout.
for _cand in (
    "/models/sauers_build/gam_fable/examples",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gam", "examples"),
    "/Users/user/gam/examples",
):
    if os.path.isfile(os.path.join(_cand, "residual_shard_io.py")):
        sys.path.insert(0, _cand)
        break
from residual_shard_io import (  # noqa: E402
    MANIFEST_NAME,
    ShardWriter,
    tokenizer_hash,
)


def compute_t0(mean: np.ndarray, norm: np.ndarray) -> dict:
    """Derive the T0 block from per-dim mean and RMS (== ShardWriter 'norm').

    std_d = sqrt(max(rms_d^2 - mean_d^2, 0)).  Rogue (massive-activation) dims
    are the coordinates whose RMS is a robust outlier above the bulk; we report
    them by both a >5x-median rule and a robust MAD z-score so downstream code
    can pick either. 'scale' is the robust central per-dim std (median), the
    natural whitening reference.
    """
    mean = np.asarray(mean, dtype=np.float64)
    norm = np.asarray(norm, dtype=np.float64)
    std = np.sqrt(np.maximum(norm * norm - mean * mean, 0.0))

    med = float(np.median(norm))
    mad = float(np.median(np.abs(norm - med))) or 1.0
    z = (norm - med) / (1.4826 * mad)
    ratio = norm / (med if med > 0 else 1.0)
    rogue_mask = (ratio > 5.0) | (z > 8.0)
    rogue_idx = np.nonzero(rogue_mask)[0]
    # order rogue dims by descending norm for readability
    rogue_idx = rogue_idx[np.argsort(-norm[rogue_idx])]

    return {
        "d_model": int(mean.shape[0]),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "rms": norm.astype(np.float32).tolist(),
        "scale_median_std": float(np.median(std)),
        "scale_median_rms": med,
        "rogue_dims": {
            "index": [int(i) for i in rogue_idx],
            "rms": [float(norm[i]) for i in rogue_idx],
            "rms_over_median": [float(ratio[i]) for i in rogue_idx],
            "mad_z": [float(z[i]) for i in rogue_idx],
            "rule": "rms>5*median_rms OR MAD-z>8",
        },
    }


def _provisional_manifest(writer: ShardWriter) -> None:
    """Dump a readable manifest mid-run without closing the writer."""
    if writer._cur_file is not None:
        writer._cur_file.flush()
    man = writer._build_manifest()
    man["provisional"] = True
    with open(os.path.join(writer.out_dir, MANIFEST_NAME), "w") as f:
        json.dump(man, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    ap.add_argument("--caps", required=True, help="comma-separated per-layer token caps")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-seqs", type=int, default=24)
    ap.add_argument("--rows-per-shard", type=int, default=1_000_000)
    ap.add_argument("--flush-every", type=int, default=2_000_000)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    caps = [int(x) for x in args.caps.split(",")]
    if len(layers) != len(caps):
        raise SystemExit("--layers and --caps must have equal length")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_gpu = torch.cuda.device_count()
    print(f"[harvest] visible GPUs={n_gpu} "
          f"({os.environ.get('CUDA_VISIBLE_DEVICES','?')})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto" if n_gpu > 1 else 0,
        low_cpu_mem_usage=True,
    ).eval()
    in_device = next(model.parameters()).device

    dec = model.model.layers
    for L in layers:
        if not (0 <= L < len(dec)):
            raise SystemExit(f"layer {L} out of range (model has {len(dec)})")
    d_model = int(model.config.hidden_size)
    print(f"[harvest] d_model={d_model} n_layers={len(dec)} layers={layers}", flush=True)

    grabbed: dict[int, torch.Tensor] = {}

    def make_hook(L: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            grabbed[L] = h.detach()
        return hook

    handles = [dec[L].register_forward_hook(make_hook(L)) for L in layers]

    meta_common = {
        "model_name": args.model,
        "tokenizer_hash": tokenizer_hash(tok),
        "text_dataset": f"{args.dataset}/{args.config}",
        "text_subset": args.split,
        "seq_len": args.seq_len,
        "workstream": "WS-D",
        "harvest_args": vars(args),
    }
    writers: dict[int, ShardWriter] = {}
    out_dirs: dict[int, str] = {}
    for L, cap in zip(layers, caps):
        d = os.path.join(args.out_root, f"{args.tag}_l{L}")
        out_dirs[L] = d
        writers[L] = ShardWriter(
            d, d_model=d_model, rows_per_shard=args.rows_per_shard,
            meta={**meta_common, "layer": L, "token_cap": cap},
        )

    counts = {L: 0 for L in layers}
    caps_by_layer = dict(zip(layers, caps))
    last_flush = 0
    t_start = time.time()

    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)
    buf: list[int] = []
    batch: list[list[int]] = []

    def all_full() -> bool:
        return all(counts[L] >= caps_by_layer[L] for L in layers)

    def flush_batch() -> None:
        if not batch:
            return
        ids = torch.tensor(batch, device=in_device)
        with torch.no_grad():
            model(ids)
        for L in layers:
            room = caps_by_layer[L] - counts[L]
            if room <= 0:
                continue
            h = grabbed[L]
            h = h[:, 1:, :].reshape(-1, h.shape[-1])  # drop BOS/position-0 per seq
            take = min(h.shape[0], room)
            writers[L].append(h[:take].float().cpu().numpy())
            counts[L] += take
        grabbed.clear()
        batch.clear()

    for doc in ds:
        text = doc.get(args.text_field) or ""
        if not text.strip():
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        while len(buf) >= args.seq_len:
            batch.append(buf[: args.seq_len])
            buf = buf[args.seq_len:]
            if len(batch) >= args.batch_seqs:
                flush_batch()
                total = sum(counts.values())
                if total - last_flush >= args.flush_every:
                    last_flush = total
                    for L in layers:
                        _provisional_manifest(writers[L])
                    el = time.time() - t_start
                    rate = total / el if el > 0 else 0.0
                    print(f"[harvest] {dict(counts)} total={total} "
                          f"{rate:,.0f} rows/s elapsed={el:,.0f}s", flush=True)
        if all_full():
            break
    flush_batch()
    for h in handles:
        h.remove()

    summary = {}
    for L in layers:
        man = writers[L].close()
        t0 = compute_t0(man["stats"]["mean"], man["stats"]["norm"])
        man["t0"] = t0
        with open(os.path.join(out_dirs[L], MANIFEST_NAME), "w") as f:
            json.dump(man, f)
        summary[L] = {
            "dir": out_dirs[L],
            "tokens": man["total_tokens"],
            "shards": len(man["shards"]),
            "n_rogue": len(t0["rogue_dims"]["index"]),
            "scale_median_std": t0["scale_median_std"],
        }
        print(f"[harvest] L{L}: {json.dumps(summary[L])}", flush=True)

    with open(os.path.join(args.out_root, f"{args.tag}_SUMMARY.json"), "w") as f:
        json.dump({"model": args.model, "layers": layers, "caps": caps,
                   "per_layer": summary,
                   "elapsed_s": time.time() - t_start}, f, indent=2)
    print("[harvest] DONE", flush=True)
    # All shards/manifests are flushed to disk above. Skip interpreter
    # finalization: datasets/pyarrow + torch background threads abort with a
    # PyGILState_Release fatal error at teardown on this stack. os._exit avoids
    # it without risking the (already-written) outputs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
