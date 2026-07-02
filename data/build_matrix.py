#!/usr/bin/env python3
"""Lane-1 DATA loader: raw activation caches -> clean train/held-out matrices.

Handles two external cache formats behind one pipeline:

  --format safetensors  each file has tensors acts_L11/L17/L23 = [tok, d] fp16
                        (Qwen3.6-35B-A3B SuperGPQA rollouts, L17 cache); the
                        contiguous split unit is the SHARD FILE.
  --format npy          each file is one [~50k, d] fp16 chunk (creditscope
                        Qwen3.5-35B L30 cache); the split unit is the CHUNK FILE.

Both split at FILE granularity — NEVER by row. Adjacent tokens within a
rollout/chunk are correlated; a row-level split leaks correlated neighbours into
held-out and inflates held-out EV for every downstream lane. Whole-file split is
the coarsest rollout-safe boundary and is recorded in split_manifest.json.

Tier-0 (per-dim mean, massive-activation "rogue" dims, global RMS scale) is
computed on TRAIN files ONLY and stored as a *transform description* in
tier0.json — the raw matrices are written un-normalised so every lane applies
the identical transform and the artifacts stay auditable.

Outputs (<prefix> defaults to the layer name):
  <out>/<prefix>_train.f32.npy      raw fp32, [n_train, d]
  <out>/<prefix>_heldout.f32.npy    raw fp32, [n_heldout, d]
  <out>/tier0.json
  <out>/split_manifest.json

Python (not Rust) because this is pure external-format I/O (safetensors / npy),
which SPEC explicitly carves out for the thin-Python rule.
"""
import argparse, json, os, sys, glob
import numpy as np


# ---- format readers: (row_count, dim) probe + full fp32 load, one file at a time ----

def _st_probe(path, key):
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        shape = f.get_slice(key).get_shape()
    return int(shape[0]), int(shape[1])

def _st_load(path, key):
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        return np.asarray(f.get_tensor(key), dtype=np.float32)

def _npy_probe(path, key):
    # key ignored; header read only (mmap doesn't pull data)
    a = np.load(path, mmap_mode="r")
    return int(a.shape[0]), int(a.shape[1])

def _npy_load(path, key):
    return np.asarray(np.load(path), dtype=np.float32)

READERS = {
    "safetensors": (_st_probe, _st_load),
    "npy": (_npy_probe, _npy_load),
}


def assign_splits(n_files, tokens, heldout_target, train_target, seed):
    """Greedy WHOLE-FILE assignment to disjoint held-out and train token targets.

    Deterministic shuffle (fixed seed); fill held-out to heldout_target first,
    then fill train from the remaining files to train_target (<=0 means take all
    remaining). Files beyond both targets are left unused. Whole files only -> no
    rollout/chunk is split across the train/held-out boundary.
    """
    order = list(range(n_files))
    np.random.default_rng(seed).shuffle(order)
    held, held_tok = [], 0
    train, train_tok = [], 0
    for i in order:
        if held_tok < heldout_target:
            held.append(i); held_tok += tokens[i]
        elif train_target <= 0 or train_tok < train_target:
            train.append(i); train_tok += tokens[i]
    return sorted(train), sorted(held)


def write_matrix(out_path, files, idxs, tokens, key, dim, load_fn):
    """Stream files into a memmapped fp32 .npy — never holds the full matrix in RAM."""
    n = sum(tokens[i] for i in idxs)
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, dim))
    row, n_nan = 0, 0
    for i in idxs:
        t = load_fn(files[i], key)
        if t.shape[1] != dim:
            raise ValueError(f"{files[i]}: dim {t.shape[1]} != {dim}")
        n_nan += int((~np.isfinite(t)).sum())
        arr[row:row + t.shape[0]] = t
        row += t.shape[0]
    arr.flush(); del arr
    return n, n_nan


def tier0_from_train(train_path, dim, n_rogue_max=3):
    """Per-dim mean, massive-activation (rogue) dims, and global RMS scale.

    Rogue dims = the massive-activation channels standard in transformer residual
    streams: dims whose per-dim RMS is a robust far-outlier (robust z > 6 via
    median/MAD) above the population. Global RMS scale is E[(x-mean)^2] averaged
    over the NON-rogue dims so the scale is not dominated by those channels.
    Moments are accumulated in chunks so memory stays bounded.
    """
    X = np.load(train_path, mmap_mode="r")
    n = X.shape[0]
    s1 = np.zeros(dim, dtype=np.float64)
    s2 = np.zeros(dim, dtype=np.float64)
    step = 50_000
    for start in range(0, n, step):
        blk = np.asarray(X[start:start + step], dtype=np.float64)
        s1 += blk.sum(0)
        s2 += np.einsum("ij,ij->j", blk, blk)   # no full blk*blk temp
    mean = s1 / n
    ms = s2 / n
    rms_dim = np.sqrt(np.maximum(ms, 0.0))
    med = np.median(rms_dim)
    mad = np.median(np.abs(rms_dim - med))
    sigma = 1.4826 * mad if mad > 0 else (rms_dim.std() or 1.0)
    z = (rms_dim - med) / sigma
    cand = np.argsort(z)[::-1]
    rogue = [int(d) for d in cand[:n_rogue_max] if z[d] > 6.0]
    var_dim = np.maximum(ms - mean * mean, 0.0)
    keep = np.ones(dim, dtype=bool); keep[rogue] = False
    global_rms = float(np.sqrt(var_dim[keep].mean())) if keep.any() else float(np.sqrt(var_dim.mean()))
    return {
        "dim": dim,
        "n_train_tokens": int(n),
        "per_dim_mean": mean.astype(np.float32).tolist(),
        "rogue_dims": rogue,
        "rogue_dim_rms": [float(rms_dim[d]) for d in rogue],
        "rogue_dim_robust_z": [float(z[d]) for d in rogue],
        "global_rms_scale": global_rms,
        "scale_excludes_rogue": True,
        "transform": "x' = (x - per_dim_mean) / global_rms_scale  (rogue dims kept, not removed)",
    }


def verify(path, dim):
    """Full chunked finiteness scan + dtype/shape asserts. Memory-bounded."""
    X = np.load(path, mmap_mode="r")
    assert X.dtype == np.float32, X.dtype
    assert X.shape[1] == dim, X.shape
    n = X.shape[0]
    n_nan = 0
    for start in range(0, n, 50_000):
        n_nan += int((~np.isfinite(X[start:start + 50_000])).sum())
    assert n_nan == 0, f"{n_nan} non-finite in {path}"
    return {"shape": [int(n), int(X.shape[1])], "dtype": "float32", "n_nonfinite": n_nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="glob for cache files")
    ap.add_argument("--format", choices=list(READERS), required=True)
    ap.add_argument("--layer", default="acts_L17", help="tensor key (safetensors only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-target", type=int, default=200_000)
    ap.add_argument("--train-target", type=int, default=0, help="cap train tokens (<=0 = all remaining)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--skip-write", action="store_true",
                    help="reuse existing matrices (deterministic split); recompute tier0 + manifest only")
    args = ap.parse_args()
    probe_fn, load_fn = READERS[args.format]
    prefix = args.prefix or (args.layer if args.format == "safetensors" else "L")

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.files))
    if not files:
        print(f"NO FILES matched {args.files}", file=sys.stderr); sys.exit(2)
    print(f"{len(files)} files", flush=True)

    tokens, dim = [], None
    for p in files:
        nt, d = probe_fn(p, args.layer)
        tokens.append(nt)
        if dim is None:
            dim = d
        elif d != dim:
            raise ValueError(f"{p}: dim {d} != {dim}")
    total = sum(tokens)
    print(f"total tokens={total:,} dim={dim}", flush=True)

    train_idx, held_idx = assign_splits(len(files), tokens, args.heldout_target, args.train_target, args.seed)
    n_train = sum(tokens[i] for i in train_idx)
    n_held = sum(tokens[i] for i in held_idx)
    n_unused = len(files) - len(train_idx) - len(held_idx)
    print(f"train files={len(train_idx)} tokens={n_train:,} | heldout files={len(held_idx)} tokens={n_held:,} | unused files={n_unused}", flush=True)

    train_path = os.path.join(args.out, f"{prefix}_train.f32.npy")
    held_path = os.path.join(args.out, f"{prefix}_heldout.f32.npy")
    if args.skip_write:
        nan_tr = nan_he = -1   # matrices reused; nan counts not recomputed here (verify still checks finiteness)
        print("skip-write: reusing existing matrices", flush=True)
    else:
        _, nan_tr = write_matrix(train_path, files, train_idx, tokens, args.layer, dim, load_fn)
        _, nan_he = write_matrix(held_path, files, held_idx, tokens, args.layer, dim, load_fn)
        print(f"wrote matrices (nan train={nan_tr}, heldout={nan_he})", flush=True)

    tier0 = tier0_from_train(train_path, dim)
    with open(os.path.join(args.out, "tier0.json"), "w") as f:
        json.dump(tier0, f)
    print(f"tier0: rogue_dims={tier0['rogue_dims']} global_rms={tier0['global_rms_scale']:.4f}", flush=True)

    v_tr = verify(train_path, dim)
    v_he = verify(held_path, dim)
    print(f"VERIFY train={v_tr} heldout={v_he}", flush=True)
    if args.skip_write:
        nan_tr, nan_he = v_tr["n_nonfinite"], v_he["n_nonfinite"]

    manifest = {
        "format": args.format,
        "layer": args.layer if args.format == "safetensors" else None,
        "dim": dim,
        "n_files": len(files),
        "seed": args.seed,
        "heldout_target": args.heldout_target,
        "train_target": args.train_target,
        "split_policy": "whole-file (rollout/chunk-safe); no row-level split",
        "n_train_tokens": n_train,
        "n_heldout_tokens": n_held,
        "n_unused_files": n_unused,
        "train_files": [os.path.basename(files[i]) for i in train_idx],
        "heldout_files": [os.path.basename(files[i]) for i in held_idx],
        "file_tokens": {os.path.basename(files[i]): tokens[i] for i in range(len(files))},
        "nan_counts": {"train": nan_tr, "heldout": nan_he},
        "artifacts": {
            "train": os.path.basename(train_path),
            "heldout": os.path.basename(held_path),
            "tier0": "tier0.json",
        },
    }
    with open(os.path.join(args.out, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
