"""Resilient, resumable, pipelined self/qualia harvest across OLMo checkpoints.

For each checkpoint revision it: loads the model once, harvests the hand-written
prompt bank (last token, all layers), then runs the steering + cloze probes on
the SAME loaded model while the NEXT checkpoint downloads in a background thread.
That overlap hides the (dominant) download latency behind useful GPU work.

Resilience / resumability:
  * Each checkpoint writes its outputs atomically and a ``done.json`` marker;
    on restart, checkpoints whose marker exists are skipped.
  * A per-checkpoint failure is logged and the sweep continues (no set -e).
  * Each revision uses an isolated HF cache dir, so the finished checkpoint can
    be deleted without disturbing the one prefetching next. At most two
    checkpoints (~128 GB) live on disk at once.

Run on the Azure A100 VM from the repo root, e.g.:

  .venv/bin/python -m experiments.run_self_qualia_trajectory \
      --model allenai/Olmo-3-1125-32B \
      --prompts-file experiments/self_qualia_prompts.jsonl \
      --out-parent runs/OLMO3_32B_TRAJ \
      --cache-root /mnt/nvme/hf_traj \
      --batch-size 16

DOES NOT launch anything on import; pure driver.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import threading
import time
import traceback
from pathlib import Path

import numpy as np


# Default base trajectory: log-spaced, early-weighted, pretrain -> mid -> long-ctx.
DEFAULT_REVS = [
    "stage1-step0", "stage1-step1000", "stage1-step2000", "stage1-step4000",
    "stage1-step8000", "stage1-step16000", "stage1-step32000", "stage1-step64000",
    "stage1-step128000", "stage1-step256000", "stage1-step512000", "stage1-step656000",
    "stage2-step4000", "stage2-step12000", "stage2-step23842",
    "stage3-step4000", "stage3-step11921",
]


def _rev_cache(cache_root: Path, rev: str) -> Path:
    return cache_root / rev.replace("/", "_")


def _prefetch(model: str, rev: str, cache_dir: Path, result: dict) -> None:
    """Download one revision's snapshot into an isolated cache dir."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model, revision=rev, cache_dir=str(cache_dir))
        result["ok"] = True
    except Exception as e:  # noqa: BLE001 - record and let caller decide
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"


def _start_prefetch(model: str, rev: str, cache_root: Path):
    res: dict = {}
    cache_dir = _rev_cache(cache_root, rev)
    cache_dir.mkdir(parents=True, exist_ok=True)
    th = threading.Thread(target=_prefetch, args=(model, rev, cache_dir, res), daemon=True)
    th.start()
    return th, res


def _free_disk_gb(path: Path) -> float:
    st = shutil.disk_usage(str(path))
    return st.free / 1e9


def run(args) -> None:
    from experiments.self_qualia_olmo import load_bank_jsonl, load_model, harvest
    from experiments.self_qualia_steer_cloze import run_steer_cloze

    revs = (
        [r.strip() for r in Path(args.revs_file).read_text().split() if r.strip()]
        if args.revs_file else list(DEFAULT_REVS)
    )
    out_parent = Path(args.out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    records = load_bank_jsonl(Path(args.prompts_file))
    prompts = [str(r["prompt"]) for r in records]

    # resume: drop checkpoints already finished
    todo = [r for r in revs if not (out_parent / r / "done.json").exists()]
    skipped = [r for r in revs if r not in todo]
    log = out_parent / "RUN.log"
    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    say(f"trajectory start: {len(todo)} to run, {len(skipped)} already done "
        f"({','.join(skipped) or 'none'})")
    if not todo:
        say("nothing to do."); return

    # block on the first checkpoint, then pipeline subsequent prefetches
    first_cache = _rev_cache(cache_root, todo[0])
    say(f"downloading first checkpoint {todo[0]} ...")
    res0: dict = {}
    _prefetch(args.model, todo[0], first_cache, res0)
    if not res0.get("ok"):
        say(f"FATAL: first checkpoint download failed: {res0.get('error')}"); return

    for i, rev in enumerate(todo):
        out_dir = out_parent / rev
        cache_dir = _rev_cache(cache_root, rev)
        prefetch_th = prefetch_res = next_rev = None
        try:
            say(f"=== {rev} ({i + 1}/{len(todo)})  free={_free_disk_gb(cache_root):.0f}GB ===")
            model, tok, n_layers = load_model(
                args.model, rev, args.dtype, args.device, cache_dir=str(cache_dir))

            # kick off NEXT checkpoint download now, to overlap with compute below
            if i + 1 < len(todo):
                next_rev = todo[i + 1]
                say(f"prefetching next: {next_rev}")
                prefetch_th, prefetch_res = _start_prefetch(args.model, next_rev, cache_root)

            tmp = out_dir.with_name(out_dir.name + ".partial")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)
            with open(tmp / "prompts.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            t0 = time.time()
            X = harvest(
                model_name=args.model, revision=rev, prompts=prompts, out_dir=tmp,
                batch_size=args.batch_size, dtype=args.dtype, device=args.device,
                pooling="last_token", model=model, tokenizer=tok,
            )
            t_harvest = time.time() - t0
            say(f"{rev}: harvested {X.shape} in {t_harvest:.0f}s")

            steer_layer = (args.steer_layer if args.steer_layer is not None
                           else int(round(args.steer_layer_percent * (n_layers - 1))))
            if not args.no_steer:
                try:
                    t1 = time.time()
                    run_steer_cloze(
                        model=model, tok=tok, device=args.device, X=X, records=records,
                        steer_layer=steer_layer, out_dir=tmp)
                    say(f"{rev}: steer+cloze in {time.time() - t1:.0f}s")
                except Exception as e:  # noqa: BLE001
                    say(f"{rev}: steer+cloze FAILED (non-fatal): {e}")

            (tmp / "done.json").write_text(json.dumps({
                "revision": rev, "model": args.model, "shape": list(X.shape),
                "steer_layer": int(steer_layer), "harvest_seconds": round(t_harvest, 1),
            }, indent=2))
            # atomic publish
            if out_dir.exists():
                shutil.rmtree(out_dir)
            tmp.rename(out_dir)
            say(f"{rev}: published -> {out_dir}")

        except Exception as e:  # noqa: BLE001 - never abort the whole sweep
            say(f"{rev}: ERROR (skipping): {e}\n{traceback.format_exc()}")
        finally:
            # free GPU + this checkpoint's weights before next iteration
            try:
                import torch
                del model
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            shutil.rmtree(cache_dir, ignore_errors=True)
            # ensure the prefetch finished before we try to load it next loop
            if prefetch_th is not None:
                prefetch_th.join()
                if not prefetch_res.get("ok"):
                    say(f"WARN prefetch of {next_rev} failed: {prefetch_res.get('error')} "
                        f"(will retry blocking next iter)")

    say("trajectory ALLDONE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="allenai/Olmo-3-1125-32B")
    ap.add_argument("--prompts-file", default="experiments/self_qualia_prompts.jsonl")
    ap.add_argument("--out-parent", default="runs/OLMO3_32B_TRAJ")
    ap.add_argument("--cache-root", default="/mnt/nvme/hf_traj")
    ap.add_argument("--revs-file", default=None,
                    help="whitespace-separated revisions; default = built-in trajectory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steer-layer", type=int, default=None)
    ap.add_argument("--steer-layer-percent", type=float, default=0.40)
    ap.add_argument("--no-steer", action="store_true",
                    help="harvest only; skip the steering+cloze window-filler")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
