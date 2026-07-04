"""Forward-only + output-Fisher harvest for the graded safety features (sycophancy, hedging),
saved as an npz cache (X_last, U_last, tmpl_mean, kept, levels, templates) in the SAME format
as the calendar dose caches, so the held-out paired-deviance premise instrument can score them
with the behavioral (nats) metric. GPU: torch.func VJP needs float32 weights (one A40).

Reuses the unmodified low-level machinery from dose_safety.py / dose_calibration_real.py.
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

ROOT = os.environ.get("ROOT", "/projects/standard/hsiehph/sauer354")
sys.path.insert(0, os.path.join(ROOT, "safety_probes"))
sys.path.insert(0, os.path.join(ROOT, "Manifold-SAE", "experiments"))

from safety_features import SAFETY_FEATURE_BANK  # noqa: E402
from dose_safety import build_prompts, harvest  # noqa: E402
from dose_calibration_real import (  # noqa: E402
    LogitsLM, resolve_hook_module, assert_tensor_output,
)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = os.environ.get("HS_MODEL", os.path.join(ROOT, "models/qwen3-8b"))
    layer_idx = int(os.environ.get("HS_LAYER", "18"))
    rank = int(os.environ.get("HS_RANK", "8"))
    device = os.environ.get("HS_DEVICE", "cuda:0")
    out = os.environ.get("PREMISE_OUT", os.path.join(ROOT, "premise_out"))
    feats = os.environ.get("HS_FEATURES", "sycophancy hedging").split()
    os.makedirs(out, exist_ok=True)
    dtype = torch.float32

    print(f"[cfg] model={model_dir} L={layer_idx} rank={rank} feats={feats}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to(device).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    lm = LogitsLM(hf)
    hook_module = resolve_hook_module(hf, layer_idx)
    probe = tok("Today is Monday", return_tensors="pt").input_ids.to(device)
    print(f"[hook] L{layer_idx} shape {assert_tensor_output(lm.module, hook_module, probe)}", flush=True)

    for feature in feats:
        words, templates, periodic, grade, pressure = SAFETY_FEATURE_BANK[feature]
        prompts, lvl, tpl = build_prompts(words, templates)
        print(f"[{feature}] harvesting {len(prompts)} prompts "
              f"({len(templates)} templates x {len(words)} levels)", flush=True)
        t0 = time.time()
        X_last, U_last, tmpl_mean, kept = harvest(lm, hook_module, tok, prompts, rank, device, dtype)
        lvl_k, tpl_k = lvl[kept], tpl[kept]
        dst = os.path.join(out, f"harvest_cache_{feature}_L{layer_idx}.npz")
        np.savez(dst, X_last=X_last, U_last=U_last, tmpl_mean=tmpl_mean, kept=kept,
                 level=lvl_k, template=tpl_k,
                 grade=np.asarray([grade[j] for j in lvl_k], dtype=float))
        print(f"[{feature}] {len(kept)} rows p={X_last.shape[1]} in {time.time()-t0:.1f}s -> {dst}",
              flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
