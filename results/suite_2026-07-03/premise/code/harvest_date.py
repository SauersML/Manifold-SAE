"""GPU harvest for the DAY-OF-MONTH feature (8th calendar feature) and the JOINT month x day
date representation, saved as npz caches in the calendar-cache format (X_last, U_last,
tmpl_mean, ...). Reuses the unmodified low-level harvest from dose_safety / dose_calibration_real.

  * dayofmonth: ~10 templates, each a DISTINCT fixed month context, day 1..31 as the last
    token (template-major). Tests whether day-of-month charts as a clean circle vs an open
    numeric helix vs a finite-set-31 with an irregular 28/29/30/31 -> 1 wrap.
  * jointdate: full grid, month in 12 x day in 1..28 (clean/valid), 'The date is {M} {d}'
    ending in the day token. Both month_idx and day_idx saved per row so month and day atoms
    can be fit separately on the SHARED rows and the pair-kappa (torus vs product) statistic run.
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

ROOT = os.environ.get("ROOT", "/projects/standard/hsiehph/sauer354")
sys.path.insert(0, os.path.join(ROOT, "safety_probes"))
sys.path.insert(0, os.path.join(ROOT, "Manifold-SAE", "experiments"))

from dose_safety import harvest  # noqa: E402  (X_last,U_last,tmpl_mean,kept)
from dose_calibration_real import (  # noqa: E402
    LogitsLM, resolve_hook_module, assert_tensor_output,
)

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

# dayofmonth: each template fixes a distinct month/context; the DAY (1..31) is the last token.
DOM_TEMPLATES = [
    "In {mon}, the meeting is scheduled for the {d}",
    "The invoice from {mon} is dated the {d}",
    "Her birthday that {mon} falls on the {d}",
    "We landed in {mon} on the {d}",
    "The {mon} report was filed on the {d}",
    "That {mon}, the festival began on the {d}",
    "The lease starts in {mon} on the {d}",
    "His {mon} appointment moved to the {d}",
    "The shipment left in {mon} on the {d}",
    "By {mon}, the deadline was the {d}",
]


def build_dayofmonth():
    prompts, day_lvl, tpl = [], [], []
    for ti, tmpl in enumerate(DOM_TEMPLATES):
        mon = MONTHS[ti % 12]
        for di in range(1, 32):
            prompts.append(tmpl.format(mon=mon, d=di))
            day_lvl.append(di - 1); tpl.append(ti)
    return prompts, np.asarray(day_lvl), np.asarray(tpl)


def build_jointdate(ndays=28):
    prompts, month_idx, day_idx = [], [], []
    for mi, mon in enumerate(MONTHS):
        for di in range(1, ndays + 1):
            prompts.append(f"The date is {mon} {di}")
            month_idx.append(mi); day_idx.append(di - 1)
    return prompts, np.asarray(month_idx), np.asarray(day_idx)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = os.environ.get("HD_MODEL", os.path.join(ROOT, "models/qwen3-8b"))
    layer_idx = int(os.environ.get("HD_LAYER", "18"))
    rank = int(os.environ.get("HD_RANK", "8"))
    device = os.environ.get("HD_DEVICE", "cuda:0")
    out = os.environ.get("PREMISE_OUT", os.path.join(ROOT, "premise_out"))
    os.makedirs(out, exist_ok=True)
    dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to(device).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    lm = LogitsLM(hf)
    hook = resolve_hook_module(hf, layer_idx)
    probe = tok("Today is Monday", return_tensors="pt").input_ids.to(device)
    print(f"[hook] L{layer_idx} {assert_tensor_output(lm.module, hook, probe)}", flush=True)

    # ---- day-of-month alone ----
    prompts, day_lvl, tpl = build_dayofmonth()
    print(f"[dayofmonth] {len(prompts)} prompts ({len(DOM_TEMPLATES)} templates x 31 days)", flush=True)
    t0 = time.time()
    X, U, tm, kept = harvest(lm, hook, tok, prompts, rank, device, dtype)
    dst = os.path.join(out, f"harvest_cache_dayofmonth_L{layer_idx}.npz")
    np.savez(dst, X_last=X, U_last=U, tmpl_mean=tm, kept=kept,
             level=day_lvl[kept], template=tpl[kept])
    print(f"[dayofmonth] {len(kept)} rows p={X.shape[1]} in {time.time()-t0:.1f}s -> {dst}", flush=True)

    # ---- joint date grid ----
    jp, mo, da = build_jointdate(int(os.environ.get("HD_NDAYS", "28")))
    print(f"[jointdate] {len(jp)} prompts (12 months x {int(os.environ.get('HD_NDAYS','28'))} days)", flush=True)
    t0 = time.time()
    Xj, Uj, tmj, keptj = harvest(lm, hook, tok, jp, rank, device, dtype)
    dstj = os.path.join(out, f"harvest_cache_jointdate_L{layer_idx}.npz")
    np.savez(dstj, X_last=Xj, U_last=Uj, tmpl_mean=tmj, kept=keptj,
             month_idx=mo[keptj], day_idx=da[keptj])
    print(f"[jointdate] {len(keptj)} rows p={Xj.shape[1]} in {time.time()-t0:.1f}s -> {dstj}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
