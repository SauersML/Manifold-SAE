"""Harvest next-token distributions for the 70 weekday prompts (CPU forward).

The behavioral summary of each weekday-token activation is the model's
next-token distribution at that (final) position — the SAME quantity the dose
experiment's measured output-KL scores. We save the probability mass over the
union of each row's top-K tokens (captures ~all between-row variation) so the
sphere-tangent behavior chart is over a manageable token set, plus the captured
mass per row for an honest faithfulness report.

Row order matches weekday_probe_harvest.build_prompts (10 templates x 7 days),
same as harvest_cache_weekday_L18_n70.npz's X_last, so behavior rows are
activation-aligned.
"""
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = "/projects/standard/hsiehph/sauer354"
MODEL = f"{R}/models/qwen3-8b"
OUT = f"{R}/dose_qwen8b_out/behavior_nexttoken_weekday_L18_n70.npz"
TOPK = 256

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
# Templates must match weekday_probe_harvest.py exactly (row alignment with X_last).
TEMPLATES = [
    "Today is {w}",
    "The day after tomorrow is {w}",
    "My favorite day of the week is {w}",
    "The meeting is scheduled for {w}",
    "Yesterday was {w}",
    "We will travel on {w}",
    "The store is closed on {w}",
    "Her birthday falls on {w}",
    "The exam takes place on {w}",
    "It always rains on {w}",
]

def build_prompts():
    prompts, labels, tids = [], [], []
    for ti, tmpl in enumerate(TEMPLATES):
        for wi, w in enumerate(WEEKDAYS):
            prompts.append(tmpl.format(w=w))
            labels.append(wi)
            tids.append(ti)
    return prompts, np.asarray(labels), np.asarray(tids)

def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32).eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
    prompts, labels, tids = build_prompts()
    n = len(prompts)
    last_logits = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt")
            out = model(**ids)
            lg = out.logits[0, -1, :].float().numpy()  # (V,)
            last_logits.append(lg)
            if i % 10 == 0:
                print(f"  {i}/{n}  {time.time()-t0:.1f}s", flush=True)
    L = np.stack(last_logits, 0)  # (n, V)
    # Full softmax per row.
    L = L - L.max(1, keepdims=True)
    P = np.exp(L)
    P = P / P.sum(1, keepdims=True)  # (n, Vfull)
    # Union of each row's top-K token ids.
    union = set()
    for i in range(n):
        top = np.argpartition(P[i], -TOPK)[-TOPK:]
        union.update(top.tolist())
    union = np.array(sorted(union), dtype=np.int64)
    Pr = P[:, union]  # (n, V)
    captured = Pr.sum(1)  # mass captured per row
    # Also record the first-subtoken ids of each weekday word (for reference).
    wk_ids = []
    for w in WEEKDAYS:
        wk_ids.append(tok(" " + w, add_special_tokens=False)["input_ids"][0])
    np.savez(OUT,
             probs=Pr.astype(np.float64),
             token_ids=union,
             captured_mass=captured.astype(np.float64),
             labels=labels.astype(np.int64),
             template_ids=tids.astype(np.int64),
             weekday_first_token_ids=np.asarray(wk_ids, dtype=np.int64))
    print(f"saved {OUT}  V={union.size}  captured "
          f"[{captured.min():.4f},{captured.max():.4f}]  "
          f"{time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
