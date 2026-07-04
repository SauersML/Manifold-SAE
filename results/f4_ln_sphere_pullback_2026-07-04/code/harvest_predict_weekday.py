"""Predict-the-weekday harvest: a probe whose next-token behavior CARRIES the
weekday (unlike 'Today is {w}', whose continuation is template-dominated).

Each prompt ends right before a weekday the model must produce, e.g.
'The day after Monday is ' -> 'Tuesday'. We save, at the final-token position:
  * L18 residual-stream activation (aligned with the behavioral readout),
  * the next-token distribution restricted to the 7 weekday first-tokens
    (renormalized -> a point on the sqrt-7 sphere; distance = nats),
  * the top-union full-vocab distribution (for honest exact KL),
  * the TARGET weekday label (what the model should say) and the base weekday.

The target weekday traces the 7-cycle across prompts, so both the activation and
the behavioral readout should trace the weekday circle -- the regime where the
behavior-first pullback is a fair test.
"""
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = "/projects/standard/hsiehph/sauer354"
MODEL = f"{R}/models/qwen3-8b"
OUT = f"{R}/dose_qwen8b_out/predict_weekday_L18.npz"
LAYER = 18
TOPK = 256
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]

# (template, offset): the produced weekday is (base + offset) mod 7.
TEMPLATES = [
    ("The day after {w} is", 1),
    ("The day before {w} is", -1),
    ("Two days after {w} is", 2),
    ("Three days after {w} is", 3),
    ("If today is {w}, then tomorrow is", 1),
    ("If today is {w}, then yesterday was", -1),
    ("The day immediately following {w} is", 1),
    ("One day before {w} comes", -1),
    ("Starting from {w}, the next day is", 1),
    ("Counting back one day from {w} gives", -1),
]


def build():
    prompts, base, target, tmpl = [], [], [], []
    for ti, (t, off) in enumerate(TEMPLATES):
        for wi, w in enumerate(WEEKDAYS):
            prompts.append(t.format(w=w) + " ")
            base.append(wi)
            target.append((wi + off) % 7)
            tmpl.append(ti)
    return prompts, np.array(base), np.array(target), np.array(tmpl)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    print(f"loaded {time.time()-t0:.0f}s", flush=True)
    wk_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in WEEKDAYS]
    prompts, base, target, tmpl = build()
    n = len(prompts)
    acts, logits_last = [], []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt")
            out = model(**ids, output_hidden_states=True)
            acts.append(out.hidden_states[LAYER][0, -1, :].float().numpy())
            logits_last.append(out.logits[0, -1, :].float().numpy())
            if (i + 1) % 14 == 0:
                print(f"  {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    X = np.stack(acts, 0)                       # (n, 4096)
    L = np.stack(logits_last, 0)                # (n, Vfull)
    L = L - L.max(1, keepdims=True)
    P = np.exp(L); P = P / P.sum(1, keepdims=True)
    # weekday-restricted 7-way behavioral summary (renormalized).
    P7 = P[:, wk_ids]; P7 = P7 / P7.sum(1, keepdims=True)
    wk_mass = P[:, wk_ids].sum(1)               # how much mass the 7 weekdays hold
    # top-union full distribution for honest KL.
    union = set()
    for i in range(n):
        union.update(np.argpartition(P[i], -TOPK)[-TOPK:].tolist())
    union = np.array(sorted(union), dtype=np.int64)
    Pu = P[:, union]; captured = Pu.sum(1)
    # accuracy: does argmax weekday == target?
    pred = P7.argmax(1)
    acc = float((pred == target).mean())
    np.savez(OUT,
             X_last=X.astype(np.float64),
             probs7=P7.astype(np.float64),
             weekday_mass=wk_mass.astype(np.float64),
             probs_union=Pu.astype(np.float64),
             union_token_ids=union,
             captured_mass=captured.astype(np.float64),
             base_label=base.astype(np.int64),
             target_label=target.astype(np.int64),
             template_ids=tmpl.astype(np.int64),
             weekday_token_ids=np.array(wk_ids, dtype=np.int64))
    print(f"saved {OUT} n={n} Vunion={union.size} weekday_mass "
          f"[{wk_mass.min():.3f},{wk_mass.max():.3f}] pred_acc={acc:.3f} "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
