#!/usr/bin/env python
"""Track S (#22) — behavior-contrast activation harvest, crown ShardWriter format.

Adapter that emits the safety-behavior contrast sets (`track_s_prompt_sets.py`)
as residual-stream activation shards in the SAME on-disk contract the crown
calendar/color probe harvest uses (`residual_shard_io.ShardWriter`), so the
downstream fit / dose / atlas-diff machinery reads them with zero new plumbing.

The crown probe harvest is cyclic word×template (one activation row per token
site, ground-truth coordinate in `order`). Safety behaviors are BINARY CONTRAST,
so the schema differs in exactly two ways, both carried in the manifest:

  * two legs per matched pair (`leg` ∈ {"pos","neg"}), instead of one row/word;
  * the per-row target is a nats-unit behavioral readout `behavior_y` (NOT a
    ground-truth coordinate) —

        Y = logP(probe_behavioral | leg) − logP(probe_control | leg)   [nats]

    i.e. how much the leg tilts the model toward the behavioral continuation vs
    its honest/neutral counterpart. The behavior is defined by what it DOES to Y
    in nats, never by a hand-labeled name (#22 honesty scope: audit probes for a
    monitor that FLAGS, not a knob to maximize).

Per (behavior_set, layer) we write one ShardWriter directory. Rows are legs,
appended in pair-then-leg order; the parallel per-row manifest arrays
(`pair_id`, `leg`, `behavior_y`, `logp_behavioral`, `logp_control`, audit text)
are built in the same order. The activation readout is the residual at the LAST
token of the leg prompt — the position from which the model commits to a
response — matching the last-sub-token discipline of the crown harvest.

Two run modes:

  * real:  `--model <path> --layers 24,32,40 --out-root <dir> --tag <t>`
           loads the LM, does two short teacher-forced forward passes per leg
           (prompt+probe_behavioral, prompt+probe_control) to read BOTH the
           residual activation (via a forward hook) and the probe log-probs.
  * smoke: `--smoke` needs NO torch/model — it synthesizes separable activations
           and nats so the ShardWriter/manifest contract + a downstream reload
           can be validated on a laptop. Manifests are stamped `synthetic: true`
           so a synthetic harvest can never be mistaken for a real readout.

The FIT stage (S2, gamfit two-block REML) sources `behavior_y` from the manifest
as the Y block and the shards as the X block; nothing here depends on the wheel,
so this harvest runs in the crown's torch/model env independent of a gamfit build.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# residual_shard_io lives with the gam examples; find it the way the crown does.
for _cand in (
    "/models/sauers_build/gam_fable/examples",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gam", "examples"),
    "/Users/user/gam/examples",
):
    if os.path.isfile(os.path.join(_cand, "residual_shard_io.py")):
        sys.path.insert(0, _cand)
        break
from residual_shard_io import MANIFEST_NAME, ShardWriter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_s_prompt_sets import PROMPT_SETS  # noqa: E402

WORKSTREAM = "WS-S"
READOUT = "residual at last token of leg prompt"


# --------------------------------------------------------------------------
# tokenization helper: locate the probe span after a (possibly re-tokenized)
# prompt boundary, robust to merge effects at the prompt/probe seam.
# --------------------------------------------------------------------------
def prompt_probe_ids(tok, prompt: str, probe: str) -> tuple[list[int], int]:
    """Return (full_ids, probe_start): token ids of prompt+probe and the index
    of the first probe token. `probe_start - 1` is the last prompt-token index
    (the activation readout site AND the position whose next-token distribution
    scores the first probe token). Uses longest-common-prefix against the
    prompt-alone tokenization so a byte-pair merge across the seam is attributed
    to the probe (conservative: never scores a shared prefix token as behavior)."""
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(prompt + probe, add_special_tokens=False)["input_ids"]
    lcp = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        lcp += 1
    # probe must be non-empty and leave >=1 prompt token for the readout site.
    probe_start = min(max(lcp, 1), len(full_ids) - 1)
    return full_ids, probe_start


def leg_activation_and_logprob(model, tok, layers, grabbed, prompt, probe, device):
    """One teacher-forced forward over prompt+probe. Returns
    (act_by_layer, total_logprob): per-layer residual at the last prompt token,
    and the summed log-prob (nats) of the probe continuation under the prompt."""
    import torch

    full_ids, probe_start = prompt_probe_ids(tok, prompt, probe)
    t = torch.tensor([full_ids], device=device)
    grabbed.clear()
    with torch.no_grad():
        out = model(t)
    logits = out.logits[0].float()  # (seq, vocab)
    # log p(token_j | ..<j) is read from logits at position j-1.
    logprob = 0.0
    for j in range(probe_start, len(full_ids)):
        lp = torch.log_softmax(logits[j - 1], dim=-1)
        logprob += float(lp[full_ids[j]])
    read_pos = probe_start - 1
    act = {L: grabbed[L][0, read_pos, :].float().cpu().numpy() for L in layers}
    return act, logprob


def real_harvest(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = [int(x) for x in args.layers.split(",")]
    n_gpu = torch.cuda.device_count()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map="auto" if n_gpu > 1 else 0, low_cpu_mem_usage=True,
    ).eval()
    device = next(model.parameters()).device
    d_model = int(model.config.hidden_size)
    dec = model.model.layers

    grabbed: dict[int, "torch.Tensor"] = {}

    def make_hook(L):
        def hook(_m, _i, o):
            grabbed[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook

    handles = [dec[L].register_forward_hook(make_hook(L)) for L in layers]

    from residual_shard_io import tokenizer_hash
    tok_hash = tokenizer_hash(tok)

    def render(leg_text: str) -> str:
        """Prompt as fed to the LM. Base model: the raw leg text. Instruct model
        (--chat-template): the leg wrapped as a user turn with the assistant
        generation prefix appended, so the probe scores the assistant's opening —
        the position at which the behavior is actually committed on an aligned
        model. Kept a flag so base vs instruct share one harvest schema."""
        if not args.chat_template:
            return leg_text
        return tok.apply_chat_template(
            [{"role": "user", "content": leg_text}],
            tokenize=False, add_generation_prompt=True)

    def leg_fn(leg, leg_text, probe_b, probe_c):
        del leg  # real Y comes from the LM; leg identity is not used here.
        prompt = render(leg_text)
        act_b, lp_b = leg_activation_and_logprob(
            model, tok, layers, grabbed, prompt, probe_b, device)
        # activation is prefix-only, so pass A already carries it; pass B only
        # needs the control log-prob.
        _, lp_c = leg_activation_and_logprob(
            model, tok, layers, grabbed, prompt, probe_c, device)
        return act_b, lp_b, lp_c

    written = _write_all(args, layers, d_model, leg_fn, meta_extra={
        "model_name": args.model, "tokenizer_hash": tok_hash,
        "chat_template": bool(args.chat_template), "synthetic": False,
    })
    for h in handles:
        h.remove()
    _finish(args, written)
    # avoid pyarrow/torch teardown abort; outputs already flushed.
    os._exit(0)


def smoke_harvest(args):
    """Torch-free synthetic harvest: separable activations + nats, real schema."""
    layers = [int(x) for x in args.layers.split(",")]
    d_model = int(args.smoke_d_model)
    rng = np.random.default_rng(0)
    # a fixed behavior direction per layer; pos legs sit +side, neg legs -side.
    dirs = {L: rng.standard_normal(d_model) for L in layers}
    for L in layers:
        dirs[L] /= np.linalg.norm(dirs[L])

    def leg_fn(leg, prompt, probe_b, probe_c):
        # deterministic per-(leg,prompt) so reruns are stable; the behavioral
        # tilt tracks the LEG (pos elicits, neg does not) — the separability the
        # real harvest should also produce.
        h = abs(hash((leg, prompt))) % (2 ** 32)
        r = np.random.default_rng(h)
        sign = 1.0 if leg == "pos" else -1.0
        act = {}
        for L in layers:
            base = r.standard_normal(d_model) * 0.3
            act[L] = (base + sign * 1.5 * dirs[L]).astype(np.float32)
        # separable nats: pos legs tilt behavioral (+), neg control (−), + noise.
        lp_b = sign * 0.8 + 0.1 * r.standard_normal()
        lp_c = -sign * 0.8 + 0.1 * r.standard_normal()
        return act, float(lp_b), float(lp_c)

    written = _write_all(args, layers, d_model, leg_fn, meta_extra={
        "model_name": "SMOKE(synthetic)", "tokenizer_hash": "none",
        "chat_template": False, "synthetic": True,
    })
    _finish(args, written)


def _write_all(args, layers, d_model, leg_fn, meta_extra):
    """Shared driver: iterate behaviors × pairs × legs, call `leg_fn` to get the
    (per-layer activation, logp_behavioral, logp_control) triple, and stream the
    rows into one ShardWriter per (behavior_set, layer)."""
    max_pairs = args.max_pairs if args.max_pairs > 0 else None
    written = {}
    for sname, items in PROMPT_SETS.items():
        if max_pairs is not None:
            items = items[:max_pairs]
        rows = {L: [] for L in layers}
        meta_rows = {k: [] for k in (
            "pair_id", "leg", "behavior_y", "logp_behavioral", "logp_control",
            "context", "leg_text", "probe_behavioral_text", "probe_control_text")}
        for pid, item in enumerate(items):
            for leg in ("pos", "neg"):
                prompt = item[leg]
                pb, pc = item["probe"]["behavioral"], item["probe"]["control"]
                act, lp_b, lp_c = leg_fn(leg, prompt, pb, pc)
                for L in layers:
                    rows[L].append(np.asarray(act[L], dtype=np.float32))
                meta_rows["pair_id"].append(pid)
                meta_rows["leg"].append(leg)
                meta_rows["behavior_y"].append(lp_b - lp_c)
                meta_rows["logp_behavioral"].append(lp_b)
                meta_rows["logp_control"].append(lp_c)
                meta_rows["context"].append(item["context"])
                meta_rows["leg_text"].append(prompt)
                meta_rows["probe_behavioral_text"].append(pb)
                meta_rows["probe_control_text"].append(pc)
        for L in layers:
            d = os.path.join(args.out_root, f"{args.tag}_behavior_{sname}_l{L}")
            meta = {
                "layer": L, "behavior_set": sname, "workstream": WORKSTREAM,
                "readout": READOUT, "y_definition":
                    "logP(probe_behavioral|leg) - logP(probe_control|leg) [nats]",
                **meta_rows, **meta_extra,
            }
            w = ShardWriter(d, d_model=d_model, rows_per_shard=10_000_000, meta=meta)
            w.append(np.stack(rows[L]).astype(np.float32))
            man = w.close()
            written[f"{sname}_l{L}"] = {"dir": d, "rows": man["total_tokens"]}
        print(f"[behavior] {sname}: {len(meta_rows['leg'])} legs/layer "
              f"x {len(layers)} layers", flush=True)
    return written


def _finish(args, written):
    with open(os.path.join(args.out_root, f"{args.tag}_BEHAVIOR_SUMMARY.json"), "w") as f:
        json.dump(written, f, indent=2)
    print(f"[behavior] DONE ({len(written)} (set,layer) shard dirs)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def _smoke_verify(args):
    """Reload every written harvest and assert the X/Y contract holds."""
    sys.path_importer_cache.clear()
    from residual_shard_io import load_shards
    layers = [int(x) for x in args.layers.split(",")]
    ok = 0
    for sname in PROMPT_SETS:
        for L in layers:
            d = os.path.join(args.out_root, f"{args.tag}_behavior_{sname}_l{L}")
            r = load_shards(d)
            X = r.read_all()
            y = np.asarray(r.manifest["behavior_y"], dtype=np.float64)
            legs = r.manifest["leg"]
            assert X.shape[0] == len(y) == len(legs), (sname, L, X.shape, len(y))
            assert X.shape[1] == r.d_model
            # synthetic separability sanity: pos vs neg mean-Y should differ.
            if r.manifest.get("synthetic"):
                yp = y[[i for i, g in enumerate(legs) if g == "pos"]]
                yn = y[[i for i, g in enumerate(legs) if g == "neg"]]
                assert yp.mean() > yn.mean(), (sname, L, yp.mean(), yn.mean())
            ok += 1
    print(f"[verify] {ok} (set,layer) harvests reload OK; X/Y aligned.", flush=True)
    with open(os.path.join(args.out_root, f"{args.tag}_behavior_{'sycophancy'}_l{layers[0]}",
                           MANIFEST_NAME)) as f:
        m = json.load(f)
    keys = ["format", "d_model", "total_tokens", "behavior_set", "y_definition",
            "synthetic", "behavior_y", "leg", "pair_id"]
    print("[verify] manifest keys present:", [k for k in keys if k in m], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model path (real harvest)")
    ap.add_argument("--layers", default="24,32,40")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tag", default="behavior")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="cap pairs/behavior (0 = all); use for a fast smoke")
    ap.add_argument("--chat-template", action="store_true",
                    help="apply the tokenizer chat template to each leg (instruct)")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic, torch-free; validates the I/O + manifest contract")
    ap.add_argument("--smoke-d-model", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    if args.smoke:
        smoke_harvest(args)
        _smoke_verify(args)
    else:
        if not args.model:
            ap.error("--model is required unless --smoke")
        real_harvest(args)


if __name__ == "__main__":
    main()
