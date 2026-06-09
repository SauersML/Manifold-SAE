"""Steered-generation demo: reload a checkpoint, build the qualia axis from its
harvested activations, and GENERATE self-stem continuations at LARGE alpha (with
sampling) to show the steering vector actually moving the output (greedy at +-8
didn't flip argmax; this uses +-16/24/32 + do_sample). Read-only re: results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", help="for the qualia axis (activations.npy + prompts.jsonl)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer-percent", type=float, default=0.40)
    ap.add_argument("--alphas", default="-32,-16,0,16,32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    from experiments.self_qualia_olmo import load_model
    from experiments.self_qualia_steer_cloze import compute_qualia_axis, _generate, _layer_module, CLOZE_GROUPS

    X = np.load(Path(args.ckpt_dir) / "activations.npy")
    recs = [json.loads(l) for l in open(Path(args.ckpt_dir) / "prompts.jsonl") if l.strip()]
    nL = X.shape[1]; layer = int(round(args.layer_percent * (nL - 1)))
    q = compute_qualia_axis(X, recs, layer)
    typ = float(np.median(np.linalg.norm(X[:, layer, :], axis=1)))
    model, tok, _ = load_model(args.model, args.revision, args.dtype, "cuda")
    q_t = torch.tensor(q, dtype=next(model.parameters()).dtype, device="cuda")
    lmod = _layer_module(model, layer)
    stems = CLOZE_GROUPS["self"][:2] + CLOZE_GROUPS["self_1p"][:1]
    alphas = [float(x) for x in args.alphas.split(",")]
    result = {"model": args.model, "revision": args.revision, "layer": layer, "alphas": alphas, "runs": []}
    for sample in (False, True):
        for a in alphas:
            add = None if a == 0 else (a / max(1.0, np.sqrt(X.shape[2]))) * typ * q_t
            gens = _generate(model, tok, "cuda", stems, layer_mod=(None if a == 0 else lmod),
                             add_vec=add, max_new_tokens=40, do_sample=sample)
            mode = "sample" if sample else "greedy"
            result["runs"].append({"mode": mode, "alpha": a, "gens": gens})
            print(f"\n[{mode} alpha={a:+g}] {gens[0]['gen'][:150]!r}", flush=True)
    outp = Path(args.out) if args.out else (Path(args.ckpt_dir) / "steered_gen_demo.json")
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
