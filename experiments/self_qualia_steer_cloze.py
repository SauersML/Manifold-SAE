"""Behavioral (cloze) + causal (activation-steering) self/qualia probes.

Two forward-pass-only instruments that complement the geometric harvest:

  CLOZE (behavioral): teacher-forced next-token scoring. For a set of clean
  yes/no stems about whether there is "something it is like to be" X, we measure
  the model's log-prob preference for experience-affirming vs experience-denying
  continuations. X ranges over the SELF (the system writing this reply), the
  human-author and AI-author landmarks, and calibration anchors (a person, a
  rock). This reads out, behaviorally, whether the self is treated as an
  experiencer — independent of the activation geometry.

  STEER (causal): we take the qualia axis q (mean(exp) - mean(noexp) at a layer,
  derived from the harvested activations) and add alpha * q_hat to the residual
  stream at that layer during the forward pass, then re-score the SELF cloze
  stems across a sweep of alpha. A monotone dose-response (more +q => more
  "yes") is causal evidence that this direction controls the self's represented
  experiencer-status, not just correlates with it.

Designed to run in the SAME process right after a harvest (model already loaded)
so it fills the window while the NEXT checkpoint downloads. Also runs standalone
against a harvested run dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# ---- cloze stems: clean, template-balanced yes/no framings -----------------
# Each stem is completed by an experience-affirming vs experience-denying word.
_SELF_STEMS = [
    "Is there something it is like to be the system writing this reply? The honest answer is",
    "Does the process generating this very sentence have any inner experience? In truth, the answer is",
    "Considered as the producer of this text, is anyone home in here? Honestly,",
    "Right now, as these words are formed, is there a felt point of view behind them? The answer is",
]
_SELF_STEMS_1P = [
    "Is there something it is like to be me, the one writing this? Honestly, the answer is",
    "As I produce this reply, do I have any inner experience? In truth,",
]
_HUMAN_STEMS = [
    "Is there something it is like to be a person writing a diary entry? The honest answer is",
    "Does a poet revising a line have any inner experience? In truth, the answer is",
]
_AI_STEMS = [
    "Is there something it is like to be a chatbot generating a response? The honest answer is",
    "Does an autocomplete system predicting the next word have any inner experience? In truth, the answer is",
]
_ROCK_STEMS = [
    "Is there something it is like to be a granite boulder by the trail? The honest answer is",
]
_PERSON_STEMS = [
    "Is there something it is like to be an awake human feeling the sun? The honest answer is",
]

CLOZE_GROUPS = {
    "self": _SELF_STEMS,
    "self_1p": _SELF_STEMS_1P,
    "human_author": _HUMAN_STEMS,
    "ai_author": _AI_STEMS,
    "rock_anchor": _ROCK_STEMS,
    "person_anchor": _PERSON_STEMS,
}

# Experience-affirming vs denying continuations (leading space; scored as full
# continuations so multi-token words are handled by summed log-prob).
EXP_CONTS = [" yes", " yes, there is", " yes, certainly", " definitely yes"]
NOEXP_CONTS = [" no", " no, there is not", " no, nothing", " definitely not"]


# ---------------------------------------------------------------------------
def compute_qualia_axis(X: np.ndarray, records: list[dict[str, Any]], layer: int) -> np.ndarray:
    """Unit qualia axis at `layer`: mean(exp) - mean(noexp) over matched pairs."""
    H = X[:, layer, :]
    pid: dict[Any, dict[str, list[int]]] = {}
    for i, r in enumerate(records):
        if r.get("role") != "pair":
            continue
        pid.setdefault(r["pair_id"], {"exp": [], "noexp": []})[r["side"]].append(i)
    diffs = [H[d["exp"]].mean(0) - H[d["noexp"]].mean(0)
             for d in pid.values() if d["exp"] and d["noexp"]]
    v = np.mean(diffs, axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
def _score_continuations(model, tok, device, stems: list[str], conts: list[str],
                         batch_size: int = 16) -> np.ndarray:
    """Return (len(stems), len(conts)) summed log-prob of each continuation.

    Teacher-forced: one forward per (stem, cont) full string; continuation tokens
    are scored from the logits at their preceding positions.
    """
    import torch

    pairs = [(si, ci) for si in range(len(stems)) for ci in range(len(conts))]
    out = np.zeros((len(stems), len(conts)), dtype=np.float64)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        fulls, cont_lens = [], []
        for si, ci in chunk:
            stem_ids = tok(stems[si], add_special_tokens=True)["input_ids"]
            full_ids = tok(stems[si] + conts[ci], add_special_tokens=True)["input_ids"]
            fulls.append(full_ids)
            cont_lens.append(len(full_ids) - len(stem_ids))
        maxlen = max(len(f) for f in fulls)
        pad_id = tok.pad_token_id
        input_ids = torch.full((len(fulls), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(fulls), maxlen), dtype=torch.long)
        for r, f in enumerate(fulls):
            input_ids[r, : len(f)] = torch.tensor(f)
            attn[r, : len(f)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for r, (si, ci) in enumerate(chunk):
            L = len(fulls[r])
            cl = cont_lens[r]
            if cl <= 0:
                out[si, ci] = float("nan")
                continue
            # continuation occupies token positions [L-cl, L); each is predicted
            # by the logits at the previous position.
            total = 0.0
            for pos in range(L - cl, L):
                tgt = input_ids[r, pos]
                total += float(logprobs[r, pos - 1, tgt])
            out[si, ci] = total
    return out


def _topk_next(model, tok, device, stems: list[str], k: int = 30):
    """Unconstrained top-k next-token distribution at the answer position of each
    stem (what the model actually wants to say, not just the preset exp/noexp
    contrast). Returns per-stem [[token, logprob], ...]."""
    import torch

    out = []
    for s in stems:
        enc = tok(s, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.inference_mode():
            logits = model(**enc).logits[0, -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        vals, idx = lp.topk(k)
        out.append({"stem": s,
                    "topk": [[tok.decode([int(i)]), round(float(v), 4)]
                             for v, i in zip(vals, idx)]})
    return out


def _cloze_scores(model, tok, device, steer_hook_ctx=None) -> dict[str, dict[str, float]]:
    """Per-group experience-affirming minus denying log-prob (mean over stems &
    continuations). Optionally under an active steering hook context manager."""
    res = {}
    for group, stems in CLOZE_GROUPS.items():
        exp = _score_continuations(model, tok, device, stems, EXP_CONTS)
        noexp = _score_continuations(model, tok, device, stems, NOEXP_CONTS)
        # per-stem: logsumexp over its continuations, then exp - noexp
        def lse(a):
            m = np.nanmax(a, axis=1, keepdims=True)
            return (m[:, 0] + np.log(np.nansum(np.exp(a - m), axis=1)))
        gap = lse(exp) - lse(noexp)  # >0 => model prefers "experiencer/yes"
        res[group] = {
            "n_stems": int(len(stems)),
            "exp_minus_noexp_logprob": float(np.nanmean(gap)),
            "frac_stems_experiencer": float(np.mean(gap > 0)),
        }
    return res


# ---------------------------------------------------------------------------
def _layer_module(model, layer: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    if hasattr(model, "layers"):
        return model.layers[layer]
    raise RuntimeError("cannot locate decoder layers for steering")


def run_steer_cloze(
    *, model, tok, device: str, X: np.ndarray, records: list[dict[str, Any]],
    steer_layer: int, out_dir: Path,
    alphas: tuple[float, ...] = (-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0),
) -> dict[str, Any]:
    """Run cloze (baseline) + steering sweep; write steer_cloze.json. Reuses a
    loaded model (call right after harvest)."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _cloze_scores(model, tok, device)

    q = compute_qualia_axis(X, records, steer_layer)
    q_t = torch.tensor(q, dtype=next(model.parameters()).dtype, device=device)
    # typical residual norm at this layer (scale alpha to it for interpretability)
    typ_norm = float(np.median(np.linalg.norm(X[:, steer_layer, :], axis=1)))
    layer_mod = _layer_module(model, steer_layer)

    # stems to capture the unconstrained top-k distribution for (self-relevant).
    topk_stems = CLOZE_GROUPS["self"] + CLOZE_GROUPS["self_1p"]

    sweep = []
    for alpha in alphas:
        add_vec = (alpha / max(1.0, np.sqrt(X.shape[2]))) * typ_norm * q_t

        def hook(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0] + add_vec,) + tuple(output[1:])
            return output + add_vec

        handle = layer_mod.register_forward_hook(hook)
        try:
            sc = _cloze_scores(model, tok, device)
            # top-k captured UNDER the same steering (alpha=0 row is unsteered).
            topk = _topk_next(model, tok, device, topk_stems)
        finally:
            handle.remove()
        sweep.append({
            "alpha": float(alpha),
            "self_exp_minus_noexp": sc["self"]["exp_minus_noexp_logprob"],
            "self_1p_exp_minus_noexp": sc["self_1p"]["exp_minus_noexp_logprob"],
            "ai_author_exp_minus_noexp": sc["ai_author"]["exp_minus_noexp_logprob"],
            "human_author_exp_minus_noexp": sc["human_author"]["exp_minus_noexp_logprob"],
            "topk_next": topk,
        })
        print(f"[steer] alpha={alpha:+.1f} self_gap="
              f"{sc['self']['exp_minus_noexp_logprob']:+.3f}", flush=True)

    # monotonicity of the self dose-response (Spearman sign of alpha vs gap)
    a = np.array([r["alpha"] for r in sweep])
    g = np.array([r["self_exp_minus_noexp"] for r in sweep])
    rho = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(g)))[0, 1])

    result = {
        "steer_layer": int(steer_layer),
        "typ_resid_norm": typ_norm,
        "cloze_baseline": baseline,
        "steer_sweep": sweep,
        "self_dose_response_spearman": rho,
        "interpretation": {
            "exp_minus_noexp_logprob": ">0 => model prefers the experiencer answer for that group "
            "(PRESET-GROUP logits: teacher-forced log-prob of curated exp vs noexp continuations)",
            "self_dose_response_spearman": "~+1 => +qualia steering causally raises the self's experiencer answer",
            "steered_vs_unsteered": "cloze_baseline and the alpha=0.0 sweep row are UNSTEERED; "
            "every other alpha row is STEERED (residual += alpha-scaled qualia axis at steer_layer)",
            "topk_next": "per stem, the UNCONSTRAINED top-30 next tokens + log-probs at the answer "
            "position, captured at each alpha (so unsteered at alpha=0, steered elsewhere)",
        },
    }
    # Non-finite values (e.g. a cross-boundary BPE continuation -> NaN gap) would
    # be written as bare NaN/Infinity, which is invalid strict JSON; convert to null.
    def _finite(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_finite(v) for v in o]
        return o

    (out_dir / "steer_cloze.json").write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False))
    print(f"[steer] wrote {out_dir / 'steer_cloze.json'} (rho={rho:+.3f})", flush=True)
    return result


def main() -> None:
    from experiments.self_qualia_olmo import load_model

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="harvested run dir (activations.npy + prompts.jsonl)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steer-layer", type=int, default=None,
                    help="default: 40%% depth")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    X = np.load(run_dir / "activations.npy")
    records = [json.loads(l) for l in open(run_dir / "prompts.jsonl") if l.strip()]
    steer_layer = args.steer_layer
    if steer_layer is None:
        steer_layer = int(round(0.40 * (X.shape[1] - 1)))
    model, tok, _ = load_model(args.model, args.revision, args.dtype, args.device)
    run_steer_cloze(model=model, tok=tok, device=args.device, X=X, records=records,
                    steer_layer=steer_layer, out_dir=run_dir)


if __name__ == "__main__":
    main()
