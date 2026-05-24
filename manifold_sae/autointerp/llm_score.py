"""LLM-based simulation scoring (Bills 2023 / Paulo 2024 style).

Given an atom's natural-language explanation, ask the LLM to predict
whether a held-out (color, template) example would activate the atom.
Score = accuracy / AUC of predicted vs ground-truth activations.

Sits alongside `score.py`'s regression-based simulation R²:
  - score.py            : feature-engineered R², deterministic, free
  - llm_score.py        : LLM simulation accuracy, costs API tokens

Uses the same prompt-caching trick as `llm_explain.py`: cache the system
prompt + the explanation block (per-atom, reusable across many examples).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .llm_explain import (
    DEFAULT_MODEL,
    LLMAtomExplanation,
    PRICE_CACHE_READ_PER_MTOK,
    PRICE_CACHE_WRITE_PER_MTOK,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    _make_client,
)


SCORE_SYSTEM_PROMPT = """You are simulating an SAE atom. You will be given:
  1. A natural-language hypothesis of what the atom detects.
  2. A list of held-out (color, template) examples.

For EACH example, output 1 if the atom would activate (above its firing
threshold), else 0. Output STRICT JSON: a list of 0/1 ints in the same
order as the examples. No prose, no fences."""


@dataclass
class SimulationScore:
    atom_id: int
    model_name: str
    n_examples: int
    accuracy: float
    n_positive_true: int
    n_positive_pred: int
    raw_predictions: list[int]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def render_score_prompt(
    explanation: str,
    examples: list[dict],
    color_hsv_lookup: Optional[np.ndarray] = None,
) -> tuple[str, str]:
    """Returns (cached_block_text, fresh_user_text).

    cached_block_text: explanation (reusable across examples for the SAME atom).
    fresh_user_text  : just the examples list (unique per request).
    """
    cached = f"HYPOTHESIS for this atom:\n  {explanation}\n"
    lines = ["EXAMPLES (predict 0/1 in order):"]
    for i, e in enumerate(examples):
        ci = e.get("color_idx", -1)
        if color_hsv_lookup is not None and 0 <= ci < len(color_hsv_lookup):
            h, s, v = color_hsv_lookup[ci]
            lines.append(f"  [{i}] color='{e['color']}' (HSV=({h:.2f},{s:.2f},{v:.2f})) "
                         f"template={e['template_id']:2d}")
        else:
            lines.append(f"  [{i}] color='{e['color']}' template={e['template_id']:2d}")
    lines.append("")
    lines.append(f"Return ONLY a JSON list of exactly {len(examples)} integers (0 or 1).")
    return cached, "\n".join(lines)


_JSON_LIST_RE = re.compile(r"\[\s*[\d,\s]*\]", re.DOTALL)


def parse_score_response(text: str, n_expected: int) -> list[int]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_LIST_RE.search(text)
        if not m:
            raise ValueError(f"could not parse list from: {text[:160]!r}")
        v = json.loads(m.group(0))
    if not isinstance(v, list):
        raise ValueError(f"expected list, got {type(v)}")
    out = [int(bool(int(x))) for x in v[:n_expected]]
    # pad with 0s if model truncated
    while len(out) < n_expected:
        out.append(0)
    return out


def call_score_batch(
    client: Any,
    cached_block: str,
    user_message: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> tuple[str, dict]:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SCORE_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": cached_block,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            text = ""
            for block in resp.content:
                t = getattr(block, "text", None)
                if t:
                    text += t
            usage = getattr(resp, "usage", None)
            udict = {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_read_input_tokens":
                    getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens":
                    getattr(usage, "cache_creation_input_tokens", 0) or 0,
            }
            return text, udict
        except Exception as e:  # noqa: BLE001
            last_err = e
            status = getattr(e, "status_code", None)
            msg = str(e)
            retriable = (
                status in (429, 500, 502, 503, 504)
                or "rate" in msg.lower()
                or "overloaded" in msg.lower()
            )
            if not retriable or attempt == max_retries - 1:
                raise
            time.sleep(backoff_base ** attempt)
    raise RuntimeError(f"unreachable; last={last_err}")


def llm_score_atom(
    exp: LLMAtomExplanation,
    eval_examples: list[dict],
    ground_truth: Sequence[int],
    *,
    color_hsv_lookup: Optional[np.ndarray] = None,
    examples_per_call: int = 20,
    llm_model: str = DEFAULT_MODEL,
    client: Optional[Any] = None,
    max_retries: int = 3,
) -> SimulationScore:
    """Run LLM-simulation for one atom against held-out examples.

    eval_examples : list of {"color", "color_idx", "template_id"} dicts.
    ground_truth  : 0/1 list, parallel to eval_examples (1 = atom actually fired).
    """
    if client is None:
        client = _make_client()
    assert len(eval_examples) == len(ground_truth)

    all_preds: list[int] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0,
                   "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    for start in range(0, len(eval_examples), examples_per_call):
        chunk = eval_examples[start:start + examples_per_call]
        cached, user_msg = render_score_prompt(
            exp.explanation, chunk, color_hsv_lookup=color_hsv_lookup,
        )
        try:
            text, usage = call_score_batch(
                client, cached, user_msg, model=llm_model, max_retries=max_retries,
            )
            preds = parse_score_response(text, n_expected=len(chunk))
        except Exception:
            preds = [0] * len(chunk)
            usage = {k: 0 for k in total_usage}
        all_preds.extend(preds)
        for k in total_usage:
            total_usage[k] += usage[k]

    gt = np.asarray(ground_truth, dtype=np.int32)
    pr = np.asarray(all_preds, dtype=np.int32)
    acc = float((gt == pr).mean()) if len(gt) > 0 else 0.0

    return SimulationScore(
        atom_id=exp.atom_id,
        model_name=exp.model_name,
        n_examples=len(gt),
        accuracy=acc,
        n_positive_true=int(gt.sum()),
        n_positive_pred=int(pr.sum()),
        raw_predictions=all_preds,
        input_tokens=total_usage["input_tokens"],
        output_tokens=total_usage["output_tokens"],
        cache_read_tokens=total_usage["cache_read_input_tokens"],
        cache_creation_tokens=total_usage["cache_creation_input_tokens"],
    )


def build_eval_examples(
    acts_val: np.ndarray,
    atom_id: int,
    row_color: np.ndarray,
    row_template: np.ndarray,
    color_names: list[str],
    n_pos: int = 10,
    n_neg: int = 10,
    seed: int = 0,
) -> tuple[list[dict], list[int]]:
    """Stratified eval set: top-n_pos firing + n_neg sampled non-firing rows."""
    col = acts_val[:, atom_id]
    pos_idx = np.where(col > 0)[0]
    neg_idx = np.where(col <= 0)[0]
    rng = np.random.default_rng(seed)
    # take strongest positives — avoid noise on borderline
    if len(pos_idx) > 0:
        pos_order = pos_idx[np.argsort(-col[pos_idx])][:n_pos]
    else:
        pos_order = np.array([], dtype=np.int64)
    if len(neg_idx) >= n_neg:
        neg_order = rng.choice(neg_idx, size=n_neg, replace=False)
    else:
        neg_order = neg_idx
    all_idx = np.concatenate([pos_order, neg_order])
    rng.shuffle(all_idx)
    examples = []
    gt = []
    for r in all_idx:
        examples.append({
            "color": color_names[int(row_color[r])],
            "color_idx": int(row_color[r]),
            "template_id": int(row_template[r]),
        })
        gt.append(1 if col[r] > 0 else 0)
    return examples, gt


def estimate_score_cost(scores: Sequence[SimulationScore]) -> dict:
    in_tok = sum(s.input_tokens for s in scores)
    out_tok = sum(s.output_tokens for s in scores)
    cache_r = sum(s.cache_read_tokens for s in scores)
    cache_w = sum(s.cache_creation_tokens for s in scores)
    cost = (
        in_tok * PRICE_INPUT_PER_MTOK
        + out_tok * PRICE_OUTPUT_PER_MTOK
        + cache_r * PRICE_CACHE_READ_PER_MTOK
        + cache_w * PRICE_CACHE_WRITE_PER_MTOK
    ) / 1_000_000.0
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_read_tokens": cache_r,
        "cache_creation_tokens": cache_w,
        "usd": cost,
    }
