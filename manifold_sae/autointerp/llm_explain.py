"""LLM-based autointerp explainer (Anthropic API).

Bills 2023 / Paulo 2024 protocol:
  For each SAE atom, collect top-N activating (color, template) pairs,
  prompt Claude with the examples, and ask for a structured JSON
  explanation. Batching: pack several atoms per request to amortize
  latency; prompt-cache the system prompt + per-model rubric so only
  the per-atom payload is uncached.

This sits ALONGSIDE the rule-based `explain.py` (intentionally augmenting,
not replacing). Output type `LLMAtomExplanation` is converted to
`AtomHypothesis` so downstream scoring (`score.py` / `llm_score.py`) is
unchanged.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .explain import AtomHypothesis


# Default model — Haiku 4.5 (cheap, fast, good for short structured prompts).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Pricing (USD per 1M tokens) for cost-estimate reporting. Approximate.
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00
PRICE_CACHE_READ_PER_MTOK = 0.10
PRICE_CACHE_WRITE_PER_MTOK = 1.25


SYSTEM_PROMPT = """You are an interpretability researcher analyzing Sparse Autoencoder atoms trained on color-name LLM activations.

Each atom fires on (color, template) pairs. Your job: given the top activating examples for an atom, hypothesize what concept it detects.

Output STRICT JSON only, one object per atom, in a JSON array. Each object MUST have keys:
  - "atom_id"            : int, matches the input
  - "explanation"        : string, one concise sentence
  - "hue_range_HSV"      : [low, high] in [0,1] (wrap allowed: low>high means circular)
  - "saturation_range"   : [low, high] in [0,1]
  - "lightness_range"    : [low, high] in [0,1] (V channel)
  - "name_pattern_regex" : python regex over color name, or "" if no name pattern
  - "template_pattern"   : list of int template ids that activate (subset of 0..27), or [] if all
  - "confidence_0_1"     : float in [0,1]

No prose outside the JSON. No markdown fences."""


USER_INSTRUCTIONS = """Below are atoms. Each atom has its top-activating (color, template, activation) examples.
Produce one JSON object per atom in a single JSON array, in the same order as listed."""


@dataclass
class LLMAtomExplanation:
    atom_id: int
    model_name: str
    explanation: str
    hue_range: tuple[float, float]
    saturation_range: tuple[float, float]
    lightness_range: tuple[float, float]
    name_pattern_regex: str
    template_pattern: list[int]
    confidence: float
    top_examples: list[dict]
    n_active: int
    raw_response: str = ""
    # token accounting (per batch, attributed equally to each atom in the batch)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


# ----------------------------------------------------------------------
# Prompt rendering
# ----------------------------------------------------------------------


def _format_examples(top_examples: list[dict], color_hsv_lookup: Optional[np.ndarray] = None) -> str:
    lines = []
    for e in top_examples:
        ci = e.get("color_idx", -1)
        if color_hsv_lookup is not None and 0 <= ci < len(color_hsv_lookup):
            h, s, v = color_hsv_lookup[ci]
            lines.append(f"  - color='{e['color']}' (HSV=({h:.2f},{s:.2f},{v:.2f})) "
                         f"template={e['template_id']:2d} act={e['act']:.3f}")
        else:
            lines.append(f"  - color='{e['color']}' template={e['template_id']:2d} act={e['act']:.3f}")
    return "\n".join(lines)


def render_batch_prompt(
    atoms_payload: list[dict],
    color_hsv_lookup: Optional[np.ndarray] = None,
) -> str:
    """Render a batch of atoms into a single user-message payload.

    atoms_payload: list of {"atom_id": int, "top_examples": [...]}.
    """
    parts = [USER_INSTRUCTIONS, ""]
    for a in atoms_payload:
        parts.append(f"### atom_id={a['atom_id']}  (n_examples={len(a['top_examples'])})")
        parts.append(_format_examples(a["top_examples"], color_hsv_lookup))
        parts.append("")
    parts.append("Return a JSON array with exactly "
                 f"{len(atoms_payload)} objects, one per atom, in order.")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------


_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def parse_llm_response(text: str) -> list[dict]:
    """Extract a JSON array from a model response, tolerating fences/prose."""
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # last-ditch: single object → wrap
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]!r}")


def _coerce_range(val: Any, default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            return float(val[0]), float(val[1])
        except (TypeError, ValueError):
            return default
    return default


def _coerce_atom(raw: dict, atom_id: int, model_name: str,
                 top_examples: list[dict]) -> LLMAtomExplanation:
    return LLMAtomExplanation(
        atom_id=int(raw.get("atom_id", atom_id)),
        model_name=model_name,
        explanation=str(raw.get("explanation", "")),
        hue_range=_coerce_range(raw.get("hue_range_HSV")),
        saturation_range=_coerce_range(raw.get("saturation_range")),
        lightness_range=_coerce_range(raw.get("lightness_range")),
        name_pattern_regex=str(raw.get("name_pattern_regex", "")),
        template_pattern=[int(t) for t in (raw.get("template_pattern") or [])
                          if isinstance(t, (int, float))],
        confidence=float(raw.get("confidence_0_1", 0.5)),
        top_examples=top_examples,
        n_active=len(top_examples),
    )


# ----------------------------------------------------------------------
# Anthropic client + retry
# ----------------------------------------------------------------------


def _make_client():
    """Lazily construct Anthropic client. Raises RuntimeError if SDK/key absent."""
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError("anthropic SDK not installed; `pip install anthropic`") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic()


def call_llm_batch(
    client: Any,
    user_message: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> tuple[str, dict]:
    """One API call with prompt-caching on the system block.

    Returns (text, usage_dict).
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            # extract text
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
        except Exception as e:  # noqa: BLE001 — broad catch for retry
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
            sleep_s = backoff_base ** attempt
            time.sleep(sleep_s)
    raise RuntimeError(f"unreachable; last={last_err}")


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------


def llm_explain_atoms(
    atoms: list[dict],
    *,
    model_name: str = "unknown",
    color_hsv_lookup: Optional[np.ndarray] = None,
    batch_size: int = 8,
    llm_model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    client: Optional[Any] = None,
) -> list[LLMAtomExplanation]:
    """Explain a list of atoms via the Anthropic API.

    `atoms` is a list of {"atom_id": int, "top_examples": [dicts]} as
    produced by `collect_top_activating`.

    Returns one LLMAtomExplanation per input atom (in input order).
    Atoms that fail 3x get a low-confidence placeholder explanation.
    """
    if client is None:
        client = _make_client()

    results: list[Optional[LLMAtomExplanation]] = [None] * len(atoms)
    for start in range(0, len(atoms), batch_size):
        batch = atoms[start:start + batch_size]
        # only send atoms with examples; degenerate dead atoms get placeholder
        payload = []
        idx_map = []
        for i, a in enumerate(batch):
            if a["top_examples"]:
                payload.append(a)
                idx_map.append(i)
            else:
                results[start + i] = LLMAtomExplanation(
                    atom_id=int(a["atom_id"]), model_name=model_name,
                    explanation="(dead atom — no activations)",
                    hue_range=(0.0, 1.0), saturation_range=(0.0, 1.0),
                    lightness_range=(0.0, 1.0), name_pattern_regex="",
                    template_pattern=[], confidence=0.0,
                    top_examples=[], n_active=0,
                )
        if not payload:
            continue

        prompt = render_batch_prompt(payload, color_hsv_lookup=color_hsv_lookup)
        try:
            text, usage = call_llm_batch(
                client, prompt, model=llm_model, max_retries=max_retries,
            )
            parsed = parse_llm_response(text)
        except Exception as e:  # noqa: BLE001
            # whole batch failed — placeholder for each
            for i, a in zip(idx_map, payload):
                results[start + i] = LLMAtomExplanation(
                    atom_id=int(a["atom_id"]), model_name=model_name,
                    explanation=f"(LLM failed: {type(e).__name__}: {str(e)[:80]})",
                    hue_range=(0.0, 1.0), saturation_range=(0.0, 1.0),
                    lightness_range=(0.0, 1.0), name_pattern_regex="",
                    template_pattern=[], confidence=0.0,
                    top_examples=a["top_examples"], n_active=len(a["top_examples"]),
                )
            continue

        # attribute usage equally to each atom in this batch
        n = max(1, len(payload))
        per_atom_usage = {
            "input_tokens": usage["input_tokens"] // n,
            "output_tokens": usage["output_tokens"] // n,
            "cache_read_input_tokens": usage["cache_read_input_tokens"] // n,
            "cache_creation_input_tokens": usage["cache_creation_input_tokens"] // n,
        }

        # align parsed responses to payload; if model returns fewer/different,
        # fall back to placeholder on the missing ones
        by_atom_id = {int(p.get("atom_id", -1)): p for p in parsed if isinstance(p, dict)}
        for k, (i, a) in enumerate(zip(idx_map, payload)):
            aid = int(a["atom_id"])
            raw = by_atom_id.get(aid)
            if raw is None and k < len(parsed) and isinstance(parsed[k], dict):
                raw = parsed[k]  # positional fallback
            if raw is None:
                exp = LLMAtomExplanation(
                    atom_id=aid, model_name=model_name,
                    explanation="(LLM omitted this atom)",
                    hue_range=(0.0, 1.0), saturation_range=(0.0, 1.0),
                    lightness_range=(0.0, 1.0), name_pattern_regex="",
                    template_pattern=[], confidence=0.0,
                    top_examples=a["top_examples"], n_active=len(a["top_examples"]),
                )
            else:
                exp = _coerce_atom(raw, aid, model_name, a["top_examples"])
                exp.input_tokens = per_atom_usage["input_tokens"]
                exp.output_tokens = per_atom_usage["output_tokens"]
                exp.cache_read_tokens = per_atom_usage["cache_read_input_tokens"]
                exp.cache_creation_tokens = per_atom_usage["cache_creation_input_tokens"]
            results[start + i] = exp

    # type-narrowed
    return [r for r in results if r is not None]


def llm_explanation_to_hypothesis(exp: LLMAtomExplanation) -> AtomHypothesis:
    """Convert LLM explanation → AtomHypothesis so the existing scorer works."""
    return AtomHypothesis(
        atom_id=exp.atom_id,
        model_name=exp.model_name,
        n_active=exp.n_active,
        top_examples=exp.top_examples,
        explanation=exp.explanation,
        hue_range=exp.hue_range,
        lightness_range=exp.lightness_range,
        saturation_range=exp.saturation_range,
        name_pattern_regex=exp.name_pattern_regex,
        template_pattern=exp.template_pattern,
        name_top_tokens=[],
        hsv_compactness=1.0 - exp.confidence,
    )


def estimate_cost(explanations: Sequence[LLMAtomExplanation]) -> dict:
    """Total USD cost across a list of explanations."""
    in_tok = sum(e.input_tokens for e in explanations)
    out_tok = sum(e.output_tokens for e in explanations)
    cache_r = sum(e.cache_read_tokens for e in explanations)
    cache_w = sum(e.cache_creation_tokens for e in explanations)
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


def explanation_to_dict(e: LLMAtomExplanation) -> dict:
    return asdict(e)
