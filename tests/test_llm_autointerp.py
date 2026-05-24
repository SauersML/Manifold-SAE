"""Tests for the LLM autointerp pipeline (mocked anthropic.Client)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from manifold_sae.autointerp import llm_explain as le
from manifold_sae.autointerp import llm_score as ls


# ---------------------------------------------------------------------- helpers


def _mock_resp(text: str,
               in_tok: int = 100, out_tok: int = 50,
               cache_r: int = 0, cache_w: int = 0):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_input_tokens=cache_r,
            cache_creation_input_tokens=cache_w,
        ),
    )


def _atoms(n=2, n_ex=3):
    out = []
    for a in range(n):
        out.append({
            "atom_id": 10 + a,
            "top_examples": [
                {"color": f"red_{i}", "color_idx": i,
                 "template_id": i % 28, "act": 1.5 - 0.1 * i}
                for i in range(n_ex)
            ],
        })
    return out


# ---------------------------------------------------------------------- tests


def test_explain_prompt_format_contains_atom_blocks():
    """render_batch_prompt embeds each atom with its examples."""
    atoms = _atoms(n=3, n_ex=4)
    prompt = le.render_batch_prompt(atoms)
    for a in atoms:
        assert f"atom_id={a['atom_id']}" in prompt
        for ex in a["top_examples"]:
            assert ex["color"] in prompt
    # explicit JSON-array instruction with the right count
    assert "JSON array" in prompt
    assert f"{len(atoms)} objects" in prompt


def test_score_prompt_format_separates_cached_and_fresh():
    """render_score_prompt returns (cached, fresh): cached holds the hypothesis,
    fresh holds only the per-example list."""
    examples = [{"color": "blue", "color_idx": 1, "template_id": 4},
                {"color": "green", "color_idx": 2, "template_id": 7}]
    cached, fresh = ls.render_score_prompt("Atom fires on blue-ish colors.", examples)
    assert "HYPOTHESIS" in cached
    assert "blue-ish" in cached
    # cached MUST NOT contain the eval examples (otherwise cache miss every call)
    assert "color='blue'" not in cached
    assert "color='blue'" in fresh
    assert "color='green'" in fresh
    assert "2 integers" in fresh


def test_batch_dispatch_one_call_per_batch():
    """llm_explain_atoms should send 1 API call per batch of `batch_size` atoms."""
    atoms = _atoms(n=10, n_ex=2)  # 10 atoms, batch_size=4 → 3 calls
    payload = json.dumps([
        {
            "atom_id": a["atom_id"],
            "explanation": f"feature {a['atom_id']}",
            "hue_range_HSV": [0.0, 0.2],
            "saturation_range": [0.5, 1.0],
            "lightness_range": [0.3, 0.9],
            "name_pattern_regex": "red",
            "template_pattern": [0, 1],
            "confidence_0_1": 0.8,
        }
        for a in atoms  # the mock returns the same payload regardless;
                        # per-call slicing happens in llm_explain_atoms
    ])
    client = MagicMock()
    # Different response per call: return only the batch slice
    def side_effect(**kwargs):
        msg = kwargs["messages"][0]["content"]
        # parse atom_ids from prompt
        import re
        ids = [int(x) for x in re.findall(r"atom_id=(\d+)", msg)]
        objs = [
            {"atom_id": i, "explanation": f"feature {i}",
             "hue_range_HSV": [0.0, 0.2], "saturation_range": [0.0, 1.0],
             "lightness_range": [0.0, 1.0], "name_pattern_regex": "",
             "template_pattern": [], "confidence_0_1": 0.7}
            for i in ids
        ]
        return _mock_resp(json.dumps(objs))
    client.messages.create.side_effect = side_effect

    out = le.llm_explain_atoms(atoms, model_name="topk", batch_size=4, client=client)
    assert len(out) == 10
    # ceil(10 / 4) = 3 API calls
    assert client.messages.create.call_count == 3
    # ids preserved and in order
    assert [e.atom_id for e in out] == [a["atom_id"] for a in atoms]
    # caching enabled on system block
    call = client.messages.create.call_args_list[0].kwargs
    assert "cache_control" in call["system"][0]


def test_retry_on_429_then_success():
    """call_llm_batch retries on 429 and eventually returns."""
    client = MagicMock()
    err = Exception("429 rate_limit_error")
    err.status_code = 429  # type: ignore[attr-defined]
    good = _mock_resp('[{"atom_id":1,"explanation":"x","hue_range_HSV":[0,1],'
                      '"saturation_range":[0,1],"lightness_range":[0,1],'
                      '"name_pattern_regex":"","template_pattern":[],'
                      '"confidence_0_1":0.5}]')
    client.messages.create.side_effect = [err, err, good]
    text, usage = le.call_llm_batch(client, "hello", max_retries=3, backoff_base=1.0)
    assert "atom_id" in text
    assert client.messages.create.call_count == 3
    assert usage["input_tokens"] == 100


def test_retry_exhausts_then_raises():
    """If 429s never resolve, the call raises."""
    client = MagicMock()
    err = Exception("429 rate_limit_error")
    err.status_code = 429  # type: ignore[attr-defined]
    client.messages.create.side_effect = [err, err, err]
    with pytest.raises(Exception):
        le.call_llm_batch(client, "hello", max_retries=3, backoff_base=1.0)


def test_parse_score_response_tolerates_fences_and_padding():
    """parse_score_response handles ```json fences and pads short outputs."""
    out = ls.parse_score_response("```json\n[1, 0, 1]\n```", n_expected=5)
    assert out == [1, 0, 1, 0, 0]
