"""Autointerp pipeline for SAE atoms on cogito-L40.

Local, rule-based hypothesizer (no API). See explain.py / score.py.
"""
from .explain import (
    load_sae_activations,
    collect_top_activating,
    hypothesize_atom,
    causal_score_atom,
    AtomHypothesis,
)
from .score import (
    simulation_features,
    score_hypothesis,
    aggregate_model_scores,
    bootstrap_ci,
)
from .llm_explain import (
    LLMAtomExplanation,
    llm_explain_atoms,
    llm_explanation_to_hypothesis,
    estimate_cost as llm_estimate_explain_cost,
    render_batch_prompt as llm_render_batch_prompt,
    parse_llm_response as llm_parse_response,
    DEFAULT_MODEL as LLM_DEFAULT_MODEL,
)
from .llm_score import (
    SimulationScore,
    llm_score_atom,
    build_eval_examples,
    estimate_score_cost as llm_estimate_score_cost,
    render_score_prompt as llm_render_score_prompt,
    parse_score_response as llm_parse_score_response,
)

__all__ = [
    "load_sae_activations",
    "collect_top_activating",
    "hypothesize_atom",
    "causal_score_atom",
    "AtomHypothesis",
    "simulation_features",
    "score_hypothesis",
    "aggregate_model_scores",
    "bootstrap_ci",
    "LLMAtomExplanation",
    "llm_explain_atoms",
    "llm_explanation_to_hypothesis",
    "llm_estimate_explain_cost",
    "llm_render_batch_prompt",
    "llm_parse_response",
    "LLM_DEFAULT_MODEL",
    "SimulationScore",
    "llm_score_atom",
    "build_eval_examples",
    "llm_estimate_score_cost",
    "llm_render_score_prompt",
    "llm_parse_score_response",
]
