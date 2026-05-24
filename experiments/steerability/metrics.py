"""Scoring metrics for cogito_intervention_results.jsonl.

Read-only over the JSONL produced by cogito_intervene.py.  All four
functions return cheap pandas / float results.

Usage (e.g. in a notebook or downstream script):

    from metrics import (
        load_results, kl_per_axis_per_alpha, directional_shift,
        monotone_dose_response, null_axis_control,
    )
    df_raw = load_results("cogito_intervention_results.jsonl")
    print(kl_per_axis_per_alpha(df_raw))
    print(directional_shift(df_raw, "hue", +2.0))
    print(monotone_dose_response(df_raw, "hue"))
    print(null_axis_control(df_raw))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def kl_per_axis_per_alpha(results: pd.DataFrame) -> pd.DataFrame:
    """Mean +- sem KL by (axis, alpha)."""
    g = results.groupby(["axis", "alpha"])["kl_intervened_vs_baseline"]
    return g.agg(["mean", "std", "count"]).reset_index()


def _lookup_lp(d: dict | None, tok: str) -> float:
    if not isinstance(d, dict):
        return float("nan")
    if tok in d:
        return float(d[tok])
    # try with/without leading space
    alt = tok[1:] if tok.startswith(" ") else " " + tok
    if alt in d:
        return float(d[alt])
    return float("nan")


def directional_shift(results: pd.DataFrame, axis: str, alpha: float,
                       baseline_color_token: str | None = None) -> float:
    """Mean over prompts of:
        Delta logit(expected_color_token) - Delta logit(baseline_color_token)
    Delta is intervened-baseline logprob.  baseline_color_token defaults
    to the lexically-most-likely color word that ISN'T expected_top_token
    for each prompt (we use a fixed list)."""
    color_pool = [" red", " orange", " yellow", " green", " blue",
                  " purple", " pink", " brown", " black", " white", " gray"]
    sub = results[(results["axis"] == axis) & (np.isclose(results["alpha"], alpha))]
    if sub.empty:
        return float("nan")
    deltas: list[float] = []
    for _, r in sub.iterrows():
        exp = r["expected_top_token"]
        if baseline_color_token is None:
            base_tok = next((c for c in color_pool if c != exp), " gray")
        else:
            base_tok = baseline_color_token
        d_exp = _lookup_lp(r["intervened_top_logprobs"], exp) \
                - _lookup_lp(r["baseline_top_logprobs"], exp)
        d_base = _lookup_lp(r["intervened_top_logprobs"], base_tok) \
                 - _lookup_lp(r["baseline_top_logprobs"], base_tok)
        if not (np.isnan(d_exp) or np.isnan(d_base)):
            deltas.append(d_exp - d_base)
    if not deltas:
        return float("nan")
    return float(np.mean(deltas))


def monotone_dose_response(results: pd.DataFrame, axis: str,
                            min_increase_per_step: float = 0.0) -> bool:
    """True if the mean directional_shift at each alpha increases
    monotonically with alpha for this axis."""
    sub = results[results["axis"] == axis]
    if sub.empty:
        return False
    alphas = sorted(sub["alpha"].unique())
    shifts = [directional_shift(results, axis, a) for a in alphas]
    if any(np.isnan(s) for s in shifts):
        return False
    diffs = np.diff(shifts)
    return bool(np.all(diffs >= min_increase_per_step))


def null_axis_control(results: pd.DataFrame, null_axis: str = "random",
                       alpha: float | None = None) -> float:
    """Mean KL for the null/random axis.  A meaningful steering axis
    should have HIGHER KL than this at matched |alpha|."""
    sub = results[results["axis"] == null_axis]
    if alpha is not None:
        sub = sub[np.isclose(np.abs(sub["alpha"]), abs(alpha))]
    if sub.empty:
        return float("nan")
    return float(sub["kl_intervened_vs_baseline"].mean())


def summary(results: pd.DataFrame) -> dict:
    """One-shot report combining all four metrics."""
    axes = [a for a in results["axis"].unique() if a != "random"]
    out: dict = {
        "kl_per_axis_per_alpha": kl_per_axis_per_alpha(results).to_dict(orient="records"),
        "directional_shift_at_alpha_plus2": {
            a: directional_shift(results, a, 2.0) for a in axes
        },
        "monotone_dose_response": {a: monotone_dose_response(results, a) for a in axes},
        "null_axis_control_kl": null_axis_control(results),
    }
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results", default="cogito_intervention_results.jsonl", nargs="?")
    args = ap.parse_args()
    df = load_results(args.results)
    print(json.dumps(summary(df), indent=2, default=float))
