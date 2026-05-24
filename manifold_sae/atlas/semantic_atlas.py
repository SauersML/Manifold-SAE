"""Semantic atlas for cogito-L40 Manifold-SAE atoms.

Per-atom card construction:
  - top-20 (color, template) activating pairs
  - HSV centroid + hue-arc span + lightness span of top-20 colors
  - causal Δ-R² (zero atom on val, re-compute val R²)
  - explanation (LLM if available, else rule-based)
  - category in {hue-arc, lightness-band, name-token, modifier-count,
                 template-specific, dead, polysemantic}

Public:
  build_semantic_atlas(model_path, X, model_kind, ...) -> list[AtomCard]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from manifold_sae.autointerp.explain import (
    rgb_to_hsv,
    collect_top_activating,
    hypothesize_atom,
    causal_score_atom,
    load_sae_activations,
)


# ---------------------------------------------------------------------------
# AtomCard
# ---------------------------------------------------------------------------


@dataclass
class AtomCard:
    atom_id: int
    n_active: int
    top_examples: list[dict]
    hsv_centroid: tuple[float, float, float]
    hue_arc_span: float           # ∈ [0, 1) smallest arc containing top-20
    lightness_span: float          # max(V) - min(V) over top-20
    saturation_span: float
    causal_delta_r2: float
    explanation: str
    explanation_source: str        # "llm" | "rule" | "dead"
    category: str
    name_top_tokens: list[str] = field(default_factory=list)
    template_concentration: float = 0.0   # 1 - entropy_norm of template histogram
    top_template_id: int = -1
    top_color: str = ""
    top_color_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hsv_compactness: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TEMPLATE_KEYWORDS = ["modifier", "monoword", "template"]


def _circular_arc_span(hues: np.ndarray) -> float:
    """Smallest arc on [0,1) containing all hues. Returns span ∈ [0, 1)."""
    if len(hues) == 0:
        return 1.0
    a = np.sort(hues % 1.0)
    if len(a) == 1:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 1.0]]))
    g_max = float(gaps.max())
    return float(1.0 - g_max)


def _template_entropy_norm(template_ids: np.ndarray, n_templates: int = 28) -> float:
    counts = np.bincount(template_ids, minlength=n_templates).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    Hmax = float(np.log(n_templates))
    return H / Hmax if Hmax > 0 else 0.0


def _categorize(
    *,
    n_active: int,
    hue_arc_span: float,
    lightness_span: float,
    saturation_span: float,
    template_concentration: float,
    name_tokens: list[str],
    hsv_compactness: float,
) -> str:
    """Heuristic categorization."""
    if n_active == 0:
        return "dead"
    # template-specific: most top-20 are in a small set of templates
    if template_concentration > 0.5:
        return "template-specific"
    # hue-arc: narrow hue, wide lightness
    if hue_arc_span < 0.18 and lightness_span > 0.25:
        return "hue-arc"
    # lightness-band: narrow lightness, wide hue
    if lightness_span < 0.18 and hue_arc_span > 0.30:
        return "lightness-band"
    # name-token: a name token explains it (low compactness elsewhere ok)
    if name_tokens and len(name_tokens) >= 1 and hsv_compactness > 0.25:
        # is the top token a modifier-counter clue?
        joined = " ".join(name_tokens).lower()
        if any(k in joined for k in ("light", "dark", "pale", "bright", "deep", "dull", "neon")):
            return "modifier-count"
        return "name-token"
    # polysemantic: nothing tight in HSV, no clear template/name handle
    if hsv_compactness > 0.45:
        return "polysemantic"
    return "hue-arc"


# ---------------------------------------------------------------------------
# LLM-explanation loader (optional)
# ---------------------------------------------------------------------------


def _load_llm_explanations(
    path: Path | None,
) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    out: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            aid = rec.get("atom_id")
            expl = rec.get("explanation") or rec.get("explain") or rec.get("text")
            if aid is None or not expl:
                continue
            out[int(aid)] = str(expl)
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_semantic_atlas(
    *,
    sae_module: torch.nn.Module,
    model_kind: str,
    X_val_np: np.ndarray,
    val_var: float,
    row_color: np.ndarray,
    row_template: np.ndarray,
    color_names: list[str],
    color_rgb: np.ndarray,
    n_top: int = 20,
    device: str = "cpu",
    llm_explanations_path: Path | None = None,
    compute_causal: bool = True,
    causal_batch: int = 2048,
    verbose: bool = True,
) -> list[AtomCard]:
    """Build an AtomCard per feature.

    Args:
      sae_module: trained SAE in eval mode on `device`.
      model_kind: 'manifold' | 'topk' | 'l1'.
      X_val_np: (N_val, D) centred (mean-subtracted) val activations.
      val_var: scalar variance of X_val (matches train_sae_comparison.py).
      row_color: (N_val,) int color-idx per row.
      row_template: (N_val,) int template-idx per row.
      color_names: list of all color names (length = N_COLORS).
      color_rgb: (N_COLORS, 3) RGB in [0,1].
    """
    color_hsv = rgb_to_hsv(color_rgb)
    llm = _load_llm_explanations(llm_explanations_path)

    if verbose:
        print(f"[atlas] computing val activations ...", flush=True)
    acts = load_sae_activations(
        sae_module, X_val_np, model_kind, device=device, batch_size=1024
    )  # (N_val, F)
    F = acts.shape[1]
    if verbose:
        print(f"[atlas] acts shape={acts.shape}", flush=True)

    cards: list[AtomCard] = []
    for k in range(F):
        top = collect_top_activating(
            acts, k, row_color, row_template, color_names, n_top=n_top
        )
        if not top:
            cards.append(
                AtomCard(
                    atom_id=k,
                    n_active=0,
                    top_examples=[],
                    hsv_centroid=(0.0, 0.0, 0.0),
                    hue_arc_span=1.0,
                    lightness_span=0.0,
                    saturation_span=0.0,
                    causal_delta_r2=0.0,
                    explanation="(dead atom — no activations on val set)",
                    explanation_source="dead",
                    category="dead",
                )
            )
            continue

        cids = np.array([e["color_idx"] for e in top])
        tids = np.array([e["template_id"] for e in top])
        hsv = color_hsv[cids]
        # circular hue centroid via vector mean
        ang = hsv[:, 0] * 2 * np.pi
        hue_c = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi))
        hue_c = hue_c % 1.0
        sat_c = float(hsv[:, 1].mean())
        val_c = float(hsv[:, 2].mean())
        hue_span = _circular_arc_span(hsv[:, 0])
        sat_span = float(hsv[:, 1].max() - hsv[:, 1].min())
        light_span = float(hsv[:, 2].max() - hsv[:, 2].min())
        tpl_conc = 1.0 - _template_entropy_norm(tids, n_templates=28)

        hyp = hypothesize_atom(
            atom_id=k,
            model_name="model_manifold",
            top_examples=top,
            color_hsv=color_hsv,
            color_names_all=color_names,
            n_templates=28,
        )
        cat = _categorize(
            n_active=len(top),
            hue_arc_span=hue_span,
            lightness_span=light_span,
            saturation_span=sat_span,
            template_concentration=tpl_conc,
            name_tokens=hyp.name_top_tokens,
            hsv_compactness=hyp.hsv_compactness,
        )

        if k in llm:
            explanation = llm[k]
            source = "llm"
        else:
            explanation = hyp.explanation
            source = "rule"

        delta_r2 = 0.0
        if compute_causal:
            # truncate val for causal speed; full pass would dominate runtime
            X_causal = (
                X_val_np if X_val_np.shape[0] <= causal_batch
                else X_val_np[:causal_batch]
            )
            try:
                d_r2, _ = causal_score_atom(
                    sae_module, X_causal, val_var, model_kind, k,
                    device=device, batch_size=512,
                )
                delta_r2 = float(d_r2)
            except Exception as e:
                if verbose and k < 5:
                    print(f"[atlas] causal score failed for atom {k}: {e}", flush=True)

        top0 = top[0]
        c0 = int(top0["color_idx"])
        cards.append(
            AtomCard(
                atom_id=k,
                n_active=len(top),
                top_examples=top,
                hsv_centroid=(hue_c, sat_c, val_c),
                hue_arc_span=hue_span,
                lightness_span=light_span,
                saturation_span=sat_span,
                causal_delta_r2=delta_r2,
                explanation=explanation,
                explanation_source=source,
                category=cat,
                name_top_tokens=hyp.name_top_tokens,
                template_concentration=tpl_conc,
                top_template_id=int(top0["template_id"]),
                top_color=color_names[c0],
                top_color_rgb=tuple(color_rgb[c0].tolist()),
                hsv_compactness=hyp.hsv_compactness,
            )
        )
        if verbose and (k + 1) % 64 == 0:
            print(f"[atlas]   {k+1}/{F} atoms processed", flush=True)
    return cards


def cards_to_json(cards: list[AtomCard]) -> list[dict]:
    return [asdict(c) for c in cards]


def category_counts(cards: list[AtomCard]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cards:
        out[c.category] = out.get(c.category, 0) + 1
    return out
