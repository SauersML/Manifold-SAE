"""Autointerp `explain` stage — collect activating examples, hypothesize, causal score.

Pipeline per atom k:
  1. collect_top_activating: top-N (color, template) val pairs by activation magnitude.
  2. hypothesize_atom: rule-based "LLM stub" emits
       - 1-sentence natural-language explanation
       - structured hypothesis (HSV ranges, name regex, template pattern)
     Honest design note: this is INTENTIONALLY rule-based, not an LLM API. The
     hypothesizer is the formal procedure Anthropic's autointerp papers
     simulate — given the top-activating examples, produce a structured
     guess. Doing it in code makes the inductive bias auditable and
     reproducible; the simulation-accuracy R² in score.py is a meaningful
     metric of "how well does THIS hypothesis predict held-out activations,"
     regardless of whether the hypothesis was written by GPT-4 or a Python
     function. The bottleneck for "is the SAE interpretable" is the encoding
     of structure into the hypothesis space — not the verbal fluency of the
     hypothesizer.
  3. causal_score_atom: zero atom on a held-out batch, re-run SAE, measure
     Δ-R² (reconstruction quality drop) and Δ-cosine (representation shift).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np
import torch


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class AtomHypothesis:
    """Structured hypothesis emitted for one atom."""
    atom_id: int
    model_name: str
    n_active: int                         # # val rows where atom fires
    top_examples: list[dict]              # [{color, template_id, act}, ...]
    explanation: str                      # 1-sentence NL
    # Structured fields (these are the inputs to score.py's regression).
    hue_range: tuple[float, float]        # (low, high) — circular handled in scorer
    lightness_range: tuple[float, float]  # value channel of HSV
    saturation_range: tuple[float, float]
    name_pattern_regex: str               # python regex over color name
    template_pattern: list[int]           # template indices that activate
    # Hypothesis metadata
    name_top_tokens: list[str] = field(default_factory=list)
    hsv_compactness: float = 0.0          # 0=tight, 1=diffuse (uniform)
    # Causal scores (filled by causal_score_atom)
    causal_delta_r2: float = 0.0          # drop in val R² when atom zeroed
    causal_delta_cosine: float = 0.0      # mean (1 - cos) of recon shift


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] (N,3) → hsv in [0,1] (N,3)."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(1); mn = rgb.min(1); df = mx - mn
    h = np.zeros_like(mx)
    mask = df > 1e-8
    rm = mask & (mx == r); gm = mask & (mx == g); bm = mask & (mx == b)
    h[rm] = ((g[rm] - b[rm]) / df[rm]) % 6
    h[gm] = ((b[gm] - r[gm]) / df[gm]) + 2
    h[bm] = ((r[bm] - g[bm]) / df[bm]) + 4
    h = h / 6.0
    s = np.where(mx > 1e-8, df / np.maximum(mx, 1e-8), 0.0)
    v = mx
    return np.stack([h, s, v], 1)


def load_sae_activations(
    sae_module: torch.nn.Module,
    X: np.ndarray,
    model_kind: str,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """Run an SAE (kind ∈ {topk, l1, manifold}) over X. Returns (N, F)."""
    sae_module.eval()
    acts: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i+batch_size])).to(device)
            if model_kind == "topk":
                z = sae_module.encode(xb)
            elif model_kind == "l1":
                z = sae_module.encode(xb)
            elif model_kind == "manifold":
                z = sae_module.encode_for_eval(xb)
            else:
                raise ValueError(model_kind)
            acts.append(z.detach().cpu().numpy())
    return np.concatenate(acts, 0)


def collect_top_activating(
    acts: np.ndarray,
    atom_id: int,
    row_color: np.ndarray,
    row_template: np.ndarray,
    color_names: list[str],
    n_top: int = 20,
) -> list[dict]:
    """Top-N val rows for atom_id, returned as list of dicts."""
    col = acts[:, atom_id]
    order = np.argsort(-col)[:n_top]
    out = []
    for r in order:
        if col[r] <= 0:
            continue
        out.append({
            "color": color_names[int(row_color[r])],
            "color_idx": int(row_color[r]),
            "template_id": int(row_template[r]),
            "act": float(col[r]),
        })
    return out


# ----------------------------------------------------------------------
# Rule-based "LLM" hypothesizer
# ----------------------------------------------------------------------


_STOPWORDS = {"the", "a", "an", "of", "and", "or", "is", "are"}


def _circular_range(angles: np.ndarray) -> tuple[float, float, float]:
    """Smallest arc containing >=80% of angles ∈ [0,1). Returns (lo, hi, spread).

    For a circular distribution, the "range" may wrap (e.g. (0.95, 0.05))
    meaning [0.95, 1.0) ∪ [0.0, 0.05). spread = arc length ∈ [0, 1].
    """
    if len(angles) == 0:
        return 0.0, 1.0, 1.0
    a = np.sort(angles % 1.0)
    # gaps between consecutive points, including wrap-around
    gaps = np.diff(np.concatenate([a, [a[0] + 1.0]]))
    g_max = int(np.argmax(gaps))
    # smallest arc covering all points starts just after the largest gap
    lo = a[(g_max + 1) % len(a)]
    hi = a[g_max]
    if hi < lo:
        spread = (hi + 1.0) - lo
    else:
        spread = hi - lo
    return float(lo), float(hi), float(spread)


def _name_token_tfidf(top_names: list[str], all_names: list[str], k: int = 3) -> list[str]:
    """Top-k tokens by simple TF-IDF: token frequency in top_names / log(1 + df_in_all)."""
    tokenize = lambda n: [t for t in re.findall(r"[a-z]+", n.lower()) if t not in _STOPWORDS]
    # df across all names
    df: dict[str, int] = {}
    for n in all_names:
        for t in set(tokenize(n)):
            df[t] = df.get(t, 0) + 1
    # tf in top_names
    tf: dict[str, int] = {}
    for n in top_names:
        for t in tokenize(n):
            tf[t] = tf.get(t, 0) + 1
    scored = [(t, c / np.log(1 + df.get(t, 1))) for t, c in tf.items() if c >= 2]
    scored.sort(key=lambda x: -x[1])
    return [t for t, _ in scored[:k]]


def hypothesize_atom(
    atom_id: int,
    model_name: str,
    top_examples: list[dict],
    color_hsv: np.ndarray,
    color_names_all: list[str],
    n_templates: int = 28,
) -> AtomHypothesis:
    """Rule-based structured hypothesizer (the autointerp LLM stub).

    Bins activations by HSV octants + template-id frequency + name-token TF-IDF.
    """
    if not top_examples:
        return AtomHypothesis(
            atom_id=atom_id, model_name=model_name, n_active=0,
            top_examples=[],
            explanation="(dead atom — no activations on val set)",
            hue_range=(0.0, 1.0),
            lightness_range=(0.0, 1.0),
            saturation_range=(0.0, 1.0),
            name_pattern_regex="",
            template_pattern=[],
            hsv_compactness=1.0,
        )

    color_ids = np.array([e["color_idx"] for e in top_examples])
    template_ids = np.array([e["template_id"] for e in top_examples])
    weights = np.array([e["act"] for e in top_examples])
    weights = weights / weights.sum()

    hsv = color_hsv[color_ids]
    # weighted hue (circular)
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]
    h_lo, h_hi, h_spread = _circular_range(h)
    s_lo, s_hi = float(np.quantile(s, 0.1)), float(np.quantile(s, 0.9))
    v_lo, v_hi = float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))

    # template pattern: templates that appear with above-baseline frequency
    tcounts = np.bincount(template_ids, minlength=n_templates)
    baseline = len(template_ids) / n_templates
    template_pattern = [int(t) for t in np.where(tcounts >= max(2, 1.5 * baseline))[0]]
    if not template_pattern:
        template_pattern = [int(t) for t in np.argsort(-tcounts)[:3]]

    # name tokens
    top_names = [e["color"] for e in top_examples]
    tokens = _name_token_tfidf(top_names, color_names_all, k=3)
    if tokens:
        name_regex = r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b"
    else:
        name_regex = ""

    # compactness: combine hue spread + s/v IQR
    sat_iqr = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
    val_iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
    compact = 0.5 * h_spread + 0.25 * sat_iqr + 0.25 * val_iqr

    # NL explanation
    hue_name = _hue_to_name(h_lo, h_hi, sat_iqr, v_lo)
    template_desc = (
        f"templates {template_pattern[:3]}" if len(template_pattern) < n_templates // 2
        else "all templates"
    )
    if tokens:
        explanation = (
            f"Atom {atom_id}: fires on {hue_name} colors with names containing "
            f"{'/'.join(tokens)}; active in {template_desc}."
        )
    else:
        explanation = (
            f"Atom {atom_id}: fires on {hue_name} colors; active in {template_desc}."
        )

    return AtomHypothesis(
        atom_id=atom_id,
        model_name=model_name,
        n_active=len(top_examples),
        top_examples=top_examples,
        explanation=explanation,
        hue_range=(h_lo, h_hi),
        lightness_range=(v_lo, v_hi),
        saturation_range=(s_lo, s_hi),
        name_pattern_regex=name_regex,
        template_pattern=template_pattern,
        name_top_tokens=tokens,
        hsv_compactness=float(compact),
    )


_HUE_NAMES = [
    (0.00, "red"), (0.08, "orange"), (0.17, "yellow"),
    (0.33, "green"), (0.50, "cyan"), (0.67, "blue"),
    (0.83, "magenta"), (1.00, "red"),
]


def _hue_to_name(lo: float, hi: float, sat_iqr: float, v_lo: float) -> str:
    if v_lo < 0.2:
        return "dark"
    if sat_iqr > 0.5:
        return "varied"
    # pick the hue name at the midpoint of [lo, hi] (handling wrap)
    if hi < lo:
        mid = ((lo + hi + 1.0) / 2.0) % 1.0
    else:
        mid = (lo + hi) / 2.0
    name = "red"
    for thr, n in _HUE_NAMES:
        if mid <= thr:
            name = n
            break
    return name


# ----------------------------------------------------------------------
# Causal score
# ----------------------------------------------------------------------


def _forward_recon(sae_module: torch.nn.Module, xb: torch.Tensor, model_kind: str) -> torch.Tensor:
    if model_kind == "topk" or model_kind == "l1":
        recon, _ = sae_module(xb)
        return recon
    if model_kind == "manifold":
        recon, _, _ = sae_module(xb, tau=0.3, hard=False)
        return recon
    raise ValueError(model_kind)


def _recon_with_atom_zeroed(
    sae_module: torch.nn.Module,
    xb: torch.Tensor,
    model_kind: str,
    atom_id: int,
) -> torch.Tensor:
    """Recompute reconstruction with atom_id's contribution removed."""
    sae_module.eval()
    with torch.no_grad():
        if model_kind == "topk":
            z = sae_module.encode(xb)
            z = z.clone()
            z[:, atom_id] = 0.0
            recon = z @ sae_module.W_d + sae_module.b_d
            return recon
        if model_kind == "l1":
            z = sae_module.encode(xb)
            z = z.clone()
            z[:, atom_id] = 0.0
            recon = z @ sae_module.W_d + sae_module.b_d
            return recon
        if model_kind == "manifold":
            # Mirror ManifoldSAE.forward but zero gate*amp for this atom.
            xc = xb - sae_module.b_d
            gate_logit = xc @ sae_module.W_gate + sae_module.b_gate
            gate = torch.sigmoid(gate_logit)
            amp_raw = xc @ sae_module.W_amp
            amp = torch.nn.functional.softplus(amp_raw) * torch.exp(sae_module.log_ard)
            cs = sae_module.theta(xb)
            phi = sae_module.fourier_basis(cs)
            w = (gate * amp).clone()
            w[:, atom_id] = 0.0
            w_phi = (w.unsqueeze(-1) * phi).reshape(xb.shape[0], -1)
            D_flat = sae_module.D_k.reshape(-1, sae_module.D_k.shape[-1])
            return w_phi @ D_flat + sae_module.b_d
    raise ValueError(model_kind)


def causal_score_atom(
    sae_module: torch.nn.Module,
    X_val: np.ndarray,
    val_var: float,
    model_kind: str,
    atom_id: int,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Returns (delta_r2, delta_cosine).

    delta_r2     = R²_full − R²_atom_zeroed (positive = atom contributes)
    delta_cosine = mean over rows of (1 − cos(recon_full, recon_zeroed))
    """
    sae_module.eval()
    sse_full = 0.0
    sse_zero = 0.0
    cos_sum = 0.0
    n_rows = 0
    n_elems = 0
    with torch.no_grad():
        for i in range(0, X_val.shape[0], batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X_val[i:i+batch_size])).to(device)
            recon_full = _forward_recon(sae_module, xb, model_kind)
            recon_zero = _recon_with_atom_zeroed(sae_module, xb, model_kind, atom_id)
            sse_full += float(((xb - recon_full) ** 2).sum().item())
            sse_zero += float(((xb - recon_zero) ** 2).sum().item())
            # cosine between recon_full and recon_zero per row
            a = recon_full
            b = recon_zero
            cs = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            cos_sum += float((1.0 - cs).sum().item())
            n_rows += xb.shape[0]
            n_elems += xb.numel()
    mse_full = sse_full / n_elems
    mse_zero = sse_zero / n_elems
    r2_full = 1.0 - mse_full / val_var
    r2_zero = 1.0 - mse_zero / val_var
    return float(r2_full - r2_zero), float(cos_sum / max(1, n_rows))


def hypothesis_to_dict(h: AtomHypothesis) -> dict:
    return asdict(h)
