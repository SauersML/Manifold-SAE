"""Concept intrinsic dim v2 — structured GT + template quotient.

The v1 measurement reported corr_dim ~3 for color/brightness/temperature
concepts. That measurement pools many templates (e.g. "Her dress was a
{adj} shade." + "The room was {adj}." + ...), so template-variation
directions sit ON TOP of any actual concept axis. The reported "3D"
might be (1 concept dim) + (2 template dims).

This script controls for that. For each concept, we have:
  * an external structured ground truth with multiple labeled axes
    (color: R, G, B, hue, sat, value;  temperature: Fahrenheit)
  * MANY varied templates (~12) per concept value

Measurements per (layer, concept):
  * Global intrinsic dim — pool all (value × template) activations
  * Per-template intrinsic dim — for each template separately,
    measure dim across the value axis only. Average.
  * Template-only intrinsic dim — for each value, measure dim across
    templates. Average. (How much variation does the template itself
    introduce?)
  * For each GT axis (R, G, B, hue, sat, val, F): top-K PC ↔ axis
    Spearman correlations. Reports which PCs the LM encodes.

If per-template dim << global dim, templates were inflating things and
v1's "concepts aren't 1D" overshot. If they're similar, real concept
clouds are genuinely high-D.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


# -----------------------------------------------------------------------------
# Color GT — curated xkcd-style subset with RGB (50 named colors)
# -----------------------------------------------------------------------------
COLORS = [
    ("red", 255, 0, 0), ("crimson", 220, 20, 60), ("maroon", 128, 0, 0),
    ("orange", 255, 165, 0), ("amber", 255, 191, 0), ("rust", 183, 65, 14),
    ("yellow", 255, 255, 0), ("gold", 255, 215, 0), ("mustard", 255, 219, 88),
    ("lime", 50, 205, 50), ("olive", 128, 128, 0), ("chartreuse", 127, 255, 0),
    ("green", 0, 128, 0), ("forest green", 34, 139, 34), ("emerald", 80, 200, 120),
    ("mint", 152, 255, 152), ("teal", 0, 128, 128), ("turquoise", 64, 224, 208),
    ("cyan", 0, 255, 255), ("sky blue", 135, 206, 235), ("azure", 0, 127, 255),
    ("blue", 0, 0, 255), ("navy", 0, 0, 128), ("cobalt", 0, 71, 171),
    ("indigo", 75, 0, 130), ("violet", 143, 0, 255), ("purple", 128, 0, 128),
    ("magenta", 255, 0, 255), ("pink", 255, 192, 203), ("rose", 255, 0, 127),
    ("salmon", 250, 128, 114), ("peach", 255, 218, 185), ("coral", 255, 127, 80),
    ("brown", 165, 42, 42), ("tan", 210, 180, 140), ("beige", 245, 245, 220),
    ("cream", 255, 253, 208), ("ivory", 255, 255, 240), ("white", 255, 255, 255),
    ("silver", 192, 192, 192), ("gray", 128, 128, 128), ("charcoal", 54, 69, 79),
    ("black", 0, 0, 0), ("lavender", 230, 230, 250), ("plum", 142, 69, 133),
    ("orchid", 218, 112, 214), ("aquamarine", 127, 255, 212), ("khaki", 240, 230, 140),
    ("burgundy", 128, 0, 32), ("scarlet", 255, 36, 0),
]


def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return h, s, v


COLOR_TEMPLATES = [
    "She wore a {x} dress to the party.",
    "The walls were painted {x}.",
    "He drove a {x} truck through the desert.",
    "Her favorite color is {x}.",
    "The sunset turned the sky {x}.",
    "I bought a {x} notebook from the store.",
    "The cat had {x} fur.",
    "They lived in a {x} house at the end of the street.",
    "She had a {x} ribbon in her hair.",
    "The artist mixed paint to get a {x} shade.",
    "He pointed at the {x} balloon floating away.",
    "A {x} bird landed on the fence.",
]


# -----------------------------------------------------------------------------
# Temperature GT — named adjectives + Fahrenheit, monotone single axis
# -----------------------------------------------------------------------------
TEMPERATURES = [
    ("freezing cold", 10), ("icy", 25), ("frigid", 30), ("very cold", 35),
    ("cold", 45), ("chilly", 55), ("cool", 60), ("mild", 65), ("lukewarm", 75),
    ("warm", 80), ("toasty", 85), ("hot", 90), ("very hot", 100),
    ("scorching", 110), ("sweltering", 115), ("boiling", 212),
    ("scalding", 180), ("piping hot", 200), ("steaming", 175), ("blazing", 130),
]

TEMP_TEMPLATES = [
    "The water in the kettle was {x}.",
    "She touched the {x} mug.",
    "Outside it felt {x} this morning.",
    "The metal handle was {x} to the touch.",
    "The soup he served was {x}.",
    "The shower turned {x} after a minute.",
    "We stood in the {x} sun for too long.",
    "The drink came out {x}.",
    "The room felt unbearably {x}.",
    "The wind off the lake was {x}.",
    "He wrapped his hands around the {x} cup.",
    "The pavement was {x} under bare feet.",
]


@dataclass
class Config:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layers: tuple[int, ...] = field(default_factory=lambda: (4, 8, 12, 18))
    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/CONCEPT_INTRINSIC_DIM_V2",
    )


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("no blocks")


def harvest(model, tok, blocks, layer: int, prompts: list[str], device) -> torch.Tensor:
    cap = {}
    h = blocks[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    feats = []
    with torch.no_grad():
        for p in prompts:
            inp = tok(p, return_tensors="pt").to(device)
            model(**inp)
            feats.append(cap["h"][0, -1, :].cpu())
    h.remove()
    return torch.stack(feats, dim=0).float()


def per_dim_normalize(X: torch.Tensor) -> torch.Tensor:
    mu = X.mean(0, keepdim=True)
    sigma = X.std(0, keepdim=True).clamp(min=1e-6)
    return (X - mu) / sigma


def pca_dims(X: torch.Tensor) -> tuple[int, int, int]:
    Xc = X - X.mean(0, keepdim=True)
    _, s, _ = torch.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    cum = torch.cumsum(var, dim=0) / var.sum().clamp(min=1e-12)
    def fge(t): return int((cum >= t).float().argmax().item()) + 1
    return fge(0.5), fge(0.9), fge(0.99)


def correlation_dim(X: torch.Tensor) -> float:
    Xn = X.numpy().astype(np.float64)
    N = Xn.shape[0]
    if N < 8:
        return float("nan")
    sq = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=-1)
    d = np.sqrt(sq[np.triu_indices(N, k=1)])
    d_med = float(np.median(d))
    if not np.isfinite(d_med) or d_med == 0:
        return float("nan")
    rs = np.geomspace(0.05 * d_med, 2 * d_med, 12)
    cs = np.array([(d < r).mean() for r in rs])
    mask = (cs > 0.05) & (cs < 0.5) & (cs > 0)
    if mask.sum() < 3:
        mask = cs > 0
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(rs[mask]), np.log(cs[mask]), 1)[0])


def local_pca_dim(X: torch.Tensor, thresh: float = 0.95) -> float:
    Xn = X.numpy().astype(np.float64)
    N, D = Xn.shape
    if N < 8:
        return float("nan")
    k = max(8, int(math.sqrt(N)))
    sq = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=-1)
    nn = np.argsort(sq, axis=1)[:, 1:k+1]
    dims = []
    for i in range(N):
        nbr = Xn[nn[i]] - Xn[i]
        _, s, _ = np.linalg.svd(nbr, full_matrices=False)
        cum = np.cumsum(s**2) / (s**2).sum()
        dims.append(int((cum >= thresh).argmax()) + 1)
    return float(np.median(dims))


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = float(np.sqrt((rx*rx).sum() * (ry*ry).sum()))
    return float((rx*ry).sum() / denom) if denom > 0 else 0.0


def measure(X: torch.Tensor) -> dict:
    k50, k90, k99 = pca_dims(X)
    return {
        "n": X.shape[0],
        "k50": k50, "k90": k90, "k99": k99,
        "corr_dim": correlation_dim(X),
        "local_pca_dim": local_pca_dim(X),
    }


def pc_axis_spearmans(X: torch.Tensor, axes: dict[str, np.ndarray], n_pcs: int = 5) -> dict:
    Xc = X - X.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(Xc, full_matrices=False)
    pcs = (Xc @ V.T[:, :n_pcs]).numpy()  # (N, n_pcs)
    out = {}
    for ax_name, ax_vals in axes.items():
        rhos = [spearman(pcs[:, k], ax_vals) for k in range(n_pcs)]
        out[ax_name] = {"per_pc": rhos, "best_pc": int(np.argmax(np.abs(rhos))),
                         "best_rho": float(max(rhos, key=abs))}
    return out


def build_color_data():
    """Returns (prompts, (label_per_prompt, template_per_prompt), axes_dict)."""
    prompts, c_idx, t_idx = [], [], []
    for ci, (name, _, _, _) in enumerate(COLORS):
        for ti, tpl in enumerate(COLOR_TEMPLATES):
            prompts.append(tpl.format(x=name))
            c_idx.append(ci); t_idx.append(ti)
    c_idx = np.array(c_idx); t_idx = np.array(t_idx)
    rgb = np.array([(r, g, b) for _, r, g, b in COLORS], dtype=np.float64)
    hsv = np.array([rgb_to_hsv(*c[1:]) for c in COLORS], dtype=np.float64)
    axes = {
        "R": rgb[c_idx, 0], "G": rgb[c_idx, 1], "B": rgb[c_idx, 2],
        "hue": hsv[c_idx, 0], "sat": hsv[c_idx, 1], "value": hsv[c_idx, 2],
        "luminance": (0.299 * rgb[c_idx, 0] + 0.587 * rgb[c_idx, 1] + 0.114 * rgb[c_idx, 2]),
    }
    return prompts, c_idx, t_idx, axes


def build_temp_data():
    prompts, c_idx, t_idx = [], [], []
    for ci, (adj, _) in enumerate(TEMPERATURES):
        for ti, tpl in enumerate(TEMP_TEMPLATES):
            prompts.append(tpl.format(x=adj))
            c_idx.append(ci); t_idx.append(ti)
    c_idx = np.array(c_idx); t_idx = np.array(t_idx)
    F_vals = np.array([t[1] for t in TEMPERATURES], dtype=np.float64)
    axes = {"fahrenheit": F_vals[c_idx]}
    return prompts, c_idx, t_idx, axes


def analyze_concept(X: torch.Tensor, c_idx: np.ndarray, t_idx: np.ndarray,
                     axes: dict[str, np.ndarray]) -> dict:
    Xn = per_dim_normalize(X)
    global_m = measure(Xn)
    # Per-template: dim across concept values
    n_t = int(t_idx.max()) + 1
    per_t = []
    for ti in range(n_t):
        m = (t_idx == ti)
        if m.sum() < 8:
            continue
        per_t.append(measure(Xn[m]))
    # Per-concept: dim across templates
    n_c = int(c_idx.max()) + 1
    per_c = []
    for ci in range(n_c):
        m = (c_idx == ci)
        if m.sum() < 8:
            continue
        per_c.append(measure(Xn[m]))
    def avg(d_list, k):
        v = [d[k] for d in d_list if not (isinstance(d[k], float) and math.isnan(d[k]))]
        return float(np.mean(v)) if v else float("nan")
    # GT-axis correlations
    pc_axes = pc_axis_spearmans(Xn, axes, n_pcs=5)
    return {
        "global": global_m,
        "per_template_avg": {k: avg(per_t, k) for k in ["k50", "k90", "k99", "corr_dim", "local_pca_dim"]},
        "per_concept_avg":  {k: avg(per_c, k) for k in ["k50", "k90", "k99", "corr_dim", "local_pca_dim"]},
        "n_templates": n_t, "n_concepts": n_c,
        "pc_vs_axes": pc_axes,
    }


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {cfg.model_name} layers={cfg.layers} device={device}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model.model if hasattr(model, "model") else model.transformer)

    color_prompts, color_c, color_t, color_axes = build_color_data()
    temp_prompts, temp_c, temp_t, temp_axes = build_temp_data()
    print(f"[color] {len(color_prompts)} prompts ({len(COLORS)} colors × {len(COLOR_TEMPLATES)} templates)", flush=True)
    print(f"[temp ] {len(temp_prompts)} prompts ({len(TEMPERATURES)} temps × {len(TEMP_TEMPLATES)} templates)", flush=True)

    results = {}
    for layer in cfg.layers:
        print(f"\n=== L={layer} ===", flush=True)
        for cname, (prompts, c_idx, t_idx, axes) in [
            ("color", (color_prompts, color_c, color_t, color_axes)),
            ("temperature", (temp_prompts, temp_c, temp_t, temp_axes)),
        ]:
            X = harvest(model, tok, blocks, layer, prompts, device)
            r = analyze_concept(X, c_idx, t_idx, axes)
            results.setdefault(f"L{layer}", {})[cname] = r
            g = r["global"]; pt = r["per_template_avg"]; pc = r["per_concept_avg"]
            print(f"  {cname:11}  global: k90={g['k90']:3d} corr={g['corr_dim']:5.2f} lpca={g['local_pca_dim']:4.1f}", flush=True)
            print(f"  {' '*11}  per-template (across values): "
                  f"k90={pt['k90']:5.1f} corr={pt['corr_dim']:5.2f} lpca={pt['local_pca_dim']:4.1f}", flush=True)
            print(f"  {' '*11}  per-concept (across templates): "
                  f"k90={pc['k90']:5.1f} corr={pc['corr_dim']:5.2f} lpca={pc['local_pca_dim']:4.1f}", flush=True)
            for ax_name, info in r["pc_vs_axes"].items():
                print(f"  {' '*11}  PC↔{ax_name:9}  best PC={info['best_pc']}  |ρ|={abs(info['best_rho']):.3f}  "
                      f"(per-PC: {['%+.2f' % v for v in info['per_pc']]})", flush=True)

    (out_dir / "results.json").write_text(json.dumps({
        "config": asdict(cfg),
        "color_meta": {"n_colors": len(COLORS), "n_templates": len(COLOR_TEMPLATES)},
        "temp_meta": {"n_temps": len(TEMPERATURES), "n_templates": len(TEMP_TEMPLATES)},
        "results": results,
    }, indent=2, default=float))
    print(f"\n[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
