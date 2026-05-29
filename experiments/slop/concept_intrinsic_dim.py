"""Measure intrinsic dimensionality of concept-firing token clouds.

This is the foundational measurement the Manifold-SAE architecture
implicitly assumes: that continuous concepts (magnitude, polarity,
brightness, etc.) live as 1D smooth manifolds in LM residuals. If
the intrinsic dim is genuinely 1, curve SAE wins. If 2-3, 2D SAE
should help. If >>2, we need a different parameterization.

For each concept c at each layer L:
  1. Harvest residuals on N prompts spanning the concept axis.
  2. Apply per-dim normalization (matches the post-fix convention).
  3. Estimate intrinsic dim three ways:
     (a) PCA — k90 = # of PCs for 90% variance
     (b) Correlation dimension (Grassberger-Procaccia): slope of
         log(C(r)) vs log(r) in the linear scaling regime
     (c) Local PCA — median # of local PCs for 95% variance in
         k-NN neighborhoods (k = sqrt(N))

Pre-fix: with the rank-1 normalization, k90 was always 1 because
the data WAS rank-1. Post-fix the answers should differ across
concepts; if magnitude is 1D while polarity is 2D, we learn
something the architecture should encode.
"""

from __future__ import annotations

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


@dataclass
class Config:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layers: tuple[int, ...] = field(default_factory=lambda: (4, 8, 12, 18))
    n_per_concept: int = 300
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/CONCEPT_INTRINSIC_DIM")


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def make_concept_prompts() -> dict:
    """Build (prompts, labels) per concept. Labels are continuous-ish
    scalars on the concept axis (magnitude, brightness, etc).
    """
    out = {}

    # Magnitude
    templates_m = [
        "There were {N} apples in the basket.",
        "She counted {N} items.",
        "The total was {N} dollars.",
        "He had {N} marbles in his pocket.",
        "We saw {N} birds in the sky.",
        "There are {N} chairs in the room.",
        "I bought {N} pencils.",
        "We need {N} servings of soup.",
        "There were {N} students in the class.",
        "He paid {N} cents for the candy.",
    ]
    mags = [1, 2, 3, 5, 8, 12, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    p, lbl = [], []
    for N in mags:
        for t in templates_m:
            p.append(t.format(N=N)); lbl.append(N)
    out["magnitude"] = (p, lbl)

    # Brightness / luminance
    templates_b = [
        "The room was {adj}.",
        "Her dress was a {adj} shade.",
        "We sat in the {adj} sun.",
        "The light was {adj}.",
        "It glowed in a {adj} way.",
    ]
    bright = [
        ("pitch black", 0), ("very dark", 1), ("dim", 2), ("dusky", 3), ("shaded", 4),
        ("medium", 5), ("bright", 6), ("lit", 7), ("brilliant", 8), ("dazzling", 9),
        ("blinding", 10),
    ]
    p, lbl = [], []
    for adj, val in bright:
        for t in templates_b:
            p.append(t.format(adj=adj)); lbl.append(val)
    out["brightness"] = (p, lbl)

    # Temperature
    templates_t = [
        "The water was {adj}.",
        "She touched the {adj} mug.",
        "Outside it felt {adj}.",
        "The metal was {adj} to the touch.",
        "The drink was {adj}.",
    ]
    temps = [
        ("freezing cold", 0), ("icy", 1), ("very cold", 2), ("cold", 3), ("cool", 4),
        ("lukewarm", 5), ("warm", 6), ("hot", 7), ("very hot", 8), ("scalding", 9), ("boiling", 10),
    ]
    p, lbl = [], []
    for adj, val in temps:
        for t in templates_t:
            p.append(t.format(adj=adj)); lbl.append(val)
    out["temperature"] = (p, lbl)

    return out


def harvest(model, tok, blocks, layer: int, prompts: list[str], device):
    """Return (N, D) tensor of last-token residuals."""
    cap = {}
    h = blocks[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    feats = []
    with torch.no_grad():
        for prompt in prompts:
            inp = tok(prompt, return_tensors="pt").to(device)
            model(**inp)
            feats.append(cap["h"][0, -1, :].cpu())
    h.remove()
    return torch.stack(feats, dim=0)


def per_dim_normalize(X: torch.Tensor) -> torch.Tensor:
    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (X - mu) / sigma


def k90(X: torch.Tensor) -> tuple[int, int, int]:
    """Return (k50, k90, k99) — number of PCs for 50/90/99% of variance."""
    X = X - X.mean(dim=0, keepdim=True)
    # SVD to get singular values, square for variance
    _, s, _ = torch.linalg.svd(X.float(), full_matrices=False)
    var = s ** 2
    cum = torch.cumsum(var, dim=0) / var.sum().clamp(min=1e-12)
    def first_ge(thresh):
        return int((cum >= thresh).float().argmax().item()) + 1
    return first_ge(0.5), first_ge(0.9), first_ge(0.99)


def correlation_dim(X: torch.Tensor, n_r: int = 12) -> float:
    """Grassberger-Procaccia correlation dimension. Slope of log C(r) vs log r."""
    Xn = X.numpy().astype(np.float64)
    N = Xn.shape[0]
    # pairwise euclidean distances (excluding self)
    sq = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=-1)
    d = np.sqrt(sq[np.triu_indices(N, k=1)])
    d_med = np.median(d)
    # Range of r: 0.05 * median to 2 * median in log scale
    rs = np.geomspace(0.05 * d_med, 2 * d_med, n_r)
    cs = np.array([(d < r).mean() for r in rs])
    # Keep only the linear-scaling regime: 0.05 < C(r) < 0.5
    mask = (cs > 0.05) & (cs < 0.5) & (cs > 0)
    if mask.sum() < 3:
        # fall back to all positive points
        mask = cs > 0
    if mask.sum() < 2:
        return float("nan")
    lr = np.log(rs[mask])
    lc = np.log(cs[mask])
    slope = np.polyfit(lr, lc, 1)[0]
    return float(slope)


def local_pca_dim(X: torch.Tensor, var_thresh: float = 0.95) -> float:
    """For each point, take its k=sqrt(N) nearest neighbors and PCA them.
    Return median number of local PCs needed to hit var_thresh."""
    Xn = X.numpy().astype(np.float64)
    N, D = Xn.shape
    k = max(8, int(math.sqrt(N)))
    sq = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=-1)
    nn_idx = np.argsort(sq, axis=1)[:, 1:k+1]   # exclude self
    dims = []
    for i in range(N):
        nbrs = Xn[nn_idx[i]] - Xn[i]
        _, s, _ = np.linalg.svd(nbrs, full_matrices=False)
        var = s ** 2
        cum = np.cumsum(var) / var.sum()
        dims.append(int((cum >= var_thresh).argmax()) + 1)
    return float(np.median(dims))


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

    concepts = make_concept_prompts()
    print(f"[concepts] {[(k, len(v[0])) for k, v in concepts.items()]}", flush=True)

    all_results = {}
    for layer in cfg.layers:
        print(f"\n=== layer L={layer} ===", flush=True)
        layer_res = {}
        for cname, (prompts, labels) in concepts.items():
            prompts = prompts[:cfg.n_per_concept]
            labels = labels[:cfg.n_per_concept]
            X = harvest(model, tok, blocks, layer, prompts, device)
            Xn = per_dim_normalize(X)
            k50, k90_v, k99 = k90(Xn)
            cdim = correlation_dim(Xn)
            lpca = local_pca_dim(Xn)
            # Also report leading-PC concept-Spearman (Phase 1 substrate test
            # under proper norm)
            from scipy import stats as _st  # late import
            _, _, V = torch.linalg.svd(Xn - Xn.mean(0, keepdim=True), full_matrices=False)
            pc1 = (Xn @ V[0]).numpy()
            rho = float(_st.spearmanr(pc1, labels).correlation)
            print(f"  {cname:12} k50={k50:3d} k90={k90_v:3d} k99={k99:3d}  "
                  f"corr_dim={cdim:5.2f}  local_pca={lpca:5.2f}  PC1↔concept ρ={rho:+.3f}",
                  flush=True)
            layer_res[cname] = {
                "k50": k50, "k90": k90_v, "k99": k99,
                "corr_dim": cdim, "local_pca_dim": lpca,
                "pc1_spearman": rho,
                "n": len(prompts),
            }
        all_results[f"L{layer}"] = layer_res

    summary = {"config": asdict(cfg), "results": all_results}
    (out_dir / "intrinsic_dim.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] {out_dir / 'intrinsic_dim.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
