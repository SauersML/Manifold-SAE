"""Manifold-SAE scaling: F vs EV at PROPORTIONAL top_k (constant sparsity).

Post-fix sweep used fixed top_k=2 across F={16..128}. At F=128 that's
1.5% sparsity; at F=16 it's 12.5%. The "curve EV is flat in F" finding
might be a sparsity artifact: at F=128 with top_k=2, only 2/128 atoms
can fire per token, which over-constrains discovery.

Standard SAE practice (Anthropic, Templeton et al.) holds *sparsity*
constant — TopK = round(0.02 · F) ≈ 2% — and scales F. This isolates
"does adding more dictionary capacity help?" from "does sparsity
matter?".

Sweep
-----
F ∈ {64, 128, 256, 512, 1024}, top_k = max(2, F // 64) — fixed at
~1.5% sparsity. Train both vanilla and curve. Measure:
  * EV (under per-dim norm, the convention)
  * alive atom count (alive = fires > 30 tokens out of 8192 eval)
  * elbow F: F at which marginal EV gain < 0.005

If curve EV scales properly with F (instead of saturating at 0.42),
the negative verdict was reading an under-budgeted configuration. If
curve still saturates at 0.42 while vanilla scales to 0.55+, that's
real architectural capacity limit.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


@dataclass
class Config:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "18"))
    n_tokens: int = 40_000
    seq_len: int = 256
    F_values: tuple[int, ...] = field(default_factory=lambda: (64, 128, 256, 512))
    sparsity_frac: float = 0.015        # ~1.5% of F
    sae_n_basis: int = 10
    sae_R: int = 2
    n_steps_vanilla: int = 1500
    n_steps_curve: int = 1500
    batch_size_vanilla: int = 1024
    batch_size_curve: int = 256
    lr: float = 1e-3
    archs: tuple[str, ...] = field(default_factory=lambda: ("vanilla", "curve"))
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/SCALING_PROP_TOPK")


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("no blocks")


def harvest_and_norm(cfg: Config, device) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model.model if hasattr(model, "model") else model.transformer)
    cap = {}
    h = blocks[cfg.layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=False)
    feats = []
    n = 0
    with torch.no_grad():
        for ex in ds:
            t = ex.get("text", "").strip()
            if len(t) < 200:
                continue
            enc = tok(t, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to(device)
            model(**enc)
            feats.append(cap["h"][0].cpu())
            n += cap["h"].shape[1]
            if n >= cfg.n_tokens:
                break
    h.remove()
    del model; torch.cuda.empty_cache()
    X = torch.cat(feats, dim=0)[:cfg.n_tokens].float()
    mu = X.mean(0, keepdim=True)
    sigma = X.std(0, keepdim=True).clamp(min=1e-6)
    return (X - mu) / sigma


def train_vanilla(cfg: Config, X: torch.Tensor, device, F: int, top_k: int):
    torch.manual_seed(0)
    D = X.shape[1]
    enc = nn.Linear(D, F).to(device)
    dec = nn.Parameter(torch.randn(F, D, device=device) / D**0.5)
    bias = nn.Parameter(torch.zeros(D, device=device))
    opt = torch.optim.Adam(list(enc.parameters()) + [dec, bias], lr=cfg.lr)
    X = X.to(device)
    for step in range(cfg.n_steps_vanilla):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size_vanilla,))
        batch = X[idx]
        opt.zero_grad()
        z = F_nn.relu(enc(batch - bias))
        vals, idx_top = torch.topk(z, top_k, dim=1)
        z_sparse = torch.zeros_like(z).scatter(1, idx_top, vals)
        recon = z_sparse @ dec + bias
        loss = F_nn.mse_loss(recon, batch) + 1e-4 * z_sparse.abs().mean()
        loss.backward()
        opt.step()
    return {"enc": enc, "dec": dec, "bias": bias, "top_k": top_k, "F": F}


def vanilla_eval(model_d, X: torch.Tensor):
    enc, dec, bias = model_d["enc"], model_d["dec"], model_d["bias"]
    z = F_nn.relu(enc(X - bias))
    vals, idx = torch.topk(z, k=model_d["top_k"], dim=1)
    z_sparse = torch.zeros_like(z).scatter(1, idx, vals)
    recon = z_sparse @ dec + bias
    fire = (z_sparse > 1e-6).sum(dim=0)
    alive = int((fire > 30).sum().item())
    return recon, alive


def train_curve(cfg: Config, X: torch.Tensor, device, F: int, top_k: int):
    torch.manual_seed(0)
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    sae_cfg = ManifoldSAEConfig(
        input_dim=X.shape[1], n_features=F, n_basis=cfg.sae_n_basis,
        top_k=top_k, intrinsic_rank=cfg.sae_R,
        encoder_type="linear", continuous_amp=True,
    )
    sae = ManifoldSAE(sae_cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    X = X.to(device)
    for step in range(cfg.n_steps_curve):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size_curve,))
        batch = X[idx]
        opt.zero_grad()
        out = sae(batch)
        loss = total_loss(out, batch, sae_cfg)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
    sae.eval()
    sae.update_snapshot(X[:2048])
    sae.inference_mode = True
    return sae


def curve_eval(sae, X: torch.Tensor):
    with torch.no_grad():
        out = sae(X)
    fire = (out.amplitudes > 1e-6).sum(dim=0)
    alive = int((fire > 30).sum().item())
    return out.reconstruction, alive


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {cfg.model_name} L={cfg.layer}  Fs={cfg.F_values}  sparsity≈{cfg.sparsity_frac:.1%}",
          flush=True)

    X = harvest_and_norm(cfg, device)
    var = float(X.var().item())
    print(f"[data] X={X.shape}  var={var:.4f}", flush=True)

    results = {}
    for F in cfg.F_values:
        top_k = max(2, int(round(F * cfg.sparsity_frac)))
        for arch in cfg.archs:
            print(f"\n=== F={F} top_k={top_k} arch={arch} ===", flush=True)
            try:
                if arch == "vanilla":
                    obj = train_vanilla(cfg, X, device, F, top_k)
                    recon, alive = vanilla_eval(obj, X.to(device))
                else:
                    sae = train_curve(cfg, X, device, F, top_k)
                    recon, alive = curve_eval(sae, X.to(device))
                mse = float(F_nn.mse_loss(recon, X.to(device)).item())
                ev = 1.0 - mse / var
                print(f"  alive={alive}/{F}  EV={ev:.3f}", flush=True)
                results[f"{arch}__F={F}"] = {"F": F, "top_k": top_k, "alive": alive, "ev": ev}
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                results[f"{arch}__F={F}"] = {"error": str(e)}

    summary = {"config": asdict(cfg), "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== scaling table ===", flush=True)
    print(f"{'F':>6}  {'top_k':>5}  {'van EV':>7} {'van alive':>9} {'crv EV':>7} {'crv alive':>9}  {'Δ EV':>7}",
          flush=True)
    for F in cfg.F_values:
        van = results.get(f"vanilla__F={F}", {})
        crv = results.get(f"curve__F={F}", {})
        if "ev" in van and "ev" in crv:
            d = crv["ev"] - van["ev"]
            print(f"{F:6d}  {van['top_k']:5d}  {van['ev']:7.3f} {van['alive']:9d} "
                  f"{crv['ev']:7.3f} {crv['alive']:9d}  {d:+7.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
