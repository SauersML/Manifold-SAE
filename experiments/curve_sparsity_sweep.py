"""Curve SAE sparsity sweep — is the dead-atom problem just over-sparsification?

Post-fix Q1.5B L18 result: curve EV is flat (0.417-0.421) across F=16..128
and alive atom count COLLAPSES (7/128 at F=128). That signature — EV
saturated, alive << F, alive doesn't scale with F — is diagnostic of a
capacity wall, not architectural failure.

Default top_k=2 means at F=128 the encoder selects only 1.5% of atoms
per token. Atoms that never fire never get gradient → die → smoothness
prior keeps them dead. The fair test: hold top_k as a *fraction* of F,
not absolute.

Sweep top_k ∈ {2, 4, 8, 16, 32} at F=128 on Q1.5B L18 post-fix data.
If alive scales with top_k and EV improves, the negative verdict was
measuring an over-sparsified configuration, not the architecture.

Same experiment on vanilla as control — does it benefit from higher
top_k or is it already saturating?
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
    sae_F: int = 128
    top_k_values: tuple[int, ...] = field(default_factory=lambda: (2, 4, 8, 16, 32))
    sae_n_basis: int = 10
    sae_R: int = 2
    n_steps_curve: int = 1500
    n_steps_vanilla: int = 1200
    batch_size_vanilla: int = 1024
    batch_size_curve: int = 256
    lr: float = 1e-3
    archs: tuple[str, ...] = field(default_factory=lambda: ("vanilla", "curve"))
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/CURVE_SPARSITY_SWEEP")


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


def train_curve(cfg: Config, X: torch.Tensor, device, top_k: int):
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    sae_cfg = ManifoldSAEConfig(
        input_dim=X.shape[1], n_features=cfg.sae_F, n_basis=cfg.sae_n_basis,
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
        if step % 400 == 0:
            print(f"      [crv top_k={top_k} step {step:4d}] mse={F_nn.mse_loss(out.reconstruction,batch).item():.4e}", flush=True)
    sae.eval()
    sae.update_snapshot(X[:2048])
    sae.inference_mode = True
    return sae


def train_vanilla(cfg: Config, X: torch.Tensor, device, top_k: int):
    torch.manual_seed(0)
    D, F = X.shape[1], cfg.sae_F
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
    return {"enc": enc, "dec": dec, "bias": bias, "top_k": top_k}


def vanilla_eval(model_d, X: torch.Tensor):
    enc, dec, bias = model_d["enc"], model_d["dec"], model_d["bias"]
    k = model_d["top_k"]
    z = F_nn.relu(enc(X - bias))
    vals, idx = torch.topk(z, k=k, dim=1)
    z_sparse = torch.zeros_like(z).scatter(1, idx, vals)
    recon = z_sparse @ dec + bias
    alive = int((z_sparse > 1e-6).any(dim=0).sum().item())
    return recon, alive


def curve_eval(sae, X: torch.Tensor):
    with torch.no_grad():
        out = sae(X)
    alive = int(((out.amplitudes > 1e-6).sum(dim=0) > 30).sum().item())
    return out.reconstruction, alive


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {cfg.model_name} L={cfg.layer} F={cfg.sae_F} top_k_values={cfg.top_k_values}", flush=True)

    X = harvest_and_norm(cfg, device)
    print(f"[data] X={X.shape}  var={X.var().item():.4f}", flush=True)
    var = float(X.var().item())

    results = {}
    for arch in cfg.archs:
        for top_k in cfg.top_k_values:
            print(f"\n=== arch={arch} top_k={top_k} ===", flush=True)
            try:
                if arch == "vanilla":
                    obj = train_vanilla(cfg, X, device, top_k)
                    recon, alive = vanilla_eval(obj, X.to(device))
                else:
                    sae = train_curve(cfg, X, device, top_k)
                    recon, alive = curve_eval(sae, X.to(device))
                mse = float(F_nn.mse_loss(recon, X.to(device)).item())
                ev = 1.0 - mse / var
                print(f"  alive={alive}/{cfg.sae_F}  EV={ev:.3f}", flush=True)
                results[f"{arch}__top_k={top_k}"] = {"alive": alive, "ev": ev}
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                results[f"{arch}__top_k={top_k}"] = {"error": str(e)}

    summary = {"config": asdict(cfg), "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== alive-atom scaling with top_k ===", flush=True)
    for arch in cfg.archs:
        line = f"  {arch:8}"
        for top_k in cfg.top_k_values:
            r = results.get(f"{arch}__top_k={top_k}", {})
            if "alive" in r:
                line += f"  top_k={top_k}:{r['alive']:3d}({r['ev']:.2f})"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
