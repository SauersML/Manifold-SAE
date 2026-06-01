"""Diagnostics: effective rank of LM activations + intra-SAE atom redundancy.

Answers three questions that several interpretations rest on:

  Q1. Is Qwen-1.5B layer 18 residual stream really low-rank
      after our (mean / global-std) normalization?
      → PCA of harvested activations, report variance-explained
      cumulative curve. If 99% lives in <10 PCs, the SAE saturation
      is expected. If it lives in 100s of PCs, the SAE is genuinely
      under-allocating its dictionary.

  Q2. Are the 49 alive Manifold-SAE atoms redundant directions of
      the same low-dim subspace, or genuinely distinct?
      → For each alive atom in the curve SAE checkpoint, take the
      effective direction `W_k @ mean_over_firing(g_k(t_k))` —
      the curve's centroid lifted into ambient. Compute pairwise
      |cos|. High intra-SAE cosine (>0.8) = redundancy. Near-zero
      = distinct features.

  Q3. Does the curve actually do work, or is it a regularizer?
      → For each alive curve atom, measure the variance ALONG the
      curve relative to the variance OF the direction. If the curve
      barely moves (low along-curve variance) the t parameterization
      isn't doing much — atoms are essentially vanilla-style.

Outputs a single results.json + a markdown report + 3 figures.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


@dataclass
class Config:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layers: tuple[int, ...] = (4, 8, 12, 18)
    n_tokens: int = 20_000
    curve_checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT",
        "<repo_root>/runs/llm_sweep_q15b_L18/curve_F128.pt",
    )
    vanilla_checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT_V",
        "<repo_root>/runs/llm_sweep_q15b_L18/vanilla_F128.pt",
    )
    # For intra-SAE atom redundancy / cross-SAE alignment, SAEs are trained
    # at this layer. Must match the saved checkpoints' training layer.
    sae_layer: int = int(os.environ.get("MSAE_LAYER", "18"))
    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/DIAGNOSTICS",
    )


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


# ---------------------------------------------------------------------------
# Q1: per-layer effective rank
# ---------------------------------------------------------------------------


def harvest_many_layers(model_name, layers, n_tokens, device) -> dict[int, torch.Tensor]:
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    def make_hook(L):
        def hook(_m, _i, output):
            captured[L] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook
    for L in layers:
        handles.append(blocks[L].register_forward_hook(make_hook(L)))

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    collected = 0
    with torch.no_grad():
        for d in ds:
            if collected >= n_tokens: break
            text = d.get("text", "")
            if not isinstance(text, str) or len(text) < 100: continue
            inputs = tok(text[:1500], return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**inputs)
            T = inputs["input_ids"].shape[1]
            for i in range(min(T, 32)):
                if collected >= n_tokens: break
                for L in layers:
                    out[L].append(captured[L][0, i, :].cpu())
                collected += 1
    for h in handles: h.remove()
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return {L: torch.stack(out[L][:n_tokens], dim=0) for L in layers}


def effective_rank_report(X: torch.Tensor, n_pcs: int = 100) -> dict:
    """Report cumulative variance-explained curve under both raw and
    our-normalization. Returns dict with PCs + ranks at various thresholds.
    """
    # Raw (just center)
    X_raw = X - X.mean(0, keepdim=True)
    # Our normalization (center + global std)
    sigma = float(X.std().item())
    X_norm = (X - X.mean(0, keepdim=True)) / max(sigma, 1e-6)
    # Per-feature normalization (center + per-dim std)
    per_dim_std = X.std(0).clamp(min=1e-6)
    X_perdim = (X - X.mean(0, keepdim=True)) / per_dim_std

    out = {}
    for label, Y in [("raw_centered", X_raw),
                     ("global_std",   X_norm),
                     ("per_dim_std",  X_perdim)]:
        U, S, _ = torch.linalg.svd(Y.to(torch.float64), full_matrices=False)
        var = (S ** 2).numpy()
        cum_var = np.cumsum(var) / var.sum()
        out[label] = {
            "n_pcs_for_50pct": int((cum_var >= 0.50).argmax()) + 1,
            "n_pcs_for_90pct": int((cum_var >= 0.90).argmax()) + 1,
            "n_pcs_for_99pct": int((cum_var >= 0.99).argmax()) + 1,
            "cum_var_first_100": cum_var[:min(n_pcs, len(cum_var))].tolist(),
            "participation_ratio": float((var.sum() ** 2) / (var ** 2).sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Q2: intra-SAE atom redundancy
# ---------------------------------------------------------------------------


class VanillaSAE(nn.Module):
    def __init__(self, D, F, top_k):
        super().__init__()
        self.F = F; self.top_k = top_k
        H = max(4 * D, 2 * F)
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H); self.act = nn.GELU()
        self.head = nn.Linear(H, F)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x):
        import torch.nn.functional as F_nn
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        if self.top_k < self.F:
            vals, idx = torch.topk(z, self.top_k, dim=1)
            gate = torch.zeros_like(z).scatter_(1, idx, vals)
            z = gate
        return z @ self.W_dec, z


def load_vanilla_sae(path: Path, D: int, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    sae = VanillaSAE(D, sig["F"], sig["top_k"]).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    return sae


def load_curve_sae(path: Path, D: int, device):
    from manifold_sae.sae import load_sae
    return load_sae(path, input_dim=D, device=device)


def curve_atom_effective_direction(sae, atom_k: int, device) -> torch.Tensor:
    """The atom's effective ambient direction = mean of its ambient curve.

    Cutover: the decoder block lives in ambient R^D, so the curve is read
    straight off the primitive (no W_k lift).
    """
    from manifold_sae.sae import lift_atom_curve
    t_grid = torch.linspace(0.05, 0.95, 21, dtype=torch.float64, device=device)
    curve_ambient = lift_atom_curve(sae, atom_k, t_grid).to(device)  # (21, D)
    return curve_ambient.mean(dim=0)                  # (D,)


def curve_atom_curve_span(sae, atom_k: int, device) -> tuple[float, float]:
    """For atom k: variance of the curve along its parameterization
    relative to the variance of its mean direction. If along-curve
    variance / mean-direction variance is ~0, the curve is doing
    nothing (atom is essentially vanilla).
    """
    from manifold_sae.sae import lift_atom_curve
    t_grid = torch.linspace(0.05, 0.95, 41, dtype=torch.float64, device=device)
    curve_ambient = lift_atom_curve(sae, atom_k, t_grid).to(device)  # (41, D)
    mean_dir = curve_ambient.mean(dim=0)               # (D,)
    deviations = curve_ambient - mean_dir.unsqueeze(0) # (41, D)
    along_var = float((deviations ** 2).mean().item())
    mean_var = float((mean_dir ** 2).sum().item() / mean_dir.numel())
    return along_var, mean_var


def intra_sae_redundancy(directions: torch.Tensor) -> dict:
    """Return cosine-similarity stats among a set of direction vectors."""
    F = directions.shape[0]
    if F < 2: return {"error": f"only {F} directions"}
    nrm = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
    sim = nrm @ nrm.t()                                # (F, F)
    abs_sim = sim.abs()
    # mask self
    mask = ~torch.eye(F, dtype=torch.bool, device=sim.device)
    off = abs_sim[mask]
    return {
        "n_atoms": int(F),
        "mean_abs_cos": float(off.mean().item()),
        "median_abs_cos": float(off.median().item()),
        "max_abs_cos": float(off.max().item()),
        "n_pairs_above_0.5": int((off > 0.5).sum().item()) // 2,
        "n_pairs_above_0.8": int((off > 0.8).sum().item()) // 2,
        "n_pairs_above_0.9": int((off > 0.9).sum().item()) // 2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] device={device} out={out_dir}", flush=True)

    print("\n=== Q1: per-layer effective rank ===", flush=True)
    X_layers = harvest_many_layers(cfg.model_name, list(cfg.layers), cfg.n_tokens, device)
    rank_report = {}
    for L, X in X_layers.items():
        rep = effective_rank_report(X)
        rank_report[L] = rep
        print(f"  L={L}: D={X.shape[1]}, n_tokens={X.shape[0]}", flush=True)
        for label, r in rep.items():
            print(f"    [{label}] PCs for 50%/90%/99%: "
                  f"{r['n_pcs_for_50pct']}/{r['n_pcs_for_90pct']}/{r['n_pcs_for_99pct']}  "
                  f"participation_ratio={r['participation_ratio']:.1f}", flush=True)
    # Save rank report
    (out_dir / "rank.json").write_text(json.dumps(rank_report, indent=2))

    print("\n=== Q2 + Q3: intra-SAE atom redundancy + curve-span ===", flush=True)
    X = X_layers[cfg.sae_layer]
    D = X.shape[1]
    mu = X.mean(0, keepdim=True); sigma = float(X.std().item())
    X_n = (X - mu) / max(sigma, 1e-6)

    redundancy = {}
    if Path(cfg.vanilla_checkpoint).exists():
        van = load_vanilla_sae(Path(cfg.vanilla_checkpoint), D, device)
        # Use only alive atoms' W_dec rows
        with torch.no_grad():
            _, gate = van(X_n.to(device))
        fire_v = (gate > 1e-6).sum(0).cpu().numpy()
        alive_v = [k for k in range(van.F) if fire_v[k] >= 5]
        W_dec_alive = van.W_dec[alive_v].detach()
        red_v = intra_sae_redundancy(W_dec_alive)
        red_v["alive_count"] = len(alive_v)
        redundancy["vanilla"] = red_v
        print(f"  Vanilla: {len(alive_v)} alive atoms, "
              f"intra-SAE |cos|: mean={red_v['mean_abs_cos']:.3f}  "
              f"pairs>0.5={red_v.get('n_pairs_above_0.5','?')}  "
              f"pairs>0.8={red_v.get('n_pairs_above_0.8','?')}", flush=True)

    curve_diag = None
    if Path(cfg.curve_checkpoint).exists():
        sae = load_curve_sae(Path(cfg.curve_checkpoint), D, device)
        with torch.no_grad():
            out = sae(X_n.to(device=device, dtype=sae.cfg.dtype))
        amp = out.amplitudes.cpu().numpy()
        fire_c = (amp > 1e-6).sum(0)
        alive_c = [k for k in range(sae.cfg.n_atoms) if fire_c[k] >= 5]

        # Effective direction per atom
        eff_dirs = torch.stack([curve_atom_effective_direction(sae, k, device) for k in alive_c])
        red_c = intra_sae_redundancy(eff_dirs)
        red_c["alive_count"] = len(alive_c)
        redundancy["curve"] = red_c
        print(f"  Curve:   {len(alive_c)} alive atoms, "
              f"intra-SAE |cos|: mean={red_c['mean_abs_cos']:.3f}  "
              f"pairs>0.5={red_c.get('n_pairs_above_0.5','?')}  "
              f"pairs>0.8={red_c.get('n_pairs_above_0.8','?')}", flush=True)

        # Q3: curve-span vs mean-direction-magnitude
        ratios = []
        for k in alive_c[: min(40, len(alive_c))]:
            along, mean_v = curve_atom_curve_span(sae, k, device)
            ratios.append({"atom": int(k), "along_var": along, "mean_var": mean_v,
                           "ratio": along / max(mean_v, 1e-12)})
        ratios_arr = np.array([r["ratio"] for r in ratios])
        curve_diag = {
            "per_atom": ratios,
            "median_curve_to_direction_ratio": float(np.median(ratios_arr)),
            "n_atoms_with_curve_dominant": int((ratios_arr > 1.0).sum()),
            "n_atoms_with_direction_dominant": int((ratios_arr < 0.1).sum()),
        }
        print(f"  Curve span / direction magnitude:")
        print(f"    median ratio: {curve_diag['median_curve_to_direction_ratio']:.3f}")
        print(f"    atoms where curve dominates direction: "
              f"{curve_diag['n_atoms_with_curve_dominant']}/{len(ratios)}")
        print(f"    atoms where direction dominates (curve trivial): "
              f"{curve_diag['n_atoms_with_direction_dominant']}/{len(ratios)}", flush=True)

    final = {
        "config": asdict(cfg),
        "Q1_rank_report": rank_report,
        "Q2_redundancy": redundancy,
        "Q3_curve_span": curve_diag,
    }
    (out_dir / "results.json").write_text(json.dumps(final, indent=2, default=float))

    # Render markdown
    md = ["# Diagnostics: rank + redundancy\n\n"]
    md.append("## Q1: per-layer effective rank\n\n")
    md.append("Cumulative-variance PCs needed for 50% / 90% / 99% of total variance, under three normalizations.\n\n")
    md.append("| Layer | normalization | PCs for 50% | for 90% | for 99% | participation ratio |\n")
    md.append("| --- | --- | --- | --- | --- | --- |\n")
    for L, rep in rank_report.items():
        for label, r in rep.items():
            md.append(f"| L{L} | {label} | {r['n_pcs_for_50pct']} | {r['n_pcs_for_90pct']} | {r['n_pcs_for_99pct']} | {r['participation_ratio']:.1f} |\n")
    md.append("\n")
    md.append("## Q2: intra-SAE atom redundancy\n\n")
    if redundancy:
        md.append("| arch | alive | mean \|cos\| | median \|cos\| | pairs > 0.5 | > 0.8 | > 0.9 |\n")
        md.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for arch, r in redundancy.items():
            md.append(f"| {arch} | {r.get('alive_count','?')} | "
                      f"{r.get('mean_abs_cos',0):.3f} | "
                      f"{r.get('median_abs_cos',0):.3f} | "
                      f"{r.get('n_pairs_above_0.5','?')} | "
                      f"{r.get('n_pairs_above_0.8','?')} | "
                      f"{r.get('n_pairs_above_0.9','?')} |\n")
    md.append("\n")
    if curve_diag:
        md.append("## Q3: curve-span vs direction-magnitude per atom\n\n")
        md.append(f"Median along-curve / direction-magnitude ratio: **{curve_diag['median_curve_to_direction_ratio']:.3f}**\n\n")
        md.append(f"* atoms where curve dominates: {curve_diag['n_atoms_with_curve_dominant']}\n")
        md.append(f"* atoms where direction dominates (curve trivial): {curve_diag['n_atoms_with_direction_dominant']}\n")
    (out_dir / "report.md").write_text("".join(md))
    print(f"\n[done] {out_dir / 'results.json'} + report.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
