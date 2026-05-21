#!/usr/bin/env python3
"""Feature dashboard for a trained Manifold-SAE.

The architectural claim of Manifold-SAE is that continuous features in
residual stream are represented as a single curve atom whose position
``t_k ∈ [0, 1]`` tracks the feature's value. The MSE-vs-vanilla
comparison is structurally uninformative at saturated layers — but
*looking at what each atom encodes* is the qualitative test.

This script loads a trained curve SAE checkpoint, runs the LM over a
corpus (or reuses cached activations), and for each alive atom
produces a sorted list of top-firing tokens by ``t_k``. If the atom
is a magnitude detector, the list reads "small → medium → large"; if
polarity, "negative → positive"; if uninformative, the order is noise.

Inputs
======
* ``--checkpoint`` curve SAE checkpoint (``runs/llm_sweep/curve_FN.pt``)
* ``--model`` HF model name (default Qwen/Qwen2.5-0.5B)
* ``--layer`` layer to harvest from (must match the layer the SAE was trained on)
* ``--corpus`` dataset spec (default wikitext-2 / a small custom file)
* ``--n-tokens`` how many tokens to score (default 20000)
* ``--top-k-per-atom`` how many top tokens to show per atom (default 30)
* ``--output`` markdown report path

Output: a markdown file with one section per alive atom containing the
top-K tokens sorted by ``t_k``, plus their surrounding-context snippet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check
from manifold_sae._cluster_bridge import require_cuda_if_env

bypass_gamfit_cuda_check()


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers", "encoder_layer"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def harvest_residuals(model_name: str, layer: int, n_tokens: int, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    """Stream tokens from wikitext-2, harvest residuals at `layer`, return
    (activations of shape (n_tokens, D), token strings of length n_tokens).
    Drops PAD and BOS.
    """
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    print(f"[dash] loading {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model)

    captured: dict[int, torch.Tensor] = {}
    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = blocks[layer].register_forward_hook(hook)

    print("[dash] loading wikitext-2", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [d["text"] for d in ds if isinstance(d.get("text"), str) and len(d["text"]) > 100]

    all_acts: list[torch.Tensor] = []
    all_tokens: list[str] = []
    collected = 0
    torch.set_grad_enabled(False)
    for text in texts:
        if collected >= n_tokens:
            break
        inputs = tok(text[:2000], return_tensors="pt", truncation=True, max_length=256).to(device)
        model(**inputs)
        h = captured["h"]                                # (1, T, D)
        T = h.shape[1]
        ids = inputs["input_ids"][0]
        for i in range(T):
            if collected >= n_tokens:
                break
            tok_str = tok.decode(ids[i])
            if tok_str in ("<|endoftext|>", "<pad>") or not tok_str.strip():
                continue
            all_acts.append(h[0, i, :].cpu())
            all_tokens.append(tok_str)
            collected += 1
    torch.set_grad_enabled(True)
    handle.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(all_acts, dim=0), all_tokens


def load_curve_sae(checkpoint_path: Path, D: int, device: torch.device) -> nn.Module:
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    cfg = ManifoldSAEConfig(
        input_dim=D,
        n_features=sig["F"],
        n_basis=sig.get("n_basis", 10),
        top_k=sig["top_k"],
        intrinsic_rank=sig.get("intrinsic_rank", 2),
        encoder_type="linear",
        continuous_amp=True,
    )
    sae = ManifoldSAE(cfg).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    sae.inference_mode = bool(sae.has_snapshot.item())
    return sae


def make_dashboard(sae: nn.Module, X: torch.Tensor, tokens: list[str],
                   top_k_per_atom: int, output_path: Path, device: torch.device) -> dict:
    """For each alive atom, find top-firing tokens sorted by position."""
    with torch.no_grad():
        out = sae(X.to(device))
    pos = out.positions.cpu().numpy()                     # (N, F)
    amp = out.amplitudes.cpu().numpy()                    # (N, F)
    F = pos.shape[1]

    # An atom is "alive" if at least 10 tokens fire on it.
    fire_counts = (amp > 1e-6).sum(axis=0)
    alive = [k for k in range(F) if fire_counts[k] >= 10]
    print(f"[dash] alive atoms (>= 10 fires): {len(alive)}/{F}", flush=True)

    md = ["# Manifold-SAE feature dashboard\n",
          f"corpus: {len(tokens)} tokens | atoms checked: {F} | alive: {len(alive)}\n"]

    atom_summaries = []
    for k in alive:
        firing = np.where(amp[:, k] > 1e-6)[0]
        if len(firing) < top_k_per_atom:
            top = firing
        else:
            # Rank by amplitude, take top
            order = np.argsort(-amp[firing, k])[: top_k_per_atom * 2]
            top = firing[order]
        # Sort by position
        top = top[np.argsort(pos[top, k])][:top_k_per_atom]
        rows = []
        for i in top:
            context_start = max(0, i - 3)
            context_end = min(len(tokens), i + 1)
            ctx = "".join(tokens[context_start:context_end]).strip()
            rows.append((float(pos[i, k]), float(amp[i, k]), tokens[i], ctx))
        atom_summaries.append({
            "atom": int(k),
            "n_fires": int(fire_counts[k]),
            "position_min": float(pos[firing, k].min()),
            "position_max": float(pos[firing, k].max()),
            "position_std": float(pos[firing, k].std()),
            "top_tokens_by_position": rows,
        })

        md.append(f"\n## Atom #{k} — {fire_counts[k]} fires, position span "
                  f"[{pos[firing, k].min():.3f}, {pos[firing, k].max():.3f}]\n\n")
        md.append("| t_k | amp | token | preceding context |\n| --- | --- | --- | --- |\n")
        for t, a, tok, ctx in rows:
            tok_disp = tok.replace("|", "\\|").strip() or "(blank)"
            ctx_disp = ctx[:80].replace("\n", " ").replace("|", "\\|")
            md.append(f"| {t:.3f} | {a:.2f} | `{tok_disp}` | …{ctx_disp} |\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(md))
    # JSON sidecar for downstream use.
    (output_path.with_suffix(".json")).write_text(json.dumps({
        "atoms": atom_summaries,
        "n_tokens": len(tokens),
        "n_alive": len(alive),
        "F": F,
    }, indent=2, default=float))
    return {"alive": len(alive), "F": F, "n_tokens": len(tokens)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="curve SAE .pt path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--n-tokens", type=int, default=20000)
    parser.add_argument("--top-k-per-atom", type=int, default=30)
    parser.add_argument("--output", default="runs/dashboard.md")
    args = parser.parse_args()

    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dash] device={device}", flush=True)

    X, tokens = harvest_residuals(args.model, args.layer, args.n_tokens, device)
    print(f"[dash] harvested {X.shape}", flush=True)
    # Normalize the same way llm_sweep does (unit variance, zero mean).
    mu = X.mean(0, keepdim=True)
    sigma = float(X.std().item())
    X_n = (X - mu) / max(sigma, 1e-6)

    sae = load_curve_sae(Path(args.checkpoint), D=X.shape[1], device=device)
    print(f"[dash] loaded SAE: F={sae.config.n_features} top_k={sae.config.top_k}", flush=True)
    summary = make_dashboard(sae, X_n, tokens, args.top_k_per_atom, Path(args.output), device)
    print(f"[dash] wrote {args.output}: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
