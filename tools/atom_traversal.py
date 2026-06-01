"""Unsupervised atom interpretation via curve traversal in unembedding space.

The Manifold-SAE architectural advantage that vanilla can't match:
each atom HAS a curve `g_k: [0,1] → ℝ^D`. Sampling `t = 0, 0.1, ..., 1`
gives a sequence of directions in residual stream. Decoding each
direction through the LM's unembedding (or just looking at top tokens
nearby in residual space) gives the SEMANTIC GRADIENT the atom encodes.

This is NOT post-hoc token clustering. It's DECODER-SIDE inspection of
what the atom DOES — the atom traces a path through the LM's
representation space, and we just read off the tokens it passes
through.

For a magnitude atom: sampling t = [0.05, 0.2, 0.5, 0.8, 0.95] should
produce a direction sequence whose nearest tokens look like
[small-number, ..., large-number]. The list of "traversed tokens"
IS the atom's semantic interpretation.

Output: a markdown catalog per atom + JSON sidecar.

Usage:
  python tools/atom_traversal.py \\
      --checkpoint runs/llm_sweep_q15b_L18/curve_F128.pt \\
      --model Qwen/Qwen2.5-1.5B --layer 18 --n-samples 12
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

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def load_curve_sae(path: Path, D: int, device: torch.device):
    from manifold_sae.sae import load_sae
    return load_sae(path, input_dim=D, device=device)


def curve_directions_along_t(sae, atom_k: int, n_samples: int, device: torch.device) -> torch.Tensor:
    """Sample atom k's curve at `n_samples` t values; return the corresponding
    ambient directions `g_k(t)` as a `(n_samples, D)` tensor.

    Cutover: the gamfit-native decoder block already lives in ambient R^D, so
    there is no separate `directions` (`W_k`) lift. We read the per-atom ambient
    curve straight off the primitive via `lift_atom_curve`.
    """
    from manifold_sae.sae import lift_atom_curve
    t_grid = torch.linspace(0.05, 0.95, n_samples, dtype=torch.float64, device=device)
    return lift_atom_curve(sae, atom_k, t_grid).to(device)


def nearest_tokens_to_direction(directions: torch.Tensor, embed_table: torch.Tensor,
                                 tok, top_k: int = 5) -> list[list[tuple[str, float]]]:
    """For each direction (n_samples, D), find the top-K nearest tokens in
    the LM's input embedding table. Returns per-sample lists of
    (token, similarity).
    """
    # Use cosine similarity in the embedding space.
    dirs_norm = directions / (directions.norm(dim=1, keepdim=True) + 1e-9)
    embed_norm = embed_table / (embed_table.norm(dim=1, keepdim=True) + 1e-9)
    sim = dirs_norm.to(embed_norm.dtype) @ embed_norm.t()                     # (n, V)
    top = sim.topk(top_k, dim=1)
    out = []
    for i in range(directions.shape[0]):
        toks = [(tok.decode(int(idx)), float(sim_val))
                for idx, sim_val in zip(top.indices[i], top.values[i])]
        out.append(toks)
    return out


def per_atom_alive_count(sae, X_corpus: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run SAE on a corpus, return per-atom firing count."""
    with torch.no_grad():
        out = sae(X_corpus.to(device=device, dtype=sae.cfg.dtype))
    amp = out.amplitudes.cpu().numpy()
    return (amp > 1e-6).sum(axis=0)


def _heuristic_label(traversal: list[list[tuple[str, float]]]) -> str:
    """Heuristic auto-label by looking at the traversed token sequence.

    * If all top tokens look numeric → magnitude.
    * If top tokens spell out a known sequence (months, days, letters) → cyclic.
    * If top tokens at t=0 are opposites of top tokens at t=1 → bipolar.
    * Otherwise: report the first-and-last top tokens.
    """
    if not traversal:
        return "no signal"
    first_tok = traversal[0][0][0].strip().lower()
    last_tok = traversal[-1][0][0].strip().lower()
    all_tops = [tok.strip().lower() for sample in traversal for tok, _ in sample[:2]]

    def is_numeric(s: str) -> bool:
        try: float(s.replace(",", "")); return True
        except: return False
    numeric_frac = sum(is_numeric(t) for t in all_tops) / max(len(all_tops), 1)
    if numeric_frac > 0.5:
        return f"likely magnitude (numeric tokens; {first_tok!r} → {last_tok!r})"

    MONTHS = {"january","february","march","april","may","june",
              "july","august","september","october","november","december"}
    DAYS = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
    if any(t in MONTHS for t in all_tops):
        return f"likely months (cyclic? {first_tok!r} → {last_tok!r})"
    if any(t in DAYS for t in all_tops):
        return f"likely weekdays (cyclic? {first_tok!r} → {last_tok!r})"

    return f"traversal: {first_tok!r} → {last_tok!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=os.environ.get("MSAE_CHECKPOINT"),
                        required=False, help="curve SAE .pt")
    parser.add_argument("--model", default=os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-0.5B"))
    parser.add_argument("--layer", type=int, default=int(os.environ.get("MSAE_LAYER", "12")))
    parser.add_argument("--n-samples", type=int, default=12,
                        help="how many t values to sample along each atom's curve")
    parser.add_argument("--top-k-per-sample", type=int, default=5,
                        help="how many nearest tokens to report per t value")
    parser.add_argument("--max-atoms", type=int, default=64,
                        help="cap on alive atoms to catalog (sorted by firing count)")
    parser.add_argument("--n-corpus-tokens", type=int, default=5000,
                        help="how many wikitext tokens to compute firing rates from")
    parser.add_argument("--output",
                        default=(os.environ.get("MANIFOLD_SAE_OUTPUT_DIR") or "runs") + "/atom_catalog")
    args = parser.parse_args()

    if not args.checkpoint:
        print("[catalog] --checkpoint or MSAE_CHECKPOINT required", file=sys.stderr)
        return 2

    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[catalog] device={device} output_dir={out_dir}", flush=True)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[catalog] loading {args.model}", flush=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()

    # Find the LM's input-embedding table for unembedding lookup.
    if hasattr(model, "embed_tokens"):
        embed_table = model.embed_tokens.weight.detach()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_table = model.model.embed_tokens.weight.detach()
    elif hasattr(model, "wte"):
        embed_table = model.wte.weight.detach()
    else:
        print("[catalog] could not find embedding table", file=sys.stderr)
        return 1

    D = embed_table.shape[1]
    sae = load_curve_sae(Path(args.checkpoint), D, device)
    F = sae.cfg.n_atoms
    print(f"[catalog] loaded SAE F={F} top_k={sae.cfg.sparsity.target_k}", flush=True)

    # Compute firing rates on a small wikitext sample to rank atoms.
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    blocks = _find_blocks(model)
    captured = {}
    h_hook = blocks[args.layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    Xs = []
    with torch.no_grad():
        for d in ds:
            if len(Xs) >= args.n_corpus_tokens: break
            text = d.get("text", "")
            if not isinstance(text, str) or len(text) < 100: continue
            inputs = tok(text[:1500], return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**inputs)
            for i in range(min(captured["h"].shape[1], 32)):
                Xs.append(captured["h"][0, i, :].cpu())
                if len(Xs) >= args.n_corpus_tokens: break
    h_hook.remove()
    X = torch.stack(Xs[: args.n_corpus_tokens], dim=0)
    mu = X.mean(0, keepdim=True); sigma = X.std(0).clamp(min=1e-6)  # per-dim std (was scalar — see _normalize.py)
    X_n = (X - mu) / sigma
    fire_counts = per_atom_alive_count(sae, X_n, device)
    alive = [k for k in range(F) if fire_counts[k] >= 20]
    if len(alive) > args.max_atoms:
        # Sort by firing count descending
        alive = sorted(alive, key=lambda k: -fire_counts[k])[: args.max_atoms]
    print(f"[catalog] alive atoms to catalog: {len(alive)}", flush=True)

    # For each alive atom: sample curve, find nearest tokens.
    catalog = []
    for k in alive:
        directions = curve_directions_along_t(sae, k, args.n_samples, device)
        traversal = nearest_tokens_to_direction(
            directions, embed_table, tok, top_k=args.top_k_per_sample,
        )
        label = _heuristic_label(traversal)
        catalog.append({
            "atom": int(k),
            "n_fires": int(fire_counts[k]),
            "heuristic_label": label,
            "traversal": [
                {"t": float(0.05 + (0.95 - 0.05) * i / max(args.n_samples - 1, 1)),
                 "top_tokens": traversal[i]}
                for i in range(args.n_samples)
            ],
        })
        print(f"  atom #{k:<4}  fires={fire_counts[k]:<5}  {label}", flush=True)

    # Render markdown
    md = [f"# Unsupervised atom catalog — {args.model} L{args.layer}\n",
          f"Checkpoint: `{args.checkpoint}`\n",
          f"{len(alive)} alive atoms catalogued (out of F={F}).\n\n"]
    for entry in catalog:
        md.append(f"## Atom #{entry['atom']} — {entry['n_fires']} fires\n")
        md.append(f"*{entry['heuristic_label']}*\n\n")
        md.append("| t | top-3 nearest tokens (cosine) |\n| --- | --- |\n")
        for tr in entry["traversal"]:
            top3 = " · ".join(f"`{tk!r}` ({s:.2f})" for tk, s in tr["top_tokens"][:3])
            md.append(f"| {tr['t']:.2f} | {top3} |\n")
        md.append("\n")
    (out_dir / "atom_catalog.md").write_text("".join(md))
    (out_dir / "atom_catalog.json").write_text(json.dumps(catalog, indent=2, default=float))
    print(f"[done] wrote {out_dir / 'atom_catalog.md'} ({len(alive)} atoms)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
