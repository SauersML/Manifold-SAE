"""Build an attribution graph for the trained skip-transcoder.

For a target output atom ``j``, compute the top-k input atoms that drive it
via the linearized circuit edge weight

    edge(i → j) = E_b [ z_in,i(b) ] · ⟨W_dec[i], W_dec[j]⟩

(the same operational definition the Paulo et al. paper uses for circuit
weight on tied-decoder transcoders).

Emits:
- ``edges.json``       — full edge list with weights (top-k per target atom).
- ``graph_top50.dot``  — Graphviz dot file restricted to the top 50 atoms
                         by total in-degree weight, ready for
                         ``dot -Tpng graph_top50.dot -o graph.png``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from manifold_sae.transcoder import TranscoderConfig
from gamfit.torch import SkipAffineSmooth


def _load_smooth(ckpt_path: Path) -> SkipAffineSmooth:
    blob = torch.load(ckpt_path, map_location="cpu")
    cfg = TranscoderConfig(**blob["config"])
    smooth = SkipAffineSmooth(
        in_dim=cfg.in_dim,
        out_dim=cfg.out_dim,
        n_atoms=cfg.n_atoms,
        rank_skip=cfg.rank_skip,
        jumprelu_threshold=cfg.jumprelu_threshold,
        learnable_threshold=cfg.learnable_threshold,
        smoothing_eps=cfg.smoothing_eps,
        dtype=torch.float32,
    )
    smooth.load_state_dict(blob["state_dict"])
    smooth.eval()
    return smooth


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default="/Users/user/Manifold-SAE/runs/COGITO_SKIP_TRANSCODER/transcoder.pt")
    p.add_argument("--paired",
                   default="/Users/user/Manifold-SAE/runs/COGITO_PAIRED_L20_L40_STANDIN/paired.pt")
    p.add_argument("--out_dir",
                   default="/Users/user/Manifold-SAE/runs/COGITO_SKIP_TRANSCODER")
    p.add_argument("--top_k_edges", type=int, default=10)
    p.add_argument("--top_n_atoms_for_dot", type=int, default=50)
    args = p.parse_args()

    smooth = _load_smooth(Path(args.ckpt))
    blob = torch.load(args.paired, map_location="cpu")
    X_in = blob["X_in"][:2000]                              # cap for memory
    hsv = blob.get("hsv")

    with torch.no_grad():
        _, z = smooth(X_in.to(torch.float32))
        z_mean = z.mean(dim=0)                              # (F,)
        dec = smooth.W_dec                                  # (F, out)
        # Normalize decoder rows so the inner-product alignment is cosine.
        dec_n = dec / dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        align = dec_n @ dec_n.t()                           # (F, F)
        # edge(i → j) = z_mean[i] * align[i, j]
        contrib = z_mean.unsqueeze(1) * align               # (F_in, F_out)
        contrib.fill_diagonal_(float("-inf"))

    F = contrib.shape[0]
    edges = []
    in_degree = torch.zeros(F)
    for j in range(F):
        col = contrib[:, j]
        vals, idx = torch.topk(col, k=min(args.top_k_edges, F - 1))
        for v, i in zip(vals, idx):
            v_f = float(v.item())
            if not (v_f == v_f) or v_f == float("-inf"):
                continue
            edges.append({"from": int(i.item()), "to": int(j), "weight": v_f})
            in_degree[j] += abs(v_f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "edges.json", "w") as f:
        json.dump(
            {
                "n_atoms": int(F),
                "n_edges": len(edges),
                "top_k_edges_per_atom": int(args.top_k_edges),
                "edges": edges,
            },
            f,
        )
    print(f"[attribution_graph] wrote {out_dir/'edges.json'} ({len(edges)} edges)")

    # Top-50 sub-graph as Graphviz .dot.
    top_atoms = set(int(i.item()) for i in in_degree.topk(args.top_n_atoms_for_dot).indices)
    sub_edges = [e for e in edges if e["from"] in top_atoms and e["to"] in top_atoms]
    lines = ["digraph SkipTranscoder {", "  rankdir=LR;", "  node [shape=circle];"]
    for a in sorted(top_atoms):
        label = f"a{a}"
        if hsv is not None:
            try:
                h = float(hsv[a, 0].item()) if a < hsv.shape[0] else 0.5
            except Exception:
                h = 0.5
            lines.append(f'  {label} [label="{label}", color="0.{int(h*999):03d} 0.8 0.9"];')
        else:
            lines.append(f"  {label};")
    max_w = max((abs(e["weight"]) for e in sub_edges), default=1.0)
    for e in sub_edges:
        w_norm = abs(e["weight"]) / max(max_w, 1e-12)
        lines.append(
            f'  a{e["from"]} -> a{e["to"]} [penwidth={0.5 + 3.0*w_norm:.2f},'
            f' label="{e["weight"]:.3f}"];'
        )
    lines.append("}")
    (out_dir / "graph_top50.dot").write_text("\n".join(lines))
    print(f"[attribution_graph] wrote {out_dir/'graph_top50.dot'} ({len(sub_edges)} sub-edges)")


if __name__ == "__main__":
    main()
