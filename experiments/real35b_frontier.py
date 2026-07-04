"""Real 35B L17 block-lane frontier from the scale_evidence frame-health run.

We do NOT re-fit the block lane here: the K in {4k,16k,32k} block-sparse dictionaries
on real Qwen3.6-35B-A3B L17 activations are already measured (scale_evidence/
t1_frame_health.json -- EV 0.707/0.906/0.990, 0 dead blocks, L0=2 blocks/token). This
script lifts those into the frontier schema (EV, realized FLOPs, MDL bits) so the same
plots/analysis apply, and records the honest curved-refinement feasibility envelope at
that massive K.

Conventions (documented, not hidden):
  * p = 2048  -- the dense-width the L17 scale runs used (scale_evidence README:
    "dense-width joint fit at p=2048", "150k x 2048 stagewise"). N ~= 150000 tokens.
  * A block-sparse atom is a width-`block_size` block: block_size decoder vectors of
    dim p. dv_per_atom = block_size = 2. Active blocks per token s = 2 (from
    utilization_mean * K_capacity = 2.0 across all K). Coded coordinates / token =
    s * block_size = 4.
  * Activations are train-PCA-whitened to ~unit per-coordinate variance, so signal_var
    per active coord = 1 and the matched-distortion floor delta2 is taken as the best
    achieved normalized residual (1 - EV at K=32k = 0.0105).
  * Selection bits: combinatorial log2 C(K, s) only. Empirical support entropy H(S)
    needs the per-token routing dump (not in frame_health); flagged as owed.

Curved-refinement feasibility at massive K: the manifold REML lane fits in ~O(minutes)
at K~10 (measured: ~70 s at p=12, and >5 min at p=256/N=6k/K=24 on a 16-core node). It
does not reach K in the thousands with the current solver -- a single K=4000 curved fit
extrapolates to days. So on real 35B at K in {4k,16k,32k} the block lane is the only
feasible tool; curved refinement's per-atom efficiency is a *small-K* property (see the
synthetic frontier). We report that complementarity as the honest result, not a curved
number we cannot produce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.frontier_bench import flop_model, mdl_bits


def build(frame_health_path: str, *, p: int = 2048, n_tokens: int = 150000,
          param_bits: float = 16.0) -> dict:
    fh = json.loads(Path(frame_health_path).read_text())
    rows = []
    best_ev = max(r["explained_variance"] for r in fh)
    delta2 = max(1e-4, 1.0 - best_ev)          # normalized residual floor at matched distortion
    for r in fh:
        K = int(r["K_capacity"] // r["block_size"])     # number of blocks (atoms)
        bs = int(r["block_size"])
        s_blocks = round(r["utilization_mean"] * r["K_capacity"])   # active blocks/token
        coded_coords = s_blocks * bs
        flops = flop_model([bs] * K, s_blocks, p, curved=False,
                           passes=1, n=n_tokens)         # block lane: 1 streamed pass accounting
        row = {"ok": True, "lane": "block", "K": K, "k_realized": K, "active": s_blocks,
               "p": p, "block_size": bs, "heldout_ev": r["explained_variance"],
               "heldout_mse": 1.0 - r["explained_variance"],
               "mean_l0": float(coded_coords), "dv_per_atom": [bs] * min(K, 1),
               "frac_dead_blocks": r["frac_dead_blocks"], "wall_s": r["wall_s"],
               "flops": flops}
        row["mdl"] = mdl_bits(mean_l0=coded_coords, signal_var=1.0, delta2=delta2, K=K,
                              decoder_params=flops["decoder_params"], param_bits=param_bits,
                              n_tokens=n_tokens, hs_bits=None)   # H(S) owed: no routing dump
        rows.append(row)
    return {
        "harness": "real35b_frontier",
        "source": "scale_evidence/t1_frame_health.json (Qwen3.6-35B-A3B L17, block lane)",
        "conventions": {"p": p, "n_tokens": n_tokens, "param_bits": param_bits,
                        "matched_distortion_delta2_normalized": delta2,
                        "signal_var_per_coord": 1.0, "selection": "combinatorial only (H(S) owed)"},
        "curved_feasibility": ("manifold REML lane is a small-K tool (~70s at p=12; "
                               ">5min at p=256/N=6k/K=24 on 16 cores); K in the thousands "
                               "extrapolates to days -- not run. Block lane owns massive K; "
                               "curved owns small-K EV-per-atom + bits (see synthetic frontier)."),
        "config": {"dgp": "real35b_L17", "p": p, "n": n_tokens, "concepts": None,
                   "lanes": ["block"], "k": [r["K"] for r in rows]},
        "matched_distortion_delta2": delta2,
        "results": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame-health",
                    default="results/suite_2026-07-03/scale_evidence/t1_frame_health.json")
    ap.add_argument("--p", type=int, default=2048)
    ap.add_argument("--n-tokens", type=int, default=150000)
    ap.add_argument("--out", default="results/suite_2026-07-03/frontiers/real_l17_block.json")
    args = ap.parse_args(argv)
    payload = build(args.frame_health, p=args.p, n_tokens=args.n_tokens)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[real35b] wrote {args.out}")
    for r in payload["results"]:
        print(f"  K={r['K']:6d} EV={r['heldout_ev']:.4f} L0coords={r['mean_l0']:.0f} "
              f"infer_macs/tok={r['flops']['infer_macs_per_token']:.3g} "
              f"bits/tok(comb)={r['mdl']['bits_per_token_combinatorial']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
