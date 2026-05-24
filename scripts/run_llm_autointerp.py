"""LLM-based autointerp on the three trained SAEs (Anthropic API).

Pipeline per model:
  1. Pick top-firing N atoms.
  2. Collect top-K activating (color, template) examples per atom.
  3. Batch-explain via Anthropic Haiku 4.5 with prompt caching.
  4. Score each explanation by LLM-simulation accuracy on a held-out
     stratified eval set.
  5. Also compute the local feature-regression simulation R² for comparison.

Outputs to runs/autointerp_llm/:
  - explanations_{topk,l1,manifold}.jsonl
  - sim_scores_{topk,l1,manifold}.jsonl
  - report.md  (head-to-head table + sample explanations + cost)
  - summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Re-use the SAE class definitions + helpers from the existing rule-based driver.
# Importing the module is safe: it only defines classes + a main() guarded by __main__.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_autointerp_local", ROOT / "scripts" / "run_autointerp.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

TopKSAE = _mod.TopKSAE
L1SAE = _mod.L1SAE
ManifoldSAE = _mod.ManifoldSAE
F_ATOMS = _mod.F_ATOMS
load_xkcd_colors = _mod.load_xkcd_colors
make_split = _mod.make_split

from manifold_sae.autointerp.explain import (
    rgb_to_hsv,
    load_sae_activations,
    collect_top_activating,
)
from manifold_sae.autointerp.score import score_hypothesis
from manifold_sae.autointerp.llm_explain import (
    DEFAULT_MODEL,
    llm_explain_atoms,
    llm_explanation_to_hypothesis,
    explanation_to_dict,
    estimate_cost,
)
from manifold_sae.autointerp.llm_score import (
    build_eval_examples,
    llm_score_atom,
    estimate_score_cost,
)


def run_model(
    model_name: str,
    kind: str,
    ctor,
    ckpt_path: Path,
    X_val_np: np.ndarray,
    val_idx,
    row_color_all,
    row_template_all,
    color_names,
    color_hsv,
    *,
    max_atoms: int,
    top_k: int,
    eval_per_atom: int,
    batch_size: int,
    n_score_atoms: int,
    device: str,
    llm_model: str,
    client,
):
    print(f"\n=== {model_name} ({kind}) ===")
    sae = ctor().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sae.load_state_dict(state)
    sae.eval()

    acts_val = load_sae_activations(sae, X_val_np, kind, device=device, batch_size=1024)
    row_color = row_color_all[val_idx]
    row_template = row_template_all[val_idx]

    firing_rate = (acts_val > 1e-3).mean(0)
    chosen_atoms = np.argsort(-firing_rate)[:max_atoms]

    atoms_payload = []
    for aid in chosen_atoms:
        aid = int(aid)
        top_ex = collect_top_activating(
            acts_val, aid, row_color, row_template, color_names, n_top=top_k,
        )
        atoms_payload.append({"atom_id": aid, "top_examples": top_ex})

    t0 = time.time()
    explanations = llm_explain_atoms(
        atoms_payload,
        model_name=model_name,
        color_hsv_lookup=color_hsv,
        batch_size=batch_size,
        llm_model=llm_model,
        client=client,
    )
    t_explain = time.time() - t0
    print(f"  explain: {len(explanations)} atoms in {t_explain:.1f}s")

    # Local-regression simulation R² (cheap sanity baseline)
    local_r2s = []
    for exp in explanations:
        h = llm_explanation_to_hypothesis(exp)
        sc = score_hypothesis(h, acts_val, color_hsv, color_names, row_color, row_template)
        local_r2s.append(sc["r2"])

    # LLM-simulation: only the top-firing N (LLM-scoring is expensive)
    scoring_subset = explanations[:n_score_atoms]
    sim_scores = []
    t0 = time.time()
    for i, exp in enumerate(scoring_subset):
        if exp.n_active == 0:
            continue
        ex, gt = build_eval_examples(
            acts_val, exp.atom_id, row_color, row_template, color_names,
            n_pos=eval_per_atom // 2, n_neg=eval_per_atom // 2,
            seed=exp.atom_id,
        )
        if not ex:
            continue
        s = llm_score_atom(
            exp, ex, gt,
            color_hsv_lookup=color_hsv,
            examples_per_call=eval_per_atom,
            llm_model=llm_model,
            client=client,
        )
        sim_scores.append(s)
        if i % 8 == 0:
            print(f"  score atom {exp.atom_id:4d}  acc={s.accuracy:.3f}  "
                  f"(true+={s.n_positive_true}, pred+={s.n_positive_pred})")
    t_score = time.time() - t0
    print(f"  score: {len(sim_scores)} atoms in {t_score:.1f}s")

    accs = [s.accuracy for s in sim_scores]
    summary = {
        "model": model_name,
        "kind": kind,
        "n_atoms_explained": len(explanations),
        "n_atoms_scored": len(sim_scores),
        "mean_sim_accuracy": float(np.mean(accs)) if accs else 0.0,
        "median_sim_accuracy": float(np.median(accs)) if accs else 0.0,
        "frac_above_0.7": float(np.mean([a >= 0.7 for a in accs])) if accs else 0.0,
        "mean_local_r2": float(np.mean(local_r2s)) if local_r2s else 0.0,
        "explain_cost": estimate_cost(explanations),
        "score_cost": estimate_score_cost(sim_scores),
        "explain_seconds": t_explain,
        "score_seconds": t_score,
    }
    return explanations, sim_scores, local_r2s, summary


def write_report(out_dir: Path, all_summaries, all_explanations):
    lines = [
        "# LLM Autointerp — Head-to-Head Report",
        "",
        f"Model: {DEFAULT_MODEL}  (Anthropic Haiku 4.5, prompt-cached)",
        "",
        "## Simulation Accuracy Table",
        "",
        "| SAE      | n_atoms | mean LLM-sim acc | median acc | frac ≥ 0.7 | local-reg R² | explain $ | score $ |",
        "|----------|--------:|-----------------:|-----------:|-----------:|-------------:|----------:|--------:|",
    ]
    total_cost = 0.0
    for s in all_summaries:
        c = s["explain_cost"]["usd"] + s["score_cost"]["usd"]
        total_cost += c
        lines.append(
            f"| {s['model']:8s} | {s['n_atoms_scored']:7d} | "
            f"{s['mean_sim_accuracy']:.3f}            | "
            f"{s['median_sim_accuracy']:.3f}      | "
            f"{s['frac_above_0.7']:.3f}      | "
            f"{s['mean_local_r2']:+.3f}        | "
            f"${s['explain_cost']['usd']:.4f}  | "
            f"${s['score_cost']['usd']:.4f} |"
        )
    lines += ["", f"**Total cost: ${total_cost:.4f}**", ""]
    lines += [
        "## Sample Explanations (top-3 per model)",
        "",
    ]
    for model_name, exps in all_explanations.items():
        lines.append(f"### {model_name}")
        for e in exps[:3]:
            top = ", ".join(f"{x['color']}" for x in e.top_examples[:5])
            lines.append(
                f"- **atom {e.atom_id}** (conf={e.confidence:.2f}): {e.explanation}"
            )
            lines.append(f"  - top examples: {top}")
        lines.append("")
    out_dir.joinpath("report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "runs" / "sae_comparison"))
    ap.add_argument("--output", default=str(ROOT / "runs" / "autointerp_llm"))
    ap.add_argument("--max_atoms", type=int, default=64)
    ap.add_argument("--n_score_atoms", type=int, default=32,
                    help="subset of explained atoms to actually LLM-score")
    ap.add_argument("--top_k", type=int, default=20,
                    help="top-K activating examples per atom in explain prompt")
    ap.add_argument("--eval_per_atom", type=int, default=20,
                    help="held-out examples per atom in score prompt")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="atoms per explain API call")
    ap.add_argument("--llm_model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry_run", action="store_true",
                    help="print cost estimate and exit without calling API")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    sae_dir = Path(args.model_dir)

    # --- data ---
    X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
    N, D = X.shape
    N_COLORS, N_TPL = 949, 28
    train_idx, val_idx, row_color_all, row_template_all = make_split(N_COLORS, N_TPL)

    X_train_np = np.ascontiguousarray(X[train_idx]).astype(np.float32)
    X_val_np = np.ascontiguousarray(X[val_idx]).astype(np.float32)
    mu = X_train_np.mean(0)
    X_train_np -= mu
    X_val_np -= mu

    color_names, color_rgb = load_xkcd_colors(
        ROOT / "experiments" / "xkcd_colors.txt", n=N_COLORS,
    )
    color_hsv = rgb_to_hsv(color_rgb)
    print(f"[data] X={X.shape}  val rows={len(val_idx)}  colors={len(color_names)}")

    # --- cost estimate ---
    # ~200 tokens prompt + 100 tokens response per atom (Haiku 4.5).
    n_models = 3
    explain_calls = n_models * (args.max_atoms // args.batch_size + 1)
    score_calls = n_models * args.n_score_atoms
    est_input = (explain_calls * (500 + args.batch_size * 400)
                 + score_calls * (200 + args.eval_per_atom * 30))
    est_output = explain_calls * args.batch_size * 150 + score_calls * args.eval_per_atom * 3
    from manifold_sae.autointerp.llm_explain import (
        PRICE_INPUT_PER_MTOK, PRICE_OUTPUT_PER_MTOK,
    )
    est_cost = (est_input * PRICE_INPUT_PER_MTOK + est_output * PRICE_OUTPUT_PER_MTOK) / 1e6
    print(f"[cost] estimate ~${est_cost:.3f}  (input~{est_input}, output~{est_output} tokens)")
    if args.dry_run:
        print("[dry-run] exiting without API calls.")
        return 0

    # --- client ---
    from manifold_sae.autointerp.llm_explain import _make_client
    client = _make_client()

    model_specs = [
        ("topk",     "topk",     sae_dir / "model_topk.pt",
         lambda: TopKSAE(D, F_ATOMS, top_k=32)),
        ("l1",       "l1",       sae_dir / "model_l1.pt",
         lambda: L1SAE(D, F_ATOMS)),
        ("manifold", "manifold", sae_dir / "model_manifold.pt",
         lambda: ManifoldSAE(D, F_ATOMS, M_F=3)),
    ]

    all_summaries = []
    all_explanations = {}
    for model_name, kind, ckpt_path, ctor in model_specs:
        explanations, sim_scores, local_r2s, summary = run_model(
            model_name, kind, ctor, ckpt_path,
            X_val_np, val_idx, row_color_all, row_template_all,
            color_names, color_hsv,
            max_atoms=args.max_atoms,
            top_k=args.top_k,
            eval_per_atom=args.eval_per_atom,
            batch_size=args.batch_size,
            n_score_atoms=args.n_score_atoms,
            device=args.device,
            llm_model=args.llm_model,
            client=client,
        )
        all_summaries.append(summary)
        all_explanations[model_name] = explanations

        with open(out_dir / f"explanations_{model_name}.jsonl", "w") as f:
            for e in explanations:
                f.write(json.dumps(explanation_to_dict(e)) + "\n")
        with open(out_dir / f"sim_scores_{model_name}.jsonl", "w") as f:
            for s in sim_scores:
                f.write(json.dumps(asdict(s)) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"per_model": all_summaries,
                   "config": vars(args)}, f, indent=2)
    write_report(out_dir, all_summaries, all_explanations)
    print(f"\n[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
