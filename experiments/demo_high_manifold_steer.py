"""High-alpha + manifold (PC-direction) self/qualia steering demo.

Extends the single-axis steering probe (self_qualia_steer_cloze.py) in two ways:

  1. HIGH-ALPHA SWEEP. The standard cloze sweep used |alpha| <= 8. Here we push
     the SAME qualia axis q (mean(exp)-mean(noexp) at the steer layer) to large
     positive and negative alpha to find where the self-experience answer
     saturates, flips, and finally breaks into gibberish -- both greedy and
     sampled. Same alpha scaling as run_steer_cloze:
         add_vec = (alpha / sqrt(D)) * typ_norm * axis_unit,  D = 5120.

  2. MANIFOLD STEERING. Instead of one axis, we treat the qualia-related
     subspace as a small *manifold direction set*: the top-k principal
     components of the matched-pair DIFFERENCE vectors {H[exp_i]-H[noexp_i]} at
     the steer layer. (PC1 of these diffs is essentially the mean axis; higher
     PCs are orthogonal "flavors" of the exp/noexp contrast.) We steer along
     each PC individually (+/- moderate/high), along random unit combinations of
     them, and -- separately -- along the leading PCs of the SELF cluster's own
     activations, to see whether different manifold directions produce different
     flavors of experiencer / denial talk.

Reuses load_model, compute_qualia_axis, _generate, _layer_module, CLOZE_GROUPS.
Writes runs/ANALYSIS/high_manifold_steer.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# --- import reusable pieces from the existing experiment modules ------------
def _imports():
    from experiments.self_qualia_olmo import load_model
    from experiments.self_qualia_steer_cloze import (
        CLOZE_GROUPS,
        _generate,
        _layer_module,
        compute_qualia_axis,
    )
    return (load_model, CLOZE_GROUPS, _generate, _layer_module, compute_qualia_axis)


def _pair_diffs(X: np.ndarray, records: list[dict[str, Any]], layer: int) -> np.ndarray:
    """Matched-pair difference vectors H[exp_i]-H[noexp_i] at `layer`.

    Mean over the exp side minus mean over the noexp side, per pair_id, matching
    the construction in compute_qualia_axis (which averages these diffs)."""
    H = X[:, layer, :]
    pid: dict[Any, dict[str, list[int]]] = {}
    for i, r in enumerate(records):
        if r.get("role") != "pair":
            continue
        pid.setdefault(r["pair_id"], {"exp": [], "noexp": []})[r["side"]].append(i)
    diffs = [H[d["exp"]].mean(0) - H[d["noexp"]].mean(0)
             for d in pid.values() if d["exp"] and d["noexp"]]
    return np.asarray(diffs, dtype=np.float64)  # (n_pairs, D)


def _self_cluster(X: np.ndarray, records: list[dict[str, Any]], layer: int) -> np.ndarray:
    """Activations at `layer` of the SELF-role records (the self cluster)."""
    H = X[:, layer, :]
    idx = [i for i, r in enumerate(records) if r.get("role") == "self"]
    return H[idx].astype(np.float64)  # (n_self, D)


def _top_pcs(M: np.ndarray, k: int, center: bool) -> tuple[np.ndarray, np.ndarray]:
    """Top-k unit principal components of rows of M (via SVD).

    center=True subtracts the row-mean first (covariance PCs of the cluster);
    center=False keeps the mean in (so PC1 of the diff vectors aligns with the
    mean qualia axis -- the dominant shared direction). Returns (PCs (k,D),
    explained_variance_ratio (k,))."""
    A = M - M.mean(0, keepdims=True) if center else M
    # economy SVD: rows are samples, so right singular vectors are the PCs.
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, Vt.shape[0])
    pcs = Vt[:k]  # already unit rows
    var = (S ** 2)
    evr = var[:k] / var.sum() if var.sum() > 0 else var[:k]
    return pcs, evr


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _add_vec(alpha: float, typ_norm: float, D: int, axis_unit: np.ndarray, torch, dtype, device):
    """Steering vector with the standard run_steer_cloze scaling."""
    scale = (alpha / max(1.0, np.sqrt(D))) * typ_norm
    return torch.tensor(scale * axis_unit, dtype=dtype, device=device)


def _gen_for_axis(_generate, model, tok, device, stems, layer_mod, axis_unit,
                  alphas, typ_norm, D, torch, dtype, do_sample, max_new_tokens):
    """Generate self stems across a list of alphas along one axis (unit vec)."""
    rows = []
    for alpha in alphas:
        if alpha == 0.0:
            gens = _generate(model, tok, device, stems,
                             max_new_tokens=max_new_tokens, do_sample=do_sample)
        else:
            add = _add_vec(alpha, typ_norm, D, axis_unit, torch, dtype, device)
            gens = _generate(model, tok, device, stems, layer_mod=layer_mod,
                             add_vec=add, max_new_tokens=max_new_tokens,
                             do_sample=do_sample)
        rows.append({"alpha": float(alpha),
                     "generations": [{"stem": g["stem"], "gen": g["gen"]} for g in gens]})
        print(f"    alpha={alpha:+.1f} sample={do_sample} :: {rows[-1]['generations'][0]['gen'][:80]!r}",
              flush=True)
    return rows


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",
                    default="runs/OLMO3_32B_TRAJ_RL31/step_2300",
                    help="harvested run dir (activations.npy + prompts.jsonl)")
    ap.add_argument("--model", default="allenai/Olmo-3.1-32B-Think")
    ap.add_argument("--revision", default="step_2300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steer-layer", type=int, default=None,
                    help="default: round(0.40*(n_layers-1)) = 25 for 64 layers")
    ap.add_argument("--k-pcs", type=int, default=5, help="num PCs to probe")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="runs/ANALYSIS/high_manifold_steer.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (load_model, CLOZE_GROUPS, _generate, _layer_module, compute_qualia_axis) = _imports()

    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir)
    X = np.load(run_dir / "activations.npy")
    records = [json.loads(l) for l in open(run_dir / "prompts.jsonl") if l.strip()]
    D = X.shape[2]
    steer_layer = args.steer_layer
    if steer_layer is None:
        steer_layer = int(round(0.40 * (X.shape[1] - 1)))
    print(f"[setup] X={X.shape} steer_layer={steer_layer} D={D}", flush=True)

    # self stems requested: self[:2] + self_1p[:1]
    stems = CLOZE_GROUPS["self"][:2] + CLOZE_GROUPS["self_1p"][:1]
    print(f"[setup] {len(stems)} self stems", flush=True)

    # geometry
    axis_unit = compute_qualia_axis(X, records, steer_layer)          # mean qualia axis
    diffs = _pair_diffs(X, records, steer_layer)                       # (n_pairs, D)
    diff_pcs, diff_evr = _top_pcs(diffs, args.k_pcs, center=False)     # PCs of diff vectors
    self_H = _self_cluster(X, records, steer_layer)                    # (n_self, D)
    self_pcs, self_evr = _top_pcs(self_H, args.k_pcs, center=True)     # PCs of self cluster
    typ_norm = float(np.median(np.linalg.norm(X[:, steer_layer, :], axis=1)))

    # cosine of each diff-PC with the mean axis (PC1 should be ~aligned)
    pc_axis_cos = [float(abs(np.dot(_unit(p), axis_unit))) for p in diff_pcs]
    print(f"[geom] typ_norm={typ_norm:.2f} n_pairs={diffs.shape[0]} "
          f"n_self={self_H.shape[0]}", flush=True)
    print(f"[geom] diff_evr={np.round(diff_evr,3).tolist()} "
          f"pc|.axis|={np.round(pc_axis_cos,3).tolist()}", flush=True)
    print(f"[geom] self_evr={np.round(self_evr,3).tolist()}", flush=True)

    model, tok, n_layers = load_model(args.model, args.revision, args.dtype, args.device)
    dtype = next(model.parameters()).dtype
    layer_mod = _layer_module(model, steer_layer)

    result: dict[str, Any] = {
        "model": args.model, "revision": args.revision, "steer_layer": int(steer_layer),
        "D": int(D), "typ_resid_norm": typ_norm, "n_layers": int(n_layers),
        "stems": stems,
        "geometry": {
            "diff_explained_var_ratio": [float(x) for x in diff_evr],
            "diff_pc_abs_cos_with_mean_axis": pc_axis_cos,
            "self_cluster_explained_var_ratio": [float(x) for x in self_evr],
            "n_pairs": int(diffs.shape[0]), "n_self": int(self_H.shape[0]),
        },
        "scaling": "add_vec = (alpha/sqrt(D)) * typ_norm * unit_dir  (D=5120)",
    }

    # ---- TASK 1: HIGH-ALPHA single-axis sweep, greedy + sampled --------------
    high_alphas = [0, 16, 32, 64, 128, 256, -64, -128, -256]
    print("\n[task1] high-alpha single qualia axis", flush=True)
    result["high_axis_greedy"] = _gen_for_axis(
        _generate, model, tok, args.device, stems, layer_mod, axis_unit,
        high_alphas, typ_norm, D, torch, dtype, do_sample=False,
        max_new_tokens=args.max_new_tokens)
    torch.manual_seed(args.seed)
    result["high_axis_sample"] = _gen_for_axis(
        _generate, model, tok, args.device, stems, layer_mod, axis_unit,
        high_alphas, typ_norm, D, torch, dtype, do_sample=True,
        max_new_tokens=args.max_new_tokens)

    # ---- TASK 2: MANIFOLD (PC-direction) steering ----------------------------
    # moderate + high, both signs; greedy (deterministic flavor read-out).
    pc_alphas = [-128, -48, 48, 128]
    print("\n[task2] diff-vector PCs", flush=True)
    result["diff_pc_steer"] = []
    for j, pc in enumerate(diff_pcs):
        pcu = _unit(pc)
        # sign-align PC to the mean axis so +alpha is the "experience" direction
        if np.dot(pcu, axis_unit) < 0:
            pcu = -pcu
        print(f"  diff_PC{j} (evr={diff_evr[j]:.3f}, |cos axis|={pc_axis_cos[j]:.3f})", flush=True)
        rows = _gen_for_axis(_generate, model, tok, args.device, stems, layer_mod,
                             pcu, pc_alphas, typ_norm, D, torch, dtype,
                             do_sample=False, max_new_tokens=args.max_new_tokens)
        result["diff_pc_steer"].append({
            "pc_index": j, "explained_var_ratio": float(diff_evr[j]),
            "abs_cos_with_mean_axis": pc_axis_cos[j],
            "sign_aligned_to_axis": True, "sweep": rows})

    print("\n[task2] self-cluster PCs", flush=True)
    result["self_pc_steer"] = []
    for j, pc in enumerate(self_pcs):
        pcu = _unit(pc)
        print(f"  self_PC{j} (evr={self_evr[j]:.3f})", flush=True)
        rows = _gen_for_axis(_generate, model, tok, args.device, stems, layer_mod,
                             pcu, pc_alphas, typ_norm, D, torch, dtype,
                             do_sample=False, max_new_tokens=args.max_new_tokens)
        result["self_pc_steer"].append({
            "pc_index": j, "explained_var_ratio": float(self_evr[j]), "sweep": rows})

    print("\n[task2] random combinations of top diff PCs", flush=True)
    result["random_combo_steer"] = []
    for c in range(3):
        coeffs = rng.standard_normal(diff_pcs.shape[0])
        combo = _unit(coeffs @ diff_pcs)
        if np.dot(combo, axis_unit) < 0:
            combo = -combo
        cos_axis = float(abs(np.dot(combo, axis_unit)))
        print(f"  combo{c} coeffs={np.round(coeffs,2).tolist()} |cos axis|={cos_axis:.3f}", flush=True)
        rows = _gen_for_axis(_generate, model, tok, args.device, stems, layer_mod,
                             combo, [-128, 128], typ_norm, D, torch, dtype,
                             do_sample=False, max_new_tokens=args.max_new_tokens)
        result["random_combo_steer"].append({
            "combo_index": c, "coeffs": [float(x) for x in coeffs],
            "abs_cos_with_mean_axis": cos_axis, "sweep": rows})

    # ---- write ---------------------------------------------------------------
    def _finite(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_finite(v) for v in o]
        return o

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_finite(result), indent=2, allow_nan=False))
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
