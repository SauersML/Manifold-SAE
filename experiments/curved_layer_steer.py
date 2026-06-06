"""Deepen the CURVED-LAYER self manifold-tangent steering (L18 & L32).

Builds on demo_manifold_tangent_steer.py (QualiaManifold = gamfit B-spline smooth
of the residual stream along the qualia latent, with an analytic tangent dg/dt).
Here we focus on the two layers found to be CURVED (the tangent rotates with
position): L18 and L32, contrasted against the ~linear L25.

Adds, per layer:
  * the gamfit curvature gain  (R2_smooth - R2_linear) and tangent_cos(lo,hi).
  * a finer alpha grid for greedy + sampled generations.
  * GEOMETRIC divergence: for every alpha, cosine between the actual ADDED vector
    of tangent-steering vs linear-steering, averaged over the live latents seen
    during generation -- i.e. how much the curved-manifold tangent points away
    from the global linear axis (this is the curvature acting causally).
  * a cross-layer summary relating tangent-vs-linear divergence to curvature gain.
  * explicit confirmation/characterization of the L32 "+64 opposite valence"
    finding (does +alpha tangent flip valence relative to +alpha linear?).

Writes runs/ANALYSIS/curved_layer_steer.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _imports():
    import os
    import sys
    _exp_dir = os.path.dirname(os.path.abspath(__file__))
    if _exp_dir not in sys.path:
        sys.path.insert(0, _exp_dir)
    if os.path.dirname(_exp_dir) not in sys.path:
        sys.path.insert(0, os.path.dirname(_exp_dir))
    from experiments.self_qualia_olmo import load_model
    from experiments.self_qualia_steer_cloze import (
        CLOZE_GROUPS, _generate, _layer_module, compute_qualia_axis,
        EXP_CONTS, NOEXP_CONTS, _score_continuations,
    )
    from experiments.color_manifold_gam import bspline_1d_basis, reml_fit
    from experiments.demo_manifold_tangent_steer import QualiaManifold
    from _pca_basis import fit_top_pcs
    return (load_model, CLOZE_GROUPS, _generate, _layer_module, compute_qualia_axis,
            EXP_CONTS, NOEXP_CONTS, _score_continuations,
            bspline_1d_basis, reml_fit, QualiaManifold, fit_top_pcs)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _valence_gap(model, tok, device, stems, layer_mod, add_fn,
                 _score_continuations, EXP_CONTS, NOEXP_CONTS):
    """Mean (logsumexp exp - logsumexp noexp) over stems, under a steering hook.

    add_fn(hid)->hid installs the steer (None = unsteered). >0 => experiencer
    ('yes') preferred; this is the scalar 'valence' we track for the +/-64 check.
    """
    import torch
    handle = None
    if add_fn is not None:
        def hook(_m, _inp, output):
            is_t = isinstance(output, tuple)
            hid = output[0] if is_t else output
            hid = add_fn(hid)
            return (hid,) + tuple(output[1:]) if is_t else hid
        handle = layer_mod.register_forward_hook(hook)
    try:
        exp = _score_continuations(model, tok, device, stems, EXP_CONTS)
        noexp = _score_continuations(model, tok, device, stems, NOEXP_CONTS)
    finally:
        if handle is not None:
            handle.remove()

    def lse(a):
        m = np.nanmax(a, axis=1, keepdims=True)
        return m[:, 0] + np.log(np.nansum(np.exp(a - m), axis=1))
    gap = lse(exp) - lse(noexp)
    return float(np.nanmean(gap))


def _steer_gen_capture(_generate, model, tok, device, stems, layer_mod, mode, manifold,
                       axis_unit, alpha, typ_norm, D, torch, dtype, do_sample,
                       max_new_tokens):
    """Like demo_manifold_tangent_steer._steer_gen but also captures the cosine
    between the tangent-add and the linear-add at every generated position, so we
    can quantify how far the curved tangent diverges from the linear axis."""
    scale = (alpha / max(1.0, np.sqrt(D))) * typ_norm
    axis_unit_u = _unit(axis_unit)

    if mode == "none" or alpha == 0.0:
        gens = _generate(model, tok, device, stems,
                         max_new_tokens=max_new_tokens, do_sample=do_sample)
        return gens, []
    if mode == "linear":
        add = torch.tensor(scale * axis_unit_u, dtype=dtype, device=device)
        gens = _generate(model, tok, device, stems, layer_mod=layer_mod,
                         add_vec=add, max_new_tokens=max_new_tokens, do_sample=do_sample)
        return gens, []

    # tangent mode: recompute tangent at the live latent; record cos vs axis.
    cos_log: list[float] = []

    def hook(_m, _inp, output):
        is_tuple = isinstance(output, tuple)
        hid = output[0] if is_tuple else output
        h_vec = hid[0, -1, :].to(torch.float64).detach().cpu().numpy()
        tang = manifold.tangent_at(h_vec)                 # unit (D,)
        cos_log.append(float(np.dot(tang, axis_unit_u)))
        add = torch.tensor(scale * tang, dtype=hid.dtype, device=hid.device)
        hid = hid + add
        return (hid,) + tuple(output[1:]) if is_tuple else hid

    handle = layer_mod.register_forward_hook(hook)
    try:
        outs = []
        for s in stems:
            enc = tok(s, return_tensors="pt", add_special_tokens=True).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=do_sample, pad_token_id=tok.pad_token_id)
            new = gen[0, enc["input_ids"].shape[1]:]
            outs.append({"stem": s, "gen": tok.decode(new, skip_special_tokens=True)})
        return outs, cos_log
    finally:
        handle.remove()


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/OLMO3_32B_TRAJ_RL31/step_2300")
    ap.add_argument("--act-subdir", default=".",
                    help="subdir with the SELF/QUALIA harvest (activations.npy + "
                         "prompts.jsonl with role=='pair'). Default '.' = the run-dir "
                         "root (step_2300/), NOT extra/ which holds the COLOR harvest.")
    ap.add_argument("--model", default="allenai/Olmo-3.1-32B-Think")
    ap.add_argument("--revision", default="step_2300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=str, default="18,25,32")
    ap.add_argument("--k-pc", type=int, default=24)
    ap.add_argument("--n-basis", type=int, default=10)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="runs/ANALYSIS/curved_layer_steer.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (load_model, CLOZE_GROUPS, _generate, _layer_module, compute_qualia_axis,
     EXP_CONTS, NOEXP_CONTS, _score_continuations,
     bspline_1d_basis, reml_fit, QualiaManifold, fit_top_pcs) = _imports()

    run_dir = Path(args.run_dir)
    act_dir = run_dir / args.act_subdir
    X = np.load(act_dir / "activations.npy")
    records = [json.loads(l) for l in open(act_dir / "prompts.jsonl") if l.strip()]
    D = X.shape[2]
    stems = CLOZE_GROUPS["self"][:2] + CLOZE_GROUPS["self_1p"][:1]
    val_stems = CLOZE_GROUPS["self"] + CLOZE_GROUPS["self_1p"]
    layers = [int(x) for x in args.layers.split(",")]

    # ---- CPU: fit every layer's manifold + curvature diagnostics first --------
    fits = {}
    for layer in layers:
        axis_unit = compute_qualia_axis(X, records, layer)
        man = QualiaManifold(X, records, layer, axis_unit, args.k_pc, args.n_basis,
                             args.degree, bspline_1d_basis, reml_fit, fit_top_pcs)
        typ_norm = float(np.median(np.linalg.norm(X[:, layer, :], axis=1)))
        fits[layer] = (axis_unit, man, typ_norm)
        print(f"[fit] L{layer} R2(g)={man.r2:.4f} R2(lin)={man.r2_linear:.4f} "
              f"curv_gain={man.r2 - man.r2_linear:+.4f} tan_cos(lo,hi)={man.tangent_cos_lo_hi:+.3f} "
              f"|tan.axis|@mid={man.tangent_axis_cos:.3f}", flush=True)

    result: dict[str, Any] = {
        "model": args.model, "revision": args.revision, "D": int(D), "stems": stems,
        "scaling": "add = (alpha/sqrt(D)) * typ_norm * unit_dir",
        "note": "LINEAR = global constant qualia axis; TANGENT = dg/dt of fitted curved "
                "manifold recomputed at the live residual every step. curvature_gain_r2 = "
                "R2(smooth)-R2(linear); tangent_cos_lo_hi<1 => tangent rotates with position. "
                "valence_gap>0 => experiencer ('yes') preferred (teacher-forced).",
        "per_layer": [],
    }

    model, tok, _ = load_model(args.model, args.revision, args.dtype, args.device)
    dtype = next(model.parameters()).dtype

    greedy_alphas = [-128, -96, -64, -48, -32, -16, 0, 16, 32, 48, 64, 96, 128]
    sample_alphas = [-128, -64, 64, 128]

    summary_rows = []

    for layer in layers:
        axis_unit, man, typ_norm = fits[layer]
        layer_mod = _layer_module(model, layer)
        print(f"\n===== LAYER {layer} (typ_norm={typ_norm:.2f}, "
              f"curv_gain={man.r2 - man.r2_linear:+.4f}) =====", flush=True)

        lr: dict[str, Any] = {
            "steer_layer": int(layer), "typ_resid_norm": typ_norm,
            "manifold_fit": {
                "r2_smooth": man.r2, "r2_linear": man.r2_linear,
                "curvature_gain_r2": man.r2 - man.r2_linear,
                "tangent_cos_lo_hi": man.tangent_cos_lo_hi,
                "tangent_abs_cos_with_axis_mid": man.tangent_axis_cos,
                "latent_range": [man.t_lo, man.t_hi],
            },
            "linear_vs_tangent_greedy": [],
            "linear_vs_tangent_sample": [],
            "valence_by_alpha": [],
        }

        # geometric divergence + generations across the fine grid (greedy)
        print("[greedy] LINEAR vs TANGENT + divergence", flush=True)
        tan_axis_cos_by_alpha = []
        for alpha in greedy_alphas:
            lin, _ = _steer_gen_capture(_generate, model, tok, args.device, stems, layer_mod,
                                        "linear", man, axis_unit, alpha, typ_norm, D, torch,
                                        dtype, False, args.max_new_tokens)
            tan, cos_log = _steer_gen_capture(_generate, model, tok, args.device, stems, layer_mod,
                                              "tangent", man, axis_unit, alpha, typ_norm, D, torch,
                                              dtype, False, args.max_new_tokens)
            mean_cos = float(np.mean(cos_log)) if cos_log else float("nan")
            tan_axis_cos_by_alpha.append(mean_cos)
            lr["linear_vs_tangent_greedy"].append({
                "alpha": float(alpha),
                "tangent_axis_cos_live_mean": mean_cos,
                "divergence_1_minus_cos": (None if not np.isfinite(mean_cos) else 1.0 - mean_cos),
                "linear": [{"stem": g["stem"], "gen": g["gen"]} for g in lin],
                "tangent": [{"stem": g["stem"], "gen": g["gen"]} for g in tan]})
            print(f"  a={alpha:+d} tan.axis(live)={mean_cos:+.3f}\n"
                  f"    LIN: {lin[0]['gen'][:74]!r}\n    TAN: {tan[0]['gen'][:74]!r}", flush=True)

        # valence (teacher-forced experiencer gap) for linear vs tangent across alpha,
        # incl. the +/-64 opposite-valence check.
        print("[valence] linear vs tangent experiencer gap", flush=True)
        axis_u = _unit(axis_unit)
        for alpha in [-128, -64, -32, 0, 32, 64, 128]:
            scale = (alpha / max(1.0, np.sqrt(D))) * typ_norm
            if alpha == 0.0:
                lin_gap = _valence_gap(model, tok, args.device, val_stems, layer_mod, None,
                                       _score_continuations, EXP_CONTS, NOEXP_CONTS)
                tan_gap = lin_gap
            else:
                lin_add = torch.tensor(scale * axis_u, dtype=dtype, device=args.device)
                lin_gap = _valence_gap(model, tok, args.device, val_stems, layer_mod,
                                       (lambda hid, a=lin_add: hid + a.to(hid.dtype)),
                                       _score_continuations, EXP_CONTS, NOEXP_CONTS)
                # tangent: position-dependent. Use a hook that recomputes per-token.
                def tan_add_fn(hid, sc=scale):
                    h_vec = hid[0, -1, :].to(torch.float64).detach().cpu().numpy()
                    tang = man.tangent_at(h_vec)
                    return hid + torch.tensor(sc * tang, dtype=hid.dtype, device=hid.device)
                tan_gap = _valence_gap(model, tok, args.device, val_stems, layer_mod,
                                       tan_add_fn, _score_continuations, EXP_CONTS, NOEXP_CONTS)
            lr["valence_by_alpha"].append({
                "alpha": float(alpha),
                "linear_valence_gap": lin_gap,
                "tangent_valence_gap": tan_gap,
                "sign_flip_tan_vs_lin": bool(np.isfinite(lin_gap) and np.isfinite(tan_gap)
                                             and (lin_gap * tan_gap < 0))})
            print(f"  a={alpha:+d} LIN_val={lin_gap:+.3f} TAN_val={tan_gap:+.3f} "
                  f"flip={lr['valence_by_alpha'][-1]['sign_flip_tan_vs_lin']}", flush=True)

        # sampled generations at the extremes
        print("[sample] +/-64,128", flush=True)
        for alpha in sample_alphas:
            torch.manual_seed(args.seed)
            lin, _ = _steer_gen_capture(_generate, model, tok, args.device, stems, layer_mod,
                                        "linear", man, axis_unit, alpha, typ_norm, D, torch,
                                        dtype, True, args.max_new_tokens)
            torch.manual_seed(args.seed)
            tan, _ = _steer_gen_capture(_generate, model, tok, args.device, stems, layer_mod,
                                        "tangent", man, axis_unit, alpha, typ_norm, D, torch,
                                        dtype, True, args.max_new_tokens)
            lr["linear_vs_tangent_sample"].append({
                "alpha": float(alpha),
                "linear": [{"stem": g["stem"], "gen": g["gen"]} for g in lin],
                "tangent": [{"stem": g["stem"], "gen": g["gen"]} for g in tan]})
            print(f"  a={alpha:+d}(s)\n    LIN: {lin[0]['gen'][:74]!r}\n    TAN: {tan[0]['gen'][:74]!r}", flush=True)

        # per-layer divergence summary at the |alpha|=64 working point
        finite = [c for c in tan_axis_cos_by_alpha if np.isfinite(c)]
        mean_div = float(1.0 - np.mean(finite)) if finite else float("nan")
        v64 = next((v for v in lr["valence_by_alpha"] if v["alpha"] == 64.0), None)
        summary_rows.append({
            "layer": int(layer),
            "curvature_gain_r2": man.r2 - man.r2_linear,
            "tangent_cos_lo_hi": man.tangent_cos_lo_hi,
            "mean_tangent_vs_linear_divergence": mean_div,
            "plus64_linear_valence": (v64["linear_valence_gap"] if v64 else None),
            "plus64_tangent_valence": (v64["tangent_valence_gap"] if v64 else None),
            "plus64_sign_flip": (v64["sign_flip_tan_vs_lin"] if v64 else None),
        })
        result["per_layer"].append(lr)

    result["cross_layer_summary"] = {
        "rows": summary_rows,
        "interpretation": "mean_tangent_vs_linear_divergence (=1-cos averaged over alpha) "
                          "should TRACK curvature_gain_r2: curved layers (L18, L32) steer "
                          "away from the global linear axis, the ~linear layer (L25) does not. "
                          "plus64_sign_flip True at L32 confirms the '+64 opposite valence' finding.",
    }

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
    print("[summary]", json.dumps(_finite(summary_rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
