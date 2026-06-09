"""Fitted-manifold-tangent self/qualia (and color) steering, via gamfit.

This is the CORRECTED "manifold steering": not a linear PCA direction, but the
TANGENT of the actual curved manifold the residual stream traces out as a smooth
function of the experience/qualia latent coordinate, fit with gamfit.

Pipeline (qualia):
  1. LATENT COORD t. For every harvested activation H_i at the steer layer,
     t_i = H_i . axis_unit, where axis_unit is the mean exp-noexp qualia axis.
     t is the position along the experience<->no-experience continuum.
  2. FIT THE MANIFOLD with gamfit. Reduce the (centered) activations to the top
     rep-PCs (denoise + tractable), and for the joint multi-output rep-PC target
     Z fit a single penalized B-spline smooth  Z ~ g(t)  with gamfit's REML
     choosing the smoothing parameter (gamfit owns knots + 2nd-order penalty +
     lambda). g(t) is allowed CURVATURE -- that is the whole point; the manifold
     is not the linear axis. rep(t) = mean + g(t) @ Vt   (back in hidden dim).
  3. MANIFOLD TANGENT. The steering direction at a point x is dg/dt evaluated at
     that point's latent t0, mapped to hidden dim:  tangent(x) = (dB/dt(t0) @ B_coef) @ Vt.
     gamfit.torch.bspline_basis_derivative gives dB/dt on the SAME knots, so the
     tangent is the analytic derivative of the fitted curve. It VARIES with t0
     (position-dependent) -- unlike the single global linear axis.
  4. STEER. Add alpha-scaled unit tangent (computed at the CURRENT residual's t0
     every forward step) to the residual at the steer layer, and compare to plain
     LINEAR-axis steering (global constant direction) at matched alpha.

Scaling matches run_steer_cloze:  add = (alpha/sqrt(D)) * typ_norm * unit_dir.

Color (optional): fit the hue circle rep ~ g(theta) with a PERIODIC spline and
steer along dg/dtheta (rotate around the hue circle).

Reuses load_model, compute_qualia_axis, _generate, _layer_module, CLOZE_GROUPS,
and the gamfit wrappers (bspline_1d_basis / reml_fit) from color_manifold_gam.
Writes runs/ANALYSIS/manifold_tangent_steer.json.
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
    # color_manifold_gam imports `_pca_basis` as a top-level module (expects the
    # experiments/ dir on sys.path); ensure that holds when we run as a package.
    _exp_dir = os.path.dirname(os.path.abspath(__file__))
    if _exp_dir not in sys.path:
        sys.path.insert(0, _exp_dir)
    from experiments.self_qualia_olmo import load_model
    from experiments.self_qualia_steer_cloze import (
        CLOZE_GROUPS,
        _generate,
        _layer_module,
        compute_qualia_axis,
    )
    # gamfit wrappers already worked out in the color manifold experiment.
    from experiments.color_manifold_gam import bspline_1d_basis, reml_fit
    from _pca_basis import fit_top_pcs
    return (load_model, CLOZE_GROUPS, _generate, _layer_module, compute_qualia_axis,
            bspline_1d_basis, reml_fit, fit_top_pcs)


def _pair_records(records):
    pid: dict[Any, dict[str, list[int]]] = {}
    for i, r in enumerate(records):
        if r.get("role") != "pair":
            continue
        pid.setdefault(r["pair_id"], {"exp": [], "noexp": []})[r["side"]].append(i)
    return pid


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# --- the fitted manifold object: g(t) and dg/dt in hidden dim ----------------
class QualiaManifold:
    """rep(t) ~ mean + g(t) @ Vt, g a gamfit B-spline smooth of the latent t."""

    def __init__(self, X, records, layer, axis_unit, k_pc, n_basis, degree,
                 bspline_1d_basis, reml_fit, fit_top_pcs):
        import torch
        import gamfit.torch as gt
        from gamfit.torch._basis import _resolve_knots_tensor

        H = X[:, layer, :].astype(np.float64)          # (N, D)
        self.D = H.shape[1]
        self.mean = H.mean(0)
        Hc = H - self.mean
        # rep-PCs as the regression-target basis (denoise + tractable joint fit).
        _, Vt = fit_top_pcs(Hc, d=min(k_pc, *Hc.shape), standardize=False)
        self.Vt = Vt                                    # (k, D), unit rows
        Z = Hc @ Vt.T                                   # (N, k) target
        # latent coordinate = projection on the qualia axis (in raw units).
        self.axis_unit = axis_unit
        t = H @ axis_unit                               # (N,)
        self.t_mean, self.t_std = float(t.mean()), float(t.std() + 1e-9)
        self.t_lo, self.t_hi = float(t.min()), float(t.max())
        tn = (t - self.t_mean) / self.t_std            # standardized for the spline
        self.tn = tn
        self.degree = degree
        # gamfit owns the knots: resolve ONE knot vector and reuse it for the
        # basis, the smoothness penalty, AND the derivative basis, so dB/dt is the
        # exact analytic derivative of the fitted curve g(t).
        tn_t = torch.from_numpy(np.ascontiguousarray(tn))
        knots, eff_deg = _resolve_knots_tensor(tn_t, None, degree=degree)
        self.knots = knots
        self.degree = int(eff_deg)
        with torch.no_grad():
            B = gt.bspline_basis(tn_t, knots, degree=self.degree, periodic=False).numpy()
            P, _null = gt.smoothness_penalty(knots, degree=self.degree, order=2)
            P = P.numpy()
        self.B_coef, _ = reml_fit(B, Z, P)             # (K, k): coefficients per rep-PC
        # fit diagnostics
        fit = B @ self.B_coef
        ss_res = float(((Z - fit) ** 2).sum())
        ss_tot = float(((Z - Z.mean(0)) ** 2).sum())
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # curvature check: how non-linear is g(t)? compare to best linear fit.
        A = np.c_[np.ones_like(tn), tn]
        lin = A @ np.linalg.lstsq(A, Z, rcond=None)[0]
        ss_lin = float(((Z - lin) ** 2).sum())
        self.r2_linear = 1.0 - ss_lin / ss_tot if ss_tot > 0 else float("nan")
        # tangent variability: cos between tangent at lo vs hi quantiles of t.
        tlo = np.quantile(tn, 0.1)
        thi = np.quantile(tn, 0.9)
        glo = _unit(self.tangent_from_tn(tlo))
        ghi = _unit(self.tangent_from_tn(thi))
        self.tangent_cos_lo_hi = float(np.dot(glo, ghi))
        self.tangent_axis_cos = float(abs(np.dot(_unit(self.tangent_from_tn(0.0)), axis_unit)))

    def tangent_from_tn(self, tn_val):
        """dg/dt at standardized latent tn_val -> hidden-dim direction (raw)."""
        import torch
        import gamfit.torch as gt
        tt = torch.tensor([float(tn_val)], dtype=torch.float64)
        dB = gt.bspline_basis_derivative(tt, self.knots, degree=self.degree,
                                         order=1, periodic=False).detach().cpu().numpy()  # (1,K)
        dZ = dB @ self.B_coef                  # (1, k) in rep-PC space
        # chain rule: d/d(t_raw) = (1/t_std) * d/d(tn); constant scale -> drop after unit-norm.
        return (dZ @ self.Vt)[0]               # (D,)

    def latent_of(self, h):
        """Standardized latent t of a raw hidden vector h."""
        return (float(h @ self.axis_unit) - self.t_mean) / self.t_std

    def tangent_at(self, h):
        """Unit manifold tangent at hidden vector h (position-dependent)."""
        return _unit(self.tangent_from_tn(self.latent_of(h)))


def _steer_gen(_generate, model, tok, device, stems, layer_mod, mode, manifold,
               axis_unit, alpha, typ_norm, D, torch, dtype, do_sample, max_new_tokens):
    """Generate self stems under steering.

    mode='linear' : add constant alpha-scaled axis (baseline).
    mode='tangent': add alpha-scaled manifold tangent recomputed at the CURRENT
                    residual's latent every step (position-dependent direction).
    mode='none'   : unsteered.
    """
    scale = (alpha / max(1.0, np.sqrt(D))) * typ_norm
    if mode == "none" or alpha == 0.0:
        return _generate(model, tok, device, stems,
                         max_new_tokens=max_new_tokens, do_sample=do_sample)
    if mode == "linear":
        add = torch.tensor(scale * axis_unit, dtype=dtype, device=device)
        return _generate(model, tok, device, stems, layer_mod=layer_mod,
                         add_vec=add, max_new_tokens=max_new_tokens, do_sample=do_sample)

    # tangent mode: custom hook that recomputes the tangent from the live residual.
    axis_t = torch.tensor(axis_unit, dtype=torch.float64, device=device)

    def hook(_m, _inp, output):
        is_tuple = isinstance(output, tuple)
        hid = output[0] if is_tuple else output                   # (B,T,D)
        # steer using the latent of the LAST position (the one being generated).
        h_vec = hid[0, -1, :].to(torch.float64).detach().cpu().numpy()
        tang = manifold.tangent_at(h_vec)                         # unit (D,)
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
        return outs
    finally:
        handle.remove()


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/OLMO3_32B_TRAJ_RL31/step_2300")
    ap.add_argument("--model", default="allenai/Olmo-3.1-32B-Think")
    ap.add_argument("--revision", default="step_2300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steer-layer", type=int, default=None)
    ap.add_argument("--layers", type=str, default=None,
                    help="comma list of layers to run (one model load). "
                         "default: the standard steer layer 25 + the most-curved layer 32")
    ap.add_argument("--k-pc", type=int, default=24, help="rep-PCs as fit targets")
    ap.add_argument("--n-basis", type=int, default=10)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="runs/ANALYSIS/manifold_tangent_steer.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (load_model, CLOZE_GROUPS, _generate, _layer_module, compute_qualia_axis,
     bspline_1d_basis, reml_fit, fit_top_pcs) = _imports()

    run_dir = Path(args.run_dir)
    X = np.load(run_dir / "activations.npy")
    records = [json.loads(l) for l in open(run_dir / "prompts.jsonl") if l.strip()]
    D = X.shape[2]
    stems = CLOZE_GROUPS["self"][:2] + CLOZE_GROUPS["self_1p"][:1]

    # Which layers to compare. Default: the STANDARD steer layer 25 (where the
    # manifold turns out ~linear) AND layer 32 (the most-curved qualia layer), so
    # the tangent-vs-linear contrast is shown both where it should and shouldn't
    # matter. (Curvature scan: layers 18/32 curve; 12/25/40/50 ~linear.)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    elif args.steer_layer is not None:
        layers = [args.steer_layer]
    else:
        layers = [25, 32]

    # fit every layer's manifold FIRST (cheap, CPU) so we can report curvature
    # even if the GPU step is interrupted.
    fits = {}
    for layer in layers:
        axis_unit = compute_qualia_axis(X, records, layer)
        man = QualiaManifold(X, records, layer, axis_unit, args.k_pc,
                             args.n_basis, args.degree, bspline_1d_basis, reml_fit, fit_top_pcs)
        typ_norm = float(np.median(np.linalg.norm(X[:, layer, :], axis=1)))
        fits[layer] = (axis_unit, man, typ_norm)
        print(f"[fit] L{layer} R2(g)={man.r2:.4f} R2(lin)={man.r2_linear:.4f} "
              f"curv_gain={man.r2 - man.r2_linear:+.4f} tan_cos(lo,hi)={man.tangent_cos_lo_hi:+.3f} "
              f"|tan.axis|@mid={man.tangent_axis_cos:.3f}", flush=True)

    model, tok, n_layers = load_model(args.model, args.revision, args.dtype, args.device)
    dtype = next(model.parameters()).dtype

    top_result: dict[str, Any] = {
        "model": args.model, "revision": args.revision, "D": int(D), "stems": stems,
        "scaling": "add = (alpha/sqrt(D)) * typ_norm * unit_dir (D=5120)",
        "note": "LINEAR = global constant qualia axis; TANGENT = dg/dt of fitted curved "
                "manifold recomputed at the live residual latent every step. "
                "tangent_cos_lo_hi<1 means the manifold tangent ROTATES with position "
                "(genuinely curved); =1 means the fitted manifold is ~linear there so "
                "TANGENT should coincide with LINEAR.",
        "per_layer": [],
    }

    greedy_alphas = [-128, -64, -32, 0, 32, 64, 128]
    sample_alphas = [-128, -64, 64, 128]

    for layer in layers:
        axis_unit, man, typ_norm = fits[layer]
        layer_mod = _layer_module(model, layer)
        print(f"\n===== LAYER {layer} (typ_norm={typ_norm:.2f}) =====", flush=True)

        lr: dict[str, Any] = {
            "steer_layer": int(layer), "typ_resid_norm": typ_norm,
            "manifold_fit": {
                "method": "gamfit B-spline smooth Z~g(t), REML lambda; Z=top rep-PCs",
                "k_pc": int(args.k_pc), "n_basis": int(args.n_basis), "degree": int(args.degree),
                "r2_smooth": man.r2, "r2_linear": man.r2_linear,
                "curvature_gain_r2": man.r2 - man.r2_linear,
                "tangent_cos_lo_hi": man.tangent_cos_lo_hi,
                "tangent_abs_cos_with_axis_mid": man.tangent_axis_cos,
                "latent_range": [man.t_lo, man.t_hi],
            },
            "linear_vs_tangent_greedy": [],
            "linear_vs_tangent_sample": [],
        }

        print("[cmp] LINEAR vs TANGENT (greedy)", flush=True)
        for alpha in greedy_alphas:
            lin = _steer_gen(_generate, model, tok, args.device, stems, layer_mod, "linear",
                             man, axis_unit, alpha, typ_norm, D, torch, dtype, False, args.max_new_tokens)
            tan = _steer_gen(_generate, model, tok, args.device, stems, layer_mod, "tangent",
                             man, axis_unit, alpha, typ_norm, D, torch, dtype, False, args.max_new_tokens)
            lr["linear_vs_tangent_greedy"].append({
                "alpha": float(alpha),
                "linear": [{"stem": g["stem"], "gen": g["gen"]} for g in lin],
                "tangent": [{"stem": g["stem"], "gen": g["gen"]} for g in tan]})
            print(f"  a={alpha:+d}\n    LIN: {lin[0]['gen'][:78]!r}\n    TAN: {tan[0]['gen'][:78]!r}", flush=True)

        print("[cmp] sampled at +/-64,128", flush=True)
        for alpha in sample_alphas:
            torch.manual_seed(args.seed)
            lin = _steer_gen(_generate, model, tok, args.device, stems, layer_mod, "linear",
                             man, axis_unit, alpha, typ_norm, D, torch, dtype, True, args.max_new_tokens)
            torch.manual_seed(args.seed)
            tan = _steer_gen(_generate, model, tok, args.device, stems, layer_mod, "tangent",
                             man, axis_unit, alpha, typ_norm, D, torch, dtype, True, args.max_new_tokens)
            lr["linear_vs_tangent_sample"].append({
                "alpha": float(alpha),
                "linear": [{"stem": g["stem"], "gen": g["gen"]} for g in lin],
                "tangent": [{"stem": g["stem"], "gen": g["gen"]} for g in tan]})
            print(f"  a={alpha:+d}(s)\n    LIN: {lin[0]['gen'][:78]!r}\n    TAN: {tan[0]['gen'][:78]!r}", flush=True)

        top_result["per_layer"].append(lr)

    result = top_result

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
