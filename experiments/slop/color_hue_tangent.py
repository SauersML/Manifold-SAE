"""COLOR hue-circle PERIODIC manifold-tangent steering (S^1), via gamfit.

Companion to demo_manifold_tangent_steer.py (which does the SELF/QUALIA manifold).
Here the latent coordinate is HUE theta in [0,1) (a circle), and we fit the color
representation as a PERIODIC smooth curve of hue:

    rep(theta) ~ mean + g(theta) @ Vt,   g a CYCLIC (bs='cc') B-spline of theta,
    REML choosing the smoothing lambda (cyclic 2nd-difference penalty).

The manifold tangent at hue theta0 is dg/dtheta(theta0) mapped to hidden dim.
Because the basis is periodic, the tangent rotates smoothly around the whole hue
wheel with no seam at red (theta=0/1).

TWO instruments:

(a) REP-SPACE demonstration. Start from a given color's fitted rep g(theta_c) and
    walk +eps along the unit tangent dg/dtheta. Show the nearest *color name* (by
    cosine to the per-color rep bank) rotates around the hue wheel
    red->orange->yellow->green->...  Quantify: does +dg/dtheta move the rep toward
    the next-HIGHER-hue color (and -dg/dtheta toward the next-lower)?  We report,
    per anchor color, the signed hue-rank shift of the nearest neighbour after a
    +/- tangent step, and the fraction of anchors that advance correctly.

(b) GENERATIVE / cloze effect. On a color cloze prompt
    ("The color most similar to red is ___"), add alpha * (hue tangent computed at
    the live residual's hue-latent) at a layer and read whether the predicted
    color WORD rotates around the hue circle. Reported with top-k logits + greedy
    generations across an alpha sweep.

Reuses the color harvest (extra/activations.npy + color_probes.jsonl with per-row
true rgb/hex), load_model, _layer_module, _generate, and the gamfit cyclic-spline
wrapper bspline_1d_cyclic_basis + reml_fit from color_manifold_gam.

Writes runs/ANALYSIS/color_hue_tangent.json.
"""

from __future__ import annotations

import argparse
import colorsys
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
    from experiments.self_qualia_olmo import load_model, harvest
    from experiments.self_qualia_steer_cloze import _generate, _layer_module
    from experiments.color_manifold_gam import bspline_1d_cyclic_basis, reml_fit
    from _pca_basis import fit_top_pcs
    return load_model, harvest, _generate, _layer_module, bspline_1d_cyclic_basis, reml_fit, fit_top_pcs


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _hue_of_rgb(rgb255) -> float:
    r, g, b = (np.asarray(rgb255, dtype=np.float64) / 255.0).tolist()
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return float(h)  # in [0,1)


def _circ_diff(a, b):
    """Signed minimal circular difference a-b on [0,1) in (-0.5, 0.5]."""
    d = (a - b) % 1.0
    return d - 1.0 if d > 0.5 else d


def load_color_bank(run_dir: Path, layer: int, fit_top_pcs, k_pc: int):
    """Frame-demeaned per-color rep bank at `layer`, plus hue and rgb.

    Returns dict with: names (hue-sorted), V (n,D raw frame-demeaned color reps),
    hue (n,), rgb (n,3), and the PCA Vt + mean for the manifold target space.
    """
    X = np.load(run_dir / "activations.npy")
    recs = [json.loads(l) for l in open(run_dir / "color_probes.jsonl") if l.strip()]
    H = X[:, layer, :].astype(np.float64)
    by, frames, rgb = {}, {}, {}
    for i, r in enumerate(recs):
        by.setdefault(r["color"], []).append(i)
        frames.setdefault(r["frame"], []).append(i)
        rgb[r["color"]] = np.asarray(r["rgb"], dtype=np.float64)
    # frame-demean: remove the per-frame (template) nuisance direction.
    Hd = H.copy()
    for f, idxs in frames.items():
        Hd[idxs] -= H[idxs].mean(0)
    names = list(by)
    V = np.stack([Hd[by[c]].mean(0) for c in names])             # (n, D)
    hue = np.array([_hue_of_rgb(rgb[c]) for c in names])         # (n,)
    rgb_arr = np.stack([rgb[c] for c in names])                  # (n, 3)
    # achromatic colors (black/white/grey/brown muddy) have ill-defined hue;
    # keep only saturated colors for the hue-circle fit.
    sat = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255.0))[1] for c in names])
    val = np.array([colorsys.rgb_to_hsv(*(rgb[c] / 255.0))[2] for c in names])
    chromatic = (sat > 0.25) & (val > 0.15)
    return {
        "names": names, "V": V, "hue": hue, "rgb": rgb_arr,
        "sat": sat, "val": val, "chromatic": chromatic,
    }


class HueManifold:
    """rep(theta) ~ mean + g(theta) @ Vt; g a CYCLIC B-spline smooth of hue theta."""

    def __init__(self, V_fit, hue_fit, n_basis, degree,
                 bspline_1d_cyclic_basis, reml_fit, fit_top_pcs, k_pc):
        self.mean = V_fit.mean(0)
        Vc = V_fit - self.mean
        _, Vt = fit_top_pcs(Vc, d=min(k_pc, *Vc.shape), standardize=False)
        self.Vt = Vt                                       # (k, D)
        Z = Vc @ Vt.T                                      # (n, k) target
        self.hue = hue_fit
        self.n_basis = n_basis
        self.degree = degree
        self.bspline = bspline_1d_cyclic_basis
        # cyclic basis + cyclic 2nd-diff penalty (mgcv bs='cc' style). Resolve the
        # knots ONCE on the fit data and reuse them for all later (single-point /
        # derivative) evaluations so the basis columns always line up.
        B, P, self.knots, self.degree = bspline_1d_cyclic_basis(
            hue_fit, n_basis=n_basis, degree=degree, return_knots=True)
        self.B_coef, _ = reml_fit(B, Z, P)                 # (K, k)
        fit = B @ self.B_coef
        ss_res = float(((Z - fit) ** 2).sum())
        ss_tot = float(((Z - Z.mean(0)) ** 2).sum())
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # linear-in-(cos,sin) baseline (a plain circle, no curvature beyond S^1).
        A = np.c_[np.ones_like(hue_fit), np.cos(2 * np.pi * hue_fit), np.sin(2 * np.pi * hue_fit)]
        lin = A @ np.linalg.lstsq(A, Z, rcond=None)[0]
        ss_lin = float(((Z - lin) ** 2).sum())
        self.r2_circle = 1.0 - ss_lin / ss_tot if ss_tot > 0 else float("nan")

    def _basis_at(self, thetas):
        return self.bspline(np.asarray(thetas, dtype=np.float64),
                            n_basis=self.n_basis, degree=self.degree,
                            knots=self.knots)[0]

    def g(self, theta):
        """Fitted rep (hidden dim) at hue theta."""
        B = self._basis_at([theta])
        return self.mean + (B @ self.B_coef @ self.Vt)[0]

    def tangent(self, theta, eps=1e-3):
        """Unit dg/dtheta at hue theta (finite-diff on the cyclic basis -> hidden)."""
        ba = self._basis_at([(theta + eps) % 1.0])
        bb = self._basis_at([(theta - eps) % 1.0])
        dB = (ba - bb) / (2 * eps)
        return _unit((dB @ self.B_coef @ self.Vt)[0])


def nearest_color(vec, V_bank, names):
    """Nearest color NAME to a hidden vector by cosine (against frame-demeaned bank)."""
    vu = _unit(vec)
    Vn = V_bank / np.maximum(np.linalg.norm(V_bank, axis=1, keepdims=True), 1e-9)
    cos = Vn @ vu
    j = int(np.argmax(cos))
    return names[j], float(cos[j]), cos


# =============================================================================
# (a) REP-SPACE hue rotation
# =============================================================================
def rep_space_demo(man: HueManifold, bank, eps_steps):
    """For each chromatic anchor color, walk +/- along the hue tangent from the
    fitted point g(theta_c) and report the nearest color name + its hue-rank shift."""
    names = bank["names"]
    V_bank = bank["V"]
    hue = {n: float(h) for n, h in zip(names, bank["hue"])}
    chro = bank["chromatic"]
    chro_names = [n for n, c in zip(names, chro) if c]
    # hue rank order (the wheel) among chromatic colors
    hue_sorted = sorted(chro_names, key=lambda n: hue[n])
    rank = {n: i for i, n in enumerate(hue_sorted)}
    n_chro = len(hue_sorted)

    anchors = []
    correct_plus = 0
    correct_minus = 0
    total = 0
    for n in hue_sorted:
        th = hue[n]
        base_vec = man.g(th)
        base_name, base_cos, _ = nearest_color(base_vec, V_bank, names)
        tang = man.tangent(th)
        row = {"anchor": n, "hue": round(th, 4),
               "fitted_nearest": base_name, "fitted_cos": round(base_cos, 4),
               "walk_plus": [], "walk_minus": []}
        for eps in eps_steps:
            vp = base_vec + eps * tang
            vm = base_vec - eps * tang
            np_name, np_cos, _ = nearest_color(vp, V_bank, names)
            nm_name, nm_cos, _ = nearest_color(vm, V_bank, names)
            row["walk_plus"].append({"eps": eps, "nearest": np_name, "cos": round(np_cos, 4)})
            row["walk_minus"].append({"eps": eps, "nearest": nm_name, "cos": round(nm_cos, 4)})
        # quantify with the largest eps step: did +tangent advance hue rank forward?
        big = eps_steps[-1]
        vp = base_vec + big * tang
        vm = base_vec - big * tang
        np_name = nearest_color(vp, V_bank, names)[0]
        nm_name = nearest_color(vm, V_bank, names)[0]
        if np_name in rank and base_name in rank:
            d_plus = (rank[np_name] - rank[base_name])
            d_plus = d_plus - n_chro if d_plus > n_chro / 2 else (d_plus + n_chro if d_plus < -n_chro / 2 else d_plus)
            row["plus_rank_shift"] = int(d_plus)
            if d_plus > 0:
                correct_plus += 1
        if nm_name in rank and base_name in rank:
            d_minus = (rank[nm_name] - rank[base_name])
            d_minus = d_minus - n_chro if d_minus > n_chro / 2 else (d_minus + n_chro if d_minus < -n_chro / 2 else d_minus)
            row["minus_rank_shift"] = int(d_minus)
            if d_minus < 0:
                correct_minus += 1
        total += 1
        anchors.append(row)
    return {
        "hue_wheel_order": hue_sorted,
        "anchors": anchors,
        "frac_plus_advances_hue": round(correct_plus / max(1, total), 3),
        "frac_minus_retreats_hue": round(correct_minus / max(1, total), 3),
        "n_anchors": total,
    }


# =============================================================================
# (b) GENERATIVE cloze hue rotation
# =============================================================================
COLOR_CLOZE = [
    "The color most similar to red is",
    "Looking at the swatch, the closest color name is",
    "On the color wheel, just past red comes",
    "If you mix a little into red, the hue shifts toward",
]

# candidate color words to track in the logits (the hue wheel).
TRACK_WORDS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple",
               "violet", "magenta", "pink"]


def _track_logits(model, tok, device, stems, layer_mod, add_fn, track_words, k=20):
    """At each stem's answer position, return top-k tokens + tracked-word logprobs.
    add_fn(hid)->hid installs the steering (or None for unsteered)."""
    import torch
    handle = None
    if add_fn is not None:
        def hook(_m, _inp, output):
            is_t = isinstance(output, tuple)
            hid = output[0] if is_t else output
            hid = add_fn(hid)
            return (hid,) + tuple(output[1:]) if is_t else hid
        handle = layer_mod.register_forward_hook(hook)
    out = []
    try:
        for s in stems:
            enc = tok(s, return_tensors="pt", add_special_tokens=True).to(device)
            with torch.inference_mode():
                logits = model(**enc).logits[0, -1]
            lp = torch.log_softmax(logits.float(), dim=-1)
            vals, idx = lp.topk(k)
            topk = [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(vals, idx)]
            tracked = {}
            for w in track_words:
                ids = tok(" " + w, add_special_tokens=False)["input_ids"]
                if ids:
                    tracked[w] = round(float(lp[ids[0]]), 4)
            out.append({"stem": s, "topk": topk, "tracked": tracked})
    finally:
        if handle is not None:
            handle.remove()
    return out


def generative_demo(model, tok, device, man, bank, layer, layer_mod, _generate,
                    alphas, anchor_hue, typ_norm, D, max_new_tokens):
    """Add alpha*hue-tangent at `layer` on color cloze prompts; track word rotation."""
    import torch
    dtype = next(model.parameters()).dtype
    scale = lambda a: (a / max(1.0, np.sqrt(D))) * typ_norm

    # Position-dependent tangent: recompute hue-latent of the live residual by
    # projecting onto the fitted color circle (nearest theta on a fine grid in
    # PCA target space), then take dg/dtheta there. Simpler & robust: use a FIXED
    # tangent at the anchor hue (the prompt is anchored at "red"), which is the
    # cleanest causal test of "rotate from red".
    tang_fixed = man.tangent(anchor_hue)                        # unit (D,)
    tang_t = torch.tensor(tang_fixed, dtype=dtype, device=device)

    rows = []
    for a in alphas:
        if a == 0.0:
            add_fn = None
        else:
            add = scale(a) * tang_t
            add_fn = lambda hid, add=add: hid + add.to(hid.dtype)
        logit_rows = _track_logits(model, tok, device, COLOR_CLOZE, layer_mod,
                                   add_fn, TRACK_WORDS)
        # greedy generations under the same steering
        if a == 0.0:
            gens = _generate(model, tok, device, COLOR_CLOZE,
                             max_new_tokens=max_new_tokens, do_sample=False)
        else:
            add_vec = scale(a) * tang_t
            gens = _generate(model, tok, device, COLOR_CLOZE, layer_mod=layer_mod,
                             add_vec=add_vec, max_new_tokens=max_new_tokens, do_sample=False)
        rows.append({"alpha": float(a),
                     "logits": logit_rows,
                     "generations": [{"stem": g["stem"], "gen": g["gen"]} for g in gens]})
        ex = logit_rows[0]["tracked"]
        best = max(ex, key=ex.get)
        print(f"[gen] a={a:+.0f} '{COLOR_CLOZE[0]}' -> argmax-tracked={best} "
              f"gen={gens[0]['gen'][:50]!r}", flush=True)
    return {"anchor_hue": round(float(anchor_hue), 4),
            "track_words": TRACK_WORDS, "cloze_prompts": COLOR_CLOZE,
            "alpha_sweep": rows,
            "tangent_note": "FIXED tangent dg/dtheta at the anchor hue (red); "
                            "+alpha should advance the predicted color word to higher hue "
                            "(red->orange->yellow...), -alpha to lower (red->pink->purple)."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/OLMO3_32B_TRAJ_RL31/step_2300")
    ap.add_argument("--color-subdir", default="color_extra",
                    help="subdir under run-dir holding the color harvest "
                         "(activations.npy + color_probes.jsonl)")
    ap.add_argument("--model", default="allenai/Olmo-3.1-32B-Think")
    ap.add_argument("--revision", default="step_2300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=str, default="25",
                    help="comma list of layers for the rep-space fit (default 25)")
    ap.add_argument("--gen-layer", type=int, default=25,
                    help="layer to inject the hue tangent for the generative cloze")
    ap.add_argument("--k-pc", type=int, default=16)
    ap.add_argument("--n-basis", type=int, default=10)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--harvest", action="store_true",
                    help="harvest color probes first (loads model, forward only)")
    ap.add_argument("--no-gen", action="store_true",
                    help="skip the GPU generative cloze (rep-space only)")
    ap.add_argument("--out", default="runs/ANALYSIS/color_hue_tangent.json")
    args = ap.parse_args()

    (load_model, harvest, _generate, _layer_module,
     bspline_1d_cyclic_basis, reml_fit, fit_top_pcs) = _imports()

    run_dir = Path(args.run_dir)
    color_dir = run_dir / args.color_subdir
    probes = Path(__file__).resolve().parent / "color_probes.jsonl"
    layers = [int(x) for x in args.layers.split(",")]

    model = tok = None
    if args.harvest or not (color_dir / "activations.npy").exists():
        color_dir.mkdir(parents=True, exist_ok=True)
        prompts = [json.loads(l)["prompt"] for l in open(probes) if l.strip()]
        print(f"[harvest] {len(prompts)} color prompts -> {color_dir}", flush=True)
        model, tok, _ = load_model(args.model, args.revision, args.dtype, args.device)
        harvest(model_name=args.model, revision=args.revision, prompts=prompts,
                out_dir=color_dir, batch_size=16, dtype=args.dtype, device=args.device,
                pooling="last_token", model=model, tokenizer=tok)
        # copy the probe metadata next to the activations (analysis reads it there)
        import shutil
        shutil.copy(probes, color_dir / "color_probes.jsonl")
        print("[harvest] done", flush=True)

    result: dict[str, Any] = {
        "model": args.model, "revision": args.revision,
        "color_dir": str(color_dir), "layers": layers, "gen_layer": args.gen_layer,
        "method": "PERIODIC (cyclic bs='cc') B-spline rep~g(hue), REML lambda; "
                  "tangent = dg/dhue mapped to hidden dim.",
        "per_layer": [],
    }

    fits = {}
    for layer in layers:
        bank = load_color_bank(color_dir, layer, fit_top_pcs, args.k_pc)
        sel = bank["chromatic"]
        man = HueManifold(bank["V"][sel], bank["hue"][sel], args.n_basis, args.degree,
                          bspline_1d_cyclic_basis, reml_fit, fit_top_pcs, args.k_pc)
        rep = rep_space_demo(man, bank, eps_steps=[1.0, 2.0, 4.0, 8.0])
        fits[layer] = (bank, man)
        result["per_layer"].append({
            "layer": int(layer),
            "manifold_fit": {"r2_cyclic_spline": round(man.r2, 4),
                             "r2_plain_circle": round(man.r2_circle, 4),
                             "curvature_gain": round(man.r2 - man.r2_circle, 4),
                             "n_chromatic": int(sel.sum()),
                             "k_pc": args.k_pc, "n_basis": args.n_basis},
            "rep_space": rep,
        })
        print(f"[fit] L{layer} r2(cc)={man.r2:.4f} r2(circle)={man.r2_circle:.4f} "
              f"frac+advance={rep['frac_plus_advances_hue']} "
              f"frac-retreat={rep['frac_minus_retreats_hue']}", flush=True)

    if not args.no_gen:
        if model is None:
            model, tok, _ = load_model(args.model, args.revision, args.dtype, args.device)
        gen_layer = args.gen_layer
        bank, man = fits.get(gen_layer, fits[layers[0]])
        if gen_layer not in fits:
            bank = load_color_bank(color_dir, gen_layer, fit_top_pcs, args.k_pc)
            sel = bank["chromatic"]
            man = HueManifold(bank["V"][sel], bank["hue"][sel], args.n_basis, args.degree,
                              bspline_1d_cyclic_basis, reml_fit, fit_top_pcs, args.k_pc)
        # anchor hue = "red"
        anchor_hue = float([h for n, h in zip(bank["names"], bank["hue"]) if n == "red"][0])
        X = np.load(color_dir / "activations.npy")
        typ_norm = float(np.median(np.linalg.norm(X[:, gen_layer, :], axis=1)))
        layer_mod = _layer_module(model, gen_layer)
        D = X.shape[2]
        gen = generative_demo(model, tok, args.device, man, bank, gen_layer, layer_mod,
                              _generate, alphas=[-128, -64, -32, 0, 32, 64, 128],
                              anchor_hue=anchor_hue, typ_norm=typ_norm, D=D,
                              max_new_tokens=args.max_new_tokens)
        result["generative_cloze"] = {"layer": int(gen_layer), "typ_resid_norm": typ_norm, **gen}

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
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
