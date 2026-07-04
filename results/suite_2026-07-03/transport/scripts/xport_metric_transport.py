"""JVP metric-transport arm for XPORT.

Question: can the downstream output-Fisher metric harvested ONCE at layer L_m
be PROPAGATED to a distant layer L_c via the frozen model's Jacobian (a JVP),
instead of re-harvesting it there -- and does that propagated metric calibrate
dose predictions (predicted nats vs measured output KL) as well as the metric
harvested directly at L_c?

Geometry: the residual stream is sequential, so the only downstream path from
h_{L_c} to the logits runs through h_{L_m} (L_c < L_m). Hence exactly
    G_{L_c} = J_{L_c->L_m}^T  G_{L_m}  J_{L_c->L_m}
where J is the Jacobian of the intervening decoder block stack. We test this as
a *predictive* claim on the REAL model:

  predicted_nats (harvested) = 1/2 s^2 || U_{L_c}^T tau ||^2       (G_{L_c} direct)
  predicted_nats (propagated)= 1/2 s^2 || U_{L_m}^T (J tau) ||^2   (G_{L_m} pushed)
  measured KL                = sym-KL of patching  h_{L_c} += s tau  at the last token

for a unit on-chart tangent tau (the weekday circle's tangent at each token's
base angle) over a grid of signed dose magnitudes s. If the metric transports,
the propagated column tracks measured KL as tightly as the harvested column,
across a hop of several layers -- one harvest, reused everywhere.

Indexing convention (hidden_states):  h_L := output.hidden_states[L] = INPUT to
decoder layers[L] = OUTPUT of layers[L-1]. So the Fisher/patch hook for h_L is
layers[L-1], and the map h_{L_c} -> h_{L_m} is layers[L_c:L_m].

All estimation reuses dose_calibration_real.py primitives; the chart plane is
the top-2 PCA plane of the (template-demeaned) last-position activations, which
for the planar weekday circle equals the fitted circle plane (planarity ~0.99),
so any plane imperfection cancels between the two metric columns.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np


def build_prompts():
    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    TEMPLATES = [
        "Today is {w}", "The day after tomorrow is {w}",
        "My favorite day of the week is {w}", "The meeting is scheduled for {w}",
        "Yesterday was {w}", "We will travel on {w}", "The store is closed on {w}",
        "Her birthday falls on {w}", "The exam takes place on {w}", "It always rains on {w}",
    ]
    prompts, labels, tids = [], [], []
    for ti, t in enumerate(TEMPLATES):
        for wi, w in enumerate(WEEKDAYS):
            prompts.append(t.format(w=w)); labels.append(wi); tids.append(ti)
    return prompts, np.asarray(labels), np.asarray(tids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chart-layer", type=int, default=11, help="hidden_states index L_c")
    ap.add_argument("--metric-layer", type=int, default=18, help="hidden_states index L_m")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent / "Manifold-SAE" / "experiments"))
    sys.path.insert(0, "/projects/standard/hsiehph/sauer354/Manifold-SAE/experiments")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import dose_calibration_real as D

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    lm = D.LogitsLM(hf)
    layers = D.find_decoder_layers(hf)
    Lc, Lm = a.chart_layer, a.metric_layer
    # Fisher/patch hooks: h_L is the OUTPUT of layers[L-1].
    hook_c = D.resolve_hook_module(hf, Lc - 1)
    hook_m = D.resolve_hook_module(hf, Lm - 1)
    rotary = hf.model.rotary_emb

    prompts, labels, tids = build_prompts()
    n = len(prompts)

    # ---- pass 1: per-prompt Fisher factors at L_c and L_m + last-pos acts at L_c
    U_c, U_m, xc_last = [], [], []
    t0 = time.time()
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to(a.device)
        ac, uc = D.harvest_last_position_fisher(lm, hook_c, ids, a.rank, 4, 2, 8, a.device, a.dtype)
        _am, um = D.harvest_last_position_fisher(lm, hook_m, ids, a.rank, 4, 2, 8, a.device, a.dtype)
        U_c.append(np.asarray(uc)); U_m.append(np.asarray(um))
        xc_last.append(np.asarray(ac)[-1])
        if (i + 1) % 10 == 0:
            print(f"[fisher] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    xc_last = np.asarray(xc_last, dtype=np.float64)  # (n, p)

    # ---- chart plane: top-2 PCA of template-demeaned last-position acts at L_c
    xd = xc_last.copy()
    for t in np.unique(tids):
        m = tids == t
        xd[m] -= xd[m].mean(0, keepdims=True)
    center = xc_last.mean(0)
    _, sv, vt = np.linalg.svd(xd, full_matrices=False)
    P = vt[:2].T  # (p, 2) ambient plane
    planarity = float((sv[:2] ** 2).sum() / max((sv ** 2).sum(), 1e-300))
    proj = xd @ P                       # (n, 2)
    theta = np.arctan2(proj[:, 1], proj[:, 0])
    # unit on-chart tangent per token: d/dtheta of P[cos,sin] = P[-sin,cos]
    tang2 = np.stack([-np.sin(theta), np.cos(theta)], 1)          # (n,2)
    tau = tang2 @ P.T                                            # (n,p) unit ambient
    radius = float(np.median(np.linalg.norm(proj, axis=1)))
    print(f"[chart] L{Lc} planarity={planarity:.4f} radius={radius:.3f}", flush=True)

    # ---- pass 2: JVP of layers[Lc:Lm] applied to tau at each token; doses + measured KL
    measurer = D.MeasuredKL(lm, hook_c, tok, a.device)
    sub = layers[Lc:Lm]
    rows = []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to(a.device)
        with torch.no_grad():
            out = hf(ids, output_hidden_states=True)
        h_from = out.hidden_states[Lc].float()               # (1,seq,p)
        seq = ids.shape[1]
        pos_ids = torch.arange(seq, device=a.device).unsqueeze(0)
        pos_emb = rotary(h_from, pos_ids)

        def f(h):
            x = h
            for layer in sub:
                o = layer(x, position_embeddings=pos_emb)
                x = o[0] if isinstance(o, tuple) else o
            return x[0, -1, :]

        # wiring check: substack reproduces the reference hidden at L_m
        with torch.no_grad():
            ref = out.hidden_states[Lm][0, -1, :].float()
            got = f(h_from)
            rel = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
        if rel > 1e-3:
            raise SystemExit(f"substack wiring check failed prompt {i}: rel={rel:.3e}")

        tau_i = torch.zeros_like(h_from)
        tau_i[0, -1, :] = torch.tensor(tau[i], dtype=h_from.dtype, device=a.device)
        _, Jtau = torch.func.jvp(f, (h_from,), (tau_i,))
        Jtau = Jtau.detach().float().cpu().numpy()            # (p,) in L_m space

        uc_i, um_i = U_c[i], U_m[i]
        # 1/2 tau^T G tau  = 1/2 ||U^T tau||^2   (per unit s^2)
        q_harv = 0.5 * float(np.sum((uc_i.T @ tau[i]) ** 2))
        q_prop = 0.5 * float(np.sum((um_i.T @ Jtau) ** 2))
        for frac in a.fracs:
            s_mag = frac * radius
            for sgn in (+1.0, -1.0):
                s = sgn * s_mag
                delta = s * tau[i]
                kl = measurer.kl(p, delta)
                rows.append({
                    "prompt_i": i, "day": int(labels[i]), "frac": frac, "sign": sgn,
                    "s": s, "measured_kl": kl,
                    "pred_harv": q_harv * s * s, "pred_prop": q_prop * s * s,
                })
        if (i + 1) % 10 == 0:
            print(f"[dose] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    # ---- calibration scoring: log-log slope, R^2, median ratio
    def score(key):
        pr = np.array([r[key] for r in rows]); me = np.array([r["measured_kl"] for r in rows])
        ok = (pr > 1e-12) & (me > 1e-12)
        x, y = np.log(pr[ok]), np.log(me[ok])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope, inter = float(coef[0]), float(coef[1])
        yhat = A @ coef
        ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        ratio = np.exp(np.median(y - x))     # median measured/pred
        mlr = float(np.median(np.abs(y - x)))
        return {"n": int(ok.sum()), "slope_loglog": slope, "r2": r2,
                "median_meas_over_pred": float(ratio), "median_abs_log_ratio": mlr}

    # per-token correlation of the two prediction columns (how equal are they?)
    ph = np.array([r["pred_harv"] for r in rows]); pp = np.array([r["pred_prop"] for r in rows])
    ok = (ph > 1e-12) & (pp > 1e-12)
    col_corr = float(np.corrcoef(np.log(ph[ok]), np.log(pp[ok]))[0, 1])
    col_ratio = float(np.exp(np.median(np.log(pp[ok]) - np.log(ph[ok]))))

    result = {
        "chart_layer": Lc, "metric_layer": Lm, "hop_layers": Lm - Lc,
        "n_prompts": n, "rank": a.rank, "fracs": a.fracs, "radius": radius,
        "planarity": planarity,
        "harvested": score("pred_harv"),
        "propagated": score("pred_prop"),
        "propagated_vs_harvested": {
            "pred_log_corr": col_corr, "median_prop_over_harv": col_ratio,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    np.savez(a.out.with_suffix(".rows.npz"),
             measured=np.array([r["measured_kl"] for r in rows]),
             pred_harv=ph, pred_prop=pp,
             frac=np.array([r["frac"] for r in rows]),
             day=np.array([r["day"] for r in rows]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
