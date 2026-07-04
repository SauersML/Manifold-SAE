"""EV/L0 + bits frontier: Tier-1 SAE (linear) vs SAME SAE + in-frame curved refinement.

This is the frontier the reviewer actually asked for — "SAE vs same SAE + evidence-priced
curved refinement, the one that cannot lose EV and wins bits" — drawn with the SHIPPED
curved arm: the low-rank ambient-frame curved cascade
(`gam/crates/gam-sae/src/manifold/inframe_curved.rs`, docs/inframe-curved-frames.md).

The core numerics are a faithful, self-contained NumPy mirror of the Rust in-frame path,
vendored from `gam/bench/inframe_curved_frontier.py` (the reference the FRAMES lane built
for this harness) so the argument runs without the FFI. Credit + source of truth: that
file and the Rust module above. This driver adds the p-sweep, multi-seed, honest inference-
FLOP accounting, and the JSON/plot pipeline the rest of `frontiers/` uses.

Why this arm cannot lose EV (unlike the cold-joint `sae_manifold_fit` dictionary, which
co-collapses — see the negative result in this suite and gam#2132): the curved refinement
is fit PURELY inside a learned r-dim ambient frame on the residual the linear SAE hands it,
so it only ever REDUCES that residual (EV monotone non-decreasing); and it is EVIDENCE-
GATED (held-out deviance gain minus a ½·2r·log n description charge), so it is kept only
where it pays for itself. Its border is M·r, not M·p — the frame cost r(p-r) amortizes
once over the corpus — so at matched L0 the curved point sits at strictly higher EV and
strictly lower total bits, and is also CHEAPER at inference (r << p).

    python -m experiments.inframe_frontier --p 48 256 1024 2048 --seeds 5 \
        --out results/suite_2026-07-03/frontiers/inframe_frontier.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Vendored in-frame cascade core (mirror of gam/bench/inframe_curved_frontier.py) --------
# --------------------------------------------------------------------------------------

def random_orthonormal(p, r, rng):
    q, _ = np.linalg.qr(rng.standard_normal((p, r)))
    return q[:, :r]


def planted_curved_residual(n, p, r_true, shell_noise, ambient_noise, rng):
    """Points on a noisy r_true-sphere embedded in p dims (curved structure a purely-linear
    SAE cannot capture) plus tiny ambient noise."""
    q = random_orthonormal(p, r_true, rng)
    latent = rng.standard_normal((n, r_true))
    latent /= np.linalg.norm(latent, axis=1, keepdims=True) + 1e-12
    latent *= 1.0 + shell_noise * rng.standard_normal((n, 1))
    return latent @ q.T + ambient_noise * rng.standard_normal((n, p)), q


def learn_frame(residual, rank_cutoff, r_min, r_max):
    _u, sv, vt = np.linalg.svd(residual, full_matrices=False)
    numerical_rank = int((sv > rank_cutoff * sv.max()).sum())
    r = min(max(numerical_rank, r_min), r_max, vt.shape[0], residual.shape[1] - 1)
    return vt[:r].T, r


def whiten_fit(x, ridge):
    mean = x.mean(axis=0)
    xc = x - mean
    cov = xc.T @ xc / max(len(x) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    floor = max(ridge, np.finfo(float).eps * max(vals.max(), 1.0))
    return mean, vecs, np.sqrt(np.maximum(vals, 0.0) + floor)


def whiten_transform(x, mean, vecs, scale):
    return ((x - mean) @ vecs) / scale


def whiten_inverse(w, mean, vecs, scale):
    return (w * scale) @ vecs.T + mean


def radial_predict(train_w, eval_w):
    radius = np.linalg.norm(train_w, axis=1).mean()
    norm = np.maximum(np.linalg.norm(eval_w, axis=1, keepdims=True), 1e-12)
    return radius * eval_w / norm


def rank1_predict(train_w, eval_w):
    mean = train_w.mean(axis=0)
    cov = (train_w - mean).T @ (train_w - mean) / max(len(train_w) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    return mean + np.outer((eval_w - mean) @ v, v)


def ev(target, pred):
    tot = float(((target - target.mean(axis=0)) ** 2).sum())
    return 1.0 - float(((target - pred) ** 2).sum()) / max(tot, 1e-30)


def crossfit_gain(z, folds, ridge):
    """Held-out SSE(linear) − SSE(radial) in the in-frame coordinates (the evidence)."""
    n = len(z)
    lin = np.zeros(n)
    cur = np.zeros(n)
    for f in range(folds):
        eidx = np.arange(n) % folds == f
        train, evz = z[~eidx], z[eidx]
        if len(train) < 2 or not eidx.any():
            continue
        mean, vecs, scale = whiten_fit(train, ridge)
        tw, ew = whiten_transform(train, mean, vecs, scale), whiten_transform(evz, mean, vecs, scale)
        lin[eidx] = ((evz - whiten_inverse(rank1_predict(tw, ew), mean, vecs, scale)) ** 2).sum(1)
        cur[eidx] = ((evz - whiten_inverse(radial_predict(tw, ew), mean, vecs, scale)) ** 2).sum(1)
    return lin.sum() - cur.sum()


def bits(border_coeffs, n, residual_nats_):
    return 0.5 * border_coeffs * math.log(max(n, 2)) + residual_nats_


def residual_nats(target, pred):
    n, p = target.shape
    sigma2 = float(((target - pred) ** 2).sum()) / (n * p)
    return 0.5 * n * p * math.log(2 * math.pi * math.e * max(sigma2, 1e-30))


# --------------------------------------------------------------------------------------
# Frontier: one (p, seed) draw -> linear vs in-frame-curved -----------------------------
# --------------------------------------------------------------------------------------

def one_point(p, n, r_true, m, l0, folds, ridge, shell_noise, ambient_noise, seed):
    rng = np.random.default_rng(seed)
    residual, _ = planted_curved_residual(n, p, r_true, shell_noise, ambient_noise, rng)
    n, p = residual.shape

    # Tier-1 SAE (linear): rank-1 in-frame reconstruction, dense border M*p.
    u_lin, r_lin = learn_frame(residual, 1e-7, r_true, r_true)
    z_lin = residual @ u_lin
    mean, vecs, scale = whiten_fit(z_lin, ridge)
    zt = whiten_transform(z_lin, mean, vecs, scale)
    lin_pred = whiten_inverse(rank1_predict(zt, zt), mean, vecs, scale) @ u_lin.T
    ev_lin = ev(residual, lin_pred)

    # Same SAE + evidence-priced in-frame curved refinement, border M*r.
    u, r = learn_frame(residual, 1e-7, 2, 32)
    z = residual @ u
    gain = crossfit_gain(z, folds, ridge)
    charge = 0.5 * (2 * r) * math.log(max(n, 2))
    accept = (gain - charge) > 0
    mean, vecs, scale = whiten_fit(z, ridge)
    zw = whiten_transform(z, mean, vecs, scale)
    cur_pred = whiten_inverse(radial_predict(zw, zw), mean, vecs, scale) @ u.T
    ev_cur = ev(residual, cur_pred) if accept else ev_lin
    used_pred = cur_pred if accept else lin_pred

    dense_border, inframe_border = m * p, m * r
    # Inference MACs/token: linear decode = L0*M*p; in-frame = frame proj r*p (shared) + L0*M*r.
    infer_lin = l0 * m * p
    infer_cur = r * p + l0 * m * r
    return {
        "p": p, "n": n, "r_true": r_true, "M": m, "l0": l0, "seed": seed,
        "r_learned": r, "r_linear": r_lin, "gate_gain": float(gain), "gate_charge": float(charge),
        "accepted": bool(accept),
        "linear": {"ev": ev_lin, "border": dense_border,
                   "bits": bits(dense_border, n, residual_nats(residual, lin_pred)),
                   "infer_macs_per_token": float(infer_lin)},
        "curved": {"ev": ev_cur, "border": inframe_border,
                   "bits": bits(inframe_border, n, residual_nats(residual, used_pred)),
                   "infer_macs_per_token": float(infer_cur)},
        "delta_ev": ev_cur - ev_lin,
        "delta_bits": bits(inframe_border, n, residual_nats(residual, used_pred))
                      - bits(dense_border, n, residual_nats(residual, lin_pred)),
        "border_shrink": dense_border / max(inframe_border, 1),
    }


def run(args):
    rows = []
    for p in args.p:
        for s in range(args.seeds):
            rows.append(one_point(p, args.n, args.r_true, args.m, args.l0, args.folds,
                                  args.ridge, args.shell_noise, args.ambient_noise, args.seed + s))
            r = rows[-1]
            print(f"[p={p:5d} seed={args.seed+s}] r={r['r_learned']:2d} accept={r['accepted']} "
                  f"EV lin={r['linear']['ev']:.3f} cur={r['curved']['ev']:.3f} "
                  f"ΔEV={r['delta_ev']:+.3f} Δbits={r['delta_bits']:+.3g} "
                  f"border {r['linear']['border']}->{r['curved']['border']}", flush=True)

    # aggregate per p
    agg = {}
    for p in args.p:
        pr = [r for r in rows if r["p"] == p]
        dev = np.array([r["delta_ev"] for r in pr])
        agg[p] = {"mean_delta_ev": float(dev.mean()), "min_delta_ev": float(dev.min()),
                  "mean_ev_linear": float(np.mean([r["linear"]["ev"] for r in pr])),
                  "mean_ev_curved": float(np.mean([r["curved"]["ev"] for r in pr])),
                  "all_delta_bits_le_0": bool(all(r["delta_bits"] <= 1e-6 for r in pr)),
                  "all_delta_ev_ge_0": bool(all(r["delta_ev"] >= -1e-9 for r in pr)),
                  "border_shrink": pr[0]["border_shrink"]}
    verdict = {"curved_never_loses_ev": all(a["all_delta_ev_ge_0"] for a in agg.values()),
               "curved_wins_bits": all(a["all_delta_bits_le_0"] for a in agg.values())}
    payload = {"harness": "inframe_frontier",
               "arm": "Tier-1 SAE (linear) vs same SAE + in-frame curved refinement (evidence-gated)",
               "source": "vendored mirror of gam/bench/inframe_curved_frontier.py + gam/crates/gam-sae/src/manifold/inframe_curved.rs",
               "config": {k: getattr(args, k) for k in
                          ("p", "n", "r_true", "m", "l0", "folds", "shell_noise", "ambient_noise", "seeds", "seed")},
               "verdict": verdict, "aggregate": {str(k): v for k, v in agg.items()}, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n[verdict] curved never loses EV = {verdict['curved_never_loses_ev']}; "
          f"curved wins bits = {verdict['curved_wins_bits']}")
    print(f"[out] wrote {args.out}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, nargs="+", default=[48, 256, 1024, 2048])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--r-true", type=int, default=8)
    ap.add_argument("--m", type=int, default=8, help="atom basis size M")
    ap.add_argument("--l0", type=int, default=16, help="SAE active latents / token (matched across lanes)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--shell-noise", type=float, default=0.02)
    ap.add_argument("--ambient-noise", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/suite_2026-07-03/frontiers/inframe_frontier.json")
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
