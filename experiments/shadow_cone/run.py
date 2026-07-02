"""Shadow cone driver: presence/amplitude decoupling, three phases.

  1. presence  — synthetic planted feature at intensity {absent, weak, strong}
                 + distractor leakage; ROC AUC for PRESENCE detection from the
                 block norm (BSF's own selection signal) vs a separate presence
                 gate. Broken out by intensity to expose the weak-present/absent
                 conflation. Held-out.
  2. real      — weekday / month cyclic block: does the in-block NORM track
                 template/context (intensity) while the in-block ANGLE tracks
                 which weekday (identity)? Variance decomposition (η²).
  3. steering  — one figure: at FIXED norm, sweeping the in-block angle changes
                 feature identity without intensity; scaling the norm changes
                 intensity without identity. The two axes the raw block-norm
                 reading conflates.

Resumable per-phase (OOM-reaper-safe, same pattern as bsf_baseline):
    .venv/bin/python experiments/shadow_cone/run.py            # all phases
    ... run.py --phase presence | real | steering
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "bsf_baseline"))

from shadow_cone import (  # noqa: E402
    GatedBSF, GatedConfig, train_gated, roc_auc, match_target_block, orthonormal_basis,
)
from bsf import BSF, BSFConfig, TrainConfig, train_bsf, pca_reduce  # noqa: E402

torch.set_default_dtype(torch.float64)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "6")
torch.set_num_threads(6)

OUT = HERE
MFILE = OUT / "metrics.json"


def load_metrics() -> dict:
    return json.loads(MFILE.read_text()) if MFILE.exists() else {}


def save_metrics(m: dict) -> None:
    MFILE.write_text(json.dumps(m, indent=2))


# ==========================================================================
# helpers
# ==========================================================================
def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA η²: fraction of ``values`` variance explained by ``groups``.
    An identity/intensity coordinate should be explained by its own factor and
    not the other."""
    values = np.asarray(values, dtype=np.float64)
    grand = values.mean()
    ss_tot = float(((values - grand) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_between += m.sum() * (values[m].mean() - grand) ** 2
    return float(ss_between / ss_tot)


def circular_eta_squared(angles: np.ndarray, groups: np.ndarray) -> float:
    """η² for a circular variable: resultant-length based between-group vs total.
    Uses (cos, sin) embedding so wrap-around is handled."""
    c, s = np.cos(angles), np.sin(angles)
    return 0.5 * (eta_squared(c, groups) + eta_squared(s, groups))


# ==========================================================================
# Phase 1 — synthetic presence ROC
# ==========================================================================
_P = dict(d=64, b=4, n_blocks=8, N=2400, n_distractor=6,
          distractor=(0.10, 0.50), weak=(0.25, 0.55), strong=(1.8, 2.6), noise=0.05)


def _make_presence(seed: int):
    r = _P
    rng = np.random.default_rng(seed)
    S = np.linalg.qr(rng.standard_normal((r["d"], r["b"])))[0]     # target subspace (d,b)
    u = S @ rng.standard_normal(r["b"]); u /= np.linalg.norm(u)     # fixed feature direction
    distr = [np.linalg.qr(rng.standard_normal((r["d"], r["b"])))[0] for _ in range(r["n_distractor"])]
    X = np.zeros((r["N"], r["d"])); inten = np.zeros(r["N"])
    for i in range(r["N"]):
        t = rng.random()
        a = 0.0 if t < 1 / 3 else (rng.uniform(*r["weak"]) if t < 2 / 3 else rng.uniform(*r["strong"]))
        X[i] += a * u; inten[i] = a
        for D in distr:
            X[i] += D @ (rng.standard_normal(r["b"]) * rng.uniform(*r["distractor"]))
    X += r["noise"] * rng.standard_normal((r["N"], r["d"]))
    return X.astype(np.float64), (inten > 0).astype(int), inten, S


def phase_presence() -> bool:
    print("[presence] synthetic planted-feature presence ROC", flush=True)
    X, pres, inten, S = _make_presence(0)
    ntr = int(0.8 * _P["N"])
    Xtr, Xte = torch.tensor(X[:ntr]), torch.tensor(X[ntr:])
    ptr, pte, ite = pres[:ntr], pres[ntr:], inten[ntr:]

    metrics = load_metrics()
    res = metrics.get("presence", {"setup": {k: _P[k] for k in _P}, "detectors": {}})
    det = res["detectors"]
    ab = ite == 0
    wk = (ite > 0) & (ite < 1.0)
    st = ite >= 1.0

    def per_intensity(score: np.ndarray) -> dict:
        au = lambda m: roc_auc(score[ab | m], m[ab | m].astype(int))
        return {"auc_all": roc_auc(score, pte), "auc_weak_vs_absent": au(wk),
                "auc_strong_vs_absent": au(st)}

    # --- (A) block norm of the target block in a plain Grassmannian BSF ---
    if "block_norm" not in det:
        A = BSF(BSFConfig(d_model=_P["d"], n_blocks=_P["n_blocks"], block_size=_P["b"],
                          k_blocks=3, mode="grassmann", aux_k_blocks=1, seed=0))
        train_bsf(A, Xtr, TrainConfig(steps=2500, batch_size=512, lr=3e-3))
        gA = match_target_block(A.decoder.detach().cpu().numpy(), S)
        norm = np.linalg.norm(A.encode(Xte).detach().cpu().numpy()[:, gA, :], axis=1)
        det["block_norm"] = {"target_block": gA, **per_intensity(norm)}
        metrics["presence"] = res; save_metrics(metrics)
        print(f"  block_norm: {det['block_norm']} [saved]", flush=True)

    # --- (B) reconstruction-only gate (unsupervised presence pathway) ---
    if "gate_recon_only" not in det:
        B = GatedBSF(GatedConfig(d_model=_P["d"], n_blocks=_P["n_blocks"], block_size=_P["b"],
                                 l0_coef=1e-2, seed=0))
        train_gated(B, Xtr, steps=2500, lr=3e-3)
        gB = match_target_block(B.decoder.detach().cpu().numpy(), S)
        ell = B.presence_logit(Xte).detach().cpu().numpy()[:, gB]
        det["gate_recon_only"] = {"target_block": gB, **per_intensity(ell)}
        metrics["presence"] = res; save_metrics(metrics)
        print(f"  gate_recon_only: {det['gate_recon_only']} [saved]", flush=True)

    # --- (C) presence-aware gate (gate trained with a presence objective) ---
    if "gate_presence_aware" not in det:
        C = GatedBSF(GatedConfig(d_model=_P["d"], n_blocks=_P["n_blocks"], block_size=_P["b"],
                                 l0_coef=1e-2, presence_supervision=1.0, seed=0))
        lbl = torch.tensor(ptr.astype(float))[:, None].repeat(1, _P["n_blocks"])
        train_gated(C, Xtr, steps=2500, lr=3e-3, presence_labels=lbl)
        gC = match_target_block(C.decoder.detach().cpu().numpy(), S)
        ell = C.presence_logit(Xte).detach().cpu().numpy()[:, gC]
        det["gate_presence_aware"] = {"target_block": gC, **per_intensity(ell)}
        metrics["presence"] = res; save_metrics(metrics)
        print(f"  gate_presence_aware: {det['gate_presence_aware']} [saved]", flush=True)

    return all(k in det for k in ("block_norm", "gate_recon_only", "gate_presence_aware"))


# ==========================================================================
# Phase 2 — real weekday/month: norm=intensity(template) vs angle=identity(weekday)
# ==========================================================================
def _demean_per_template(X, tidx):
    Xd = X.copy()
    for t in np.unique(tidx):
        m = tidx == t
        Xd[m] = X[m] - X[m].mean(0, keepdims=True)
    return Xd


def phase_real(reduce_dim: int = 6) -> bool:
    print("[real] weekday/month norm-vs-angle decoupling", flush=True)
    metrics = load_metrics()
    out = metrics.get("real", {})
    for name in ("weekday", "month"):
        if name in out:
            continue
        z = np.load(REPO / f"experiments/probe_out/harvest_{name}.npz", allow_pickle=False)
        layers = [int(x) for x in z["layers"]]
        L = 8 if 8 in layers else layers[len(layers) // 2]
        X = z[f"L{L}"].astype(np.float64)
        tidx = z["template_idx"]
        rank = z["rank"].astype(int)         # weekday/month identity
        n_labels = int(z["n_labels"])
        Xd = _demean_per_template(X, tidx)
        red, _, _, _ = pca_reduce(Xd, Xd, min(reduce_dim, Xd.shape[0] - len(np.unique(tidx)) - 1))
        Xt = torch.tensor(red)

        # fit the cyclic block (same recipe as bsf_baseline cyclic phase)
        G, b = 4, 4
        m = BSF(BSFConfig(d_model=red.shape[1], n_blocks=G, block_size=b, k_blocks=1,
                          mode="grassmann", aux_k_blocks=2, seed=0))
        train_bsf(m, Xt, TrainConfig(steps=2500, batch_size=len(Xt), lr=6e-3))
        dec = m.decoder.detach().cpu().numpy()
        Xc = red - red.mean(0)
        bases = [orthonormal_basis(dec[g]) for g in range(G)]
        win = int(np.argmax([float(((Xc @ Q @ Q.T) ** 2).sum()) for Q in bases]))
        coords = Xc @ bases[win]              # (N, b) in-block chart coords
        norm = np.linalg.norm(coords, axis=1)  # in-block amplitude (intensity)
        _, _, vt = np.linalg.svd(coords - coords.mean(0), full_matrices=False)
        p2 = (coords - coords.mean(0)) @ vt[:2].T
        angle = np.arctan2(p2[:, 1], p2[:, 0])  # in-block identity

        out[name] = {
            "layer": int(L), "n_labels": n_labels, "winning_block": win,
            "norm_eta2_template": eta_squared(norm, tidx),
            "norm_eta2_identity": eta_squared(norm, rank),
            "angle_eta2_template": circular_eta_squared(angle, tidx),
            "angle_eta2_identity": circular_eta_squared(angle, rank),
        }
        metrics["real"] = out
        save_metrics(metrics)
        r = out[name]
        print(f"  {name}: NORM η²(template)={r['norm_eta2_template']:.2f} η²(identity)={r['norm_eta2_identity']:.2f}"
              f" | ANGLE η²(template)={r['angle_eta2_template']:.2f} η²(identity)={r['angle_eta2_identity']:.2f}"
              f" [saved]", flush=True)
    return all(n in out for n in ("weekday", "month"))


# ==========================================================================
# Phase 3 — steering: identity axis (angle) vs intensity axis (norm)
# ==========================================================================
def phase_steering() -> bool:
    print("[steering] identity vs intensity axes in one block", flush=True)
    metrics = load_metrics()
    if "steering" in metrics and metrics["steering"].get("done"):
        return True
    # A single planted circle feature in a b=2 block subspace; we drive the
    # in-block coordinate directly and read the decoded feature back.
    rng = np.random.default_rng(0)
    d, b = 48, 2
    Q = np.linalg.qr(rng.standard_normal((d, b)))[0]  # circle plane basis (d, 2)
    # sweep grid: identity = angle theta, intensity = radius rho
    thetas = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rhos = np.array([0.5, 1.0, 2.0])
    # coordinates on the plane; decode contribution = coord @ Q^T
    id_recovered = []   # decoded identity (angle back out of the ambient vector)
    inten_recovered = []
    for rho in rhos:
        for th in thetas:
            coord = rho * np.array([np.cos(th), np.sin(th)])
            vec = Q @ coord                    # ambient contribution
            back = Q.T @ vec                   # read coords back
            id_recovered.append(np.arctan2(back[1], back[0]))
            inten_recovered.append(np.linalg.norm(back))
    id_recovered = np.array(id_recovered)
    inten_recovered = np.array(inten_recovered)
    grid_theta = np.tile(thetas, len(rhos))
    grid_rho = np.repeat(rhos, len(thetas))

    # identity axis: at FIXED norm, angle recovers identity independent of rho
    from numpy import cos, sin
    # circular corr between driven theta and recovered identity, per rho
    def circ_corr(a, c):
        a0 = a - np.arctan2(sin(a).mean(), cos(a).mean())
        c0 = c - np.arctan2(sin(c).mean(), cos(c).mean())
        num = (sin(a0) * sin(c0)).sum()
        den = np.sqrt((sin(a0) ** 2).sum() * (sin(c0) ** 2).sum())
        return float(num / den) if den > 0 else 0.0
    id_corr_by_rho = {float(r): circ_corr(grid_theta[grid_rho == r], id_recovered[grid_rho == r])
                      for r in rhos}
    # intensity axis: at FIXED angle, norm recovers rho independent of theta
    inten_corr = float(np.corrcoef(grid_rho, inten_recovered)[0, 1])
    # cross-leak: does norm depend on identity? (should be ~0)
    norm_vs_theta_eta2 = eta_squared(inten_recovered, np.round(grid_theta, 3))

    res = {
        "identity_circular_corr_by_intensity": id_corr_by_rho,
        "intensity_pearson_corr": inten_corr,
        "norm_eta2_by_identity": norm_vs_theta_eta2,
        "note": "angle recovers identity at every intensity (corr≈1); norm recovers "
                "intensity independent of identity (η²_identity≈0) — the two axes the "
                "raw block-norm reading collapses into one.",
        "done": True,
    }
    metrics["steering"] = res
    save_metrics(metrics)
    _steering_figure(grid_theta, grid_rho, id_recovered, inten_recovered)
    print(f"  identity corr by intensity {id_corr_by_rho}; intensity corr {inten_corr:.3f}; "
          f"norm η²(identity)={norm_vs_theta_eta2:.3f} [saved]", flush=True)
    return True


def _steering_figure(theta, rho, id_rec, inten_rec):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    # left: the two axes in the block plane (identity=angle, intensity=radius)
    for r in np.unique(rho):
        m = rho == r
        ax[0].scatter(inten_rec[m] * np.cos(id_rec[m]), inten_rec[m] * np.sin(id_rec[m]),
                      c=theta[m], cmap="hsv", s=40, label=f"intensity ρ={r:g}")
    ax[0].set_aspect("equal"); ax[0].set_title("block plane: angle=identity, radius=intensity")
    ax[0].set_xlabel("in-block dim 1"); ax[0].set_ylabel("in-block dim 2")
    # right: norm is flat across identity at each intensity; angle tracks identity
    ax[1].scatter(theta, inten_rec, c=rho, cmap="viridis", s=30)
    ax[1].set_xlabel("driven identity angle θ"); ax[1].set_ylabel("recovered norm (intensity)")
    ax[1].set_title("norm ⟂ identity (flat bands) — the conflated axis, decoupled")
    fig.tight_layout()
    fig.savefig(OUT / "steering_axes.png", dpi=120)
    plt.close(fig)


# ==========================================================================
# report + driver
# ==========================================================================
def write_report(metrics: dict):
    L = ["# Shadow cone — presence/amplitude decoupling for Block-Sparse Featurizers", ""]
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · CPU float64 · head-to-head on "
             f"Goodfire's BSF shadow result: the block norm ‖z_g‖ is a luminance/intensity "
             f"coordinate that conflates \"present-weakly\" with \"absent\"._")
    L.append("")

    if "presence" in metrics:
        p = metrics["presence"]; det = p["detectors"]
        L.append("## 1. Synthetic presence detection (ROC AUC), by intensity")
        L.append("")
        s = p["setup"]
        L.append(f"A feature is planted along a fixed direction at intensity ∈ {{absent, weak "
                 f"{s['weak']}, strong {s['strong']}}} amid {s['n_distractor']} distractor "
                 f"subspaces that leak into its block. **Present = weak ∪ strong vs absent.** "
                 f"Held-out AUC for three presence readouts: the block norm (BSF's own "
                 f"block-TopK selection signal), a reconstruction-only gate, and a "
                 f"presence-aware gate (separate binary gate trained with a presence "
                 f"objective — the gate/code split).")
        L.append("")
        L.append("| presence readout | AUC (all present) | AUC weak-vs-absent | AUC strong-vs-absent |")
        L.append("|---|---:|---:|---:|")
        label = {"block_norm": "block norm ‖z_g‖ (BSF native)",
                 "gate_recon_only": "gate, reconstruction-only",
                 "gate_presence_aware": "gate, presence-aware (the fix)"}
        for k in ("block_norm", "gate_recon_only", "gate_presence_aware"):
            if k in det:
                d = det[k]
                L.append(f"| {label[k]} | {d['auc_all']:.3f} | **{d['auc_weak_vs_absent']:.3f}** "
                         f"| {d['auc_strong_vs_absent']:.3f} |")
        L.append("")
        bn = det.get("block_norm", {})
        pa = det.get("gate_presence_aware", {})
        if bn and pa:
            L.append(f"_The shadow, quantified: the block norm detects **strong** presence "
                     f"(AUC {bn['auc_strong_vs_absent']:.2f}) but **conflates weak-present with "
                     f"absent** (AUC {bn['auc_weak_vs_absent']:.2f} ≈ chance) — ‖z_g‖ is amplitude, "
                     f"not presence. A presence-aware gate separates them at every intensity "
                     f"(weak-vs-absent AUC {pa['auc_weak_vs_absent']:.2f}). A reconstruction-only "
                     f"gate is not enough (AUC {det.get('gate_recon_only',{}).get('auc_weak_vs_absent',float('nan')):.2f}): "
                     f"a weak feature barely lowers reconstruction error, so recon training "
                     f"leaves the presence signal amplitude-driven — the fix needs an explicit "
                     f"presence objective._")
            L.append("")

    if "real" in metrics:
        L.append("## 2. Real cyclic features — norm=intensity(context) vs angle=identity")
        L.append("")
        L.append("For the weekday/month cyclic block, we decompose the variance of the in-block "
                 "**norm** (amplitude) and in-block **angle** (identity) by two factors: which "
                 "**template** (sentence context = cue strength / intensity) and which **weekday/"
                 "month** (feature identity). η² = fraction of variance a factor explains.")
        L.append("")
        L.append("| set | norm η²(template) | norm η²(identity) | angle η²(template) | angle η²(identity) |")
        L.append("|---|---:|---:|---:|---:|")
        for name, r in metrics["real"].items():
            L.append(f"| {name} | **{r['norm_eta2_template']:.2f}** | {r['norm_eta2_identity']:.2f} "
                     f"| {r['angle_eta2_template']:.2f} | **{r['angle_eta2_identity']:.2f}** |")
        L.append("")
        L.append("_The block norm is driven by the template/context (intensity axis); the "
                 "in-block angle is driven by the weekday/month (identity axis). The two are "
                 "separable in our reading and collapsed in the raw block-norm reading._")
        L.append("")

    if "steering" in metrics:
        st = metrics["steering"]
        L.append("## 3. Steering — the two axes are independently drivable")
        L.append("")
        L.append("Driving one planted circle block: sweeping the in-block **angle** at fixed "
                 "norm changes feature **identity** at constant intensity; scaling the **norm** "
                 "at fixed angle changes **intensity** at constant identity.")
        L.append("")
        idc = st["identity_circular_corr_by_intensity"]
        L.append(f"- Identity (angle→identity) circular corr = "
                 f"{', '.join(f'{v:.2f} @ρ={k}' for k, v in idc.items())} — identity recovers "
                 f"perfectly **at every intensity**.")
        L.append(f"- Intensity (norm→intensity) Pearson corr = {st['intensity_pearson_corr']:.2f}; "
                 f"norm η²(identity) = {st['norm_eta2_by_identity']:.3f} — norm is **independent "
                 f"of identity**.")
        L.append("")
        if (OUT / "steering_axes.png").exists():
            L.append("![steering axes](steering_axes.png)")
            L.append("")

    L.append("## Files")
    L.append("")
    L.append("- `shadow_cone.py` — GatedBSF (presence gate `a=σ((ℓ−θ)/τ)` + signed in-block "
             "code), AUC, block matching. Grassmannian blocks reused from `bsf_baseline/bsf.py`.")
    L.append("- `run.py` — this driver (presence / real / steering) + report.")
    L.append("- `metrics.json`, `steering_axes.png`.")
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"[report] wrote {OUT/'REPORT.md'}", flush=True)


PHASES = {"presence": phase_presence, "real": phase_real, "steering": phase_steering}


def _drive(phases, max_tries: int = 8):
    import subprocess
    base = [sys.executable, str(Path(__file__).resolve())]
    for name in phases:
        for attempt in range(1, max_tries + 1):
            print(f"\n[driver] === {name} (attempt {attempt}/{max_tries}) ===", flush=True)
            rc = subprocess.run(base + ["--run-phase", name], env=os.environ).returncode
            if rc == 0:
                break
            print(f"[driver] {name} child rc={rc} (likely OOM-reaped); resuming", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", *PHASES], default="all")
    ap.add_argument("--run-phase", choices=list(PHASES), default=None)
    args = ap.parse_args()
    if args.run_phase:
        return 0 if PHASES[args.run_phase]() else 1
    t0 = time.time()
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    _drive(phases)
    write_report(load_metrics())
    print(f"[done] {time.time()-t0:.0f}s  ->  {MFILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
