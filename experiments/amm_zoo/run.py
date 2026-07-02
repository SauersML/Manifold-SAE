"""AMM zoo driver: generate -> fit arms (held-out) -> metrics -> MDL -> matched
null -> incremental JSON -> figures. The Appendix-H replication/beat benchmark.

Each (seed, sigma, arm) CELL runs in its OWN subprocess with a wall-clock timeout
and a retry (the box has an OOM reaper; a killed cell must not lose the sweep).
Results are saved incrementally to ``results.json`` after every cell.

Metrics per cell (held-out TEST): Hungarian-matched per-factor contribution R²,
coordinate circular-corr, geodesic-Spearman, topology-ID + dimension accuracy, and
MDL bits/token. Every RECOVERED-STRUCTURE claim (a matched circle/arc/torus/sphere)
is gated by a MATCHED PERMUTATION NULL on its geodesic-Spearman — permuting the
recovered coordinate's token alignment destroys the coordinate<->geometry
correspondence while preserving both marginals (the matched_null.py discipline) —
so a factor is only called "recovered" at ``p < 0.05``.

Headlines: (1) R²-vs-σ crossing curves per topology (the chart denoises curved
factors onto the manifold as σ grows; the block reconstructs the noise), and
(2) the topology-ID table (the BSF baselines answer "subspace" for everything =
chance; ours reads topology off the chart).

Usage::

    # full node run (200k/50k, 5 seeds, 4 sigmas, 4 arms):
    python run.py --full
    # quick local validation:
    python run.py --quick
    # one isolated cell (internal):
    python run.py --worker spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("AMM_OUT", HERE))
SCRATCH = Path(os.environ.get(
    "AMM_SCRATCH",
    "/private/tmp/claude-501/-Users-user/8553f8a7-a419-454a-a5c1-9d6acf52ece3/scratchpad/amm_work"))

ARMS = ["topk_sae", "bsf_vanilla", "bsf_grassmann", "ours"]


# --------------------------------------------------------------------------- #
# Matched permutation null for a recovered-structure claim.
# --------------------------------------------------------------------------- #
def _null_geodesic_p(true_coord, true_topo, rec_coord, rng, b_perm=200):
    """Matched permutation null on geodesic-Spearman: obs vs the distribution under
    permuting the recovered coordinate's token alignment. Returns (obs, p)."""
    from metrics import geodesic_spearman

    m = true_coord.shape[0]
    if m < 8:
        return 0.0, 1.0
    obs = geodesic_spearman(true_coord, true_topo, rec_coord, rng)
    null = np.empty(b_perm)
    for i in range(b_perm):
        perm = rng.permutation(m)
        null[i] = geodesic_spearman(true_coord, true_topo, rec_coord[perm], rng)
    p = float((1 + np.sum(null >= obs)) / (1 + b_perm))
    return round(float(obs), 4), round(p, 4)


# --------------------------------------------------------------------------- #
# Worker: one (seed, sigma, arm) cell.
# --------------------------------------------------------------------------- #
def _worker(spec_path: str) -> None:
    sys.excepthook = sys.__excepthook__
    import torch

    torch.set_num_threads(int(os.environ.get("AMM_THREADS", "2")))
    from amm import generate_amm
    from arms import run_arm
    from metrics import score_arm

    spec = json.loads(Path(spec_path).read_text())
    ds = generate_amm(
        seed=spec["seed"],
        sigma_frac=spec["sigma_frac"],
        coherence=spec["coherence"],
        n_train=spec["n_train"],
        n_test=spec["n_test"],
    )
    recs = run_arm(ds, spec["arm"], steps=spec["steps"], seed=spec["seed"])
    rep = score_arm(ds, recs, "test", seed=spec["seed"])

    # Matched null on structural recovered factors (circle/arc/torus/sphere).
    rng = np.random.default_rng(spec["seed"] + 991)
    # Re-derive the recovered coord per matched factor for the null.
    rec_by_name = {r.name: r for r in recs}
    for pf in rep["per_factor"]:
        if pf["true_topology"] == "linear":
            continue
        rf = rec_by_name.get(pf["recovered"])
        if rf is None:
            continue
        j = pf["true_factor"]
        rows, tcoord = ds.true_intrinsic("test", j)
        rc = rf.coord[rows]
        valid = np.all(np.isfinite(rc), axis=tuple(range(1, rc.ndim))) if rc.ndim > 1 else np.isfinite(rc)
        if valid.sum() >= 8:
            obs, p = _null_geodesic_p(
                tcoord[valid], pf["true_topology"], rf.coord[rows][valid], rng,
                b_perm=spec.get("b_perm", 200),
            )
            pf["geodesic_null_p"] = p
            pf["structure_recovered"] = bool(p < 0.05)

    out = {
        "seed": spec["seed"],
        "sigma_frac": spec["sigma_frac"],
        "coherence": spec["coherence"],
        "arm": spec["arm"],
        "min_principal_angle_deg": ds.min_principal_angle_deg,
        "signal_rms": ds.signal_rms,
        "report": rep,
    }
    Path(spec["out_path"]).write_text(json.dumps(out, default=float))


def _run_cell(seed, sigma, coherence, arm, cfg) -> dict:
    """Run one cell in an isolated subprocess with timeout + one retry."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_s{seed}_n{sigma}_c{coherence}"
    spec_path = SCRATCH / f"{tag}_spec.json"
    out_path = SCRATCH / f"{tag}_out.json"
    spec = {
        "seed": seed, "sigma_frac": sigma, "coherence": coherence, "arm": arm,
        "n_train": cfg["n_train"], "n_test": cfg["n_test"], "steps": cfg["steps"],
        "b_perm": cfg["b_perm"], "out_path": str(out_path),
    }
    spec_path.write_text(json.dumps(spec))
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", str(spec_path)]
    for attempt in range(cfg["retries"] + 1):
        if out_path.exists():
            out_path.unlink()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=cfg["timeout"], env=os.environ)
        except subprocess.TimeoutExpired:
            proc = None
        wall = round(time.time() - t0, 1)
        if out_path.exists():
            rec = json.loads(out_path.read_text())
            rec["wall_s"] = wall
            return rec
        code = None if proc is None else proc.returncode
        tail = "" if proc is None else "".join(proc.stderr.splitlines(keepends=True)[-4:])
        print(f"  [{tag}] attempt {attempt+1} failed (rc={code}, {wall}s); {tail[:200]}", flush=True)
    return {"seed": seed, "sigma_frac": sigma, "coherence": coherence, "arm": arm,
            "status": "FAILED", "wall_s": wall}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def drive(cfg) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / cfg["results_name"]
    master = {"config": cfg, "cells": []}
    results_path.write_text(json.dumps(master, indent=2, default=float))
    total = len(cfg["seeds"]) * len(cfg["sigmas"]) * len(ARMS)
    done = 0
    for seed in cfg["seeds"]:
        for sigma in cfg["sigmas"]:
            for arm in ARMS:
                done += 1
                t = time.time()
                rec = _run_cell(seed, sigma, cfg["coherence"], arm, cfg)
                ov = rec.get("report", {}).get("overall", {})
                print(f"[{done}/{total}] {arm} seed={seed} sigma={sigma}: "
                      f"R2={ov.get('mean_contribution_r2')} topoID={ov.get('topology_id_accuracy')} "
                      f"({rec.get('wall_s')}s)", flush=True)
                master["cells"].append(rec)
                results_path.write_text(json.dumps(master, indent=2, default=float))
    make_figures(master, OUT)
    _write_report(master, OUT)
    return master


# --------------------------------------------------------------------------- #
# Aggregation + figures + report
# --------------------------------------------------------------------------- #
def _agg(master):
    """Aggregate cells -> {(arm, sigma): {topology: [R2...], topoID:[...]}} over seeds."""
    from collections import defaultdict

    r2 = defaultdict(lambda: defaultdict(list))     # (arm,sigma) -> topo -> [R2]
    topoid = defaultdict(lambda: defaultdict(list))  # (arm,sigma) -> topo -> [acc]
    for c in master["cells"]:
        rep = c.get("report")
        if not rep:
            continue
        key = (c["arm"], c["sigma_frac"])
        for topo, v in rep["by_topology"].items():
            if v.get("mean_contribution_r2") is not None:
                r2[key][topo].append(v["mean_contribution_r2"])
            topoid[key][topo].append(v["topology_id_accuracy"])
    return r2, topoid


def make_figures(master, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[figures] matplotlib unavailable; skipping PNGs (JSON has all data)", flush=True)
        return
    r2, _ = _agg(master)
    sigmas = sorted(master["config"]["sigmas"])
    topos = ["circle", "arc", "torus", "sphere", "linear"]
    colors = {"topk_sae": "#888", "bsf_vanilla": "#4C78A8", "bsf_grassmann": "#F58518", "ours": "#54A24B"}

    fig, axes = plt.subplots(1, len(topos), figsize=(4 * len(topos), 3.6), sharey=True)
    for ax, topo in zip(axes, topos):
        for arm in ARMS:
            ys = [float(np.mean(r2[(arm, s)][topo])) if r2[(arm, s)][topo] else np.nan for s in sigmas]
            ax.plot(sigmas, ys, "-o", label=arm, color=colors[arm], lw=2, ms=4)
        ax.set_title(topo)
        ax.set_xlabel("σ / signal")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("contribution R² (held-out)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("AMM zoo: contribution R² vs noise — chart denoises curves, not the linear control")
    fig.tight_layout()
    fig.savefig(out / "r2_vs_sigma.png", dpi=130)
    plt.close(fig)

    # Topology-ID table (accuracy per arm per topology, averaged over seeds+sigmas).
    _, topoid = _agg(master)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cell = np.full((len(ARMS), len(topos)), np.nan)
    for i, arm in enumerate(ARMS):
        for j, topo in enumerate(topos):
            vals = [a for s in sigmas for a in topoid[(arm, s)][topo]]
            if vals:
                cell[i, j] = float(np.mean(vals))
    im = ax.imshow(cell, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(topos)), topos)
    ax.set_yticks(range(len(ARMS)), ARMS)
    for i in range(len(ARMS)):
        for j in range(len(topos)):
            if np.isfinite(cell[i, j]):
                ax.text(j, i, f"{cell[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Topology-ID accuracy (BSF ≈ chance 'subspace'; ours reads the chart)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "topology_id.png", dpi=130)
    plt.close(fig)
    print(f"[figures] wrote {out/'r2_vs_sigma.png'} and {out/'topology_id.png'}", flush=True)


def _write_report(master, out: Path) -> None:
    r2, topoid = _agg(master)
    sigmas = sorted(master["config"]["sigmas"])
    topos = ["circle", "arc", "torus", "sphere", "linear"]
    lines = ["# AMM zoo — Appendix-H replication/beat\n",
             f"Config: {json.dumps(master['config'])}\n",
             "## Contribution R² (held-out), mean over seeds\n"]
    lines.append("| arm | topo | " + " | ".join(f"σ={s}" for s in sigmas) + " |")
    lines.append("|---|---|" + "---|" * len(sigmas))
    for arm in ARMS:
        for topo in topos:
            row = [f"{np.mean(r2[(arm,s)][topo]):.3f}" if r2[(arm, s)][topo] else "-" for s in sigmas]
            lines.append(f"| {arm} | {topo} | " + " | ".join(row) + " |")
    lines.append("\n## Topology-ID accuracy (mean over seeds+σ)\n")
    lines.append("| arm | " + " | ".join(topos) + " |")
    lines.append("|---|" + "---|" * len(topos))
    for arm in ARMS:
        row = []
        for topo in topos:
            vals = [a for s in sigmas for a in topoid[(arm, s)][topo]]
            row.append(f"{np.mean(vals):.2f}" if vals else "-")
        lines.append(f"| {arm} | " + " | ".join(row) + " |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out/'REPORT.md'}", flush=True)


def _config(full: bool) -> dict:
    if full:
        return {"seeds": [0, 1, 2, 3, 4], "sigmas": [0.02, 0.05, 0.1, 0.2], "coherence": 0.0,
                "n_train": 200_000, "n_test": 50_000, "steps": 4000, "b_perm": 500,
                "timeout": 1800, "retries": 2, "results_name": "results.json"}
    return {"seeds": [0], "sigmas": [0.05, 0.2], "coherence": 0.0,
            "n_train": 8000, "n_test": 2500, "steps": 600, "b_perm": 120,
            "timeout": 600, "retries": 2, "results_name": "results_quick.json"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return _worker(args.worker)
    drive(_config(full=args.full))


if __name__ == "__main__":
    main()
