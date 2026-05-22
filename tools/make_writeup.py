"""Minimal figures — data only, no decoration."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.5,
})

VANILLA  = "#D55E00"
MANIFOLD = "#0072B2"

OUT = Path("runs/figures"); OUT.mkdir(parents=True, exist_ok=True)


def fig1_cross_scale():
    runs = [
        ("Qwen 0.5B",  "runs_cluster/llm_sweep_0.5B_L18_perdim_fast"),
        ("Qwen 1.5B", "runs_cluster/llm_sweep_1.5B_L18_perdim_fast"),
        ("Qwen 3B",    "runs_cluster/llm_sweep_3B_L18_perdim_fast"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (label, run_dir) in zip(axes, runs):
        Fs, van, crv = [], [], []
        for f in sorted(glob.glob(f"{run_dir}/eval_F*.json")):
            d = json.load(open(f))["result"]
            Fs.append(d["F"]); van.append(d["vanilla_explained"]); crv.append(d["curve_explained"])
        order = np.argsort(Fs)
        Fs = [Fs[i] for i in order]; van = [van[i] for i in order]; crv = [crv[i] for i in order]
        ax.plot(Fs, van, "o-", color=VANILLA, linewidth=2.5, markersize=12, label="standard")
        ax.plot(Fs, crv, "s-", color=MANIFOLD, linewidth=2.5, markersize=12, label="curve")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Fs); ax.set_xticklabels(Fs)
        ax.set_title(label, fontsize=14)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 0.6)
    axes[0].set_ylabel("variance recovered", fontsize=13)
    axes[1].set_xlabel("dictionary size", fontsize=13)
    axes[0].legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_cross_scale.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_intrinsic_dim():
    d = json.load(open("runs_cluster/concept_intrinsic_dim_q15b/intrinsic_dim.json"))
    results = d["results"]
    layers = sorted([int(L[1:]) for L in results.keys()])
    concepts = ["magnitude", "brightness", "temperature"]
    colors = {"magnitude": "#0072B2", "brightness": "#009E73", "temperature": "#D55E00"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in concepts:
        ys = [results.get(f"L{L}", {}).get(c, {}).get("corr_dim", np.nan) for L in layers]
        ax.plot(layers, ys, "o-", color=colors[c], linewidth=2.5, markersize=12, label=c)
    ax.axhline(1, linestyle="--", color="#666", linewidth=1.5)
    ax.set_xlabel("layer", fontsize=13)
    ax.set_ylabel("concept dimensionality", fontsize=13)
    ax.set_xticks(layers)
    ax.set_ylim(0.5, 4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_intrinsic_dim.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_2d_ablation():
    rows = []
    for src in ["v2", "v3"]:
        s = json.load(open(f"runs_cluster/synthetic_2d_{src}/summary.json"))
        for v, r in s.get("results", {}).items():
            if not isinstance(r, dict): continue
            mpg = r.get("mean_per_grid")
            if mpg is None: continue
            label = v.split("_", 1)[1] if "_" in v else v
            rows.append((label, mpg))
    seen = {}
    for l, s in rows:
        if l not in seen or s > seen[l]:
            seen[l] = s
    rows = sorted(seen.items(), key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r[0] for r in rows]; scores = [r[1] for r in rows]
    ax.barh(range(len(rows)), scores, color=MANIFOLD, edgecolor="white", linewidth=1, height=0.7)
    ax.axvline(0.39, color=VANILLA, linewidth=2.5, linestyle="--", label="1D pair")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("recovery score", fontsize=13)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.25, axis="x")
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_2d_ablation.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4_synthetic_1d():
    scenarios = ["small", "medium", "large"]
    van = [0.494, 0.513, 0.452]
    crv = [0.768, 0.760, 0.643]
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.35
    ax.bar(x - w/2, van, w, color=VANILLA, label="standard", edgecolor="white", linewidth=1.5)
    ax.bar(x + w/2, crv, w, color=MANIFOLD, label="curve", edgecolor="white", linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=13)
    ax.set_ylabel("variance recovered", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_synthetic_1d_win.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5_rank_comparison():
    d = json.load(open("runs_cluster/diagnostics_q15b_L18/results.json"))
    rank = d["Q1_rank_report"]
    layers = sorted([int(L) for L in rank.keys()])

    fig, ax = plt.subplots(figsize=(8, 5))
    olds = [rank[str(L)]["global_std"]["n_pcs_for_99pct"] for L in layers]
    news = [rank[str(L)]["per_dim_std"]["n_pcs_for_99pct"] for L in layers]

    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, olds, w, color=VANILLA, label="old", edgecolor="white", linewidth=1.5)
    ax.bar(x + w/2, news, w, color=MANIFOLD, label="fixed", edgecolor="white", linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in layers], fontsize=13)
    ax.set_ylabel("directions for 99% variance", fontsize=13)
    ax.set_ylim(0, 1200)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_normalization_bug.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1_cross_scale()
    fig2_intrinsic_dim()
    fig3_2d_ablation()
    fig4_synthetic_1d()
    fig5_rank_comparison()
    print(f"saved 5 figures to {OUT}/")
