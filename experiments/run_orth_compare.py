"""Run the orthogonal-manifolds experiment in BOTH ground-truth regimes and
emit a side-by-side comparison plot.

Condition A: planted subspaces mutually orthogonal (disjoint ambient blocks).
Condition B: planted subspaces are independent random projections (they overlap).

Same manifolds, same architecture, same training — only the ground-truth
subspace geometry differs. Tiny scale so it finishes fast on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.orthogonal_manifolds import Config, main

# Tiny, fast config. 4 diverse manifolds: line, parabola, half-arc, circle.
BASE = dict(
    d_ambient=32,
    curve_indices=(0, 1, 8, 9),
    n_samples=2000,
    sparsity=0.25,
    noise=0.02,
    n_basis=10,
    intrinsic_rank=2,
    top_k=2,
    slack_features=2,
    n_steps=200,
    batch_size=128,
    curriculum_steps=100,
    lr=2e-3,
    seed=0,
    plot=True,
)

CONDS = [
    ("orthogonal", True, "runs/orth_compare/orthogonal"),
    ("overlapping", False, "runs/orth_compare/overlapping"),
]


def _summary(path):
    r = json.load(open(path))
    return r


def run() -> None:
    results = {}
    for name, orth, outdir in CONDS:
        print(f"\n{'='*60}\n== CONDITION: {name} (orthogonal={orth})\n{'='*60}", flush=True)
        cfg = Config(orthogonal=orth, output_dir=outdir, **BASE)
        main(cfg)
        results[name] = _summary(Path(outdir) / "orthogonal_manifolds_results.json")

    # Comparison table.
    print(f"\n\n{'='*60}\n== COMPARISON\n{'='*60}")
    print(f"{'metric':<22} {'orthogonal':>12} {'overlapping':>12}")
    print("-" * 48)
    for key, label in [
        ("explained_variance", "explained_var"),
        ("chamfer_mean", "chamfer_mean (↓)"),
        ("subspace_cos_mean", "subspace_cos (↑)"),
        ("leakage_mean", "leakage_mean (↓)"),
        ("leakage_max", "leakage_max (↓)"),
    ]:
        a, b = results["orthogonal"][key], results["overlapping"][key]
        print(f"{label:<22} {a:>12.3f} {b:>12.3f}")

    _comparison_plot(results)

    print(f"\nPlots:")
    for _, _, outdir in CONDS:
        print(f"  {outdir}/synthetic_recovery_curves.png  (per-feature recovery)")
    print(f"  runs/orth_compare/comparison.png  (metrics bar chart)")


def _comparison_plot(results) -> None:
    """Grouped bar chart comparing recovery metrics across the two regimes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("explained_variance", "explained var ↑"),
        ("chamfer_mean", "chamfer ↓"),
        ("subspace_cos_mean", "subspace cos ↑"),
        ("leakage_mean", "leakage ↓"),
    ]
    orth = [results["orthogonal"][k] for k, _ in metrics]
    over = [results["overlapping"][k] for k, _ in metrics]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, orth, w, label="orthogonal GT subspaces", color="C0")
    b2 = ax.bar(x + w / 2, over, w, label="overlapping GT subspaces", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_title("Manifold-SAE recovery: orthogonal vs overlapping ground truth")
    ax.legend()
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = Path("runs/orth_compare/comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    run()
