"""auto_exp_75: Cylinder-native Manifold-SAE on cogito-L40.

Hypothesis (from auto_exp_67 topology selector):
    Cylinder (S^1 x R) is the right per-atom topology for cogito-L40.
    Pure S^1 (current ManifoldSAE periodic mode) loses by ΔREML=697.

Falsifiable claim:
    Cylinder-SAE (F=512, H=3, K_ell=4) reaches val R² ≥ 0.913 — matching or
    beating the Manifold-SAE baseline reported in
    paper/manifold_sae/manifold_sae.tex (val R²=0.913, F=512 cogito-L40).

Compute policy:
    THIS SCRIPT ONLY LAUNCHES THE TRAINER. By default it does NOT spawn a
    new training process — it reads the existing
    runs/CYLINDER_SAE_COGITO/metrics.json that train_cylinder_sae.py
    produces. To force a fresh launch, set CYLINDER_SAE_FORCE_TRAIN=1.

Outputs:
    runs/auto_exp_75_cylinder/comparison.json
    runs/auto_exp_75_cylinder/comparison.png  (2-panel: train curves + R² bars)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "CYLINDER_SAE_COGITO"
METRICS = RUN_DIR / "metrics.json"
OUT_DIR = ROOT / "runs" / "auto_exp_75_cylinder"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_MANIFOLD_SAE_R2 = 0.913   # paper headline
BASELINE_L1_R2 = 0.882
BASELINE_TOPK_R2 = 0.874


def maybe_train() -> None:
    """Run trainer iff metrics.json missing AND env var requests it."""
    if METRICS.exists():
        print(f"[exp75] using cached {METRICS}", flush=True)
        return
    if os.environ.get("CYLINDER_SAE_FORCE_TRAIN", "0") != "1":
        raise SystemExit(
            f"[exp75] {METRICS} missing and CYLINDER_SAE_FORCE_TRAIN!=1; "
            f"refusing to auto-launch trainer. Run "
            f"`uv run python scripts/train_cylinder_sae.py` first."
        )
    cmd = [sys.executable, str(ROOT / "scripts" / "train_cylinder_sae.py")]
    print(f"[exp75] launching trainer: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    maybe_train()
    with open(METRICS) as f:
        m = json.load(f)

    final_r2 = float(m["final_val_r2"])
    train_curve = m["train_loss_curve"]
    val_r2_curve = m["val_r2_curve"]
    k_eff = float(m["final_k_eff"])
    dead = float(m["final_dead_rate"])
    params = int(m["config"]["params"])

    comparison = {
        "cylinder_sae": {
            "val_r2": final_r2,
            "k_eff": k_eff,
            "dead_rate": dead,
            "params": params,
            "config": m["config"],
        },
        "baselines": {
            "manifold_sae_paper": BASELINE_MANIFOLD_SAE_R2,
            "l1_paper": BASELINE_L1_R2,
            "topk_paper": BASELINE_TOPK_R2,
        },
        "delta_vs_manifold_sae": final_r2 - BASELINE_MANIFOLD_SAE_R2,
        "wins_vs_manifold_sae": final_r2 >= BASELINE_MANIFOLD_SAE_R2,
    }
    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Two-panel plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    epochs = np.arange(1, len(train_curve) + 1)
    ax.plot(epochs, train_curve, "o-", label="train loss", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(epochs, val_r2_curve, "s-", label="val R²", color="#d62728")
    ax2.axhline(BASELINE_MANIFOLD_SAE_R2, ls="--", color="gray",
                label=f"Manifold-SAE paper R²={BASELINE_MANIFOLD_SAE_R2}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss", color="#1f77b4")
    ax2.set_ylabel("val R²", color="#d62728")
    ax.set_title("Cylinder-SAE training")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=8)

    ax = axes[1]
    names = ["Cylinder-SAE\n(F=512)", "Manifold-SAE\n(paper)", "TopK\n(paper)", "L1\n(paper)"]
    vals = [final_r2, BASELINE_MANIFOLD_SAE_R2, BASELINE_TOPK_R2, BASELINE_L1_R2]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("val R²")
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.03)
    ax.set_title("Cylinder-SAE vs baselines (cogito-L40)")
    ax.axhline(BASELINE_MANIFOLD_SAE_R2, ls="--", color="gray", alpha=0.5)

    plt.tight_layout()
    fig_path = OUT_DIR / "comparison.png"
    plt.savefig(fig_path, dpi=140)
    plt.close(fig)

    print(json.dumps(comparison, indent=2), flush=True)
    print(f"[exp75] figure -> {fig_path}", flush=True)
    print(f"[exp75] cylinder val R² = {final_r2:.4f}  "
          f"(Δ vs Manifold-SAE 0.913 = {final_r2 - BASELINE_MANIFOLD_SAE_R2:+.4f})", flush=True)


if __name__ == "__main__":
    main()
