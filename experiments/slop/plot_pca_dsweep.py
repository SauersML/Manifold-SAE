"""Plot the PCA d-sweep — held-out R² vs number of PCs retained.

Reveals the intrinsic dimensionality of cogito's per-color centroid space:
where does the curve plateau? How many dims do you really need?

Overlays the best supervised model's R² so the gap is visible.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> int:
    results_path = Path(os.environ.get(
        "RESULTS_JSON",
        "/Users/user/Manifold-SAE/runs/COLOR_MANIFOLD_GAM_COGITO_L40/results.json",
    ))
    data = json.loads(results_path.read_text())
    specs = next(iter(data["per_layer"].values()))["specs"]

    pca_ds = [2, 3, 4, 8, 16, 24, 32, 48]
    pca_r2 = []
    pca_std = []
    for d in pca_ds:
        key = f"U_pca_{d}d"
        if key in specs and "r2_macro_mean" in specs[key]:
            pca_r2.append(specs[key]["r2_macro_mean"])
            pca_std.append(specs[key]["r2_macro_std"])
        else:
            pca_r2.append(np.nan); pca_std.append(0.0)
    pca_r2 = np.array(pca_r2); pca_std = np.array(pca_std)

    # Pull comparison points
    u3d = specs.get("U_3d", {}).get("r2_macro_mean", np.nan)
    u4d = specs.get("U_4d", {}).get("r2_macro_mean", np.nan)
    u3d_ms = specs.get("U_3d_multistart", {}).get("r2_macro_mean", np.nan)

    sup_specs = [k for k, v in specs.items()
                  if (k.startswith("L_") or k.startswith("M_") or k.startswith("N_"))
                  and isinstance(v, dict) and v.get("r2_macro_mean") is not None
                  and np.isfinite(v["r2_macro_mean"])]
    best_sup = max(sup_specs, key=lambda k: specs[k]["r2_macro_mean"])
    best_sup_r2 = specs[best_sup]["r2_macro_mean"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(pca_ds, pca_r2, yerr=pca_std, fmt="o-", linewidth=2,
                  markersize=10, color="#d68a4f", capsize=4,
                  label="U_pca_kd — linear top-k PC reconstruction")
    # Reference lines
    if np.isfinite(u3d):
        ax.axhline(u3d, color="#4f93bf", linestyle="--", alpha=0.7,
                    label=f"U_3d (nonlinear alternation, d=3) = {u3d:.3f}")
    if np.isfinite(u4d):
        ax.axhline(u4d, color="#356d96", linestyle=":", alpha=0.7,
                    label=f"U_4d (nonlinear alternation, d=4) = {u4d:.3f}")
    if np.isfinite(best_sup_r2):
        ax.axhline(best_sup_r2, color="#aa3333", linestyle="-", alpha=0.7,
                    label=f"best supervised ({best_sup}) = {best_sup_r2:.3f}")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    # Annotate each PCA point
    for d, r2 in zip(pca_ds, pca_r2):
        if np.isfinite(r2):
            ax.annotate(f"{r2:.2f}", (d, r2),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, color="#854d20")

    ax.set_xscale("log", base=2)
    ax.set_xticks(pca_ds)
    ax.set_xticklabels(pca_ds)
    ax.set_xlabel("number of principal components kept", fontsize=12)
    ax.set_ylabel("held-out R²_macro (5-fold CV by color)", fontsize=12)
    ax.set_title(
        "Held-out R² vs PCA dimensionality of cogito's per-color centroid space\n"
        "How many PCs do you need to predict held-out colors?",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    out_path = results_path.parent / "pca_dimensionality_sweep.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
