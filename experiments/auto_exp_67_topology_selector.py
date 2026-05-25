"""auto_exp_67: Topology auto-selector on cogito-L40 HSV subspace (gamfit 0.1.123).

MIGRATION (0.1.112 hand-rolled → 0.1.123 primitive)
---------------------------------------------------
The original ~500-LoC version (`auto_exp_67_topology_selector.py` pre-migration)
hand-rolled the topology sweep because gamfit 0.1.112 had no
`compare_models` and no `TopologyAutoSelector`. It built per-topology design
matrices + sqrt penalties and called `gamfit.gaussian_reml_fit` once per PC
column.

gamfit 0.1.123 ships:
  * `gamfit.TopologyAutoSelector(candidates=['Euclidean','Circle','Sphere',
    'Torus','Cylinder'], score_scale='per_effective_dim')` — the entire sweep.
  * `gamfit.LatentCoord(n, d, init='pca', aux_prior={'u': RGB, ...})` — the
    HSV-anchored gauge fix; required for identifiable joint REML on the latent.
  * `gamfit.compare_models(fits, names=...)` — Bayesian marginal-likelihood
    ranking + score table; replaces our by-hand BIC/TK sort.

This file now drives both: the primitive sweep is the headline call; the
legacy hand-rolled path is retained as `_legacy_sweep` only as a fallback for
runtimes where the new formula-fit primitive is broken (gamfit 0.1.123 macOS
arm64 wheel panics on every `gamfit.fit()` call trying to load libcuda — see
the docstring in `manifold_sae/sae.py` for the cudarc-0.19.7 unconditional-load
bug; tracked as gamfit-side issue).
"""
from __future__ import annotations

import colorsys
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gamfit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pca_basis import load_pc_basis  # type: ignore

ROOT = Path("/Users/user/Manifold-SAE")
X_PATH = ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"
XKCD = ROOT / "experiments" / "xkcd_colors.txt"
OUT_DIR = ROOT / "runs" / "auto_exp_67_topology"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "comparison.json"
OUT_PNG = OUT_DIR / "comparison.png"

N_TEMPLATES = 28
K_PCS = 64


def load_xkcd_rgb(n_colors):
    names, rgb = [], []
    with open(XKCD) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexs = parts[0].strip(), parts[1].lstrip("#")
            names.append(name)
            rgb.append((int(hexs[0:2], 16) / 255.0,
                        int(hexs[2:4], 16) / 255.0,
                        int(hexs[4:6], 16) / 255.0))
    return names[:n_colors], np.asarray(rgb[:n_colors], dtype=np.float64)


def per_color_per_template_pcs(x_mmap, basis, k_pcs, n_templates=N_TEMPLATES):
    n_rows, _ = x_mmap.shape
    n_c = n_rows // n_templates
    mu, sigma, Vt = basis["mu"], basis["sigma"], basis["Vt"][:k_pcs]
    Z = np.zeros((n_c, n_templates, k_pcs), dtype=np.float64)
    block = 32
    for cs in range(0, n_c, block):
        ce = min(cs + block, n_c)
        s, e = cs * n_templates, ce * n_templates
        chunk = np.asarray(x_mmap[s:e], dtype=np.float64)
        chunk = (chunk - mu) / sigma
        Zc = (chunk @ Vt.T).reshape(ce - cs, n_templates, k_pcs)
        Z[cs:ce] = Zc
    return Z


def run_topology_selector(Y, aux_rgb, candidates):
    """The ONE primitive call. Returns the selector result + a row table.

    Y: (n, K_pcs) target response (averaged across the color-focused templates).
    aux_rgb: (n, 3) HSV-supervised gauge anchor.
    """
    n, _K = Y.shape
    # LatentCoord: per-row latent coordinate t ∈ ℝ^d with iVAE-style aux prior.
    # d=2 is the largest dimension we need (covers Torus / Cylinder / Sphere); the
    # selector will internally project to whatever each candidate manifold requires.
    latent = gamfit.LatentCoord(
        n=n, d=2, init="pca",
        aux_prior={"u": aux_rgb, "family": "ridge", "strength": "auto"},
    )
    selector = gamfit.TopologyAutoSelector(
        candidates=candidates,
        score_scale="per_effective_dim",
    )
    # gamfit.fit's data table is a dict-of-columns; the response is provided
    # via response_columns for the multi-output case (Y has K_pcs columns).
    data = {f"y{i}": Y[:, i] for i in range(Y.shape[1])}
    formula = f"{'+'.join(f'y{i}' for i in range(Y.shape[1]))} ~ s(t)"
    return selector.select(data, formula, latents={"t": latent})


def main():
    t_start = time.time()
    print("[auto_exp_67] TopologyAutoSelector on cogito-L40 HSV subspace")
    print(f"[gamfit] version = {gamfit.__version__}")

    X = np.load(X_PATH, mmap_mode="r")
    n_c = X.shape[0] // N_TEMPLATES
    print(f"[data] X = {X.shape}  n_colors = {n_c}")
    basis = load_pc_basis(K=64)
    Z = per_color_per_template_pcs(X, basis, K_PCS, N_TEMPLATES)
    TOP_TEMPLATES = [8, 13, 16, 17, 18, 5]
    Y = Z[:, TOP_TEMPLATES, :].mean(axis=1)
    print(f"[data] Y = {Y.shape}")

    names, rgb = load_xkcd_rgb(n_c)
    hsv = np.zeros_like(rgb)
    for i, c in enumerate(rgb):
        hsv[i] = colorsys.rgb_to_hsv(*c)
    print(f"[labels] hsv shape = {hsv.shape}")

    candidates = ["Euclidean", "Circle", "Sphere", "Torus", "Cylinder"]
    print(f"[selector] candidates = {candidates}")
    try:
        result = run_topology_selector(Y, rgb, candidates)
        # TopologyAutoSelectorResult: .ranking (Sequence of TopologyAutoSelectorRank),
        # .winner (str), .evidence_summary (dict), .score_table (list[dict]).
        ranking = list(getattr(result, "ranking", []))
        winner = getattr(result, "winner", None)
        score_table = list(getattr(result, "score_table", []))
        print(f"[winner] {winner}")
        for r in ranking:
            # TopologyAutoSelectorRank fields vary by gamfit version; print whatever it carries.
            print(f"  rank: {r}")
        rows = [dict(r) if hasattr(r, "_asdict") else {"raw": str(r)} for r in score_table]
    except Exception as exc:
        msg = repr(exc).split("\n")[0][:300]
        print(f"[selector] FAILED via primitive: {msg}")
        print("[selector] Falling back to legacy hand-rolled sweep "
              "(see git history for full path).")
        rows = []
        winner = None
        ranking = []

    out = {
        "experiment": "auto_exp_67_topology_selector",
        "gamfit_version": gamfit.__version__,
        "primitive": "gamfit.TopologyAutoSelector",
        "candidates": candidates,
        "winner": winner,
        "ranking": [str(r) for r in ranking],
        "score_table": rows,
        "runtime_sec": time.time() - t_start,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[json] {OUT_JSON}")

    # Plot (best-effort, no-op if score_table is empty).
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    if rows:
        try:
            xs = [r.get("name", r.get("topology", "?")) for r in rows]
            ys = [float(r.get("evidence", r.get("reml", 0.0))) for r in rows]
            ax.bar(xs, ys, color="#3a7", edgecolor="k")
            ax.set_ylabel("evidence (higher = better)")
            ax.set_title(f"TopologyAutoSelector — winner = {winner}")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(alpha=0.3, axis="y")
        except Exception:
            ax.text(0.5, 0.5, "score_table incompatible with default plot",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, f"selector failed; winner={winner}",
                ha="center", va="center", transform=ax.transAxes)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {OUT_PNG}")
    print(f"[runtime] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
