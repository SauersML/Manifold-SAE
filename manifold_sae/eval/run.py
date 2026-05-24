"""CLI:
    python -m manifold_sae.eval.run \
        --models runs/sae_comparison/model_topk.pt runs/sae_comparison/model_l1.pt runs/sae_comparison/model_manifold.pt \
        --data runs/COLOR_COGITO_L40/X_L40.npy \
        --output runs/eval/leaderboard.json

Also emits:
    runs/eval/leaderboard.md
    runs/eval/leaderboard.png   (radar plot)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .harness import Harness, HarnessLabels, HarnessResult
from .registry import loader_for
from . import baselines as bl


# ---------------------------------------------------------------------------
# Data loading (matches scripts/train_sae_comparison.py conventions)
# ---------------------------------------------------------------------------

N_COLORS = 949
N_TPL = 28


def load_xkcd_colors(root: Path):
    p = root / "experiments" / "xkcd_colors.txt"
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name, hex_ = parts[0], parts[1].lstrip("#")
            r = int(hex_[0:2], 16); g = int(hex_[2:4], 16); b = int(hex_[4:6], 16)
            out.append((name, r, g, b))
    return out[:N_COLORS]


def _rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(1); mn = rgb.min(1); df = mx - mn
    h = np.zeros_like(mx)
    mask = df > 1e-8
    rm = mask & (mx == r); gm = mask & (mx == g); bm = mask & (mx == b)
    h[rm] = ((g[rm] - b[rm]) / df[rm]) % 6
    h[gm] = ((b[gm] - r[gm]) / df[gm]) + 2
    h[bm] = ((r[bm] - g[bm]) / df[bm]) + 4
    h = h / 6.0
    s = np.where(mx > 1e-8, df / np.maximum(mx, 1e-8), 0.0)
    v = mx
    return np.stack([h, s, v], 1)


def prepare(root: Path, X_path: Path, device: str, max_val_rows: int | None):
    X = np.load(X_path, mmap_mode="r")
    N, D = X.shape
    rng = np.random.default_rng(0)
    color_perm = rng.permutation(N_COLORS)
    n_val_colors = int(0.2 * N_COLORS)
    val_colors = set(color_perm[:n_val_colors].tolist())
    train_colors = set(color_perm[n_val_colors:].tolist())
    row_color = np.arange(N) // N_TPL
    train_idx = np.where(np.isin(row_color, list(train_colors)))[0]
    val_idx = np.where(np.isin(row_color, list(val_colors)))[0]
    X_train_np = np.ascontiguousarray(X[train_idx]).astype(np.float32)
    X_val_np = np.ascontiguousarray(X[val_idx]).astype(np.float32)
    mu = X_train_np.mean(0)
    X_train_np -= mu
    X_val_np -= mu

    if max_val_rows is not None and X_val_np.shape[0] > max_val_rows:
        sub = np.random.default_rng(0).choice(X_val_np.shape[0], max_val_rows, replace=False)
        X_val_np = X_val_np[sub]
        val_idx = val_idx[sub]

    X_val = torch.from_numpy(X_val_np).to(device)

    # Labels for the validation set.
    xkcd = load_xkcd_colors(root)
    color_names = [c[0] for c in xkcd]
    color_rgb = np.array([(r / 255.0, g / 255.0, b / 255.0) for _, r, g, b in xkcd], dtype=np.float32)
    color_hsv = _rgb_to_hsv(color_rgb)

    row_color_idx = (val_idx // N_TPL).astype(np.int64)
    row_hue = color_hsv[row_color_idx, 0].astype(np.float32)

    # modifier_count + monoword from color name (template_idx % N_TPL would give
    # template variation if available; here we use the name itself as proxy).
    row_mod_count = np.array(
        [len(color_names[c].split()) - 1 for c in row_color_idx], dtype=np.int64
    )
    row_mono = (row_mod_count == 0).astype(np.int64)

    # Concept labels: top-7 color buckets (red/orange/yellow/green/cyan/blue/purple).
    concept_labels = _hue_bucket_labels(row_hue)

    labels = HarnessLabels(
        row_color_idx=row_color_idx,
        color_hsv=color_hsv,
        color_rgb=color_rgb,
        row_hue=row_hue,
        row_modifier_count=row_mod_count,
        row_monoword=row_mono,
        concept_labels=concept_labels,
    )
    return X_train_np, X_val, labels, D


def _hue_bucket_labels(hue: np.ndarray, n_buckets: int = 7) -> np.ndarray:
    edges = np.linspace(0, 1, n_buckets + 1)
    bucket = np.digitize(hue, edges[1:-1])
    labels = np.zeros((hue.size, n_buckets), dtype=bool)
    labels[np.arange(hue.size), bucket] = True
    return labels


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------


def _flatten_for_table(result: HarnessResult) -> dict[str, float]:
    m = result.metrics
    out = {
        "model": result.model_name,
        "val_r2": m.get("val_r2", float("nan")),
        "L0": m["sparsity"]["L0"],
        "L1": m["sparsity"]["L1"],
        "gini": m["sparsity"]["gini"],
        "mean_active": m["sparsity"]["mean_active_fraction"],
        "dead_frac": m.get("dead_atom_fraction", float("nan")),
        "n_active": m.get("n_active_atoms", 0),
        "hsv_coh": m.get("hsv_coherence", {}).get("mean_top20_coherence", float("nan")),
        "manifold_dim": m.get("manifold_dim", {}).get("mean_effective_rank", float("nan")),
        "abs_rate": m.get("feature_absorption", {}).get("mean_absorption", float("nan")),
        "abl_dR2": m.get("ablation", {}).get("mean_delta_r2", float("nan")),
        "probe_hsv": m.get("probes", {}).get("hsv", {}).get("r2_mean",
                       m.get("probes", {}).get("hsv", {}).get("r2", float("nan"))),
        "probe_modcount": m.get("probes", {}).get("modifier_count", {}).get("r2", float("nan")),
        "probe_mono_acc": m.get("probes", {}).get("monoword", {}).get("accuracy", float("nan")),
        "steer_cos": m.get("steering", {}).get("steering_cosine", float("nan")),
        "r2_per_flop": m.get("r2_per_flop", float("nan")),
        "r2_per_active": m.get("r2_per_active_atom", float("nan")),
    }
    return out


def render_markdown(rows: list[dict], path: Path) -> None:
    cols = list(rows[0].keys())
    pretty = {
        "model": "Model", "val_r2": "R²", "L0": "L0", "L1": "L1", "gini": "Gini",
        "mean_active": "Active%", "dead_frac": "Dead%", "n_active": "N_active",
        "hsv_coh": "HSV-coh", "manifold_dim": "Mfd-dim", "abs_rate": "Absorp",
        "abl_dR2": "Abl ΔR²", "probe_hsv": "P:hsv", "probe_modcount": "P:mod",
        "probe_mono_acc": "P:mono", "steer_cos": "Steer", "r2_per_flop": "R²/FLOP",
        "r2_per_active": "R²/atom",
    }
    header = [pretty.get(c, c) for c in cols]
    with open(path, "w") as f:
        f.write("# SAE Evaluation Leaderboard\n\n")
        f.write("Generated by `manifold_sae.eval.run`. Higher is better for R²,"
                " HSV-coh, probes, steer cos, R²/FLOP, R²/atom.\n"
                "Lower is better for Dead%, Absorp, Mfd-dim (manifold atoms ≈1),"
                " Gini for non-TopK (TopK gini is inherent).\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    if c in ("L0", "L1", "n_active"):
                        cells.append(f"{v:.1f}")
                    elif c in ("r2_per_flop", "r2_per_active"):
                        cells.append(f"{v:.2e}")
                    else:
                        cells.append(f"{v:.3f}")
                elif isinstance(v, int):
                    cells.append(str(v))
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")


def render_radar(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Normalize axes that are bounded already; clamp others.
    axes = [
        ("val_r2", False),
        ("hsv_coh", False),
        ("probe_hsv", False),
        ("steer_cos", False),
        ("1-dead_frac", False),
        ("1-abs_rate", False),
    ]
    def val(row, key):
        if key.startswith("1-"):
            return float(np.nan_to_num(1.0 - row[key[2:]], nan=0.0))
        return float(np.nan_to_num(row[key], nan=0.0))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]
    for row in rows:
        vals = [max(0.0, min(1.0, val(row, k))) for k, _ in axes]
        vals += vals[:1]
        ax.plot(angles, vals, label=row["model"], lw=2)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a for a, _ in axes])
    ax.set_ylim(0, 1)
    ax.set_title("SAE leaderboard (radar; higher = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(models: list[str], data: str, output: str, device: str = "cpu",
        include_baselines: bool = True, max_val_rows: int | None = 1500,
        ablation_subset: int = 32) -> dict:
    root = Path(__file__).resolve().parents[2]
    X_train_np, X_val, labels, D = prepare(root, Path(data), device=device, max_val_rows=max_val_rows)
    results: list[HarnessResult] = []

    for path in models:
        loader = loader_for(path)
        print(f"[harness] loading {path}", flush=True)
        wrapper = loader(path, d_in=D, device=device)
        h = Harness(wrapper, X_val, labels=labels, ablation_subset=ablation_subset)
        print(f"[harness] scoring {wrapper.name}", flush=True)
        results.append(h.run())

    if include_baselines:
        print("[harness] baselines", flush=True)
        for w in [
            bl.pca_baseline(X_train_np, n_features=min(512, D), name="PCA-512", device=device),
            bl.random_projection_baseline(D, n_features=min(512, D), name="RandomProj-512", device=device),
            bl.identity_baseline(D, name="Identity", device=device),
        ]:
            try:
                results.append(Harness(w, X_val, labels=labels, ablation_subset=ablation_subset).run())
            except Exception as e:
                print(f"[harness] baseline {w.name} failed: {e}", flush=True)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, dict):
            # JSON keys must be strings.
            return {str(k): _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(v) for v in o]
        return o

    out_json = {r.model_name: _to_jsonable(r.metrics) for r in results}
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)

    rows = [_flatten_for_table(r) for r in results]
    render_markdown(rows, out_path.with_suffix(".md"))
    try:
        render_radar(rows, out_path.with_suffix(".png"))
    except Exception as e:
        print(f"[harness] radar render failed: {e}", flush=True)
    print(f"[harness] wrote {out_path} + .md + .png", flush=True)
    return out_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--max-val-rows", type=int, default=1500)
    ap.add_argument("--ablation-subset", type=int, default=32)
    args = ap.parse_args()
    run(args.models, args.data, args.output, device=args.device,
        include_baselines=not args.no_baselines, max_val_rows=args.max_val_rows,
        ablation_subset=args.ablation_subset)


if __name__ == "__main__":
    main()
