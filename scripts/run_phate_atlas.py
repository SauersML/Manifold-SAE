"""CLI: PHATE / persistent-H1 / Mapper atlas of SAE atoms.

Usage
-----
    uv run python scripts/run_phate_atlas.py \
        --models runs/sae_comparison/model_topk.pt \
                 runs/sae_comparison/model_l1.pt \
                 runs/sae_comparison/model_manifold.pt \
        --output runs/phate_atlas/

For each SAE checkpoint, writes:
    {name}_phate.png        2-D PHATE embedding, atoms colored by hue
    {name}_h1.png           H1 persistence diagram with dominant cycle annotated
    {name}_mapper.dot       Mapper graph (graphviz)
    {name}_summary.json     numeric verdict (dominant-H1-persistence, etc.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.atlas import atom_atlas, persistent_h1, mapper_atlas
from manifold_sae.atlas.phate_atlas import (
    extract_atom_directions,
    hue_label_from_color_centroids,
    mapper_to_dot,
)

N_COLORS = 949
N_TPL = 28


def _load_xkcd():
    p = ROOT / "experiments" / "xkcd_colors.txt"
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, hex_ = line.split("\t")[:2]
            hex_ = hex_.lstrip("#")
            r, g, b = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
            out.append((name, r / 255, g / 255, b / 255))
    return out[:N_COLORS]


def _rgb_to_hue(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(1), rgb.min(1)
    df = mx - mn
    h = np.zeros_like(mx)
    m = df > 1e-8
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & (mx == b) & ~(rm | gm)
    h[rm] = ((g[rm] - b[rm]) / df[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / df[gm] + 2
    h[bm] = (r[bm] - g[bm]) / df[bm] + 4
    return (h / 6.0) % 1.0


def _color_centroids_from_harvest(X_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute (N_COLORS, D) centroid per xkcd color from cogito-L40 harvest +
    the xkcd hues. Returns (centroids, hues)."""
    X = np.load(X_path, mmap_mode="r")
    assert X.shape[0] == N_COLORS * N_TPL, X.shape
    D = X.shape[1]
    centroids = np.zeros((N_COLORS, D), dtype=np.float32)
    for c in range(N_COLORS):
        sl = X[c * N_TPL : (c + 1) * N_TPL]
        centroids[c] = np.asarray(sl, dtype=np.float32).mean(0)
    centroids -= centroids.mean(0, keepdims=True)
    xkcd = _load_xkcd()
    rgb = np.array([[r, g, b] for (_, r, g, b) in xkcd], dtype=np.float32)
    hues = _rgb_to_hue(rgb)
    return centroids, hues


def _plot_phate(atlas: dict, title: str, out_path: Path) -> None:
    emb = atlas["embedding"]
    hues = atlas.get("hue_labels")
    fig, ax = plt.subplots(figsize=(6, 6))
    if hues is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=12, c="steelblue", alpha=0.7)
    else:
        sc = ax.scatter(
            emb[:, 0], emb[:, 1], s=18, c=hues, cmap="hsv", vmin=0, vmax=1, alpha=0.85
        )
        plt.colorbar(sc, ax=ax, label="atom hue (from color centroids)")
    ax.set_title(f"PHATE atom-atlas — {title}\n({atlas['backend']}, F_alive={atlas['n_atoms_alive']})")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_h1(bars: list[tuple[float, float]], title: str, out_path: Path) -> tuple[float, float, float]:
    """Persistence diagram. Returns (dominant_persistence, runner_up, ratio)."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if not bars:
        ax.text(0.5, 0.5, "no H1 bars", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=120); plt.close(fig)
        return 0.0, 0.0, 0.0
    bs = np.array([b for (b, _) in bars])
    ds_raw = np.array([d for (_, d) in bars])
    finite = np.isfinite(ds_raw)
    cap = float(np.nanmax(ds_raw[finite])) if finite.any() else float(np.max(bs) * 2 + 1)
    ds = np.where(finite, ds_raw, cap * 1.2)
    pers = ds - bs
    order = np.argsort(-pers)
    ax.scatter(bs, ds, s=30, c="tab:red", edgecolors="black", linewidth=0.4)
    lim = max(cap * 1.25, float(ds.max()), 1e-3)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4)
    if len(order) > 0:
        i = order[0]
        ax.annotate(
            f"dom. pers = {pers[i]:.3f}",
            xy=(bs[i], ds[i]),
            xytext=(bs[i] + 0.05 * lim, ds[i] + 0.05 * lim),
            arrowprops=dict(arrowstyle="->", lw=0.8),
            fontsize=9,
        )
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(f"H1 persistence — {title}")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    dom = float(pers[order[0]])
    runner = float(pers[order[1]]) if len(order) > 1 else 0.0
    ratio = dom / max(runner, 1e-9)
    return dom, runner, ratio


def process_model(path: Path, out_dir: Path, color_centroids: np.ndarray | None,
                  color_hues: np.ndarray | None) -> dict:
    name = path.stem
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    D_atoms = extract_atom_directions(sd)
    hue_labels = None
    if color_centroids is not None and color_centroids.shape[1] == D_atoms.shape[1]:
        hue_labels = hue_label_from_color_centroids(D_atoms, color_centroids, color_hues)

    atlas = atom_atlas(path, n_components=2, hue_labels=hue_labels)
    _plot_phate(atlas, name, out_dir / f"{name}_phate.png")
    bars = persistent_h1(atlas)
    dom, runner, ratio = _plot_h1(bars, name, out_dir / f"{name}_h1.png")
    mp = mapper_atlas(atlas)
    (out_dir / f"{name}_mapper.dot").write_text(mapper_to_dot(mp, title=name))
    summary = {
        "name": name,
        "model_path": str(path),
        "n_atoms_total": atlas["n_atoms_total"],
        "n_atoms_alive": atlas["n_atoms_alive"],
        "backend": atlas["backend"],
        "n_h1_bars": len(bars),
        "dominant_h1_persistence": dom,
        "runner_up_h1_persistence": runner,
        "dominance_ratio": ratio,
        "mapper_n_nodes": len(mp["nodes"]),
        "mapper_n_edges": len(mp["edges"]),
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--harvest",
        default=str(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy"),
        help="Path to cogito harvest .npy used to compute color centroids for hue labels.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    centroids, hues = None, None
    harvest = Path(args.harvest)
    if harvest.exists():
        try:
            centroids, hues = _color_centroids_from_harvest(harvest)
            print(f"[atlas] loaded color centroids from {harvest}: {centroids.shape}", flush=True)
        except Exception as e:
            print(f"[atlas] WARN failed to load harvest for hue labels: {e}", flush=True)
    else:
        print(f"[atlas] WARN harvest not found at {harvest} — hue labels skipped", flush=True)

    summaries = []
    for m in args.models:
        for p in sorted(Path().glob(m)) if any(c in m for c in "*?[") else [Path(m)]:
            try:
                s = process_model(Path(p), out_dir, centroids, hues)
                print(f"[atlas] {s['name']}: dom_H1={s['dominant_h1_persistence']:.4f} "
                      f"ratio={s['dominance_ratio']:.2f} (backend={s['backend']})", flush=True)
                summaries.append(s)
            except Exception as e:
                print(f"[atlas] ERROR processing {p}: {e}", flush=True)

    (out_dir / "atlas_all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"[atlas] wrote {len(summaries)} summaries to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
