"""auto_exp_61: Wasserstein Dictionary SAE on cogito-L40.

Pipeline
--------
1. Train (or load) WassersteinSAE F=128, M=64 on X_L40 (calls trainer).
2. Report val R² + per-atom hue-arc compactness vs F=512 Manifold-SAE
   baseline R²=0.913.
3. Plot every atom's learned hue distribution.
4. For 5 chosen xkcd colors → encode → show simplex weight distribution
   over atoms (which atoms got which weight) + barycenter histogram.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.wasserstein_sae import WassersteinSAE  # noqa: E402

OUT_ROOT = ROOT / "runs" / "AUTO_EXP_61_WASSERSTEIN"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "runs" / "WASSERSTEIN_SAE_F128_M64"


def maybe_train():
    ckpt = CKPT_DIR / "checkpoint.pt"
    if ckpt.exists():
        print(f"[exp61] reusing checkpoint at {ckpt}")
        return
    print("[exp61] no checkpoint found — running trainer")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_wasserstein_sae.py"),
           "--F", "128", "--M", "64", "--epochs", "10",
           "--out", str(CKPT_DIR)]
    subprocess.check_call(cmd, cwd=str(ROOT))


def load_model() -> WassersteinSAE:
    state = torch.load(CKPT_DIR / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = WassersteinSAE(F=cfg["F"], M=cfg["M"], D=cfg["D"], eps=cfg["eps"])
    model.load_state_dict(state["model"])
    model.eval()
    return model


def load_xkcd() -> list[tuple[str, str]]:
    out = []
    for line in (ROOT / "experiments" / "xkcd_colors.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def hex_to_hsv(hx: str) -> tuple[float, float, float]:
    hx = hx.lstrip("#")
    r = int(hx[0:2], 16) / 255.0
    g = int(hx[2:4], 16) / 255.0
    b = int(hx[4:6], 16) / 255.0
    import colorsys
    return colorsys.rgb_to_hsv(r, g, b)


def plot_atoms(model: WassersteinSAE, top_colors_per_atom: dict[int, list[str]]):
    import matplotlib.pyplot as plt
    atoms = model.atoms().detach().cpu().numpy()
    compact = model.atom_compactness().detach().cpu().numpy()
    F, M = atoms.shape
    cols = 8
    rows = (F + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.4))
    axes = np.atleast_2d(axes)
    angles_deg = np.linspace(0, 360, M, endpoint=False)
    for k in range(F):
        ax = axes[k // cols, k % cols]
        ax.bar(angles_deg, atoms[k], width=360 / M, color="C0")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, atoms[k].max() * 1.1 + 1e-6)
        ax.set_xticks([])
        ax.set_yticks([])
        top = ", ".join(top_colors_per_atom.get(k, [])[:3])
        ax.set_title(f"atom {k} c={compact[k]:.2f}\n{top}", fontsize=5)
    for j in range(F, rows * cols):
        axes[j // cols, j % cols].axis("off")
    plt.tight_layout()
    out = OUT_ROOT / "atoms_hue_distributions.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def find_top_colors_per_atom(model, X, xkcd, n_per_atom: int = 10):
    """For each atom k, find the xkcd colors whose centroid activates it most."""
    # Build per-color centroid: average over the 28 templates.
    N, D = X.shape
    n_colors = len(xkcd)
    n_templates = N // n_colors
    centroids = np.empty((n_colors, D), dtype=np.float32)
    for c in range(n_colors):
        sl = slice(c * n_templates, (c + 1) * n_templates)
        centroids[c] = np.asarray(X[sl], dtype=np.float32).mean(0)
    with torch.no_grad():
        pi = model.encode(torch.from_numpy(centroids))   # (n_colors, F)
    pi = pi.numpy()
    top: dict[int, list[str]] = {}
    for k in range(model.F):
        order = np.argsort(-pi[:, k])[:n_per_atom]
        top[k] = [xkcd[i][0] for i in order]
    return top, pi, centroids


def plot_barycenter_examples(model: WassersteinSAE, xkcd, centroids, pi_all,
                             chosen_names: list[str]):
    import matplotlib.pyplot as plt
    name_to_idx = {n: i for i, (n, _) in enumerate(xkcd)}
    rows = len(chosen_names)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.4 * rows))
    if rows == 1:
        axes = axes[None, :]
    angles_deg = np.linspace(0, 360, model.M, endpoint=False)
    atoms = model.atoms().detach().cpu().numpy()
    for r, name in enumerate(chosen_names):
        if name not in name_to_idx:
            print(f"[exp61]  warn: {name} not found in xkcd list — skipping")
            continue
        ci = name_to_idx[name]
        hx = xkcd[ci][1]
        h, s, v = hex_to_hsv(hx)
        xc = torch.from_numpy(centroids[ci:ci+1])
        with torch.no_grad():
            out = model(xc)
        bary = out["bary"].cpu().numpy()[0]
        pi = out["pi"].cpu().numpy()[0]
        top_atoms = np.argsort(-pi)[:8]

        # col 0: barycenter histogram on hue circle (+ true HSV hue marker)
        ax = axes[r, 0]
        ax.bar(angles_deg, bary, width=360 / model.M, color=hx)
        ax.axvline(h * 360, color="black", linestyle="--", lw=1,
                   label=f"true hue {h*360:.0f}°")
        ax.set_title(f"{name} ({hx}) — barycenter")
        ax.set_xlim(0, 360); ax.set_xlabel("hue°"); ax.legend(fontsize=7)

        # col 1: atom weights (top 8)
        ax = axes[r, 1]
        ax.bar(range(len(top_atoms)), pi[top_atoms], color="C2")
        ax.set_xticks(range(len(top_atoms)))
        ax.set_xticklabels([str(a) for a in top_atoms], fontsize=7)
        ax.set_title(f"top-8 atom weights (Σ={pi[top_atoms].sum():.2f})")
        ax.set_ylabel("π_k")

        # col 2: stacked atom distributions for top 4
        ax = axes[r, 2]
        for a in top_atoms[:4]:
            ax.plot(angles_deg, atoms[a], label=f"atom {a} (π={pi[a]:.2f})", lw=1.2)
        ax.set_xlim(0, 360); ax.legend(fontsize=7)
        ax.set_title("top-4 atom hue distributions")
    plt.tight_layout()
    out = OUT_ROOT / "barycenter_examples.png"
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def main():
    maybe_train()
    model = load_model()
    print(f"[exp61] loaded WassersteinSAE F={model.F} M={model.M}")

    X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
    xkcd = load_xkcd()
    print(f"[exp61] xkcd colors: {len(xkcd)}; X shape: {X.shape}")

    top, pi_all, centroids = find_top_colors_per_atom(model, X, xkcd, n_per_atom=10)

    # Val R² on a held-out slice (deterministic split: last 5%)
    N = X.shape[0]
    val_idx = np.arange(N - int(0.05 * N), N)
    sse, sst = 0.0, 0.0
    # use 4K batch mean as approximate global mean
    mean_chunk = np.asarray(X[val_idx[:2048]], dtype=np.float32).mean(0)
    bs = 256
    with torch.no_grad():
        for i in range(0, len(val_idx), bs):
            batch = np.asarray(X[val_idx[i:i+bs]], dtype=np.float32)
            t = torch.from_numpy(batch)
            recon = model(t)["recon"].numpy()
            sse += float(((recon - batch) ** 2).sum())
            sst += float(((batch - mean_chunk) ** 2).sum())
    r2 = 1.0 - sse / max(sst, 1e-9)
    compact = model.atom_compactness().detach().cpu().numpy()
    print(f"[exp61] val_R2={r2:.4f}  mean_compactness={compact.mean():.3f}  "
          f"vs baseline R²=0.913 at F=512 (this F=128, 4× fewer atoms)")

    atoms_plot = plot_atoms(model, top)
    chosen = ["red", "blue", "green", "purple", "yellow"]
    bary_plot = plot_barycenter_examples(model, xkcd, centroids, pi_all, chosen)

    report = {
        "F": model.F,
        "M": model.M,
        "D": model.D,
        "val_R2": r2,
        "mean_compactness": float(compact.mean()),
        "compactness_min": float(compact.min()),
        "compactness_max": float(compact.max()),
        "baseline_R2_F512_manifold_sae": 0.913,
        "figs": {
            "atoms": str(atoms_plot),
            "barycenters": str(bary_plot),
        },
        "top_colors_per_atom": {str(k): v[:5] for k, v in top.items()},
    }
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[exp61] report → {OUT_ROOT / 'report.json'}")
    print(f"[exp61] atoms plot → {atoms_plot}")
    print(f"[exp61] barycenter examples → {bary_plot}")


if __name__ == "__main__":
    main()
