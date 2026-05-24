"""Train a Crosscoder F=512 on the synthetic 3-layer cogito stack.

Usage
-----
    uv run python scripts/train_crosscoder.py
        [--epochs 15] [--batch 256] [--n-atoms 512]
        [--data-dir runs/COLOR_COGITO_MULTILAYER]
        [--out-dir runs/crosscoder]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_sae.crosscoder import Crosscoder


REPO = Path(__file__).resolve().parents[1]


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_layers(data_dir: Path) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    for i in (1, 2, 3):
        p = data_dir / f"X_l{i}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run scripts/synthesize_multilayer_cogito.py first."
            )
        layers.append(np.load(p, mmap_mode="r"))
    return layers


def normalize_layers(layers: list[np.ndarray]) -> tuple[list[torch.Tensor], list[dict]]:
    """Center + scale each layer to unit per-feature variance (running stats)."""
    out: list[torch.Tensor] = []
    stats: list[dict] = []
    for X in layers:
        mu = X.mean(axis=0, keepdims=True).astype(np.float32)
        sd = X.std(axis=0, keepdims=True).astype(np.float32).clip(min=1e-6)
        Z = (X - mu) / sd
        out.append(torch.from_numpy(np.ascontiguousarray(Z)))
        stats.append({"mu": mu, "sd": sd})
    return out, stats


def train(args: argparse.Namespace) -> dict:
    device = _device()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={device}  data={data_dir}  out={out_dir}")
    layers_np = load_layers(data_dir)
    Xs, _stats = normalize_layers(layers_np)
    N = Xs[0].shape[0]
    layer_dims = [int(X.shape[1]) for X in Xs]
    print(f"[train] N={N}  layer_dims={layer_dims}  F={args.n_atoms}")

    ds = TensorDataset(*Xs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = Crosscoder(
        layer_dims=layer_dims,
        n_atoms=args.n_atoms,
        sparsity_weight=args.sparsity_weight,
        activation="relu",
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        ep_mse = [0.0] * 3
        ep_l1 = 0.0
        n_batches = 0
        for batch in dl:
            xs = [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            out = model(xs)
            out["loss"].backward()
            opt.step()
            ep_loss += float(out["loss"].item())
            for l in range(3):
                ep_mse[l] += float(out["mse_per_layer"][l].item())
            ep_l1 += float(out["l1"].item())
            n_batches += 1
        ep_loss /= max(n_batches, 1)
        ep_mse = [m / max(n_batches, 1) for m in ep_mse]
        ep_l1 /= max(n_batches, 1)
        history.append({
            "epoch": epoch,
            "loss": ep_loss,
            "mse_per_layer": ep_mse,
            "l1": ep_l1,
        })
        print(
            f"[train] ep{epoch:02d}  loss={ep_loss:.4f}  "
            f"mse={[f'{m:.3f}' for m in ep_mse]}  l1={ep_l1:.3f}  "
            f"t={time.time()-t0:.1f}s"
        )

    # ---- evaluation ----
    model.eval()
    with torch.no_grad():
        # Eval R² on a held-out 4k random subset (or full set if smaller).
        n_eval = min(4096, N)
        idx = torch.randperm(N)[:n_eval]
        xs_eval = [X[idx].to(device) for X in Xs]
        per_layer_r2 = model.per_layer_r2(xs_eval)
        affinity = model.atom_layer_affinity().cpu().numpy()  # (F, 3)
        x_cat = torch.cat(xs_eval, dim=-1)
        z = model.encode(x_cat).cpu().numpy()
        firing_rate = (z > 0).mean(axis=0)  # per-atom fraction-of-rows active
        cross_mask = model.cross_layer_atom_mask(threshold=0.15).cpu().numpy()
        n_cross = int(cross_mask.sum())

    print(f"[eval] per-layer R²: {per_layer_r2}")
    print(f"[eval] cross-layer atoms (min share ≥0.15): {n_cross} / {args.n_atoms}")

    # ---- save model + summary ----
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "history": history,
            "per_layer_r2": per_layer_r2,
            "affinity": affinity,
            "firing_rate": firing_rate,
            "cross_mask": cross_mask,
        },
        out_dir / "crosscoder.pt",
    )

    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "n_atoms": args.n_atoms,
        "layer_dims": layer_dims,
        "per_layer_r2": per_layer_r2,
        "n_cross_layer_atoms": n_cross,
        "mean_firing_rate": float(firing_rate.mean()),
        "n_dead": int((firing_rate < 1e-4).sum()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] wrote {out_dir/'crosscoder.pt'} and summary.json")

    # ---- atom layer-affinity histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Left: stacked bar (per-atom shares, sorted by max share).
    order = np.argsort(-affinity.max(axis=1))
    bottom = np.zeros(args.n_atoms)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for l in range(3):
        axes[0].bar(
            np.arange(args.n_atoms), affinity[order, l],
            bottom=bottom, color=colors[l], label=f"layer {l+1}", width=1.0,
        )
        bottom += affinity[order, l]
    axes[0].set_title("Per-atom decoder-norm share by layer")
    axes[0].set_xlabel("atom (sorted by max layer share)")
    axes[0].set_ylabel("share")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_xlim(0, args.n_atoms)

    # Right: histogram of min-layer-share.
    min_share = affinity.min(axis=1)
    axes[1].hist(min_share, bins=40, color="#444", edgecolor="white")
    axes[1].axvline(0.15, color="red", linestyle="--", label="cross-layer cutoff")
    axes[1].axvline(1.0/3.0, color="green", linestyle=":", label="uniform (1/L)")
    axes[1].set_title(
        f"Atom min-layer share  ({n_cross} cross-layer / {args.n_atoms})"
    )
    axes[1].set_xlabel("min over layers of decoder share")
    axes[1].set_ylabel("# atoms")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / "atom_layer_affinity.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"[train] wrote {fig_path}")

    summary["figure"] = str(fig_path)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO / "runs/COLOR_COGITO_MULTILAYER"))
    p.add_argument("--out-dir", default=str(REPO / "runs/crosscoder"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-atoms", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sparsity-weight", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
