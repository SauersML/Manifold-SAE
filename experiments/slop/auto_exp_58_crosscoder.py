"""auto_exp_58 — Crosscoder vs. independent per-layer L1-SAE baselines.

Goal
----
Show the crosscoder (shared encoder + per-layer decoders) recovers per-layer
reconstruction quality at least as good as L independent SAEs trained
separately on each layer — while SHARING atoms across layers, which the
independents cannot do.

We then identify the 10 atoms with strongest cross-layer presence and
report their activation patterns vs. HSV / lightness probes built from
the cogito L40 color manifold (`runs/COLOR_COGITO_L40/color_manifold_layer40.npz`).

Outputs
-------
runs/crosscoder/auto_exp_58_results.json
runs/crosscoder/auto_exp_58_top10_hsv_corr.png
runs/crosscoder/circuit.dot                       (top-cross-layer atoms only)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_sae.crosscoder import Crosscoder
from manifold_sae.circuit_trace import trace_and_save


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "runs/COLOR_COGITO_MULTILAYER"
OUT = REPO / "runs/crosscoder"
PROBES = REPO / "runs/COLOR_COGITO_L40/color_manifold_probes_layer40.npz"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------
# Independent per-layer L1-SAE baseline
# --------------------------------------------------------------------

class TiedL1SAE(torch.nn.Module):
    """Tiny linear-encoder L1 SAE for baseline (D -> F -> D)."""

    def __init__(self, D: int, F: int) -> None:
        super().__init__()
        self.enc = torch.nn.Linear(D, F)
        W = torch.randn(F, D) / D ** 0.5
        self.W = torch.nn.Parameter(W)
        self.b_dec = torch.nn.Parameter(torch.zeros(D))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.enc(x))
        x_hat = z @ self.W + self.b_dec
        return x_hat, z


def train_baseline(
    X: torch.Tensor,
    *,
    n_atoms: int,
    epochs: int,
    batch: int,
    sparsity_weight: float,
    lr: float,
    device: torch.device,
) -> tuple[float, dict]:
    D = X.shape[1]
    model = TiedL1SAE(D, n_atoms).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch):
            idx = perm[s:s + batch]
            xb = X[idx].to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, z = model(xb)
            dec_norm = model.W.norm(dim=1)
            loss = ((xb - x_hat) ** 2).mean() + sparsity_weight * (z.abs() * dec_norm.unsqueeze(0)).sum(dim=-1).mean()
            loss.backward()
            opt.step()

    # R² on a 4k subset.
    model.eval()
    with torch.no_grad():
        idx = torch.randperm(N)[:min(4096, N)]
        xb = X[idx].to(device)
        x_hat, z = model(xb)
        ss_res = ((xb - x_hat) ** 2).sum().item()
        ss_tot = ((xb - xb.mean(0, keepdim=True)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return r2, {"firing_rate": float((z > 0).float().mean().item())}


# --------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"[exp58] device={device}")

    # 0. Ensure multilayer data exists.
    if not (DATA / "X_l1.npy").exists():
        print("[exp58] data missing — running synthesize step")
        subprocess.run([sys.executable, str(REPO / "scripts/synthesize_multilayer_cogito.py")], check=True)

    # 1. Train the crosscoder (call the trainer module directly to avoid spawning).
    from scripts.train_crosscoder import train as train_crosscoder, parse_args as _parse_train_args
    print("[exp58] training crosscoder")
    train_argv_backup = sys.argv
    try:
        sys.argv = ["train_crosscoder.py", "--epochs", "15", "--n-atoms", "512"]
        cross_summary = train_crosscoder(_parse_train_args())
    finally:
        sys.argv = train_argv_backup

    cross_r2 = cross_summary["per_layer_r2"]
    n_cross = cross_summary["n_cross_layer_atoms"]
    print(f"[exp58] crosscoder R²: {cross_r2}  cross-layer atoms: {n_cross}")

    # 2. Train independent per-layer L1 SAEs (same compute budget: 15 ep, F=512).
    print("[exp58] training independent per-layer baselines")
    layers_np = [np.load(DATA / f"X_l{i}.npy", mmap_mode="r") for i in (1, 2, 3)]

    baseline_r2: list[float] = []
    baseline_meta: list[dict] = []
    for l, X_np in enumerate(layers_np):
        # Normalize same way as crosscoder.
        mu = X_np.mean(0, keepdims=True)
        sd = X_np.std(0, keepdims=True).clip(min=1e-6)
        Z = (X_np - mu) / sd
        X = torch.from_numpy(np.ascontiguousarray(Z.astype(np.float32)))
        t0 = time.time()
        r2, meta = train_baseline(
            X, n_atoms=512, epochs=15, batch=256,
            sparsity_weight=1e-3, lr=1e-3, device=device,
        )
        meta["train_s"] = time.time() - t0
        print(f"[exp58] baseline l{l+1} R²={r2:.4f} fire={meta['firing_rate']:.3f} t={meta['train_s']:.1f}s")
        baseline_r2.append(r2)
        baseline_meta.append(meta)

    wins = sum(int(cr >= br - 1e-3) for cr, br in zip(cross_r2, baseline_r2))
    print(f"[exp58] crosscoder wins on {wins}/3 layers")

    # 3. Identify top-10 cross-layer atoms and their HSV correlations.
    ckpt = torch.load(OUT / "crosscoder.pt", map_location=device, weights_only=False)
    model = Crosscoder(
        layer_dims=ckpt["config"]["layer_dims"],
        n_atoms=ckpt["config"]["n_atoms"],
        sparsity_weight=ckpt["config"]["sparsity_weight"],
        activation=ckpt["config"]["activation"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    affinity = ckpt["affinity"]  # (F, 3)
    # "Cross-layer strength" = min-layer share (higher → more uniform across layers).
    min_share = affinity.min(axis=1)
    top10 = np.argsort(-min_share)[:10].tolist()
    print(f"[exp58] top-10 cross-layer atoms: {top10}")
    print(f"[exp58]   their min-shares:      {min_share[top10].round(3).tolist()}")

    # Build HSV correlation by activating the encoder on the full N rows and
    # correlating each atom's z vs each HSV-ish probe AXIS PROJECTION at L40.
    # We project L40 onto each probe direction to get a (N,) signal per probe,
    # then correlate against z[:, atom].
    if PROBES.exists():
        probes = np.load(PROBES, allow_pickle=True)
        probe_dirs = probes["directions"].astype(np.float32)  # (15, 7168)
        probe_labels = [str(s) for s in probes["labels"]]
        # L40 = X_l3.
        X3 = np.load(DATA / "X_l3.npy", mmap_mode="r")
        # Standardize layers same as in training.
        Xs = []
        for i in (1, 2, 3):
            X = np.load(DATA / f"X_l{i}.npy", mmap_mode="r")
            mu = X.mean(0, keepdims=True).astype(np.float32)
            sd = X.std(0, keepdims=True).astype(np.float32).clip(min=1e-6)
            Xs.append(torch.from_numpy(((X - mu) / sd).astype(np.float32)))

        # Build z over a sample (cap at 8k for MPS RAM).
        n_eval = min(8192, Xs[0].shape[0])
        idx = torch.randperm(Xs[0].shape[0])[:n_eval]
        with torch.no_grad():
            x_cat = torch.cat([X[idx].to(device) for X in Xs], dim=-1)
            z = model.encode(x_cat).cpu().numpy()  # (n_eval, F)
        # Probe signals at L40 (the real layer).
        x3_eval = X3[idx.cpu().numpy().astype(np.int64)].astype(np.float32)
        # Demean probe dirs not necessary — Pearson handles it.
        probe_signals = x3_eval @ probe_dirs.T  # (n_eval, 15)

        # Pearson correlations: top10 atoms × 15 probes.
        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a - a.mean(); b = b - b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            return float((a * b).sum() / den) if den > 1e-12 else 0.0

        corr = np.zeros((10, len(probe_labels)))
        for ii, atom in enumerate(top10):
            for jj in range(len(probe_labels)):
                corr[ii, jj] = _corr(z[:, atom], probe_signals[:, jj])

        # Heatmap.
        fig, ax = plt.subplots(figsize=(11, 4.5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(probe_labels)))
        ax.set_xticklabels(probe_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"atom {a}" for a in top10], fontsize=8)
        for ii in range(10):
            for jj in range(len(probe_labels)):
                if abs(corr[ii, jj]) > 0.15:
                    ax.text(jj, ii, f"{corr[ii, jj]:.2f}", ha="center",
                            va="center", fontsize=6, color="black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        ax.set_title("Top-10 cross-layer atoms — correlations with cogito-L40 color probes")
        plt.tight_layout()
        hsv_fig = OUT / "auto_exp_58_top10_hsv_corr.png"
        fig.savefig(hsv_fig, dpi=130)
        plt.close(fig)
        print(f"[exp58] wrote {hsv_fig}")
    else:
        print(f"[exp58] WARNING probes file missing — skipping HSV correlation: {PROBES}")
        corr = np.zeros((10, 0))
        probe_labels = []
        hsv_fig = None

    # 4. Circuit DOT for these top-10 atoms.
    n_eval = min(4096, Xs[0].shape[0])
    idx = torch.randperm(Xs[0].shape[0])[:n_eval]
    dot_path, edges = trace_and_save(
        model, [X[idx] for X in Xs], OUT / "circuit.dot",
        top_k_per_atom=2, min_weight=0.2, keep_atoms=top10,
    )
    print(f"[exp58] wrote {dot_path}  ({len(edges)} edges; restricted to top10)")

    # 5. Save results.
    results = {
        "crosscoder_per_layer_r2": cross_r2,
        "baseline_per_layer_r2": baseline_r2,
        "n_layers_won_or_tied": wins,
        "n_cross_layer_atoms": int(n_cross),
        "top10_cross_layer_atoms": [int(a) for a in top10],
        "top10_min_layer_share": min_share[top10].astype(float).tolist(),
        "top10_hsv_corr_labels": probe_labels,
        "top10_hsv_corr": corr.tolist(),
        "figures": {
            "affinity": str(OUT / "atom_layer_affinity.png"),
            "hsv_corr": str(hsv_fig) if hsv_fig is not None else None,
            "circuit_dot": str(dot_path),
        },
    }
    res_path = OUT / "auto_exp_58_results.json"
    res_path.write_text(json.dumps(results, indent=2))
    print(f"[exp58] wrote {res_path}")

    # Pretty summary table.
    print()
    print(f"{'layer':<8}{'crosscoder R²':>16}{'baseline R²':>16}{'Δ':>10}")
    for l in range(3):
        d = cross_r2[l] - baseline_r2[l]
        print(f"l{l+1:<7}{cross_r2[l]:>16.4f}{baseline_r2[l]:>16.4f}{d:>+10.4f}")
    print(f"\nWins/Ties: {wins}/3   Cross-layer atoms: {n_cross}/512")


if __name__ == "__main__":
    main()
