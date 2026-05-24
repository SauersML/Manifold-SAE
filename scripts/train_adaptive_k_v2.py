"""Train AdaptiveKv2 SAE with both target-K loss variants; pick winner.

Compares loss_kind ∈ {"clipped", "squared"} at matched k_target=32, k_max=80,
λ=1e-2, 15 epochs on cogito-L40 (held-out colors). Goal: R² ≥ 0.880 at
mean-K ≈ 32, beating TopK-32 baseline (R²=0.874).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.adaptive_k_v2 import AdaptiveKv2SAE  # noqa: E402

OUT = ROOT / "runs" / "adaptive_k_v2"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data():
    X = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
    N, D = X.shape
    N_COLORS, N_TPL = 949, 28
    rng = np.random.default_rng(0)
    color_perm = rng.permutation(N_COLORS)
    val_set = set(color_perm[: int(0.2 * N_COLORS)].tolist())
    row_color = np.arange(N) // N_TPL
    train_idx = np.where(~np.isin(row_color, list(val_set)))[0]
    val_idx = np.where(np.isin(row_color, list(val_set)))[0]
    X_train = np.ascontiguousarray(X[train_idx]).astype(np.float32)
    X_val = np.ascontiguousarray(X[val_idx]).astype(np.float32)
    mu = X_train.mean(0)
    X_train -= mu
    X_val -= mu
    return X_train, X_val, D


def _batches(X_np, bs):
    n = X_np.shape[0]
    order = np.random.permutation(n)
    for s in range(0, n, bs):
        yield torch.from_numpy(X_np[order[s : s + bs]]).to(DEVICE)


def train_one(
    loss_kind: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    D: int,
    *,
    epochs: int = 15,
    bs: int = 512,
    lr: float = 3e-4,
    k_target: int = 32,
    k_max: int = 80,
    sparsity_weight: float = 1e-2,
) -> dict:
    print(f"[train:{loss_kind}] starting", flush=True)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    val_var = X_val_t.var().item()
    model = AdaptiveKv2SAE(
        input_dim=D,
        F=512,
        k_target=k_target,
        k_min=4,
        k_max=k_max,
        sparsity_weight=sparsity_weight,
        loss_kind=loss_kind,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        n_b = 0
        for xb in _batches(X_train, bs):
            opt.zero_grad()
            out = model.loss(xb)
            out["loss"].backward()
            opt.step()
            ep_loss += out["recon"].item()
            n_b += 1
        # Val
        model.eval()
        with torch.no_grad():
            ks, recons, kstds = [], [], []
            for s in range(0, X_val_t.shape[0], bs):
                xb = X_val_t[s : s + bs]
                out = model.loss(xb)
                ks.append(out["mean_k_actual"].item())
                recons.append(out["recon"].item())
                kstds.append(out["k_std"].item())
            mean_k = float(np.mean(ks))
            mean_recon = float(np.mean(recons))
            k_std = float(np.mean(kstds))
        r2 = 1.0 - mean_recon / val_var
        print(
            f"[{loss_kind} ep {ep:02d}] recon={ep_loss/max(n_b,1):.4f} "
            f"r2={r2:.4f} mean_k={mean_k:.1f} k_std={k_std:.2f} "
            f"t={time.time()-t0:.1f}s",
            flush=True,
        )
        history.append({"epoch": ep, "val_r2": r2, "mean_k": mean_k, "k_std": k_std})

    # Per-row K distribution at end.
    model.eval()
    with torch.no_grad():
        all_k = []
        for s in range(0, X_val_t.shape[0], bs):
            _, _, kp = model.forward(X_val_t[s : s + bs])
            all_k.append(kp.cpu().numpy())
        all_k = np.concatenate(all_k)
        active_counts = torch.zeros(model.n_features, device=DEVICE)
        for s in range(0, X_val_t.shape[0], bs):
            _, z, _ = model.forward(X_val_t[s : s + bs])
            active_counts += (z.abs() > 0).float().sum(0)
        n_active = int((active_counts > 0).sum().item())
        dead_rate = 1.0 - n_active / model.n_features

    summary = {
        "loss_kind": loss_kind,
        "history": history,
        "final_val_r2": history[-1]["val_r2"],
        "final_mean_k": history[-1]["mean_k"],
        "k_p10": float(np.percentile(all_k, 10)),
        "k_p50": float(np.percentile(all_k, 50)),
        "k_p90": float(np.percentile(all_k, 90)),
        "k_min_obs": float(all_k.min()),
        "k_max_obs": float(all_k.max()),
        "k_std_obs": float(all_k.std()),
        "n_active": n_active,
        "dead_rate": dead_rate,
    }
    torch.save(model.state_dict(), OUT / f"model_{loss_kind}.pt")
    (OUT / f"summary_{loss_kind}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    X_train, X_val, D = load_data()
    print(f"[data] train={X_train.shape} val={X_val.shape} D={D}", flush=True)
    results = {}
    for kind in ("clipped", "squared"):
        results[kind] = train_one(kind, X_train, X_val, D)

    # Pick winner: highest R² subject to mean_k ≤ 32 + slack.
    def score(r):
        # Prefer R² but penalize being far above target K.
        over = max(0.0, r["final_mean_k"] - 32.0)
        return r["final_val_r2"] - 0.001 * over

    winner = max(results.values(), key=score)
    summary = {
        "results": results,
        "winner": winner["loss_kind"],
        "winner_r2": winner["final_val_r2"],
        "winner_mean_k": winner["final_mean_k"],
        "goal_met": (winner["final_val_r2"] >= 0.880 and winner["final_mean_k"] <= 32 + 2),
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "results"} | {
            "clipped": {k: v for k, v in results["clipped"].items() if k != "history"},
            "squared": {k: v for k, v in results["squared"].items() if k != "history"},
        },
        indent=2,
    ))
    return summary


if __name__ == "__main__":
    main()
