"""Train AdaptiveK SAE on cogito-L40 cache; compare to TopK-32 baseline."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifold_sae.adaptive_k import AdaptiveKSAE  # noqa: E402

OUT = ROOT / "runs" / "adaptive_k"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] device={DEVICE}", flush=True)


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


def batches(X_np, bs):
    n = X_np.shape[0]
    order = np.random.permutation(n)
    for s in range(0, n, bs):
        yield torch.from_numpy(X_np[order[s : s + bs]]).to(DEVICE)


def train(epochs=10, bs=512, lr=3e-4, sparsity_weight=0.02, k_min=8, k_max=80):
    X_train, X_val, D = load_data()
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    print(f"[data] train={X_train.shape} val={X_val.shape} D={D}", flush=True)

    model = AdaptiveKSAE(
        input_dim=D, F=512, k_min=k_min, k_max=k_max, sparsity_weight=sparsity_weight
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    t0 = time.time()
    # Anneal λ: warm up the encoder/decoder at λ=0 for the first warm_epochs
    # so the SAE first learns to reconstruct at the initial K_pred (~k_max),
    # then ramp λ to the target so the K-head learns to trim.
    target_lambda = float(sparsity_weight)
    warm_epochs = max(1, epochs // 3)
    for ep in range(epochs):
        if ep < warm_epochs:
            model.sparsity_weight = 0.0
        else:
            # cosine ramp from 0 to target over remaining epochs
            t = (ep - warm_epochs + 1) / max(1, epochs - warm_epochs)
            model.sparsity_weight = target_lambda * t
        model.train()
        ep_loss = 0.0
        n_b = 0
        for xb in batches(X_train, bs):
            opt.zero_grad()
            out = model.loss(xb)
            out["loss"].backward()
            opt.step()
            ep_loss += out["recon"].item()
            n_b += 1
        # Val
        model.eval()
        with torch.no_grad():
            ks, recons = [], []
            for s in range(0, X_val_t.shape[0], bs):
                xb = X_val_t[s : s + bs]
                out = model.loss(xb)
                ks.append(out["mean_k_actual"].item())
                recons.append(out["recon"].item())
            mean_k = float(np.mean(ks))
            mean_recon = float(np.mean(recons))
        val_var = X_val_t.var().item()
        r2 = 1.0 - mean_recon / val_var
        elapsed = time.time() - t0
        print(
            f"[ep {ep:02d}] train_recon={ep_loss/max(n_b,1):.4f} "
            f"val_r2={r2:.4f} mean_k={mean_k:.1f} t={elapsed:.1f}s",
            flush=True,
        )
        history.append(
            {"epoch": ep, "train_recon": ep_loss / max(n_b, 1), "val_r2": r2, "mean_k": mean_k}
        )

    # Per-row K distribution
    model.eval()
    with torch.no_grad():
        all_k = []
        for s in range(0, X_val_t.shape[0], bs):
            xb = X_val_t[s : s + bs]
            _, _, k_pred = model.forward(xb)
            all_k.append(k_pred.cpu().numpy())
        all_k = np.concatenate(all_k)
    k_min_obs = float(all_k.min())
    k_max_obs = float(all_k.max())
    k_p10, k_p50, k_p90 = (float(x) for x in np.percentile(all_k, [10, 50, 90]))

    # Dead-feature analysis
    with torch.no_grad():
        active_counts = torch.zeros(model.n_features, device=DEVICE)
        for s in range(0, X_val_t.shape[0], bs):
            xb = X_val_t[s : s + bs]
            _, z, _ = model.forward(xb)
            active_counts += (z.abs() > 0).float().sum(0)
        n_active = int((active_counts > 0).sum().item())
        dead_rate = 1.0 - n_active / model.n_features

    # Baseline reference
    base_json = json.loads((ROOT / "runs" / "sae_comparison" / "comparison.json").read_text())
    base_topk = base_json["TopK"]
    summary = {
        "history": history,
        "final_val_r2": history[-1]["val_r2"],
        "final_mean_k": history[-1]["mean_k"],
        "k_per_row_min": k_min_obs,
        "k_per_row_max": k_max_obs,
        "k_p10": k_p10,
        "k_p50": k_p50,
        "k_p90": k_p90,
        "n_active": n_active,
        "dead_rate": dead_rate,
        "baseline_topk32": {
            "val_r2": base_topk["val_r2"],
            "n_active": base_topk["n_active_atoms"],
            "mean_k": 32,
        },
    }
    (OUT / "adaptive_k_summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(model.state_dict(), OUT / "model_adaptive_k.pt")
    print(f"[done] saved to {OUT}", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    train()
