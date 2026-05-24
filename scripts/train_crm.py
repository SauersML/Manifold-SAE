"""Train Complete Replacement Model (CRM) on the 3-layer synthetic stack.

Loads `runs/COLOR_COGITO_MULTILAYER/` if available. Otherwise synthesizes
a 3-layer stack inline by linear projection of L40 + nonlinearity, so this
script is always runnable.
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

from manifold_sae.crm import CompleteReplacementModel, CRMConfig  # noqa: E402

OUT = ROOT / "runs" / "crm"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] device={DEVICE}", flush=True)


def _synth_multilayer(X_l40: np.ndarray, seed: int = 0) -> tuple[list[np.ndarray], list[int]]:
    """Build a 3-layer synthetic stack from L40 (mid-layer surrogate).

    L20-like: random Gaussian projection of L40 → 4096 then GELU.
    L30-like: random Gaussian projection of L20 → 5120 then GELU.
    L40: itself (passed in).
    """
    rng = np.random.default_rng(seed)
    D40 = X_l40.shape[1]
    D20, D30 = 4096, 5120
    P_20 = rng.standard_normal((D40, D20)).astype(np.float32) / np.sqrt(D40)
    P_30 = rng.standard_normal((D20, D30)).astype(np.float32) / np.sqrt(D20)

    def gelu_np(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    X20 = gelu_np(X_l40 @ P_20)
    X30 = gelu_np(X20 @ P_30)
    return [X20, X30, X_l40], [D20, D30, D40]


def load_multilayer():
    cache_dir = ROOT / "runs" / "COLOR_COGITO_MULTILAYER"
    saved_npz = list(cache_dir.glob("*.npz")) if cache_dir.exists() else []
    if saved_npz:
        # Heuristic: try to find per-layer X arrays.
        try:
            arrs = []
            for p in sorted(saved_npz):
                npz = np.load(p)
                for key in ("X", "activations", "x"):
                    if key in npz.files:
                        arrs.append(npz[key].astype(np.float32))
                        break
            if len(arrs) >= 3:
                arrs = arrs[:3]
                dims = [a.shape[1] for a in arrs]
                print(f"[data] loaded multilayer from disk: dims={dims}", flush=True)
                return arrs, dims
        except Exception as e:
            print(f"[data] disk load failed ({e}); falling back to synth", flush=True)
    X_l40 = np.load(ROOT / "runs" / "COLOR_COGITO_L40" / "X_L40.npy", mmap_mode="r")
    # Use a subset to keep synth memory bounded.
    N = X_l40.shape[0]
    idx = np.arange(N)
    X_l40_arr = np.ascontiguousarray(X_l40[idx]).astype(np.float32)
    arrs, dims = _synth_multilayer(X_l40_arr)
    print(f"[data] synthesized multilayer dims={dims} N={N}", flush=True)
    return arrs, dims


def train(epochs=5, bs=256, lr=3e-4):
    arrs, dims = load_multilayer()
    N = arrs[0].shape[0]
    # train/val split
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_val = int(0.2 * N)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    arrs_tr = [a[tr_idx] for a in arrs]
    arrs_val = [a[val_idx] for a in arrs]

    # center per layer using train stats; enforce float32 (MPS doesn't have f64)
    mus = [a.mean(0) for a in arrs_tr]
    arrs_tr = [(a - mu).astype(np.float32) for a, mu in zip(arrs_tr, mus)]
    arrs_val = [(a - mu).astype(np.float32) for a, mu in zip(arrs_val, mus)]

    config = CRMConfig(
        layer_dims=dims,
        n_features_per_sae=512,
        sae_top_k=32,
        transcoder_mid=1024,
        transcoder_top_k=64,
    )
    model = CompleteReplacementModel(config).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[model] CRM L={model.L} dims={dims}", flush=True)

    # Move val to device once
    xs_val = [torch.from_numpy(a).to(DEVICE) for a in arrs_val]

    history = []
    t0 = time.time()
    n_tr = arrs_tr[0].shape[0]
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n_tr)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n_tr, bs):
            idx = order[s : s + bs]
            xs = [torch.from_numpy(a[idx]).to(DEVICE) for a in arrs_tr]
            opt.zero_grad()
            out = model.loss(xs)
            out["loss"].backward()
            opt.step()
            ep_loss += out["loss"].item()
            nb += 1
        model.eval()
        with torch.no_grad():
            r2s = model.per_stage_r2(xs_val)
        print(
            f"[ep {ep}] loss={ep_loss/max(nb,1):.4f} per_stage_r2={[f'{r:.3f}' for r in r2s]} t={time.time()-t0:.1f}s",
            flush=True,
        )
        history.append({"epoch": ep, "loss": ep_loss / max(nb, 1), "per_stage_r2": r2s})

    summary = {
        "history": history,
        "final_per_stage_r2": history[-1]["per_stage_r2"],
        "layer_dims": dims,
        "config": {
            "n_features_per_sae": config.n_features_per_sae,
            "sae_top_k": config.sae_top_k,
            "transcoder_mid": config.transcoder_mid,
            "transcoder_top_k": config.transcoder_top_k,
        },
    }
    (OUT / "crm_summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(model.state_dict(), OUT / "model_crm.pt")
    print(f"[done] {OUT}", flush=True)
    print(json.dumps(summary["final_per_stage_r2"]), flush=True)
    return summary


if __name__ == "__main__":
    train()
