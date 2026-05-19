"""Pure-curve benchmark for Manifold-SAE vs vanilla TopK SAE.

The cleanest setting where a curve SAE should win: every GT feature is
a continuous 1D family along a smooth curve in R^D. A vanilla SAE has
to discretize each curve into many atoms; a curve SAE can represent
each family with one atom.

Reports:
  * Reconstruction explained variance per architecture at matched param
    budgets.
  * "Dictionary cost per family" — number of SAE atoms a single GT curve
    family activates across the dataset.

If the curve SAE achieves competitive MSE with a much smaller dictionary
(or much better MSE at matched dictionary), that's the LLM-applicability
proof.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    d_ambient: int = 256
    n_curve_features: int = 16
    n_samples: int = 30_000
    sparsity_per_token: int = 3          # exact number of curve features active per token
    noise: float = 0.02
    curve_amp_low: float = 0.7
    curve_amp_high: float = 1.3
    n_curve_anchors: int = 32

    sae_features_vanilla: int = 256       # vanilla needs many to enumerate curve positions
    sae_features_curve: int = 32          # curve only needs ~1 per family
    top_k_vanilla: int = 8                # ~ avg curve_features active per token + slack
    top_k_curve: int = 4                  # only ~ #firing GT features
    # Optional override to compare matched-budget (vanilla and curve at the same F)
    match_budget: bool = False
    matched_features: int = 32
    matched_topk: int = 4

    n_steps: int = 6000
    batch_size: int = 256
    lr: float = 1e-3
    n_basis: int = 8
    intrinsic_rank: int = 4
    sparsity_weight: float = 3e-4
    ortho_weight: float = 1e-3
    reml_weight: float = 1.0
    continuous_amp: bool = False

    seed: int = 0
    output_dir: str = "runs/PURE_CURVE_BENCH"


def _smooth_random_curve(D: int, n_anchors: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((n_anchors, D)) * 0.3
    smooth = np.cumsum(raw, axis=0)
    smooth -= smooth.mean(axis=0, keepdims=True)
    smooth /= max(np.linalg.norm(smooth) / (n_anchors ** 0.5), 1e-8)
    return smooth


def make_data(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    F = cfg.n_curve_features
    curves = np.stack([_smooth_random_curve(cfg.d_ambient, cfg.n_curve_anchors, rng) for _ in range(F)])

    # Each token has exactly k_active GT features fire, drawn without replacement.
    active = np.zeros((cfg.n_samples, F), dtype=bool)
    for n in range(cfg.n_samples):
        idx = rng.choice(F, size=cfg.sparsity_per_token, replace=False)
        active[n, idx] = True
    amps = rng.uniform(cfg.curve_amp_low, cfg.curve_amp_high, (cfg.n_samples, F))
    positions = rng.integers(0, cfg.n_curve_anchors, (cfg.n_samples, F))

    X = np.zeros((cfg.n_samples, cfg.d_ambient), dtype=np.float32)
    for k in range(F):
        mask = active[:, k]
        if not mask.any():
            continue
        idx = positions[mask, k]
        contrib = amps[mask, k, None] * curves[k][idx]
        X[mask] += contrib.astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * cfg.noise
    return {"X": torch.from_numpy(X), "curves": curves, "active": active, "positions": positions}


class VanillaSAE(nn.Module):
    def __init__(self, D, F, top_k):
        super().__init__()
        self.F = F; self.top_k = top_k
        self.norm = nn.LayerNorm(D)
        H = max(4 * D, 2 * F)
        self.fc1 = nn.Linear(D, H, bias=True)
        self.act = nn.GELU()
        self.head = nn.Linear(H, F, bias=True)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D ** 0.5)

    def forward(self, x):
        h = self.act(self.fc1(self.norm(x)))
        z = torch.nn.functional.relu(self.head(h))
        if self.top_k < self.F:
            vals, idx = torch.topk(z, self.top_k, dim=1)
            gate = torch.zeros_like(z)
            gate.scatter_(1, idx, vals)
            z = gate
        return z @ self.W_dec, z


def train(model, X, cfg, label, device, is_curve=False, sae_cfg=None):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    history = {"step": [], "mse": []}
    it = iter(loader); step = 0
    t0 = time.time()
    while step < cfg.n_steps:
        try: (batch,) = next(it)
        except StopIteration: it = iter(loader); (batch,) = next(it)
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        if is_curve:
            from manifold_sae.losses import total_loss
            out = model(batch)
            losses = total_loss(out, batch, sae_cfg)
            loss = losses["total"]
            mse = losses["mse"]
        else:
            recon, z = model(batch)
            mse = torch.mean((recon - batch) ** 2)
            loss = mse + 3e-4 * z.abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(cfg.n_steps // 10, 1) == 0:
            history["step"].append(step); history["mse"].append(float(mse.item()))
            print(f"  [{label} step {step:5d}] mse={mse.item():.4e}", flush=True)
        step += 1
    return history, time.time() - t0


def main(cfg: Config = Config()) -> int:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"[setup] device={device}", flush=True)
    data = make_data(cfg)
    X = data["X"]
    print(f"[data] X={tuple(X.shape)}  curves={cfg.n_curve_features}  positions/curve={cfg.n_curve_anchors}", flush=True)
    mu = X.mean(dim=0, keepdim=True)
    sigma = (X - mu).std().item()
    X_n = (X - mu) / max(sigma, 1e-6)
    var = float(X_n.var().item())

    F_van = cfg.matched_features if cfg.match_budget else cfg.sae_features_vanilla
    K_van = cfg.matched_topk if cfg.match_budget else cfg.top_k_vanilla
    F_crv = cfg.matched_features if cfg.match_budget else cfg.sae_features_curve
    K_crv = cfg.matched_topk if cfg.match_budget else cfg.top_k_curve

    print("[vanilla] train", flush=True)
    van = VanillaSAE(D=X.shape[1], F=F_van, top_k=K_van).to(device)
    n_v = sum(p.numel() for p in van.parameters())
    print(f"  params={n_v/1e6:.2f}M  F={F_van}  topk={K_van}", flush=True)
    vh, t_v = train(van, X_n, cfg, "van", device, is_curve=False)

    print("[curve] train", flush=True)
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    sae_cfg = ManifoldSAEConfig(
        input_dim=X.shape[1], n_features=F_crv, n_basis=cfg.n_basis,
        top_k=K_crv, intrinsic_rank=cfg.intrinsic_rank,
        sparsity_weight=cfg.sparsity_weight,
        ortho_weight=cfg.ortho_weight,
        reml_weight=cfg.reml_weight,
        encoder_type="linear",
        continuous_amp=cfg.continuous_amp,
    )
    curve = ManifoldSAE(sae_cfg).to(device)
    n_c = sum(p.numel() for p in curve.parameters())
    print(f"  params={n_c/1e6:.2f}M  F={F_crv}  topk={K_crv}", flush=True)
    ch, t_c = train(curve, X_n, cfg, "crv", device, is_curve=True, sae_cfg=sae_cfg)

    van.eval(); curve.eval()
    with torch.no_grad():
        eb = X_n[: min(4096, X_n.shape[0])].to(device)
        recon_v, z_v = van(eb)
        out_c = curve(eb)
        mse_v = float(torch.mean((recon_v - eb) ** 2).item())
        mse_c = float(torch.mean((out_c.reconstruction - eb) ** 2).item())
        alive_v = ((z_v > 0).any(dim=0)).sum().item()
        alive_c = ((out_c.amplitudes > 0.5).any(dim=0)).sum().item()
        z_v_np = (z_v > 0).cpu().numpy()
        z_c_np = (out_c.amplitudes > 0.5).cpu().numpy()
    print(f"[eval] vanilla MSE={mse_v:.4e} expl={1-mse_v/var:.3f} alive={alive_v}/{F_van}", flush=True)
    print(f"[eval] curve   MSE={mse_c:.4e} expl={1-mse_c/var:.3f} alive={alive_c}/{F_crv}", flush=True)

    coverage_v, coverage_c = [], []
    active = data["active"][: eb.shape[0]]
    for k in range(cfg.n_curve_features):
        mask = active[:, k]
        if mask.sum() < 20: continue
        v_freq = z_v_np[mask].mean(axis=0)
        c_freq = z_c_np[mask].mean(axis=0)
        coverage_v.append(int((v_freq > 0.02).sum()))
        coverage_c.append(int((c_freq > 0.02).sum()))
    print(f"[diag] avg atoms/family: vanilla={np.mean(coverage_v):.2f}  curve={np.mean(coverage_c):.2f}", flush=True)

    report = {
        "config": asdict(cfg),
        "vanilla": {"mse": mse_v, "explained": 1-mse_v/var, "alive": alive_v, "params_M": n_v/1e6, "train_seconds": t_v, "history": vh},
        "curve":   {"mse": mse_c, "explained": 1-mse_c/var, "alive": alive_c, "params_M": n_c/1e6, "train_seconds": t_c, "history": ch},
        "coverage": {"vanilla_atoms_per_family_mean": float(np.mean(coverage_v)), "curve_atoms_per_family_mean": float(np.mean(coverage_c)), "vanilla_per_family": coverage_v, "curve_per_family": coverage_c},
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[done] {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
