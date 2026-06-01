"""Synthetic cyclic recovery — falsification test for the periodic basis.

Findings note: the Q1.5B L18 weekday cyclic probe returned zero atoms
with |ρ_circ| > 0.7. Hypothesis space:
  H1 — data scarcity / weak cyclic signal at Q1.5B L18.
  H2 — periodic Duchon basis machinery is broken.

This script falsifies H2 by planting a known cyclic structure in
synthetic ambient space and asking the periodic SAE to recover it.

Plant
-----
A latent angle θ_n ∈ [0, 2π) draws each token's structure:
    x_n = cos(θ_n) · v1 + sin(θ_n) · v2 + amp · noise
where (v1, v2) is a fixed pair of orthogonal D-vectors (a planted circle).

For sparsity, multiple independent circles are planted; each token
activates only k_per_token of them.

Recovery target
---------------
A successful atom is one whose t_k (mapped to [0, 2π) by ×2π) has
circular Spearman > 0.7 with the planted θ for that token's circle.

Variants:
  perio_default — periodic=True, K=8
  perio_K12    — periodic=True, K=12 (more basis flexibility per cycle)
  noperio      — periodic=False, baseline (should do WORSE — the basis
                  can't wrap)

If perio_default recovers the cycle (|ρ_circ|>0.7) and noperio doesn't,
the periodic machinery works and the weekday-LM failure is signal-
strength, not arch. If neither recovers it, the periodic basis is
broken and needs gamfit investigation.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


@dataclass
class Config:
    d_ambient: int = 256
    n_cycles: int = 4
    k_per_token: int = 2
    n_samples: int = 30_000
    n_steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    sae_F: int = 16
    sae_top_k: int = 4
    sae_R: int = 2
    noise: float = 0.05
    seed: int = 0
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/SYNTHETIC_CYCLIC")
    variants: tuple[str, ...] = field(default_factory=lambda: (
        "perio_default", "perio_K12", "noperio",
    ))


def plant_cycles(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    D, G = cfg.d_ambient, cfg.n_cycles
    bases = []
    for _ in range(G):
        Q, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        bases.append(Q.astype(np.float32))  # (D, 2)
    active = np.zeros((cfg.n_samples, G), dtype=bool)
    for n in range(cfg.n_samples):
        idx = rng.choice(G, size=cfg.k_per_token, replace=False)
        active[n, idx] = True
    theta = rng.uniform(0.0, 2*np.pi, (cfg.n_samples, G)).astype(np.float32)
    amps = rng.uniform(0.7, 1.3, (cfg.n_samples, G)).astype(np.float32)
    X = np.zeros((cfg.n_samples, D), dtype=np.float32)
    for g in range(G):
        m = active[:, g]
        if not m.any():
            continue
        c = np.cos(theta[m, g])
        s = np.sin(theta[m, g])
        contrib = c[:, None] * bases[g][:, 0] + s[:, None] * bases[g][:, 1]
        X[m] += (amps[m, g, None] * contrib)
    X += rng.standard_normal(X.shape).astype(np.float32) * cfg.noise
    return {"X": torch.from_numpy(X), "active": active, "theta": theta, "amps": amps, "bases": bases}


def circular_spearman(t_norm: np.ndarray, theta: np.ndarray) -> float:
    """t_norm in [0, 1], theta in [0, 2π). Returns max over phase shift of
    standard Spearman between sin/cos rotations.

    Use Fisher-Lee correlation: max over phase φ of
        ρ(sort_idx(t), sort_idx((theta - φ) mod 2π))
    Cheap version: try a few phases, return best Spearman magnitude.
    """
    if len(t_norm) < 5:
        return 0.0
    angle = t_norm * 2 * np.pi
    best = 0.0
    # Sample 24 phase shifts (15° each)
    for k in range(24):
        phi = k * (2*np.pi / 24)
        shifted = (theta - phi) % (2*np.pi)
        # circular distance can't easily go through Spearman directly; use
        # ρ(sin·sin + cos·cos)-style approach by mapping both to unit
        # circle and computing the circular correlation coefficient:
        a = angle
        b = shifted
        sin_a, cos_a = np.sin(a), np.cos(a)
        sin_b, cos_b = np.sin(b), np.cos(b)
        num = float(np.mean(sin_a * sin_b + cos_a * cos_b))
        # num is mean(cos(a-b)); maximize over phi gives 1 if a==b.
        best = max(best, abs(num))
    return best


def train_variant(cfg: Config, X: torch.Tensor, device: torch.device,
                  variant: str) -> dict:
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig, SparsityConfig

    K = 12 if variant == "perio_K12" else 8
    periodic = variant != "noperio"
    # periodic -> S^1 atoms (intrinsic_rank forced to 1); non-periodic uses a
    # product manifold carrying the requested intrinsic rank (or circle if R<=1).
    if periodic:
        manifold, rank = "circle", 1
    elif cfg.sae_R <= 1:
        manifold, rank = "circle", 1
    else:
        manifold, rank = "product", cfg.sae_R
    sae_cfg = ManifoldSAEConfig(
        input_dim=cfg.d_ambient, n_atoms=cfg.sae_F, n_basis_per_atom=K,
        intrinsic_rank=rank, atom_manifold=manifold,
        sparsity=SparsityConfig(kind="softmax_topk", target_k=cfg.sae_top_k),
        dtype=torch.float64,
    )
    sae = ManifoldSAE(sae_cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    X = X.to(device=device, dtype=sae_cfg.dtype)
    for step in range(cfg.n_steps):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size,))
        batch = X[idx]
        opt.zero_grad()
        out = sae(batch)
        loss = total_loss(out, batch, sae)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            mse = F_nn.mse_loss(out.x_hat, batch).item()
            print(f"    [{variant} step {step:5d}] mse={mse:.4e}", flush=True)
    sae.eval()
    sae.fit(X[: min(2048, X.shape[0])])
    sae.lock_snapshot()
    return {"sae": sae, "cfg": sae_cfg}


def evaluate(cfg: Config, sae, data: dict, device: torch.device) -> dict:
    X = data["X"].to(device=device, dtype=sae.cfg.dtype)
    with torch.no_grad():
        out = sae(X)
    mse = F_nn.mse_loss(out.x_hat, X).item()
    var = float(X.var().item())
    ev = 1 - mse / var
    fire = (out.amplitudes > 1e-6).sum(dim=0).cpu().numpy()
    pos = out.positions[..., 0].cpu().numpy()      # (N, F) — first manifold coord, in [0,1]

    per_cycle = []
    for g in range(cfg.n_cycles):
        m = data["active"][:, g]
        if m.sum() < 50:
            continue
        gt_theta = data["theta"][m, g]
        best = 0.0
        best_atom = -1
        for k in range(cfg.sae_F):
            firing_in_cycle = (data["active"][:, g] & (out.amplitudes[:, k].cpu().numpy() > 1e-6))
            if firing_in_cycle.sum() < 30:
                continue
            sc = circular_spearman(pos[firing_in_cycle, k], data["theta"][firing_in_cycle, g])
            if sc > best:
                best = sc
                best_atom = k
        per_cycle.append({"cycle": g, "best_atom": best_atom, "rho_circ": best})
    return {
        "ev": ev,
        "alive": int((fire > 30).sum()),
        "per_cycle": per_cycle,
        "mean_rho_circ": float(np.mean([c["rho_circ"] for c in per_cycle])) if per_cycle else 0.0,
        "max_rho_circ": float(np.max([c["rho_circ"] for c in per_cycle])) if per_cycle else 0.0,
    }


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} out={out_dir}", flush=True)

    print("[data] planting cycles", flush=True)
    data = plant_cycles(cfg)
    print(f"  X={data['X'].shape}, cycles={cfg.n_cycles}", flush=True)

    all_results = {}
    for variant in cfg.variants:
        print(f"\n=== variant: {variant} ===", flush=True)
        try:
            obj = train_variant(cfg, data["X"], device, variant)
            report = evaluate(cfg, obj["sae"], data, device)
            all_results[variant] = report
            print(f"  EV={report['ev']:.3f}  alive={report['alive']}/{cfg.sae_F}  "
                  f"max_ρ_circ={report['max_rho_circ']:.3f}  "
                  f"mean_ρ_circ={report['mean_rho_circ']:.3f}", flush=True)
            for c in report["per_cycle"]:
                print(f"    cycle {c['cycle']}: best_atom={c['best_atom']} ρ_circ={c['rho_circ']:.3f}", flush=True)
        except Exception as e:
            print(f"  [{variant}] FAILED: {type(e).__name__}: {e}", flush=True)
            all_results[variant] = {"error": str(e)}

    (out_dir / "summary.json").write_text(json.dumps({
        "config": asdict(cfg), "results": all_results,
    }, indent=2, default=float))

    print("\n=== VERDICT ===", flush=True)
    print(f"  periodic-default max_ρ_circ: {all_results.get('perio_default', {}).get('max_rho_circ', 'NA')}", flush=True)
    print(f"  periodic-K12     max_ρ_circ: {all_results.get('perio_K12', {}).get('max_rho_circ', 'NA')}", flush=True)
    print(f"  non-periodic     max_ρ_circ: {all_results.get('noperio', {}).get('max_rho_circ', 'NA')}", flush=True)
    print(f"[done] {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
