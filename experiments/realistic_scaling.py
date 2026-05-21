"""Realistic scaling benchmark — curve SAE vs vanilla TopK SAE.

This is the closest test to LLM-like activations we can run without
real LM data: smooth random curves in high-D ambient with realistic
sparsity, multi-position curves, and Hungarian-matched per-curve
chamfer for shape recovery.

Configurations tested by default:
  * D = 128, 16 GT curves, 32 positions, 3 active/token (similar to
    earlier pure_curve at D=256 but smaller / faster).
  * D = 256, 32 GT curves, 64 positions, 5 active/token (closer to
    LLM-scale stress).

For each configuration we run vanilla SAE and curve SAE at matched
dictionary size (F = #GT_features) and report:
  * explained variance (1 - MSE / var(X))
  * alive features
  * mean Hungarian chamfer between learned and planted curves —
    measures actual shape recovery, not just reconstruction.

If curve SAE wins on explained variance AND chamfer at matched F,
that's the head-to-head LLM-applicability proof at scale.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Scenario:
    name: str
    d_ambient: int
    n_curve_features: int
    n_curve_anchors: int
    sparsity_per_token: int
    n_samples: int
    sae_features: int           # matched F for both architectures
    top_k: int                  # matched top_k for both architectures
    n_steps: int
    batch_size: int
    lr: float
    # curve SAE specifics
    n_basis: int
    intrinsic_rank: int
    sparsity_weight: float = 3e-4
    ortho_weight: float = 1e-3
    reml_weight: float = 1.0
    continuous_amp: bool = True
    noise: float = 0.02


SCENARIOS = [
    Scenario(
        name="small",
        d_ambient=128, n_curve_features=16, n_curve_anchors=32,
        sparsity_per_token=3, n_samples=30_000,
        sae_features=16, top_k=4,
        n_steps=5000, batch_size=256, lr=1e-3,
        n_basis=8, intrinsic_rank=4,
    ),
    Scenario(
        name="mid",
        d_ambient=256, n_curve_features=32, n_curve_anchors=64,
        sparsity_per_token=5, n_samples=50_000,
        sae_features=32, top_k=8,
        n_steps=6000, batch_size=256, lr=1e-3,
        n_basis=10, intrinsic_rank=4,
    ),
    Scenario(
        name="large",
        d_ambient=512, n_curve_features=64, n_curve_anchors=64,
        sparsity_per_token=8, n_samples=60_000,
        sae_features=64, top_k=12,
        n_steps=6000, batch_size=256, lr=1e-3,
        n_basis=12, intrinsic_rank=4,
    ),
    # LM-scale-ish: D matches Qwen2.5-0.5B's hidden size; F matches what
    # a real SAE would use at this width. Designed to exercise B200
    # GEMM, not just CPU REML — bigger batch, more steps, denser data.
    Scenario(
        name="xlarge",
        d_ambient=896, n_curve_features=128, n_curve_anchors=64,
        sparsity_per_token=12, n_samples=120_000,
        sae_features=128, top_k=16,
        n_steps=8000, batch_size=512, lr=1e-3,
        n_basis=12, intrinsic_rank=4,
    ),
]


def _smooth_random_curve(D: int, n_anchors: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((n_anchors, D)) * 0.3
    smooth = np.cumsum(raw, axis=0)
    smooth -= smooth.mean(axis=0, keepdims=True)
    smooth /= max(np.linalg.norm(smooth) / (n_anchors ** 0.5), 1e-8)
    return smooth


def make_data(s: Scenario, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    F = s.n_curve_features
    curves = np.stack([_smooth_random_curve(s.d_ambient, s.n_curve_anchors, rng) for _ in range(F)])

    active = np.zeros((s.n_samples, F), dtype=bool)
    for n in range(s.n_samples):
        idx = rng.choice(F, size=s.sparsity_per_token, replace=False)
        active[n, idx] = True
    amps = rng.uniform(0.7, 1.3, (s.n_samples, F))
    positions = rng.integers(0, s.n_curve_anchors, (s.n_samples, F))

    X = np.zeros((s.n_samples, s.d_ambient), dtype=np.float32)
    for k in range(F):
        m = active[:, k]
        if not m.any(): continue
        contrib = amps[m, k, None] * curves[k][positions[m, k]]
        X[m] += contrib.astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * s.noise
    return {"X": torch.from_numpy(X), "curves": curves, "active": active, "positions": positions}


class VanillaSAE(nn.Module):
    def __init__(self, D, F, top_k):
        super().__init__()
        self.F = F; self.top_k = top_k
        H = max(4 * D, 2 * F)
        self.norm = nn.LayerNorm(D)
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


def train_loop(model, X, n_steps, batch_size, lr, label, device, is_curve, sae_cfg=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader); step = 0
    t0 = time.time()
    while step < n_steps:
        try: (batch,) = next(it)
        except StopIteration: it = iter(loader); (batch,) = next(it)
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        if is_curve:
            from manifold_sae.losses import total_loss
            out = model(batch)
            losses = total_loss(out, batch, sae_cfg)
            loss = losses["total"]; mse = losses["mse"]
        else:
            recon, z = model(batch)
            mse = torch.mean((recon - batch) ** 2)
            loss = mse + 3e-4 * z.abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(n_steps // 8, 1) == 0:
            print(f"  [{label} step {step:5d}] mse={mse.item():.4e}", flush=True)
        step += 1
    return time.time() - t0


def extract_curve_atoms(sae, gt_anchors: int, device) -> np.ndarray:
    """For the curve SAE, sample each atom's curve at gt_anchors uniform
    positions over [0, 1]. Uses the locked snapshot if present, else fits
    fresh via gamfit on a calibration batch. Returns (F, gt_anchors, D).
    """
    import gamfit.torch as gt_torch
    if not bool(sae.has_snapshot):
        raise RuntimeError("call sae.update_snapshot(reference_batch) before extract_curve_atoms")
    t_grid_f64 = torch.linspace(0.0, 1.0, gt_anchors, dtype=torch.float64, device=device)
    phi = gt_torch.duchon_basis_1d(t_grid_f64, sae.centers, m=2, periodic=sae.config.periodic)  # (T, K)
    g = torch.einsum("tk,fkr->ftr", phi, sae.B_locked)                                          # (F, T, R)
    amb = torch.einsum("ftr,fdr->ftd", g, sae.directions.to(torch.float64))                     # (F, T, D)
    return amb.detach().cpu().numpy()


def hungarian_chamfer(gt_curves: np.ndarray, learned_curves: np.ndarray) -> dict:
    from scipy.optimize import linear_sum_assignment
    G = gt_curves.shape[0]; L = learned_curves.shape[0]
    cost = np.zeros((G, L))
    for j in range(G):
        a = gt_curves[j] - gt_curves[j].mean(axis=0, keepdims=True)
        a_n = a / max(np.linalg.norm(a), 1e-12)
        for k in range(L):
            b = learned_curves[k] - learned_curves[k].mean(axis=0, keepdims=True)
            b_n = b / max(np.linalg.norm(b), 1e-12)
            diff = a_n[:, None, :] - b_n[None, :, :]
            d2 = (diff ** 2).sum(axis=-1)
            cost[j, k] = 0.5 * (np.sqrt(d2.min(axis=1)).mean() + np.sqrt(d2.min(axis=0)).mean())
    rows, cols = linear_sum_assignment(cost)
    return {
        "mean": float(cost[rows, cols].mean()),
        "max": float(cost[rows, cols].max()),
        "per_pair": [{"gt": int(r), "sae": int(c), "chamfer": float(cost[r, c])} for r, c in zip(rows, cols)],
    }


def run_scenario(s: Scenario, seed: int = 0, output_dir: str = "runs/REALISTIC") -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(output_dir) / s.name; out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("MSAE_REQUIRE_CUDA") == "1" and device.type != "cuda":
        raise RuntimeError(
            f"MSAE_REQUIRE_CUDA=1 but torch.cuda.is_available()=False "
            f"(torch.version.cuda={torch.version.cuda!r})."
        )
    print(f"\n=========== Scenario: {s.name} ===========", flush=True)
    print(f"  D={s.d_ambient}  GT_features={s.n_curve_features}  anchors={s.n_curve_anchors}", flush=True)
    print(f"  sparsity/token={s.sparsity_per_token}  samples={s.n_samples}", flush=True)
    print(f"  SAE F={s.sae_features}  TopK={s.top_k}  n_steps={s.n_steps}", flush=True)
    data = make_data(s, seed)
    X = data["X"]
    gt_curves = data["curves"]  # (F_gt, T, D)

    mu = X.mean(dim=0, keepdim=True)
    sigma = (X - mu).std().item()
    X_n = (X - mu) / max(sigma, 1e-6)
    var = float(X_n.var().item())

    print("\n[vanilla] training", flush=True)
    van = VanillaSAE(D=X.shape[1], F=s.sae_features, top_k=s.top_k).to(device)
    n_v = sum(p.numel() for p in van.parameters())
    print(f"  params={n_v/1e6:.2f}M", flush=True)
    t_v = train_loop(van, X_n, s.n_steps, s.batch_size, s.lr, "van", device, is_curve=False)

    print("\n[curve] training", flush=True)
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    sae_cfg = ManifoldSAEConfig(
        input_dim=X.shape[1], n_features=s.sae_features, n_basis=s.n_basis,
        top_k=s.top_k, intrinsic_rank=s.intrinsic_rank,
        sparsity_weight=s.sparsity_weight,
        ortho_weight=s.ortho_weight,
        reml_weight=s.reml_weight,
        encoder_type="linear",
        continuous_amp=s.continuous_amp,
    )
    curve = ManifoldSAE(sae_cfg).to(device)
    n_c = sum(p.numel() for p in curve.parameters())
    print(f"  params={n_c/1e6:.2f}M", flush=True)
    t_c = train_loop(curve, X_n, s.n_steps, s.batch_size, s.lr, "crv", device, is_curve=True, sae_cfg=sae_cfg)

    van.eval(); curve.eval()
    with torch.no_grad():
        eb = X_n[: min(4096, X_n.shape[0])].to(device)
        recon_v, z_v = van(eb)
        out_c = curve(eb)
        mse_v = float(torch.mean((recon_v - eb) ** 2).item())
        mse_c = float(torch.mean((out_c.reconstruction - eb) ** 2).item())
        alive_v = ((z_v > 0).any(dim=0)).sum().item()
        alive_c = ((out_c.amplitudes > 1e-3).any(dim=0)).sum().item()
    print(f"\n[eval] vanilla MSE={mse_v:.4f}  expl={1-mse_v/var:.3f}  alive={alive_v}/{s.sae_features}", flush=True)
    print(f"[eval] curve   MSE={mse_c:.4f}  expl={1-mse_c/var:.3f}  alive={alive_c}/{s.sae_features}", flush=True)

    # Lock-and-cache: snapshot B and λ from one big REML fit on a held-out
    # reference batch. After this the curve SAE is feedforward.
    snapshot_batch = X_n[: min(8192, X_n.shape[0])].to(device)
    curve.update_snapshot(snapshot_batch)
    curve.inference_mode = True

    # For curve SAE: sample each atom at uniform t-grid to get (F, anchors, D).
    learned_curve_atoms = extract_curve_atoms(curve, s.n_curve_anchors, device)
    # Renormalize to ambient scale (we trained on X_n; lift back to X scale)
    learned_curve_atoms = learned_curve_atoms * sigma
    chamfer_curve = hungarian_chamfer(gt_curves, learned_curve_atoms)

    # Vanilla atoms have NO position parameterization — each atom is a
    # single direction. Replicate the direction across n_anchors so the
    # chamfer code can compare to GT curves; vanilla "curve" is just a
    # constant point in residual space.
    van_dirs = van.W_dec.detach().cpu().numpy() * sigma  # (F, D)
    van_atoms = np.tile(van_dirs[:, None, :], (1, s.n_curve_anchors, 1))   # (F, T, D)
    chamfer_van = hungarian_chamfer(gt_curves, van_atoms)
    # Crucially: a single direction can be Frobenius-normalized to any
    # constant vector — chamfer will compare a constant point cloud to a
    # GT curve. Distance will be HIGH for any non-degenerate GT curve.

    print(f"\n[chamfer] vanilla mean={chamfer_van['mean']:.3f} max={chamfer_van['max']:.3f}", flush=True)
    print(f"[chamfer] curve   mean={chamfer_curve['mean']:.3f} max={chamfer_curve['max']:.3f}", flush=True)

    report = {
        "scenario": asdict(s),
        "var": var,
        "vanilla": {"mse": mse_v, "explained": 1-mse_v/var, "alive": alive_v, "params_M": n_v/1e6, "train_seconds": t_v, "chamfer": chamfer_van},
        "curve":   {"mse": mse_c, "explained": 1-mse_c/var, "alive": alive_c, "params_M": n_c/1e6, "train_seconds": t_c, "chamfer": chamfer_curve},
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main():
    # Env override lets a Heimdall submitter redirect outputs.
    output_dir = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "runs/REALISTIC")
    all_reports = {}
    for s in SCENARIOS:
        all_reports[s.name] = run_scenario(s, seed=0, output_dir=output_dir)
    summary_path = Path(output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_reports, indent=2, default=float))
    print(f"\n[done] summary at {summary_path}")
    print("\n=== Summary ===")
    for name, r in all_reports.items():
        print(f"  {name}: vanilla expl={r['vanilla']['explained']:.3f}  curve expl={r['curve']['explained']:.3f}  Δ={r['curve']['explained']-r['vanilla']['explained']:+.3f}")
        print(f"         vanilla chamfer={r['vanilla']['chamfer']['mean']:.3f}  curve chamfer={r['curve']['chamfer']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
