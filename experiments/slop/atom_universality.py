"""Atom universality across seeds — do independent runs find the same atoms?

SAE atoms are "universal" if training the same model with different
random seeds yields a dictionary where each planted concept gets matched
to a different atom index but the *direction* the atom encodes is
similar across seeds (up to permutation).

This is a sanity-check measurement that's surprisingly absent from the
SAE literature: most papers show one run. If atoms vary wildly across
seeds, "concept X lives at atom 42" is not a property of the model — it's
a property of the seed.

Procedure
---------
1. Plant 8 known 1D smooth manifolds (random curves) in ℝ²⁵⁶.
2. Train Manifold-SAE 1D and vanilla SAE under 3 random seeds each.
3. For each planted manifold, find the best-matching atom per seed.
4. Compute pairwise cosine similarity of best-match directions across
   seeds. A universal architecture has cos ≈ 1; a seed-dependent one
   has cos near 0.

Reports:
  * Hungarian-matched atom directions across (seed_i, seed_j) for each
    architecture
  * Distribution of cos-similarities
  * "agreement rate" — fraction of planted manifolds where all 3 seeds
    found an atom with cos > 0.7 to the planted direction
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
    n_manifolds: int = 8
    sparsity_per_token: int = 2
    n_curve_knots: int = 6
    n_samples: int = 20_000
    n_steps: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    sae_F: int = 16
    sae_top_k: int = 4
    sae_n_basis: int = 10
    sae_R: int = 2
    noise: float = 0.02
    seeds: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2))
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/ATOM_UNIVERSALITY")


def plant_curves(cfg: Config, data_seed: int = 0) -> dict:
    """Plant `n_manifolds` smooth 1D curves in ℝ^D. Each curve = a random
    direction + a random smooth scalar warp."""
    rng = np.random.default_rng(data_seed)
    D, M = cfg.d_ambient, cfg.n_manifolds
    Q, _ = np.linalg.qr(rng.standard_normal((D, M)))
    dirs = Q.astype(np.float32)                  # (D, M) orthogonal planted directions
    # smooth warps — random Fourier curves f_m: [0,1] -> R
    coeffs = rng.standard_normal((M, cfg.n_curve_knots)).astype(np.float32)
    active = np.zeros((cfg.n_samples, M), dtype=bool)
    for n in range(cfg.n_samples):
        idx = rng.choice(M, size=cfg.sparsity_per_token, replace=False)
        active[n, idx] = True
    t = rng.uniform(0.0, 1.0, (cfg.n_samples, M)).astype(np.float32)
    amps = rng.uniform(0.7, 1.3, (cfg.n_samples, M)).astype(np.float32)
    X = np.zeros((cfg.n_samples, cfg.d_ambient), dtype=np.float32)
    for m in range(M):
        mask = active[:, m]
        if not mask.any():
            continue
        # warp: weighted sum of sinusoids
        warp = np.zeros_like(t[mask, m])
        for k in range(cfg.n_curve_knots):
            warp = warp + coeffs[m, k] * np.sin((k + 1) * np.pi * t[mask, m])
        X[mask] += (amps[mask, m, None] * warp[:, None] * dirs[None, :, m]).astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * cfg.noise
    return {"X": torch.from_numpy(X), "active": active, "t": t, "amps": amps,
            "planted_dirs": torch.from_numpy(dirs), "coeffs": coeffs}


def train_curve(cfg: Config, X: torch.Tensor, device, seed: int):
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig, SparsityConfig
    torch.manual_seed(seed)
    manifold = "circle" if cfg.sae_R <= 1 else "product"
    rank = 1 if cfg.sae_R <= 1 else cfg.sae_R
    sae_cfg = ManifoldSAEConfig(
        input_dim=cfg.d_ambient, n_atoms=cfg.sae_F, n_basis_per_atom=cfg.sae_n_basis,
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
    sae.eval()
    sae.fit(X[:2048])
    sae.lock_snapshot()
    return sae


def train_vanilla(cfg: Config, X: torch.Tensor, device, seed: int):
    torch.manual_seed(seed)
    D, F = cfg.d_ambient, cfg.sae_F
    enc = torch.nn.Linear(D, F).to(device)
    dec = torch.nn.Parameter(torch.randn(F, D, device=device) / D**0.5)
    bias = torch.nn.Parameter(torch.zeros(D, device=device))
    opt = torch.optim.Adam(list(enc.parameters()) + [dec, bias], lr=cfg.lr)
    k = cfg.sae_top_k
    X = X.to(device)
    for step in range(cfg.n_steps):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size,))
        batch = X[idx]
        opt.zero_grad()
        z = F_nn.relu(enc(batch - bias))
        vals, idx_top = torch.topk(z, k, dim=1)
        z_sparse = torch.zeros_like(z).scatter(1, idx_top, vals)
        recon = z_sparse @ dec + bias
        loss = F_nn.mse_loss(recon, batch) + 1e-4 * z_sparse.abs().mean()
        loss.backward()
        opt.step()
    enc.eval()
    return {"enc": enc, "dec": dec, "bias": bias}


def curve_atom_directions(sae) -> torch.Tensor:
    """For each atom, take the mean ambient curve direction over the manifold
    grid in t.

    The new gamfit decoder blocks already live in ambient ``R^D``, so the
    per-atom curve ``g_k(t)`` (shape ``(T, D)``) is obtained directly from
    ``extract_feature_curves``. We average over the grid to get a single
    representative ambient direction per atom, then normalize.

    Returns (F, D) tensor of ambient directions per atom (normalized).
    """
    F = sae.cfg.n_atoms
    curve_dict = sae.extract_feature_curves(grid_size=64)           # {k -> (T, D)}
    ordered = [curve_dict[i] for i in range(F)]
    curves = torch.stack(ordered, dim=0).to(torch.float32)          # (F, T, D)
    dirs = curves.mean(dim=1)                                       # (F, D)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return dirs


def vanilla_atom_directions(van) -> torch.Tensor:
    dec = van["dec"]
    dirs = dec / dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return dirs


def hungarian_match(planted: torch.Tensor, atoms: torch.Tensor) -> tuple[list[int], list[float]]:
    """For each planted direction (column m of (D, M)), find the best
    atom (row of (F, D)) by absolute cosine similarity (no greedy
    matching — true Hungarian via scipy)."""
    from scipy.optimize import linear_sum_assignment
    cos = atoms @ planted                                          # (F, M)
    cos_abs = cos.abs().cpu().numpy()
    # We want max abs-cos per planted; Hungarian on -cos.
    row_ind, col_ind = linear_sum_assignment(-cos_abs)
    # col_ind: planted index -> atom index. Need atom per planted.
    # cost matrix is (F, M), Hungarian gives F rows assigned to M cols
    # where F >= M. We invert to get planted -> atom.
    assignment = {col: row for row, col in zip(row_ind.tolist(), col_ind.tolist())}
    atom_per_planted = [assignment.get(m, -1) for m in range(planted.shape[1])]
    cos_per_planted = [cos_abs[atom_per_planted[m], m] if atom_per_planted[m] >= 0 else 0.0
                        for m in range(planted.shape[1])]
    return atom_per_planted, cos_per_planted


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} out={out_dir}", flush=True)

    data = plant_curves(cfg, data_seed=42)
    print(f"  X={data['X'].shape}, manifolds={cfg.n_manifolds}", flush=True)
    planted = data["planted_dirs"].to(device)            # (D, M)

    results = {"curve": {}, "vanilla": {}}
    curve_dirs_per_seed = []
    vanilla_dirs_per_seed = []

    for seed in cfg.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        print("  training curve SAE", flush=True)
        sae = train_curve(cfg, data["X"], device, seed)
        c_dirs = curve_atom_directions(sae)              # (F, D)
        curve_dirs_per_seed.append(c_dirs)
        atom, cos = hungarian_match(planted, c_dirs)
        results["curve"][str(seed)] = {"atom_per_planted": atom, "cos_per_planted": cos}
        print(f"  curve  matched cos: {[f'{c:.2f}' for c in cos]}", flush=True)

        print("  training vanilla SAE", flush=True)
        van = train_vanilla(cfg, data["X"], device, seed)
        v_dirs = vanilla_atom_directions(van)            # (F, D)
        vanilla_dirs_per_seed.append(v_dirs)
        atom, cos = hungarian_match(planted, v_dirs)
        results["vanilla"][str(seed)] = {"atom_per_planted": atom, "cos_per_planted": cos}
        print(f"  vanilla matched cos: {[f'{c:.2f}' for c in cos]}", flush=True)

    # Cross-seed agreement: for each planted manifold, did all seeds find
    # an atom with cos > 0.7?
    def agreement_rate(per_seed_results, thresh=0.7):
        n_planted = cfg.n_manifolds
        agree = 0
        for m in range(n_planted):
            if all(per_seed_results[str(s)]["cos_per_planted"][m] > thresh for s in cfg.seeds):
                agree += 1
        return agree / n_planted

    # Cross-seed direction stability: take best-match atom dir per seed,
    # compute pairwise cos across seeds, average.
    def stability(dirs_list):
        S = len(dirs_list)
        D = dirs_list[0].shape[1]
        ks = []
        for m in range(cfg.n_manifolds):
            # best atom per seed for planted m
            per_seed = []
            for s_idx, dirs in enumerate(dirs_list):
                atom = results[
                    "curve" if dirs is curve_dirs_per_seed[s_idx] else "vanilla"
                ][str(cfg.seeds[s_idx])]["atom_per_planted"][m]
                if atom < 0:
                    continue
                per_seed.append(dirs[atom])
            if len(per_seed) < 2:
                ks.append(0.0)
                continue
            # average pairwise |cos|
            cs = []
            for i in range(len(per_seed)):
                for j in range(i+1, len(per_seed)):
                    cs.append(abs(float(torch.dot(per_seed[i], per_seed[j]))))
            ks.append(float(np.mean(cs)))
        return ks

    summary = {
        "config": asdict(cfg),
        "results": results,
        "curve_agreement_rate": agreement_rate(results["curve"]),
        "vanilla_agreement_rate": agreement_rate(results["vanilla"]),
        "curve_stability_per_manifold": stability(curve_dirs_per_seed),
        "vanilla_stability_per_manifold": stability(vanilla_dirs_per_seed),
    }
    summary["curve_stability_mean"] = float(np.mean(summary["curve_stability_per_manifold"]))
    summary["vanilla_stability_mean"] = float(np.mean(summary["vanilla_stability_per_manifold"]))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n=== UNIVERSALITY ===", flush=True)
    print(f"  curve   agreement rate (cos>0.7 across all seeds): {summary['curve_agreement_rate']:.2f}", flush=True)
    print(f"  vanilla agreement rate (cos>0.7 across all seeds): {summary['vanilla_agreement_rate']:.2f}", flush=True)
    print(f"  curve   mean cross-seed |cos|: {summary['curve_stability_mean']:.3f}", flush=True)
    print(f"  vanilla mean cross-seed |cos|: {summary['vanilla_stability_mean']:.3f}", flush=True)
    print(f"\n[done] {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
