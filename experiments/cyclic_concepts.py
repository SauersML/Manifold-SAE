"""Cyclic-concept manifold recovery test.

Inspired by Engels et al. (2024) "Not All Language Model Features Are
Linear" and the Goodfire et al. (2025) manifold-steering paper. Both
show that for cyclic conceptual structures (days of the week, months
of the year) language models encode the structure as a curved 1D
manifold (a loop) in residual stream, recoverable by PCA + smooth
spline fit through per-class centroids.

This experiment asks a more architectural question: can a *single
Manifold-SAE atom* recover such a manifold natively, without
post-hoc spline fitting? The architectural claim of Manifold-SAE is
that each curve atom k learns a smooth path `g_k(t)` in residual
stream parameterized by a learned 1D coordinate `t_k ∈ [0,1]`. If
the model has internalized a cyclic conceptual manifold for
weekdays, Manifold-SAE *should* find one atom whose `g_k(t)` traces
that loop and whose `t_k` cycles with the day-of-week index.

Tasks
=====
* weekdays: "What day is {k} days after {day}?" — 49 prompts, 7 cycled answers
* months:   "What month is {k} months after {month}?" — 84 prompts, 12 cycled answers

For each task:
  1. Harvest answer-position residual at layer L (configurable).
  2. PCA to 64-d, compute concept centroids (one per result class).
  3. Train Manifold-SAE F=64 R=2 TopK=8 on the harvested activations.
  4. For each curve atom k that fires on >100 prompts, compute the
     CIRCULAR correlation between `t_k` modulo wrap and the GT result-class
     index modulo the cycle length. Spearman on (sin, cos) of both angles
     is the right metric for cyclic data.
  5. Report the best atom + visualize its curve `g_k(t)` in the same
     PCA basis used for the activation centroids. If the architectural
     claim holds, the atom's curve will trace the same loop as the
     centroids.

Predictions
-----------
* If the model's encoding is cyclic AND Manifold-SAE captures cyclic
  features in a single atom: best-atom |ρ_circ| > 0.9, and g_k(t)
  visually matches the centroid loop.
* If the encoding is cyclic but Manifold-SAE atoms fragment it:
  |ρ_circ| around 0.5-0.7 with multiple atoms each covering a
  segment. (Same failure mode as a vanilla SAE would have.)
* If the model doesn't encode the task cyclically at this layer:
  |ρ_circ| no better than random, even for vanilla baseline.

A clean win here would be a striking architectural-difference result
on real LM data, complementing the saturated MSE-Pareto on Qwen
layer 12.
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
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check
from manifold_sae._cluster_bridge import require_cuda_if_env

bypass_gamfit_cuda_check()


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
INCREMENTS_WORD = ["one", "two", "three", "four", "five", "six", "seven"]


@dataclass
class CyclicConfig:
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "18"))
    # If set, only run this task instead of all.
    only_task: str | None = os.environ.get("MSAE_ONLY_TASK")

    # SAE training
    sae_F: int = 64
    sae_top_k: int = 8
    sae_R: int = 2
    sae_n_basis: int = 12
    n_steps_curve: int = 3000
    n_steps_vanilla: int = 3000
    batch_size: int = 64
    lr: float = 1e-3
    n_replicates_per_prompt: int = 50          # extra contexts / paraphrases per (concept, increment)

    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/CYCLIC",
    )
    seed: int = 0


def circ_spearman(theta1: np.ndarray, theta2: np.ndarray) -> float:
    """Mardia's circular correlation: |ρ_circ| ∈ [0, 1].
    theta1, theta2 in radians.
    """
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s1c = s1 - s1.mean(); c1c = c1 - c1.mean()
    s2c = s2 - s2.mean(); c2c = c2 - c2.mean()
    num = (s1c * s2c).sum() * (c1c * c2c).sum() - (s1c * c2c).sum() * (c1c * s2c).sum()
    den = ((s1c**2).sum() * (c1c**2).sum() - (s1c * c1c).sum()**2) * \
          ((s2c**2).sum() * (c2c**2).sum() - (s2c * c2c).sum()**2)
    if den <= 0:
        return 0.0
    return float(np.sqrt(num**2 / den) * np.sign(num))


def _make_weekday_prompts(n_paraphrases: int) -> tuple[list[str], list[int]]:
    templates = [
        "Q: What day comes {k} days after {d}?\nA:",
        "If today is {d}, then {k} days from now is",
        "Starting from {d}, count forward {k} days. The day is",
        "{d} plus {k} days equals",
        "Beginning on {d} and waiting {k} days lands on a",
    ]
    prompts, results = [], []
    for k_idx, k_word in enumerate(INCREMENTS_WORD):
        k_int = k_idx + 1
        for d_idx, d in enumerate(DAYS):
            result_idx = (d_idx + k_int) % 7
            for t in templates[:max(1, n_paraphrases // 50)]:
                prompts.append(t.format(k=k_word, d=d))
                results.append(result_idx)
    return prompts, results


def _make_month_prompts(n_paraphrases: int) -> tuple[list[str], list[int]]:
    templates = [
        "Q: What month is {k} months after {m}?\nA:",
        "Starting in {m}, the month {k} months later is",
        "{k} months after {m} is",
        "{m} plus {k} months equals",
    ]
    prompts, results = [], []
    for k_idx, k_word in enumerate(INCREMENTS_WORD):
        k_int = k_idx + 1
        for m_idx, m in enumerate(MONTHS):
            result_idx = (m_idx + k_int) % 12
            for t in templates[:max(1, n_paraphrases // 50)]:
                prompts.append(t.format(k=k_word, m=m))
                results.append(result_idx)
    return prompts, results


def harvest(model_name: str, layer: int, prompts: list[str], device: torch.device) -> torch.Tensor:
    """Harvest the LAST-token residual at `layer` for each prompt."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[harvest] loading {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = None
    for attr in ("h", "layers", "encoder_layer"):
        if hasattr(model, attr):
            blocks = getattr(model, attr); break
    if blocks is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers

    captured = {}
    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = blocks[layer].register_forward_hook(hook)

    acts = []
    torch.set_grad_enabled(False)
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to(device)
        model(**inputs)
        acts.append(captured["h"][0, -1, :].cpu().float())
    torch.set_grad_enabled(True)
    handle.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(acts, dim=0)


class VanillaSAE(nn.Module):
    def __init__(self, D: int, F: int, top_k: int):
        super().__init__()
        H = max(4 * D, 2 * F)
        self.F = F; self.top_k = top_k
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H); self.act = nn.GELU()
        self.head = nn.Linear(H, F)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x):
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        if self.top_k < self.F:
            vals, idx = torch.topk(z, self.top_k, dim=1)
            gate = torch.zeros_like(z); gate.scatter_(1, idx, vals)
            z = gate
        return z @ self.W_dec, z


def train_van(D: int, F: int, top_k: int, X: torch.Tensor, n_steps: int, device: torch.device) -> VanillaSAE:
    sae = VanillaSAE(D, F, top_k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    X = X.to(device)
    for step in range(n_steps):
        idx = torch.randint(0, X.shape[0], (64,))
        opt.zero_grad()
        recon, z = sae(X[idx])
        loss = F_nn.mse_loss(recon, X[idx]) + 3e-4 * z.abs().mean()
        loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  [van step {step}] mse={F_nn.mse_loss(recon, X[idx]).item():.4e}", flush=True)
    return sae.eval()


def train_curve(D: int, cfg: CyclicConfig, X: torch.Tensor, device: torch.device):
    from manifold_sae.losses import total_loss
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    sae_cfg = ManifoldSAEConfig(
        input_dim=D, n_features=cfg.sae_F, n_basis=cfg.sae_n_basis,
        top_k=cfg.sae_top_k, intrinsic_rank=cfg.sae_R,
        encoder_type="linear", continuous_amp=True,
    )
    sae = ManifoldSAE(sae_cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    X = X.to(device)
    for step in range(cfg.n_steps_curve):
        idx = torch.randint(0, X.shape[0], (cfg.batch_size,))
        opt.zero_grad()
        out = sae(X[idx])
        loss = total_loss(out, X[idx], sae_cfg)["total"]
        loss.backward(); opt.step()
        if step % 500 == 0:
            mse = F_nn.mse_loss(out.reconstruction, X[idx]).item()
            print(f"  [crv step {step}] mse={mse:.4e}", flush=True)
    sae.eval()
    sae.update_snapshot(X[:min(X.shape[0], 1024)])
    sae.inference_mode = True
    return sae


def run_task(cfg: CyclicConfig, task: str, prompts: list[str], result_idx: list[int],
             cycle_len: int, device: torch.device, out_dir: Path) -> dict:
    print(f"\n=========== Task: {task} (cycle {cycle_len}) ===========", flush=True)
    print(f"  {len(prompts)} prompts", flush=True)
    X = harvest(cfg.model_name, cfg.layer, prompts, device)
    print(f"  X: {X.shape}", flush=True)
    D = X.shape[1]
    mu = X.mean(0, keepdim=True); sigma = float(X.std().item())
    X_n = (X - mu) / max(sigma, 1e-6)

    # Train both SAEs
    print("[vanilla] training", flush=True)
    sae_v = train_van(D, cfg.sae_F, cfg.sae_top_k, X_n, cfg.n_steps_vanilla, device)
    print("[curve] training", flush=True)
    sae_c = train_curve(D, cfg, X_n, device)

    # Eval: for each atom, compute circular correlation between atom-signal and result_idx
    angles = (np.array(result_idx, dtype=np.float64) / cycle_len) * 2.0 * np.pi
    with torch.no_grad():
        _, z = sae_v(X_n.to(device))
        out = sae_c(X_n.to(device))
    z_np = z.cpu().numpy()
    pos = out.positions.cpu().numpy()                     # (N, F)
    amp = out.amplitudes.cpu().numpy()

    fire = (amp > 1e-6).sum(axis=0)
    F = pos.shape[1]
    # Curve atoms: circular correlation between t_k (mapped to angle 2π·t_k) and result-angle
    curve_circ = np.zeros(F)
    for k in range(F):
        if fire[k] < 50: continue
        m = amp[:, k] > 1e-6
        atom_angles = pos[m, k] * 2.0 * np.pi
        curve_circ[k] = abs(circ_spearman(atom_angles, angles[m]))
    # Vanilla atoms: circular correlation between activation magnitude and result-angle
    fire_v = (z_np > 1e-6).sum(axis=0)
    F_v = z_np.shape[1]
    van_circ = np.zeros(F_v)
    for k in range(F_v):
        if fire_v[k] < 50: continue
        m = z_np[:, k] > 1e-6
        # Vanilla activation is scalar — use atan2(act, 0) which collapses; use as a linear projection.
        # Better metric for vanilla: max Spearman between activation magnitude and (sin, cos)(result-angle).
        van_circ[k] = max(
            abs(circ_spearman(z_np[m, k][:, None].repeat(1, 0).flatten() * np.pi, angles[m])),
            0.0,
        )
        # Simpler: how well does the activation magnitude rank-correlate with the linear "result index"?
        # Vanilla CAN'T encode a cyclic order in one scalar by definition — best |ρ| with linear index is its score.
        # Use that as the metric.
        ranks = np.argsort(np.argsort(np.array(result_idx)[m]))
        zranks = np.argsort(np.argsort(z_np[m, k]))
        rx = ranks - ranks.mean(); ry = zranks - zranks.mean()
        denom = np.sqrt((rx*rx).sum() * (ry*ry).sum())
        van_circ[k] = abs((rx*ry).sum() / denom) if denom > 0 else 0.0

    best_curve = int(np.argmax(curve_circ))
    best_van = int(np.argmax(van_circ))
    print(f"[eval] curve best atom: idx={best_curve}  |ρ_circ|={curve_circ[best_curve]:.3f}  fires={fire[best_curve]}", flush=True)
    print(f"[eval] vanilla best atom: idx={best_van}  |ρ_linear|={van_circ[best_van]:.3f}  fires={fire_v[best_van]}", flush=True)
    print(f"[eval] curve atoms above 0.7: {int((curve_circ > 0.7).sum())}; above 0.5: {int((curve_circ > 0.5).sum())}", flush=True)
    print(f"[eval] vanilla atoms above 0.7: {int((van_circ > 0.7).sum())}; above 0.5: {int((van_circ > 0.5).sum())}", flush=True)

    # Plot: PCA of centroids + best curve atom's g_k(t) in same basis
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_task_circle(task, X_n, result_idx, cycle_len, sae_c, best_curve, out_dir)

    report = {
        "task": task,
        "cycle_len": cycle_len,
        "n_prompts": len(prompts),
        "D": D,
        "curve_best_atom": best_curve,
        "curve_best_circ_rho": float(curve_circ[best_curve]),
        "curve_n_above_strong": int((curve_circ > 0.7).sum()),
        "curve_n_above_moderate": int((curve_circ > 0.5).sum()),
        "curve_per_atom_circ_rho": curve_circ.tolist(),
        "vanilla_best_atom": best_van,
        "vanilla_best_lin_rho": float(van_circ[best_van]),
        "vanilla_n_above_strong": int((van_circ > 0.7).sum()),
        "vanilla_n_above_moderate": int((van_circ > 0.5).sum()),
        "vanilla_per_atom_lin_rho": van_circ.tolist(),
    }
    (out_dir / f"{task}_results.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def plot_task_circle(task: str, X_n: torch.Tensor, result_idx: list[int],
                      cycle_len: int, sae_c, best_atom: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PCA of centroids: (cycle_len, D)
    Xn = X_n.numpy()
    centroids = np.array([Xn[np.array(result_idx) == k].mean(axis=0) for k in range(cycle_len)])
    cc = centroids - centroids.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(cc, full_matrices=False)
    proj = cc @ Vt[:3].T

    # Eval atom's curve at uniform t
    import torch as t
    t_grid = t.linspace(0.0, 1.0, 100, dtype=t.float64)
    centers = sae_c.centers
    import gamfit.torch as gt
    phi = gt.duchon_basis_1d(t_grid, centers, m=2, periodic=False).cpu().numpy()
    B = sae_c.B_locked[best_atom].cpu().numpy()           # (K, R)
    g = phi @ B                                            # (100, R)
    # Lift to ambient via the atom's directions
    dirs = sae_c.directions[best_atom].cpu().numpy()      # (D, R)
    g_ambient = g @ dirs.T                                 # (100, D)
    g_proj = (g_ambient - centroids.mean(0, keepdims=True)) @ Vt[:3].T

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              subplot_kw={"projection": "3d"})
    cmap = plt.cm.hsv if cycle_len in (7, 12) else plt.cm.viridis
    for ax in axes:
        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                    c=np.arange(cycle_len) / cycle_len, cmap=cmap, s=80, edgecolors="black")
        for k, (x, y, z) in enumerate(proj):
            ax.text(x, y, z, str(k), fontsize=9)
    axes[0].set_title(f"{task}: per-class activation centroids (PCA-3)")
    axes[1].plot(g_proj[:, 0], g_proj[:, 1], g_proj[:, 2], "-",
                  color="black", linewidth=2, alpha=0.7,
                  label=f"Manifold-SAE atom #{best_atom}")
    axes[1].set_title(f"{task}: best curve atom's g_k(t) in same PCA basis")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_manifold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cfg = CyclicConfig()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} model={cfg.model_name} layer={cfg.layer}", flush=True)

    tasks = {
        "weekdays": (_make_weekday_prompts(cfg.n_replicates_per_prompt), 7),
        "months":   (_make_month_prompts(cfg.n_replicates_per_prompt), 12),
    }
    if cfg.only_task:
        tasks = {cfg.only_task: tasks[cfg.only_task]}

    all_reports = {}
    for task, ((prompts, result_idx), cycle_len) in tasks.items():
        all_reports[task] = run_task(cfg, task, prompts, result_idx, cycle_len, device, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(all_reports, indent=2, default=float))
    print("\n=== Summary ===")
    for task, r in all_reports.items():
        print(f"  {task}: curve |ρ_circ|={r['curve_best_circ_rho']:.3f}  "
              f"vanilla |ρ_linear|={r['vanilla_best_lin_rho']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
