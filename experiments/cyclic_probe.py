"""Probe a pre-trained Manifold-SAE for cyclic concept structure.

Replaces the from-scratch SAE training in `cyclic_concepts.py` (which
failed because 49 weekday prompts is far too little data). Instead:

  1. Load an SAE already trained on wikitext at the target layer.
  2. Run it on weekday + month + hour-of-day prompts.
  3. For each alive atom, compute circular Spearman correlation
     between its position `t_k` and the GT result-class index
     (Monday=0, Tuesday=1, ..., Sunday=6).
  4. Report the atom with strongest circular correlation per cycle.

Vanilla baseline: best linear Spearman between activation magnitude
and the result index (the upper bound for what a scalar can encode).

If the curve SAE has ANY atom with |ρ_circ| > 0.7, the architecture
recovers cyclic structure natively from a wikitext-trained SAE — the
weekday-cycle exists in Qwen's residual stream and our atoms find it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_sae._cluster_bridge import bypass_gamfit_cuda_check, require_cuda_if_env

bypass_gamfit_cuda_check()


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass
class Config:
    curve_checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT",
        "/home/athuser/gnome_home/manifold_sae/runs/llm_sweep_q15b_L18/curve_F128.pt",
    )
    vanilla_checkpoint: str = os.environ.get(
        "MSAE_CHECKPOINT_V",
        "/home/athuser/gnome_home/manifold_sae/runs/llm_sweep_q15b_L18/vanilla_F128.pt",
    )
    model_name: str = os.environ.get("MSAE_MODEL", "Qwen/Qwen2.5-1.5B")
    layer: int = int(os.environ.get("MSAE_LAYER", "18"))
    n_paraphrases: int = 30
    min_fires: int = 5
    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/CYCLIC_PROBE",
    )


def _find_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def make_weekday_prompts(n_paraphrases: int) -> tuple[list[str], list[int]]:
    """Returns prompts and per-prompt result-class index (day of week)."""
    templates = [
        "What day comes {k} days after {d}? Answer:",
        "{k} days after {d} is",
        "Starting from {d}, what day is it in {k} days?",
        "If today is {d}, in {k} days it will be",
    ]
    INCREMENT_WORDS = ["one", "two", "three", "four", "five", "six", "seven"]
    prompts, idx = [], []
    rng = np.random.default_rng(0)
    for _ in range(max(1, n_paraphrases // (len(DAYS) * 7))):
        for di, d in enumerate(DAYS):
            for k in range(7):
                t = rng.choice(templates)
                k_str = rng.choice([str(k), INCREMENT_WORDS[k] if k < len(INCREMENT_WORDS) else str(k)])
                prompts.append(t.format(k=k_str, d=d))
                idx.append((di + k) % 7)
    return prompts, idx


def make_month_prompts(n_paraphrases: int) -> tuple[list[str], list[int]]:
    templates = [
        "What month comes {k} months after {m}? Answer:",
        "{k} months after {m} is",
        "If the current month is {m}, in {k} months it will be",
    ]
    prompts, idx = [], []
    rng = np.random.default_rng(1)
    for _ in range(max(1, n_paraphrases // (len(MONTHS) * 12))):
        for mi, m in enumerate(MONTHS):
            for k in range(12):
                t = rng.choice(templates)
                prompts.append(t.format(k=k, m=m))
                idx.append((mi + k) % 12)
    return prompts, idx


def harvest(model_name: str, layer: int, prompts: list[str], device) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    blocks = _find_blocks(model)
    cap = {}
    hh = blocks[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    Xs = []
    with torch.no_grad():
        for p in prompts:
            inputs = tok(p, return_tensors="pt").to(device)
            model(**inputs)
            Xs.append(cap["h"][0, -1, :].cpu())
    hh.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(Xs, dim=0)


def load_curve_sae(path: Path, D: int, device):
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    cfg = ManifoldSAEConfig(
        input_dim=D, n_features=sig["F"], n_basis=sig.get("n_basis", 10),
        top_k=sig["top_k"], intrinsic_rank=sig.get("intrinsic_rank", 2),
        encoder_type="linear", continuous_amp=True,
    )
    sae = ManifoldSAE(cfg).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    sae.inference_mode = bool(sae.has_snapshot.item())
    return sae


class VanillaSAE(nn.Module):
    def __init__(self, D, F, top_k):
        super().__init__()
        self.F = F; self.top_k = top_k
        H = max(4 * D, 2 * F)
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H); self.act = nn.GELU()
        self.head = nn.Linear(H, F)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x):
        import torch.nn.functional as F_nn
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        if self.top_k < self.F:
            vals, idx = torch.topk(z, self.top_k, dim=1)
            gate = torch.zeros_like(z).scatter_(1, idx, vals)
            z = gate
        return z @ self.W_dec, z


def load_vanilla_sae(path: Path, D: int, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    sae = VanillaSAE(D, sig["F"], sig["top_k"]).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    return sae


def circ_spearman(theta1: np.ndarray, theta2: np.ndarray) -> float:
    """Circular Spearman (Mardia) — works for periodic data."""
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    num = np.mean(s1 * s2) - np.mean(s1) * np.mean(s2)
    d1 = np.sqrt(max(np.mean(s1**2) - np.mean(s1)**2, 1e-9))
    d2 = np.sqrt(max(np.mean(s2**2) - np.mean(s2)**2, 1e-9))
    rho_ss = num / (d1 * d2)
    num = np.mean(c1 * c2) - np.mean(c1) * np.mean(c2)
    d1 = np.sqrt(max(np.mean(c1**2) - np.mean(c1)**2, 1e-9))
    d2 = np.sqrt(max(np.mean(c2**2) - np.mean(c2)**2, 1e-9))
    rho_cc = num / (d1 * d2)
    return float(np.sqrt(max((rho_ss**2 + rho_cc**2) / 2.0, 0.0)))


def run_task(task: str, cycle_len: int, prompts, idx, sae_c, sae_v, device, cfg: Config) -> dict:
    print(f"\n=== task={task} cycle_len={cycle_len} n_prompts={len(prompts)} ===", flush=True)
    X = harvest(cfg.model_name, cfg.layer, prompts, device)
    print(f"  X={X.shape}", flush=True)

    mu = X.mean(0, keepdim=True); sigma = float(X.std().item())
    X_n = (X - mu) / max(sigma, 1e-6)
    angles = (np.array(idx, dtype=np.float64) / cycle_len) * 2.0 * np.pi

    # Curve: |ρ_circ| between t_k (mapped to 2π·t_k) and angle
    with torch.no_grad():
        out_c = sae_c(X_n.to(device))
        out_v, gate_v = sae_v(X_n.to(device))
    pos = out_c.positions.cpu().numpy()
    amp = out_c.amplitudes.cpu().numpy()
    F = pos.shape[1]
    fire = (amp > 1e-6).sum(axis=0)
    curve_circ = np.zeros(F)
    for k in range(F):
        if fire[k] < cfg.min_fires: continue
        m = amp[:, k] > 1e-6
        if m.sum() < cfg.min_fires: continue
        atom_angles = pos[m, k] * 2.0 * np.pi
        curve_circ[k] = circ_spearman(atom_angles, angles[m])

    # Vanilla: best linear Spearman (vanilla can't encode cyclic in scalar)
    z_np = gate_v.cpu().numpy()
    fire_v = (z_np > 1e-6).sum(axis=0)
    F_v = z_np.shape[1]
    van_lin = np.zeros(F_v)
    for k in range(F_v):
        if fire_v[k] < cfg.min_fires: continue
        m = z_np[:, k] > 1e-6
        if m.sum() < cfg.min_fires: continue
        ranks_i = np.argsort(np.argsort(np.array(idx)[m]))
        ranks_z = np.argsort(np.argsort(z_np[m, k]))
        rx = ranks_i - ranks_i.mean(); ry = ranks_z - ranks_z.mean()
        denom = np.sqrt((rx*rx).sum() * (ry*ry).sum())
        van_lin[k] = abs((rx*ry).sum() / denom) if denom > 0 else 0.0

    best_curve = int(np.argmax(curve_circ))
    best_van = int(np.argmax(van_lin))
    print(f"  curve best atom: {best_curve}  |ρ_circ|={curve_circ[best_curve]:.3f}  fires={fire[best_curve]}", flush=True)
    print(f"  vanilla best atom: {best_van}  |ρ_lin|={van_lin[best_van]:.3f}  fires={fire_v[best_van]}", flush=True)
    print(f"  curve atoms with |ρ_circ| > 0.7: {int((curve_circ > 0.7).sum())}; > 0.5: {int((curve_circ > 0.5).sum())}", flush=True)
    print(f"  vanilla atoms with |ρ_lin| > 0.7: {int((van_lin > 0.7).sum())}; > 0.5: {int((van_lin > 0.5).sum())}", flush=True)
    return {
        "task": task, "cycle_len": cycle_len, "n_prompts": len(prompts),
        "curve_best_atom": best_curve,
        "curve_best_circ_rho": float(curve_circ[best_curve]),
        "curve_n_above_strong": int((curve_circ > 0.7).sum()),
        "curve_n_above_weak": int((curve_circ > 0.5).sum()),
        "vanilla_best_atom": best_van,
        "vanilla_best_lin_rho": float(van_lin[best_van]),
        "vanilla_n_above_strong": int((van_lin > 0.7).sum()),
        "vanilla_n_above_weak": int((van_lin > 0.5).sum()),
    }


def main() -> int:
    cfg = Config()
    require_cuda_if_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} out={out_dir}", flush=True)

    # Harvest a sniff batch to discover D.
    sniff_X = harvest(cfg.model_name, cfg.layer, ["The quick brown fox."], device)
    D = sniff_X.shape[1]
    print(f"[setup] D={D}", flush=True)

    sae_c = load_curve_sae(Path(cfg.curve_checkpoint), D, device)
    sae_v = load_vanilla_sae(Path(cfg.vanilla_checkpoint), D, device)
    print(f"[setup] curve F={sae_c.config.n_features}, vanilla F={sae_v.F}", flush=True)

    reports = {}
    for task, (cycle_len, gen) in {
        "weekdays": (7, make_weekday_prompts),
        "months":   (12, make_month_prompts),
    }.items():
        prompts, idx = gen(cfg.n_paraphrases)
        reports[task] = run_task(task, cycle_len, prompts, idx, sae_c, sae_v, device, cfg)
        (out_dir / f"{task}_results.json").write_text(json.dumps(reports[task], indent=2, default=float))

    (out_dir / "summary.json").write_text(json.dumps({
        "config": asdict(cfg), "results": reports,
    }, indent=2, default=float))
    print(f"\n[done] {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
