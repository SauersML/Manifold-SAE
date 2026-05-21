"""Planted-manifold probe: does a continuous feature live in LM residuals,
   and does Manifold-SAE recover it?

This experiment replaces the saturated MSE Pareto framing with a qualitative
manifold-recovery test, which is what Manifold-SAE actually claims to do.

The setup: a continuous feature is "planted" in the LM's input by prompting.
We use magnitude (N ∈ {1, 2, 5, 10, ..., 1000} across multiple templates).
The hypothesis is that some mid-layer residual stream encodes N continuously
along a 1D manifold.

Two phases:

  Phase 1 — Does the manifold exist?
    For each candidate layer L, harvest the residual at the N-token position
    across all (prompt, N) pairs. PCA the resulting (#prompts, D) tensor.
    The headline diagnostic is Spearman(N, PC1). If |ρ| > 0.7 for some L,
    magnitude lives there as a 1D manifold; otherwise the architecture
    comparison has nothing to recover.

  Phase 2 — Does the SAE recover it?
    Load trained curve and vanilla SAEs at the best layer. For each alive
    atom: compute Spearman(N, atom_signal). For vanilla, atom_signal is the
    TopK activation magnitude. For curve, atom_signal is the position t_k.
    Headline:
      * Count atoms with |Spearman| > 0.7 per architecture.
      * Best curve atom's (t_k, N) overlay with vanilla's best atom.

Failure modes both informative:
  * No layer has |ρ| > 0.7 → real result that magnitude isn't encoded as a
    clean manifold in Qwen-0.5B residual stream. Architecture comparison
    can't be done on this concept.
  * Some layer has |ρ| > 0.7 but SAE atoms don't track it → the SAE
    training on wikitext didn't allocate capacity to magnitude.

T4 wall time: ~5 minutes for phase 1, +5 min for phase 2 (if existing
checkpoints reused). No GPU REML, no big batches.
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


# Same gamfit bridge as the other LM drivers — see docs/known_issues.md.
def _bypass_gamfit_cuda_check() -> None:
    import gamfit._cuda as _gc

    def _no_conflicts():
        return {
            "platform": "linux", "mapped": {}, "conflicts": {},
            "packaged_nvidia_roots": [], "packaged_complete_stacks": [],
            "system_complete_stacks": [],
        }

    _gc.cuda_diagnostics = _no_conflicts
    _gc.assert_no_cuda_library_conflicts = lambda context: None
    for mod_name in ("gamfit._binding", "gamfit.torch._reml", "gamfit._api"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "assert_no_cuda_library_conflicts"):
                mod.assert_no_cuda_library_conflicts = lambda context: None
            if hasattr(mod, "cuda_diagnostics"):
                mod.cuda_diagnostics = _no_conflicts
        except ImportError:
            pass
    try:
        import gamfit._binding as _gb
        if hasattr(_gb.rust_module, "cache_clear"):
            _gb.rust_module.cache_clear()
    except ImportError:
        pass


_bypass_gamfit_cuda_check()


@dataclass
class ProbeConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    layers_to_probe: tuple[int, ...] = (4, 8, 12, 16, 20)
    # Numbers spanning ~3.5 orders of magnitude, roughly log-uniform.
    magnitudes: tuple[int, ...] = field(default_factory=lambda: (
        1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 70,
        100, 120, 150, 200, 250, 300, 400, 500, 700, 850, 1000,
    ))
    # Distinct templates so the SAE/PCA can't trivially key on the template
    # itself. Each contains exactly one {N} placeholder.
    prompt_templates: tuple[str, ...] = field(default_factory=lambda: (
        "The number is {N}.",
        "There were {N} apples in the basket.",
        "Count to {N}.",
        "She bought {N} books at the store.",
        "It costs {N} dollars total.",
        "The team scored {N} points.",
    ))
    # Where to grab the activation. The number token is the most informative
    # site; using -1 means "last token of the formatted prompt".
    target_token_position: int = -1
    # SAE checkpoint directory to probe in phase 2. If missing, only phase 1
    # runs (the manifold-existence test stands on its own).
    sae_checkpoint_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/LLM_SWEEP")
    sae_F_to_probe: int = 128

    spearman_threshold: float = 0.7         # "found the manifold" bar
    output_dir: str = os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/LLM_PROBE")
    seed: int = 0


# ---------------------------------------------------------------------------
# Phase 1: planted-feature harvest + per-layer PCA Spearman
# ---------------------------------------------------------------------------


def harvest_per_layer_target_activations(cfg: ProbeConfig, device: torch.device) -> tuple[dict[int, torch.Tensor], np.ndarray]:
    """For each layer in cfg.layers_to_probe, return a (P, D) tensor where P
    is the number of (template, N) prompt instances and D is the model
    hidden size. Also return the N values in matching order.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"[phase1] loading {cfg.model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()

    # Find the transformer-block ModuleList. Same logic as other drivers.
    blocks = None
    for attr in ("h", "layers", "encoder_layer"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            break
    if blocks is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    if blocks is None:
        raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")

    # Register one hook per layer of interest, capturing the post-block
    # residual stream into a dict.
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach()
        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in cfg.layers_to_probe]

    prompts: list[str] = []
    N_values: list[int] = []
    for N in cfg.magnitudes:
        for t in cfg.prompt_templates:
            prompts.append(t.format(N=N))
            N_values.append(N)
    N_arr = np.array(N_values, dtype=np.int64)

    print(f"[phase1] running {len(prompts)} prompts at {len(cfg.layers_to_probe)} layers", flush=True)
    activations_per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in cfg.layers_to_probe}
    torch.set_grad_enabled(False)
    try:
        for i, prompt in enumerate(prompts):
            inputs = tok(prompt, return_tensors="pt").to(device)
            model(**inputs)
            for L in cfg.layers_to_probe:
                h = captured[L]                                  # (1, T, D)
                # Use the configured target token position (default -1).
                idx = cfg.target_token_position
                if idx < 0:
                    idx = h.shape[1] + idx
                activations_per_layer[L].append(h[0, idx, :].cpu().float())
    finally:
        for h in handles:
            h.remove()
    torch.set_grad_enabled(True)

    out = {L: torch.stack(activations_per_layer[L], dim=0) for L in cfg.layers_to_probe}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, N_arr


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, no scipy dependency."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def phase1_layer_diagnostic(activations: torch.Tensor, N_arr: np.ndarray) -> dict:
    """For one layer's (P, D) activations, fit PCA and report Spearman
    correlations between top-PCs and N. If |ρ| on PC1 is high, magnitude
    lives along that direction at this layer.
    """
    X = activations.numpy().astype(np.float64)
    X_c = X - X.mean(axis=0, keepdims=True)
    # PCA via SVD on the centered point cloud.
    _, sv, vh = np.linalg.svd(X_c, full_matrices=False)
    proj = X_c @ vh[:5].T                                # top-5 PCs
    rhos = [spearman_corr(proj[:, k], N_arr) for k in range(5)]
    log_N = np.log(N_arr.astype(np.float64) + 1.0)
    rhos_log = [spearman_corr(proj[:, k], log_N) for k in range(5)]
    return {
        "singular_values_top5": [float(s) for s in sv[:5]],
        "spearman_pc_vs_N": rhos,
        "spearman_pc_vs_logN": rhos_log,
        "best_abs_rho_any_top5": float(max(abs(r) for r in rhos + rhos_log)),
    }


# ---------------------------------------------------------------------------
# Phase 2: probe trained SAEs for atoms that track N
# ---------------------------------------------------------------------------


class VanillaSAE(nn.Module):
    """Mirror of the architecture used in llm_sweep so we can load its checkpoints."""

    def __init__(self, D: int, F: int, top_k: int) -> None:
        super().__init__()
        self.F = F
        self.top_k = top_k
        H = 4 * D
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H)
        self.act = nn.GELU()
        self.head = nn.Linear(H, F)
        self.W_dec = nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F_nn.relu(self.head(self.act(self.fc1(self.norm(x)))))
        vals, idx = torch.topk(z, self.top_k, dim=1)
        gate = torch.zeros_like(z).scatter_(1, idx, vals)
        return gate @ self.W_dec, gate


def phase2_atom_correlations_vanilla(ckpt_path: Path, X: torch.Tensor, N_arr: np.ndarray, device: torch.device) -> dict:
    """Load a vanilla SAE checkpoint and report per-atom Spearman(activation, N)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    F = sig.get("F")
    top_k = sig.get("top_k")
    D = X.shape[1]
    sae = VanillaSAE(D, F, top_k).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    with torch.no_grad():
        _, z = sae(X.to(device))
    z_np = z.cpu().numpy()                                # (P, F)
    rhos = [spearman_corr(z_np[:, k], N_arr) for k in range(F)]
    rhos_log = [spearman_corr(z_np[:, k], np.log(N_arr + 1.0)) for k in range(F)]
    return {"F": F, "top_k": top_k, "spearman_N": rhos, "spearman_logN": rhos_log,
            "best_abs": float(max(abs(r) for r in rhos + rhos_log))}


def phase2_atom_correlations_curve(ckpt_path: Path, X: torch.Tensor, N_arr: np.ndarray, device: torch.device) -> dict:
    """Load a curve SAE checkpoint, run inference, report Spearman of both
    atom position t_k and atom amplitude with N.
    """
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    F = sig.get("F")
    top_k = sig.get("top_k")
    n_basis = sig.get("n_basis", 10)
    intrinsic_rank = sig.get("intrinsic_rank", 2)
    D = X.shape[1]

    cfg = ManifoldSAEConfig(
        input_dim=D, n_features=F, n_basis=n_basis, top_k=top_k,
        intrinsic_rank=intrinsic_rank, encoder_type="linear", continuous_amp=True,
    )
    sae = ManifoldSAE(cfg).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    sae.inference_mode = bool(sae.has_snapshot)

    with torch.no_grad():
        out = sae(X.to(device))
    pos_np = out.positions.cpu().numpy()                 # (P, F)
    amp_np = out.amplitudes.cpu().numpy()                # (P, F)
    rhos_pos = [spearman_corr(pos_np[:, k], N_arr) for k in range(F)]
    rhos_pos_log = [spearman_corr(pos_np[:, k], np.log(N_arr + 1.0)) for k in range(F)]
    rhos_amp = [spearman_corr(amp_np[:, k], N_arr) for k in range(F)]
    return {
        "F": F, "top_k": top_k,
        "spearman_position_N": rhos_pos,
        "spearman_position_logN": rhos_pos_log,
        "spearman_amplitude_N": rhos_amp,
        "best_position_abs": float(max(abs(r) for r in rhos_pos + rhos_pos_log)),
        "best_amplitude_abs": float(max(abs(r) for r in rhos_amp)),
        "positions": pos_np.tolist(),
        "amplitudes": amp_np.tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_phase1(per_layer: dict[int, dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = sorted(per_layer.keys())
    best_rho_lin = [max(abs(r) for r in per_layer[L]["spearman_pc_vs_N"]) for L in Ls]
    best_rho_log = [max(abs(r) for r in per_layer[L]["spearman_pc_vs_logN"]) for L in Ls]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ls, best_rho_lin, "o-", label="best |ρ| over top-5 PCs vs N (linear)")
    ax.plot(Ls, best_rho_log, "s--", label="best |ρ| over top-5 PCs vs log(N)")
    ax.axhline(0.7, color="gray", linestyle=":", label="manifold-found bar")
    ax.set_xlabel("layer")
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title("Phase 1: Is magnitude encoded as a 1D manifold at this layer?")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase1_manifold_existence.png", dpi=110)
    plt.close(fig)


def plot_phase2_best_atom(curve_result: dict, vanilla_result: dict, N_arr: np.ndarray, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = curve_result["spearman_position_N"]
    rhos_log = curve_result["spearman_position_logN"]
    best_idx = int(np.argmax([max(abs(rhos[k]), abs(rhos_log[k])) for k in range(len(rhos))]))
    pos = np.array(curve_result["positions"])[:, best_idx]
    amp = np.array(curve_result["amplitudes"])[:, best_idx]
    rho = max(abs(rhos[best_idx]), abs(rhos_log[best_idx]))

    rhos_v = vanilla_result["spearman_N"]
    rhos_v_log = vanilla_result["spearman_logN"]
    best_v = int(np.argmax([max(abs(rhos_v[k]), abs(rhos_v_log[k])) for k in range(len(rhos_v))]))
    rho_v = max(abs(rhos_v[best_v]), abs(rhos_v_log[best_v]))

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(N_arr, pos, c=amp, cmap="viridis", s=24, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("ground-truth magnitude N (log scale)")
    ax.set_ylabel(f"curve atom #{best_idx} position t_k")
    ax.set_title(
        f"Phase 2: best curve atom position vs N\n"
        f"curve best |ρ|={rho:.3f}    vanilla best |ρ|={rho_v:.3f}"
    )
    plt.colorbar(sc, ax=ax, label="curve atom amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase2_best_curve_atom.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: ProbeConfig | None = None) -> int:
    if cfg is None:
        cfg = ProbeConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("MSAE_REQUIRE_CUDA") == "1" and device.type != "cuda":
        raise RuntimeError(
            f"MSAE_REQUIRE_CUDA=1 but torch.cuda.is_available()=False "
            f"(torch.version.cuda={torch.version.cuda!r}). Likely a torch/driver "
            f"mismatch — pin torch in pyproject.toml to match the host driver."
        )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} output_dir={out_dir}", flush=True)

    # Phase 1: harvest + per-layer diagnostic.
    activations_per_layer, N_arr = harvest_per_layer_target_activations(cfg, device)
    print(f"[phase1] harvested {len(N_arr)} (prompt, N) instances at {len(cfg.layers_to_probe)} layers", flush=True)

    phase1 = {}
    for L in cfg.layers_to_probe:
        result = phase1_layer_diagnostic(activations_per_layer[L], N_arr)
        phase1[L] = result
        print(f"  layer {L:3d}: top-5 ρ vs N    = {[f'{r:+.2f}' for r in result['spearman_pc_vs_N']]}", flush=True)
        print(f"            top-5 ρ vs log(N)= {[f'{r:+.2f}' for r in result['spearman_pc_vs_logN']]}", flush=True)
    plot_phase1(phase1, out_dir)

    best_L = max(cfg.layers_to_probe, key=lambda L: phase1[L]["best_abs_rho_any_top5"])
    best_rho = phase1[best_L]["best_abs_rho_any_top5"]
    print(f"\n[phase1] best layer = {best_L}, best |ρ| = {best_rho:.3f}", flush=True)

    if best_rho < cfg.spearman_threshold:
        print(f"[phase1] manifold-existence test FAILED at threshold {cfg.spearman_threshold}: "
              f"no layer encodes magnitude as a clean 1D manifold. Skipping phase 2.", flush=True)
        (out_dir / "results.json").write_text(json.dumps(
            {"config": asdict(cfg), "N_values": N_arr.tolist(), "phase1": phase1,
             "best_layer": best_L, "phase2_skipped": "manifold-existence threshold not met"},
            indent=2, default=float))
        return 0

    print(f"[phase1] manifold exists at layer {best_L}; proceeding to phase 2", flush=True)

    # Phase 2: probe SAEs at best_L.
    if best_L != 12:
        print(f"[phase2] WARNING: best layer is {best_L}, but SAE checkpoints were "
              f"trained on layer 12 (per llm_sweep.py default). Probe will use "
              f"layer-{best_L} activations on the layer-12-trained SAE — atoms "
              f"may not generalize.", flush=True)

    X_best = activations_per_layer[best_L]
    # Normalize the probe activations the same way the SAE training did.
    mu = X_best.mean(0, keepdim=True)
    sigma = float(X_best.std().item())
    X_n = (X_best - mu) / max(sigma, 1e-6)

    ckpt_dir = Path(cfg.sae_checkpoint_dir)
    vanilla_path = ckpt_dir / f"vanilla_F{cfg.sae_F_to_probe}.pt"
    curve_path = ckpt_dir / f"curve_F{cfg.sae_F_to_probe}.pt"

    phase2: dict = {"sae_F": cfg.sae_F_to_probe}
    if vanilla_path.exists():
        phase2["vanilla"] = phase2_atom_correlations_vanilla(vanilla_path, X_n, N_arr, device)
        print(f"[phase2] vanilla SAE F={cfg.sae_F_to_probe}: best |ρ| = {phase2['vanilla']['best_abs']:.3f}", flush=True)
        v_alive = sum(1 for r in phase2["vanilla"]["spearman_N"] if abs(r) > 0.7) + \
                  sum(1 for r in phase2["vanilla"]["spearman_logN"] if abs(r) > 0.7)
        phase2["vanilla"]["n_atoms_above_threshold"] = v_alive
        print(f"           atoms with |ρ| > 0.7: {v_alive}", flush=True)
    else:
        print(f"[phase2] vanilla checkpoint not found at {vanilla_path}; skipping", flush=True)

    if curve_path.exists():
        phase2["curve"] = phase2_atom_correlations_curve(curve_path, X_n, N_arr, device)
        print(f"[phase2] curve SAE   F={cfg.sae_F_to_probe}: best |ρ| pos = {phase2['curve']['best_position_abs']:.3f}, amp = {phase2['curve']['best_amplitude_abs']:.3f}", flush=True)
        c_alive_pos = sum(1 for r in phase2["curve"]["spearman_position_N"] if abs(r) > 0.7) + \
                       sum(1 for r in phase2["curve"]["spearman_position_logN"] if abs(r) > 0.7)
        phase2["curve"]["n_atoms_position_above_threshold"] = c_alive_pos
        print(f"           atoms with |position ρ| > 0.7: {c_alive_pos}", flush=True)
        if "vanilla" in phase2:
            plot_phase2_best_atom(phase2["curve"], phase2["vanilla"], N_arr, out_dir)
    else:
        print(f"[phase2] curve checkpoint not found at {curve_path}; skipping", flush=True)

    report = {
        "config": asdict(cfg),
        "N_values": N_arr.tolist(),
        "phase1": phase1,
        "best_layer": best_L,
        "best_layer_rho": best_rho,
        "phase2": phase2,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[done] wrote {out_dir / 'results.json'}", flush=True)
    print(f"[done] plots at {out_dir}/phase1_manifold_existence.png and (if phase 2 ran) phase2_best_curve_atom.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
