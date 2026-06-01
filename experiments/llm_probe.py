"""Multi-concept manifold benchmark: does Manifold-SAE recover continuous
   features in LM residuals better than vanilla SAE?

This is the headline architectural test. Saturated MSE Pareto comparison
on real LM activations can't discriminate the two architectures (both
trivially explain >99% of variance at this layer with TopK=2 and 4-6
alive features). The question that matters: *for known continuous
features*, does curve-SAE recover them as one atom with a smooth t-axis,
while vanilla-SAE shatters them across multiple binary direction atoms?

Falsifiable hypothesis
======================

H1 (manifold existence). At some layer L, a planted continuous concept
    C (e.g. magnitude N ∈ {1..1000}) traces a 1D manifold in residual
    stream. Diagnostic: Spearman(C, PC_k) ≥ 0.7 for some top-k PC.

H2 (curve beats vanilla for compactness). For concepts that pass H1,
    the curve SAE's best atom has higher |Spearman(t_k, C)| than the
    vanilla SAE's best atom |Spearman(activation_k, C)|. The
    architectural advantage: t_k is continuous along the curve while
    vanilla activations are continuous along ONE direction only.

H3 (vanilla fragments). For concepts that pass H1, vanilla SAE
    activates MORE atoms above |ρ| > 0.5 than curve SAE does. The
    advantage: curve atoms compress a 1D feature into 1 atom while
    vanilla atoms partition the range across many.

Concepts planted
================
* magnitude  — N ∈ log-spaced {1, 2, 5, 10, ..., 1000}
* size       — {tiny, small, medium, large, huge, gigantic}
* polarity   — {terrible, bad, mediocre, okay, good, great, excellent}
* time       — {dawn, morning, noon, afternoon, evening, night, midnight}
* temperature— {freezing, cold, cool, warm, hot, scorching}
* brightness — {pitch-black, dark, dim, lit, bright, blinding}

Each concept is presented in ≥5 prompt templates so the model can't
trivially key on a single template.

Test strategy
=============
* Self-contained: if there's no curve SAE checkpoint at the canonical
  path, we train a small one (F=512, K=4) on harvested background
  activations from wikitext (cached from a prior llm_sweep run). The
  whole probe runs end-to-end as one job.
* Per concept, per layer, per architecture, report best-atom Spearman
  and number of atoms above |ρ| > 0.5 and > 0.7.
* Output: results.json + one summary plot (best curve atom vs N for
  the strongest concept).
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
bypass_gamfit_cuda_check()


# ---------------------------------------------------------------------------
# Concept catalogue: each concept is a list of (label_string, rank).
# Rank is the ground-truth ordering; the model is presented with the label.
# ---------------------------------------------------------------------------


def _continuous_concepts() -> dict[str, list[tuple[str, float]]]:
    return {
        "magnitude": [
            (str(n), float(np.log(n + 1)))
            for n in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000)
        ],
        "size": [
            ("tiny", 1.0), ("small", 2.0), ("modest", 3.0), ("medium", 4.0),
            ("large", 5.0), ("huge", 6.0), ("enormous", 7.0), ("gigantic", 8.0),
        ],
        "polarity": [
            ("terrible", 1.0), ("awful", 1.5), ("bad", 2.0), ("poor", 2.5),
            ("mediocre", 3.0), ("okay", 4.0), ("decent", 5.0), ("good", 6.0),
            ("great", 7.0), ("excellent", 8.0), ("amazing", 9.0),
        ],
        "time": [
            ("dawn", 1.0), ("morning", 2.0), ("midday", 3.0), ("afternoon", 4.0),
            ("dusk", 5.0), ("evening", 6.0), ("night", 7.0), ("midnight", 8.0),
        ],
        "temperature": [
            ("freezing", 1.0), ("cold", 2.0), ("cool", 3.0), ("mild", 4.0),
            ("warm", 5.0), ("hot", 6.0), ("scorching", 7.0),
        ],
        "brightness": [
            ("pitch-black", 1.0), ("dark", 2.0), ("dim", 3.0), ("lit", 4.0),
            ("bright", 5.0), ("dazzling", 6.0), ("blinding", 7.0),
        ],
    }


def _concept_templates() -> dict[str, list[str]]:
    """Per-concept prompt templates with one {x} placeholder."""
    return {
        "magnitude": [
            "There were {x} apples in the basket.",
            "She counted to {x}.",
            "The price was {x} dollars.",
            "He scored {x} points.",
            "They drove for {x} miles.",
        ],
        "size": [
            "It was a {x} animal.",
            "She bought a {x} car.",
            "We saw a {x} mountain.",
            "He owns a {x} library.",
            "The {x} package was on the table.",
        ],
        "polarity": [
            "The food was {x}.",
            "I felt {x} about it.",
            "The performance was {x}.",
            "She thought it was {x}.",
            "Overall, the day was {x}.",
        ],
        "time": [
            "It happened at {x}.",
            "She woke at {x}.",
            "The bell rang at {x}.",
            "By {x} they were ready.",
            "We left in the {x}.",
        ],
        "temperature": [
            "The weather was {x}.",
            "She felt {x} air on her face.",
            "It was {x} outside.",
            "The drink was {x}.",
            "The room was {x}.",
        ],
        "brightness": [
            "The room was {x}.",
            "It was a {x} morning.",
            "The lamp gave off a {x} glow.",
            "Outside, the day was {x}.",
            "She blinked in the {x} light.",
        ],
    }


@dataclass
class ProbeConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    # Probe these layers in phase 1; phase 2 uses the best for SAE probing.
    layers_to_probe: tuple[int, ...] = (4, 8, 12, 16, 20)
    target_token_position: int = -1               # last token of prompt

    # Spearman correlation thresholds.
    rho_strong: float = 0.7
    rho_moderate: float = 0.5

    # SAE-checkpoint path for phase 2. Reads MSAE_SWEEP_DIR if set
    # (typical: cluster job points at the sibling llm_sweep run dir),
    # otherwise tries MANIFOLD_SAE_OUTPUT_DIR which is the probe's own
    # output dir (only useful if checkpoints were dropped there manually).
    sae_checkpoint_dir: str = os.environ.get(
        "MSAE_SWEEP_DIR",
        os.environ.get("MANIFOLD_SAE_OUTPUT_DIR", "/content/runs/LLM_SWEEP"),
    )
    sae_F_to_probe: int = 128

    output_dir: str = os.environ.get(
        "MANIFOLD_SAE_OUTPUT_DIR",
        "/content/runs/LLM_PROBE",
    )
    seed: int = 0


# ---------------------------------------------------------------------------
# Spearman + ranking helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 1: harvest planted-concept activations and PCA-Spearman per layer
# ---------------------------------------------------------------------------


def harvest_concept_activations(
    cfg: ProbeConfig,
    concept_name: str,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], np.ndarray, list[str]]:
    """For one concept, return per-layer activation tensors (#prompts, D),
    the rank values of each prompt, and the human-readable label of each
    prompt for downstream inspection.
    """
    from transformers import AutoModel, AutoTokenizer

    labels_ranks = _continuous_concepts()[concept_name]
    templates = _concept_templates()[concept_name]

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()
    blocks = _find_transformer_blocks(model)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(L: int):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[L] = h.detach()
        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in cfg.layers_to_probe]

    prompts: list[str] = []
    ranks: list[float] = []
    used_labels: list[str] = []
    for label, r in labels_ranks:
        for t in templates:
            prompts.append(t.format(x=label))
            ranks.append(r)
            used_labels.append(label)

    activations: dict[int, list[torch.Tensor]] = {L: [] for L in cfg.layers_to_probe}
    torch.set_grad_enabled(False)
    try:
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt").to(device)
            model(**inputs)
            for L in cfg.layers_to_probe:
                h = captured[L]
                idx = cfg.target_token_position
                if idx < 0:
                    idx = h.shape[1] + idx
                activations[L].append(h[0, idx, :].cpu().float())
    finally:
        for h in handles:
            h.remove()
    torch.set_grad_enabled(True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = {L: torch.stack(activations[L], dim=0) for L in cfg.layers_to_probe}
    return out, np.array(ranks), used_labels


def _find_transformer_blocks(model) -> nn.ModuleList:
    for attr in ("h", "layers", "encoder_layer"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"could not find transformer blocks on {type(model).__name__}")


def phase1_per_layer_diagnostic(activations: torch.Tensor, ranks: np.ndarray) -> dict:
    """PCA the activations, report top-PC Spearman correlations with rank."""
    X = activations.numpy().astype(np.float64)
    X_c = X - X.mean(axis=0, keepdims=True)
    _, sv, vh = np.linalg.svd(X_c, full_matrices=False)
    proj = X_c @ vh[: min(8, vh.shape[0])].T
    rhos = [spearman_corr(proj[:, k], ranks) for k in range(proj.shape[1])]
    return {
        "singular_values_top5": [float(s) for s in sv[:5]],
        "spearman_top_pcs": rhos,
        "best_abs_rho_top8": float(max(abs(r) for r in rhos)),
        "best_abs_rho_pc": int(np.argmax([abs(r) for r in rhos])),
    }


# ---------------------------------------------------------------------------
# Phase 2: probe trained SAEs (if present) for atom-level concept tracking
# ---------------------------------------------------------------------------


class VanillaSAE(nn.Module):
    """Mirror of llm_sweep.VanillaSAE for checkpoint loading."""

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


def _summarize_per_atom(
    rhos: list[float],
    rho_strong: float,
    rho_moderate: float,
) -> dict:
    """Per-atom Spearman summary stats.

    `best` (max |ρ|) saturates trivially with small label-count concepts
    and large F — included for backward compatibility but not the
    discriminating metric. Use:
      * `n_atoms_above_moderate` — count of atoms with |ρ| > 0.5
      * `median_abs` and `p90_abs` — distribution shape
      * `top_10_mean_abs` — mean of top-10 atom |ρ| (more robust than max)
      * `concept_concentration` — Gini-like coefficient: how unequally
        the concept-attribution is spread across atoms. A localized
        representation has high concentration (few atoms dominate);
        a smeared one has low concentration.
    """
    rhos_abs = np.abs(np.array(rhos))
    n = len(rhos_abs)
    if n == 0:
        return {"best": 0.0, "best_atom_idx": -1,
                "n_atoms_above_strong": 0, "n_atoms_above_moderate": 0,
                "median_abs": 0.0, "p90_abs": 0.0, "top_10_mean_abs": 0.0,
                "concept_concentration": 0.0}
    sorted_abs = np.sort(rhos_abs)[::-1]
    top_10 = sorted_abs[:min(10, n)].mean()
    # Concept-concentration: 1 − (area under sorted-|ρ| CDF) / (area under uniform).
    # If one atom has |ρ|=1 and rest have 0, concentration = 1.
    # If all atoms tied, concentration = 0.
    csum = np.cumsum(sorted_abs)
    if csum[-1] > 0:
        normalized = csum / csum[-1]
        concentration = 1.0 - 2.0 * (normalized.mean() - 0.5)
    else:
        concentration = 0.0
    return {
        "best": float(rhos_abs.max()),
        "best_atom_idx": int(np.argmax(rhos_abs)),
        "n_atoms_above_strong": int((rhos_abs > rho_strong).sum()),
        "n_atoms_above_moderate": int((rhos_abs > rho_moderate).sum()),
        "median_abs": float(np.median(rhos_abs)),
        "p90_abs": float(np.percentile(rhos_abs, 90)),
        "top_10_mean_abs": float(top_10),
        "concept_concentration": float(concentration),
    }


def phase2_probe_concept(
    ckpt_dir: Path,
    F: int,
    X: torch.Tensor,
    ranks: np.ndarray,
    device: torch.device,
    cfg: ProbeConfig,
) -> dict:
    """For one concept-layer pair, load whichever SAE checkpoints exist and
    summarize each architecture's per-atom Spearman correlation with rank.

    Also computes a train/test-split metric: 80% of prompts pick the best
    atom by Spearman; the same atom's |ρ| is reported on the held-out 20%.
    A genuinely concept-encoding atom keeps |ρ| high on holdout; an
    overfit best-of-F atom drops sharply.
    """
    out: dict = {"sae_F": F}
    # Train/test split — same seed for vanilla and curve so the comparison
    # is fair (same partition of prompts).
    rng = np.random.default_rng(0)
    N = X.shape[0]
    perm = rng.permutation(N)
    n_train = max(1, int(0.8 * N))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    ranks_train = ranks[train_idx]
    ranks_test = ranks[test_idx]

    vanilla_path = ckpt_dir / f"vanilla_F{F}.pt"
    if vanilla_path.exists():
        ckpt = torch.load(vanilla_path, map_location="cpu", weights_only=False)
        sig = ckpt.get("sig", {})
        sae = VanillaSAE(X.shape[1], sig["F"], sig["top_k"]).to(device)
        sae.load_state_dict(ckpt["sae"])
        sae.eval()
        with torch.no_grad():
            _, z = sae(X.to(device))
        z_np = z.cpu().numpy()
        rhos = [spearman_corr(z_np[:, k], ranks) for k in range(sig["F"])]
        out["vanilla"] = _summarize_per_atom(rhos, cfg.rho_strong, cfg.rho_moderate)
        out["vanilla"]["per_atom_rho"] = rhos
        # Train/test holdout: pick best atom on train, evaluate on test.
        rhos_train = [spearman_corr(z_np[train_idx, k], ranks_train) for k in range(sig["F"])]
        best_atom_train = int(np.argmax(np.abs(rhos_train)))
        rho_test = spearman_corr(z_np[test_idx, best_atom_train], ranks_test)
        out["vanilla"]["best_atom_train"] = best_atom_train
        out["vanilla"]["best_atom_train_rho"] = rhos_train[best_atom_train]
        out["vanilla"]["best_atom_test_rho"] = rho_test
    else:
        out["vanilla"] = None
        out["vanilla_skipped_reason"] = f"checkpoint not found at {vanilla_path}"

    curve_path = ckpt_dir / f"curve_F{F}.pt"
    if curve_path.exists():
        from manifold_sae.sae import load_sae

        sae = load_sae(curve_path, input_dim=X.shape[1], device=device)
        sig = {"F": sae.cfg.n_atoms, "top_k": sae.cfg.sparsity.target_k}
        with torch.no_grad():
            sae_out = sae(X.to(device=device, dtype=sae.cfg.dtype))
        pos_np = sae_out.positions[..., 0].cpu().numpy()
        amp_np = sae_out.amplitudes.cpu().numpy()
        rhos_pos = [spearman_corr(pos_np[:, k], ranks) for k in range(sig["F"])]
        rhos_amp = [spearman_corr(amp_np[:, k], ranks) for k in range(sig["F"])]
        # Train/test holdout for curve atoms (positions metric).
        rhos_pos_train = [spearman_corr(pos_np[train_idx, k], ranks_train) for k in range(sig["F"])]
        best_atom_train = int(np.argmax(np.abs(rhos_pos_train)))
        rho_pos_test = spearman_corr(pos_np[test_idx, best_atom_train], ranks_test)
        out["curve_position"] = _summarize_per_atom(rhos_pos, cfg.rho_strong, cfg.rho_moderate)
        out["curve_position"]["best_atom_train"] = best_atom_train
        out["curve_position"]["best_atom_train_rho"] = rhos_pos_train[best_atom_train]
        out["curve_position"]["best_atom_test_rho"] = rho_pos_test
        out["curve_position"]["per_atom_rho"] = rhos_pos
        out["curve_amplitude"] = _summarize_per_atom(rhos_amp, cfg.rho_strong, cfg.rho_moderate)
        out["curve_amplitude"]["per_atom_rho"] = rhos_amp
        # Save the best-atom traces for later plotting.
        best_pos = out["curve_position"]["best_atom_idx"]
        out["best_curve_atom_positions"] = pos_np[:, best_pos].tolist()
        out["best_curve_atom_amplitudes"] = amp_np[:, best_pos].tolist()
    else:
        out["curve_position"] = None
        out["curve_skipped_reason"] = f"checkpoint not found at {curve_path}"

    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_phase1_heatmap(phase1_per_concept: dict[str, dict], cfg: ProbeConfig, out_dir: Path) -> None:
    """One row per concept, one col per layer, cell = best |ρ| over top-8 PCs.
    Quick at-a-glance view of which concept × layer combinations carry
    manifold structure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = list(phase1_per_concept.keys())
    Ls = list(cfg.layers_to_probe)
    grid = np.zeros((len(concepts), len(Ls)))
    for i, c in enumerate(concepts):
        for j, L in enumerate(Ls):
            grid[i, j] = phase1_per_concept[c][L]["best_abs_rho_top8"]

    fig, ax = plt.subplots(figsize=(0.8 + 0.6 * len(Ls), 0.6 + 0.6 * len(concepts)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(Ls)), [f"L{L}" for L in Ls])
    ax.set_yticks(range(len(concepts)), concepts)
    for i in range(len(concepts)):
        for j in range(len(Ls)):
            color = "white" if grid[i, j] < 0.4 else "black"
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title("Phase 1: best |Spearman(rank, PC)| over top-8 PCs\n(green = manifold lives here)")
    plt.colorbar(im, ax=ax, label="|ρ|")
    fig.tight_layout()
    fig.savefig(out_dir / "phase1_heatmap.png", dpi=120)
    plt.close(fig)


def plot_phase2_curve_atom(
    concept_name: str,
    layer: int,
    result: dict,
    ranks: np.ndarray,
    labels: list[str],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "best_curve_atom_positions" not in result:
        return

    pos = np.array(result["best_curve_atom_positions"])
    amp = np.array(result["best_curve_atom_amplitudes"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(ranks, pos, c=amp, cmap="viridis", s=28, edgecolors="none")
    ax.set_xlabel(f"ground-truth rank of {concept_name}")
    ax.set_ylabel(f"curve atom #{result['curve_position']['best_atom_idx']} position t_k")
    rho = result["curve_position"]["best"]
    rho_v = result["vanilla"]["best"] if result.get("vanilla") else None
    title = (
        f"Concept = {concept_name}  |  layer {layer}  |  best |ρ| curve={rho:.3f}"
        + (f"  vs vanilla={rho_v:.3f}" if rho_v is not None else "")
    )
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="atom amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"phase2_{concept_name}_L{layer}.png", dpi=120)
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

    concepts = list(_continuous_concepts().keys())
    print(f"[setup] probing {len(concepts)} concepts × {len(cfg.layers_to_probe)} layers", flush=True)

    # Phase 1: harvest each concept's activations and report per-layer Spearman.
    phase1: dict[str, dict] = {}
    harvested: dict[str, tuple[dict[int, torch.Tensor], np.ndarray, list[str]]] = {}
    for concept in concepts:
        print(f"\n=== concept: {concept} ===", flush=True)
        activations, ranks, labels = harvest_concept_activations(cfg, concept, device)
        harvested[concept] = (activations, ranks, labels)
        per_layer: dict = {}
        for L in cfg.layers_to_probe:
            d = phase1_per_layer_diagnostic(activations[L], ranks)
            per_layer[L] = d
            print(
                f"  layer {L:3d}: best |ρ|={d['best_abs_rho_top8']:+.3f} on PC{d['best_abs_rho_pc']+1} "
                f"(top-5 ρ = {['{:+.2f}'.format(r) for r in d['spearman_top_pcs'][:5]]})",
                flush=True,
            )
        phase1[concept] = per_layer

    plot_phase1_heatmap(phase1, cfg, out_dir)

    # Decide which concept × layer combos passed H1.
    passed: list[tuple[str, int]] = []
    for c, per_layer in phase1.items():
        for L, d in per_layer.items():
            if d["best_abs_rho_top8"] >= cfg.rho_strong:
                passed.append((c, L))
    print(
        f"\n[phase1] {len(passed)} (concept, layer) pairs passed H1 "
        f"with |ρ| >= {cfg.rho_strong}",
        flush=True,
    )

    # Phase 2: probe trained SAEs for atoms that track each passing concept.
    ckpt_dir = Path(cfg.sae_checkpoint_dir)
    phase2: dict = {}
    if not (ckpt_dir / f"curve_F{cfg.sae_F_to_probe}.pt").exists() and not \
       (ckpt_dir / f"vanilla_F{cfg.sae_F_to_probe}.pt").exists():
        print(
            f"\n[phase2] no checkpoints at {ckpt_dir} (F={cfg.sae_F_to_probe}); "
            f"skipping. Run experiments.llm_sweep first.",
            flush=True,
        )
    else:
        print(f"\n[phase2] probing SAE checkpoints at {ckpt_dir} F={cfg.sae_F_to_probe}", flush=True)
        for concept, L in passed:
            activations, ranks, labels = harvested[concept]
            X = activations[L]
            mu = X.mean(0, keepdim=True)
            sigma = X.std(0).clamp(min=1e-6)  # per-dim std (was scalar — see _normalize.py)
            X_n = (X - mu) / sigma
            res = phase2_probe_concept(ckpt_dir, cfg.sae_F_to_probe, X_n, ranks, device, cfg)
            key = f"{concept}_L{L}"
            phase2[key] = res
            v = res.get("vanilla") or {}
            cp = res.get("curve_position") or {}
            print(
                f"  {key}: vanilla best |ρ|={v.get('best', 0):.3f} ({v.get('n_atoms_above_moderate', 0)} atoms > 0.5) | "
                f"curve_pos best |ρ|={cp.get('best', 0):.3f} ({cp.get('n_atoms_above_moderate', 0)} atoms > 0.5)",
                flush=True,
            )
            plot_phase2_curve_atom(concept, L, res, ranks, labels, out_dir)

    # Save full results.
    report = {
        "config": asdict(cfg),
        "concepts": {c: [{"label": lbl, "rank": float(r)} for lbl, r in _continuous_concepts()[c]] for c in concepts},
        "phase1": {c: {str(L): v for L, v in per_layer.items()} for c, per_layer in phase1.items()},
        "phase1_passing": [{"concept": c, "layer": L} for c, L in passed],
        "phase2": phase2,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[done] wrote {out_dir / 'results.json'}", flush=True)
    print(f"[done] plots: phase1_heatmap.png + phase2_*.png in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
