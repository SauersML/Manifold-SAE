"""LLM-activation pipeline: harvest, train Manifold-SAE, analyze topology.

Three top-level entrypoints, each driven by a frozen Config dataclass at the
bottom of this file. No CLI. To run, edit ``DEFAULT_*_CONFIG`` or import this
module and call ``harvest(HarvestConfig(...))`` / ``train_sae(TrainConfig(...))``
/ ``analyze(AnalyzeConfig(...))``.

The headline experiment: train on cyclic-concept activations, then in
``analyze`` look for the feature whose mean amplitude is largest on the
days-of-week prompts. If the inferred ``t`` values for that feature wrap
nicely around the unit interval when sorted Mon -> Sun, we've discovered the
days circle unsupervised.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import torch

# Make sibling-package import work whether run as a script or via python -m.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_sae.data_activations import (
    ActivationDataset,
    harvest_activations,
    iter_batches,
    load_prompts,
    partition_by_category,
)


# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HarvestConfig:
    model: str = "meta-llama/Meta-Llama-3.1-8B"
    layer: int = 28
    prompts: str = "experiments/prompts_cyclic.json"
    output: str = "runs/llama31_8b_l28.pt"
    batch_size: int = 4
    device: str = "cuda"
    dtype: str = "float16"
    target_token_strategy: str = "value_token"
    max_prompts: int | None = None


@dataclass(frozen=True)
class TrainConfig:
    activations: str = "runs/llama31_8b_l28.pt"
    n_features: int = 64
    n_basis: int = 12
    n_steps: int = 5000
    batch_size: int = 64
    lr: float = 1e-3
    sparsity_weight: float = 1e-3
    reml_weight: float = 1e-2
    log_every: int = 50
    output_dir: str = "runs/sae_days_v1"
    seed: int = 0
    device: str = "cuda"


@dataclass(frozen=True)
class AnalyzeConfig:
    sae_checkpoint: str = "runs/sae_days_v1/sae.pt"
    activations: str = "runs/llama31_8b_l28.pt"
    output_dir: str = "runs/sae_days_v1/analysis"
    n_grid: int = 128
    device: str = "cpu"


DEFAULT_HARVEST_CONFIG = HarvestConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_ANALYZE_CONFIG = AnalyzeConfig()


# ----------------------------------------------------------------------
# harvest
# ----------------------------------------------------------------------


def harvest(cfg: HarvestConfig = DEFAULT_HARVEST_CONFIG) -> None:
    prompts = load_prompts(cfg.prompts)
    print(f"[harvest] {len(prompts)} prompts loaded from {cfg.prompts}")

    result = harvest_activations(
        model_name=cfg.model,
        prompts=prompts,
        layer=cfg.layer,
        target_token_strategy=cfg.target_token_strategy,
        device=cfg.device,
        batch_size=cfg.batch_size,
        output_path=cfg.output,
        dtype=cfg.dtype,
        max_prompts=cfg.max_prompts,
    )
    print(f"[harvest] done — {result['n_records']} records at {result['output_path']}")


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


def train_sae(cfg: TrainConfig = DEFAULT_TRAIN_CONFIG) -> None:
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig
    from manifold_sae.train import build_optimizer, train

    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)

    ds = ActivationDataset(cfg.activations, dtype=torch.float32)
    print(f"[train] {len(ds)} activations of dim {ds.dim}; meta={ds.meta}")

    sae_config = ManifoldSAEConfig(
        input_dim=ds.dim,
        n_features=cfg.n_features,
        n_basis=cfg.n_basis,
        sparsity_weight=cfg.sparsity_weight,
        reml_weight=cfg.reml_weight,
    )
    sae = ManifoldSAE(sae_config)

    device = torch.device(cfg.device)
    sae.to(device)
    activations = ds.activations.to(device)

    loader = iter_batches(activations, batch_size=cfg.batch_size, shuffle=True, generator=g)
    optimizer = build_optimizer(sae, lr=cfg.lr)

    history = train(sae, loader, optimizer, n_steps=cfg.n_steps, log_every=cfg.log_every)

    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, "sae.pt")
    torch.save(
        {"config": sae_config.__dict__, "state_dict": sae.state_dict(), "meta": ds.meta},
        ckpt_path,
    )
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] saved checkpoint to {ckpt_path}")


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------


def _curve_metrics(curve: torch.Tensor) -> dict:
    """Closedness, length, mean curvature for a single (T, D) curve."""
    diffs = curve[1:] - curve[:-1]
    seg_lens = diffs.norm(dim=-1)
    length = seg_lens.sum().item()
    closure_gap = (curve[-1] - curve[0]).norm().item()
    closedness = closure_gap / (length + 1e-12)

    # Discrete curvature: angle between consecutive tangents.
    if diffs.shape[0] >= 2:
        t1 = diffs[:-1]
        t2 = diffs[1:]
        n1 = t1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n2 = t2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos = ((t1 / n1) * (t2 / n2)).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.acos(cos)
        mean_curv = angles.mean().item()
        max_curv = angles.max().item()
    else:
        mean_curv = max_curv = 0.0

    return {
        "length": length,
        "closure_gap": closure_gap,
        "closedness": closedness,
        "mean_curvature": mean_curv,
        "max_curvature": max_curv,
    }


def _decode_curves_from_data(
    sae,
    activations: torch.Tensor,
    t_grid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Probe each feature's learned curve on a position grid.

    Thin wrapper over :func:`manifold_sae.decoder.extract_feature_curves` —
    kept here so the analyze subcommand has one obvious entrypoint.
    """
    from manifold_sae.decoder import extract_feature_curves

    return extract_feature_curves(sae, activations.to(device), t_grid)


def analyze(cfg: AnalyzeConfig = DEFAULT_ANALYZE_CONFIG) -> None:
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig

    os.makedirs(cfg.output_dir, exist_ok=True)

    ckpt = torch.load(cfg.sae_checkpoint, map_location="cpu")
    sae_cfg = ManifoldSAEConfig(**ckpt["config"])
    sae = ManifoldSAE(sae_cfg)
    sae.load_state_dict(ckpt["state_dict"])

    device = torch.device(cfg.device)
    sae.to(device)
    sae.eval()

    ds = ActivationDataset(cfg.activations, dtype=torch.float32)
    activations = ds.activations.to(device)

    with torch.no_grad():
        out = sae(activations)
    positions = out.positions.cpu()  # (N, F)
    amplitudes = out.amplitudes.cpu()  # (N, F)

    F = sae_cfg.n_features
    t_grid = torch.linspace(0.0, 1.0, cfg.n_grid)

    curves = _decode_curves_from_data(sae, activations, t_grid, device).cpu()  # (F, T, D)

    # Per-feature metrics
    per_feature: list[dict] = []
    for k in range(F):
        m = _curve_metrics(curves[k])
        m["feature"] = k
        m["mean_amplitude"] = float(amplitudes[:, k].mean().item())
        m["mean_position"] = float(positions[:, k].mean().item())
        m["position_var"] = float(positions[:, k].var(unbiased=False).item())
        per_feature.append(m)

    # Per-category candidate features
    partitions = partition_by_category(ds)
    category_summary: dict = {}
    for cat, sub in partitions.items():
        if len(sub) == 0:
            continue
        sub_amps = amplitudes[sub.indices]  # (n_cat, F)
        mean_per_feature = sub_amps.mean(dim=0)  # (F,)
        top = int(torch.argmax(mean_per_feature).item())
        sub_positions = positions[sub.indices, top].tolist()
        category_summary[cat] = {
            "candidate_feature": top,
            "mean_amplitude_on_category": float(mean_per_feature[top].item()),
            "values": sub.values(),
            "positions_on_candidate": sub_positions,
        }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # PCA the union of all curves -> 3D for a single 3D scatter per feature group.
    all_pts = curves.reshape(-1, curves.shape[-1]).numpy()
    pca = PCA(n_components=3).fit(all_pts)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for k in range(min(F, 16)):
        pts = pca.transform(curves[k].numpy())
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=f"feat {k}")
    ax.set_title("Learned curves in PCA(3) space")
    ax.legend(fontsize=6, loc="upper right")
    fig.savefig(os.path.join(cfg.output_dir, "curves_pca3.png"), dpi=150)
    plt.close(fig)

    # For each category, color its prompts on the candidate-feature t-axis.
    for cat, summary in category_summary.items():
        vals = summary["values"]
        ts = summary["positions_on_candidate"]
        uniq = sorted(set(vals))
        color_map = {v: i for i, v in enumerate(uniq)}
        colors = [color_map[v] for v in vals]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(ts, [0] * len(ts), c=colors, cmap="hsv", s=40)
        ax.set_yticks([])
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f"{cat}: candidate feature {summary['candidate_feature']}")
        for v in uniq:
            xs = [t for t, vv in zip(ts, vals, strict=False) if vv == v]
            if xs:
                ax.text(sum(xs) / len(xs), 0.02, v, fontsize=7, ha="center")
        fig.savefig(os.path.join(cfg.output_dir, f"category_{cat}_positions.png"), dpi=150)
        plt.close(fig)

    metrics_dump = {
        "per_feature": per_feature,
        "category_summary": category_summary,
        "config": ckpt["config"],
        "meta": ckpt.get("meta", {}),
    }
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dump, f, indent=2)
    print(f"[analyze] wrote metrics + figures to {cfg.output_dir}")


# ----------------------------------------------------------------------
# Default entrypoint: train on already-harvested activations.
# Edit the dataclass constructors below to override, or import and call.
# ----------------------------------------------------------------------


if __name__ == "__main__":
    train_sae(DEFAULT_TRAIN_CONFIG)
