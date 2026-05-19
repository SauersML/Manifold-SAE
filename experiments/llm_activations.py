"""Three-subcommand driver: harvest activations, train Manifold-SAE, analyze topology.

Usage:
    python -m experiments.llm_activations harvest --model meta-llama/Meta-Llama-3.1-8B \\
        --layer 28 --prompts experiments/prompts_cyclic.json \\
        --output runs/llama31_8b_l28.pt --batch-size 4 --device cuda

    python -m experiments.llm_activations train \\
        --activations runs/llama31_8b_l28.pt --n-features 64 --n-basis 12 \\
        --n-steps 5000 --batch-size 64 --lr 1e-3 \\
        --output-dir runs/sae_days_v1

    python -m experiments.llm_activations analyze \\
        --sae-checkpoint runs/sae_days_v1/sae.pt \\
        --activations runs/llama31_8b_l28.pt \\
        --output-dir runs/sae_days_v1/analysis

The headline experiment: train on cyclic-concept activations, then in ``analyze``
look for the feature whose mean amplitude is largest on the days-of-week prompts.
If the inferred ``t`` values for that feature wrap nicely around the unit
interval when sorted Mon -> Sun, we've discovered the days circle unsupervised.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# Make sibling-package import work whether run via ``python -m`` or directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_sae.data_activations import (  # noqa: E402
    ActivationDataset,
    harvest_activations,
    iter_batches,
    load_prompts,
    partition_by_category,
)


# ----------------------------------------------------------------------
# harvest
# ----------------------------------------------------------------------


def cmd_harvest(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts)
    print(f"[harvest] {len(prompts)} prompts loaded from {args.prompts}")

    result = harvest_activations(
        model_name=args.model,
        prompts=prompts,
        layer=args.layer,
        target_token_strategy=args.target_token_strategy,
        device=args.device,
        batch_size=args.batch_size,
        output_path=args.output,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
    )
    print(f"[harvest] done — {result['n_records']} records at {result['output_path']}")


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    # Heavy imports deferred — the harvest subcommand shouldn't pay this cost.
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig  # noqa: WPS433
    from manifold_sae.train import build_optimizer, train  # noqa: WPS433

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    ds = ActivationDataset(args.activations, dtype=torch.float32)
    print(f"[train] {len(ds)} activations of dim {ds.dim}; meta={ds.meta}")

    cfg = ManifoldSAEConfig(
        input_dim=ds.dim,
        n_features=args.n_features,
        n_basis=args.n_basis,
        sparsity_weight=args.sparsity_weight,
        reml_weight=args.reml_weight,
    )
    sae = ManifoldSAE(cfg)

    device = torch.device(args.device)
    sae.to(device)
    activations = ds.activations.to(device)

    loader = iter_batches(activations, batch_size=args.batch_size, shuffle=True, generator=g)
    optimizer = build_optimizer(sae, lr=args.lr)

    history = train(sae, loader, optimizer, n_steps=args.n_steps, log_every=args.log_every)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "sae.pt")
    torch.save(
        {
            "config": cfg.__dict__,
            "state_dict": sae.state_dict(),
            "meta": ds.meta,
        },
        ckpt_path,
    )
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
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


def cmd_analyze(args: argparse.Namespace) -> None:
    from manifold_sae.sae import ManifoldSAE, ManifoldSAEConfig  # noqa: WPS433

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.sae_checkpoint, map_location="cpu")
    cfg = ManifoldSAEConfig(**ckpt["config"])
    sae = ManifoldSAE(cfg)
    sae.load_state_dict(ckpt["state_dict"])

    device = torch.device(args.device)
    sae.to(device)
    sae.eval()

    ds = ActivationDataset(args.activations, dtype=torch.float32)
    activations = ds.activations.to(device)

    # Inference on the full dataset
    with torch.no_grad():
        out = sae(activations)
    positions = out.positions.cpu()  # (N, F)
    amplitudes = out.amplitudes.cpu()  # (N, F)

    F = cfg.n_features
    t_grid = torch.linspace(0.0, 1.0, args.n_grid)

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

    # Try plotting
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA  # type: ignore

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
        fig.savefig(os.path.join(args.output_dir, "curves_pca3.png"), dpi=150)
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
            fig.savefig(
                os.path.join(args.output_dir, f"category_{cat}_positions.png"), dpi=150
            )
            plt.close(fig)
    except ImportError as e:
        print(f"[analyze] plotting skipped ({e})")

    metrics_dump = {
        "per_feature": per_feature,
        "category_summary": category_summary,
        "config": ckpt["config"],
        "meta": ckpt.get("meta", {}),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dump, f, indent=2)
    print(f"[analyze] wrote metrics + figures to {args.output_dir}")


# ----------------------------------------------------------------------
# CLI plumbing
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manifold-SAE LLM activation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="harvest residual-stream activations")
    h.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    h.add_argument("--layer", type=int, default=28)
    h.add_argument("--prompts", default="experiments/prompts_cyclic.json")
    h.add_argument("--output", required=True)
    h.add_argument("--batch-size", type=int, default=4)
    h.add_argument("--device", default="cuda")
    h.add_argument("--dtype", default="bfloat16")
    h.add_argument(
        "--target-token-strategy",
        default="value_token",
        choices=["value_token", "last_token"],
    )
    h.add_argument("--max-prompts", type=int, default=None)
    h.set_defaults(func=cmd_harvest)

    t = sub.add_parser("train", help="train Manifold-SAE on harvested activations")
    t.add_argument("--activations", required=True)
    t.add_argument("--n-features", type=int, default=64)
    t.add_argument("--n-basis", type=int, default=12)
    t.add_argument("--n-steps", type=int, default=5000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--sparsity-weight", type=float, default=1e-3)
    t.add_argument("--reml-weight", type=float, default=1e-2)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--output-dir", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("analyze", help="post-hoc topology analysis on a trained SAE")
    a.add_argument("--sae-checkpoint", required=True)
    a.add_argument("--activations", required=True)
    a.add_argument("--output-dir", required=True)
    a.add_argument("--n-grid", type=int, default=64)
    a.add_argument("--device", default="cpu")
    a.set_defaults(func=cmd_analyze)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
