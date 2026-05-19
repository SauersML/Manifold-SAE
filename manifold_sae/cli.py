"""CLI entry point for Manifold-SAE."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Iterator

import torch

from .sae import ManifoldSAE, ManifoldSAEConfig
from .train import build_optimizer, train


def _synthetic_loader(input_dim: int, batch_size: int) -> Iterator[torch.Tensor]:
    while True:
        yield torch.randn(batch_size, input_dim)


def _load_data_loader(path: Path | None, input_dim: int, batch_size: int) -> Iterator[torch.Tensor]:
    if path is None:
        return _synthetic_loader(input_dim, batch_size)

    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "activations" in raw:
        raw = raw["activations"]
    tensor: torch.Tensor = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw, dtype=torch.float32)

    def _iter() -> Iterator[torch.Tensor]:
        n = tensor.shape[0]
        while True:
            idx = torch.randint(0, n, (batch_size,))
            yield tensor[idx]

    return _iter()


def _cmd_train(args: argparse.Namespace) -> None:
    config = ManifoldSAEConfig(
        input_dim=args.input_dim,
        n_features=args.n_features,
        n_basis=args.n_basis,
        sparsity_weight=args.sparsity_weight,
        reml_weight=args.reml_weight,
        position_spread_weight=args.position_spread_weight,
    )
    sae = ManifoldSAE(config)
    optimizer = build_optimizer(sae, lr=args.lr)
    loader = _load_data_loader(Path(args.data) if args.data else None, args.input_dim, args.batch_size)
    history = train(sae, loader, optimizer, n_steps=args.n_steps, log_every=args.log_every)
    print(f"Done. Final step={history['step'][-1] if history['step'] else 0}")


def _cmd_diagnose(args: argparse.Namespace) -> None:
    print(f"diagnose: not implemented (would load {args.checkpoint}); see experiments/ for full diagnostics.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manifold-sae")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a ManifoldSAE")
    p_train.add_argument("--input-dim", type=int, required=True)
    p_train.add_argument("--n-features", type=int, required=True)
    p_train.add_argument("--n-basis", type=int, required=True)
    p_train.add_argument("--n-steps", type=int, default=1000)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--sparsity-weight", type=float, default=1e-3)
    p_train.add_argument("--reml-weight", type=float, default=1e-2)
    p_train.add_argument("--position-spread-weight", type=float, default=1e-3)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--data", type=str, default=None, help="optional .pt file; synthetic if omitted")
    p_train.set_defaults(func=_cmd_train)

    p_diag = sub.add_parser("diagnose", help="Run diagnostics on a checkpoint (stub)")
    p_diag.add_argument("--checkpoint", type=str, required=True)
    p_diag.set_defaults(func=_cmd_diagnose)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
