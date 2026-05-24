"""Integration layer — unified registry + driver for every SAE variant.

This module ties together the parallel-developed SAE variants under a
single API:

    * :data:`SAEModelRegistry` — name → builder callable mapping.
    * :func:`train_any` — construct + train any variant on the same data,
      report a common metric dict.
    * :func:`compose_pipeline` — chain SAEs sequentially where each
      consumes the previous reconstruction.

The driver layer intentionally does NOT modify any variant's internals.
Each variant exposes its own forward signature, loss shape, and save
format; we adapt them at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Per-variant builders
# ---------------------------------------------------------------------------


def _build_manifold(D: int, F: int, **kw) -> nn.Module:
    from .sae import ManifoldSAE, ManifoldSAEConfig

    cfg = ManifoldSAEConfig(
        input_dim=D,
        n_features=F,
        n_basis=int(kw.get("n_basis", 8)),
        top_k=int(kw.get("top_k", max(1, F // 4))),
        intrinsic_rank=int(kw.get("intrinsic_rank", 2)),
        sparsity_weight=float(kw.get("sparsity_weight", 1e-3)),
        ortho_weight=float(kw.get("ortho_weight", 1e-2)),
    )
    return ManifoldSAE(cfg)


def _build_adaptive_k(D: int, F: int, **kw) -> nn.Module:
    from .adaptive_k import AdaptiveKSAE

    return AdaptiveKSAE(
        input_dim=D,
        F=F,
        k_min=int(kw.get("k_min", max(1, F // 8))),
        k_max=int(kw.get("k_max", F)),
        sparsity_weight=float(kw.get("sparsity_weight", 1e-3)),
    )


def _build_crm(D: int, F: int, **kw) -> nn.Module:
    from .crm import CompleteReplacementModel, CRMConfig

    L = int(kw.get("n_layers", 2))
    return CompleteReplacementModel(
        CRMConfig(
            layer_dims=[D] * L,
            n_features_per_sae=F,
            sae_top_k=int(kw.get("top_k", max(1, F // 4))),
            transcoder_mid=int(kw.get("transcoder_mid", 2 * F)),
            transcoder_top_k=int(kw.get("transcoder_top_k", F // 2)),
        )
    )


def _build_crosscoder(D: int, F: int, **kw) -> nn.Module:
    from .crosscoder import Crosscoder

    L = int(kw.get("n_layers", 2))
    return Crosscoder(
        layer_dims=[D] * L,
        n_atoms=F,
        sparsity_weight=float(kw.get("sparsity_weight", 1e-3)),
        activation=str(kw.get("activation", "relu")),
        top_k=int(kw.get("top_k", max(1, F // 4))),
    )


def _build_equivariant(D: int, F: int, **kw) -> nn.Module:
    from .equivariant import EquivariantSAE, EquivariantSAEConfig

    n_so2 = int(kw.get("n_so2", max(2, F // 2)))
    n_trivial = F - n_so2
    return EquivariantSAE(
        EquivariantSAEConfig(
            d_in=D,
            n_so2=n_so2,
            n_trivial=n_trivial,
            sparsity_weight=float(kw.get("sparsity_weight", 1e-3)),
            eq_weight=float(kw.get("eq_weight", 1e-2)),
            ard_weight=float(kw.get("ard_weight", 1e-4)),
        )
    )


def _build_wasserstein(D: int, F: int, **kw) -> nn.Module:
    from .wasserstein_sae import WassersteinSAE

    return WassersteinSAE(
        F=F,
        M=int(kw.get("M", 8)),
        D=D,
        eps=float(kw.get("eps", 0.01)),
        n_sinkhorn_iter=int(kw.get("n_sinkhorn_iter", 10)),
        neighbor_weight=float(kw.get("neighbor_weight", 1e-3)),
    )


def _build_sheaf(D: int, F: int, **kw) -> nn.Module:
    from .sheaf import SheafConsistencyHead

    # Sheaf isn't a full SAE — it's a consistency head. We wrap it in a tiny
    # paired-SAE driver so the registry can train it like any other.
    return _SheafPair(D=D, F=F, n_layers=int(kw.get("n_layers", 2)))


class _SheafPair(nn.Module):
    """Two linear SAEs + a SheafConsistencyHead — minimal trainable wrapper."""

    def __init__(self, D: int, F: int, n_layers: int = 2) -> None:
        super().__init__()
        from .sheaf import SheafConsistencyHead

        self.n_layers = n_layers
        self.encoders = nn.ModuleList([nn.Linear(D, F) for _ in range(n_layers)])
        self.decoders = nn.ModuleList([nn.Linear(F, D) for _ in range(n_layers)])
        self.head = SheafConsistencyHead(n_layers=n_layers, F=F)

    def forward(self, x: torch.Tensor) -> dict:
        # replicate x across layers (good-enough for smoke tests; for real
        # use the caller supplies a list).
        xs = [x for _ in range(self.n_layers)]
        codes = [torch.relu(e(xi)) for e, xi in zip(self.encoders, xs)]
        recons = [d(z) for d, z in zip(self.decoders, codes)]
        return {"codes": codes, "recons": recons}

    def loss(self, x: torch.Tensor) -> dict:
        out = self.forward(x)
        mse = sum(((r - x) ** 2).mean() for r in out["recons"]) / self.n_layers
        cons = self.head.energy(out["codes"])
        total = mse + 1e-3 * cons
        return {"loss": total, "mse": mse.detach(), "consistency": cons.detach(),
                "recon": out["recons"][0]}


def _build_transcoder(D: int, F: int, **kw) -> nn.Module:
    """Linear-skip transcoder wrapper. Falls back to a plain Linear when
    SkipAffineSmooth is unavailable in the installed gamfit."""
    from .transcoder import SkipAffineSmooth

    if SkipAffineSmooth is None:
        return _LinearTranscoderStub(D, F)
    return SkipAffineSmooth(
        in_dim=D, out_dim=D, n_atoms=F,
        rank_skip=min(F, D),
        jumprelu_threshold=0.03, learnable_threshold=True,
        smoothing_eps=1e-3,
    )


class _LinearTranscoderStub(nn.Module):
    """One-layer linear stand-in when SkipAffineSmooth isn't installed."""

    def __init__(self, D: int, F: int) -> None:
        super().__init__()
        self.enc = nn.Linear(D, F)
        self.dec = nn.Linear(F, D)

    def forward(self, x: torch.Tensor):
        z = torch.relu(self.enc(x))
        return self.dec(z), z

    def loss(self, x: torch.Tensor) -> dict:
        y, z = self.forward(x)
        mse = ((y - x) ** 2).mean()
        return {"loss": mse + 1e-3 * z.abs().mean(), "mse": mse.detach(), "recon": y}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SAEModelRegistry: Dict[str, Callable[..., nn.Module]] = {
    "manifold": _build_manifold,
    "adaptive_k": _build_adaptive_k,
    "crm": _build_crm,
    "crosscoder": _build_crosscoder,
    "equivariant": _build_equivariant,
    "wasserstein": _build_wasserstein,
    "sheaf": _build_sheaf,
    "transcoder": _build_transcoder,
}


# ---------------------------------------------------------------------------
# Unified forward / loss adapter
# ---------------------------------------------------------------------------


def _call_loss(model: nn.Module, x: torch.Tensor, name: str) -> dict:
    """Adapt each variant's forward/loss to a common ``{loss, recon}`` dict."""
    # Multi-input variants need x replicated across layers.
    if name in ("crm", "crosscoder"):
        n_layers = getattr(model, "n_layers", 2) if name == "crosscoder" else model.L
        xs = [x for _ in range(n_layers)]
        if name == "crm":
            out = model.loss(xs)
            return {"loss": out["loss"], "recon": out["out"]["recons"][0]}
        out = model(xs)
        return {"loss": out["loss"], "recon": out["recons"][0]}

    if hasattr(model, "loss"):
        out = model.loss(x)
        # normalize total-key name
        loss = out.get("loss", out.get("total"))
        # NOTE: some variants (eg AdaptiveKSAE) name the scalar MSE
        # ``recon`` — so prefer ``recon_out``/``reconstruction`` first, then
        # ``recon`` only if it's tensor-shaped.
        recon = out.get("recon_out")
        if recon is None:
            recon = out.get("reconstruction")
        if recon is None:
            r = out.get("recon")
            if isinstance(r, torch.Tensor) and r.shape == x.shape:
                recon = r
        if recon is None:
            for v in out.values():
                if isinstance(v, torch.Tensor) and v.shape == x.shape:
                    recon = v
                    break
        return {"loss": loss, "recon": recon}

    # No .loss → call forward and synthesize MSE-only loss.
    out = model(x)
    if isinstance(out, dict):
        recon = out.get("recon", out.get("reconstruction"))
    elif isinstance(out, tuple):
        recon = out[0]
    else:
        recon = getattr(out, "reconstruction", None)
    if recon is None:
        raise RuntimeError(f"{name}: cannot find reconstruction in forward output")
    mse = ((recon - x) ** 2).mean()
    return {"loss": mse, "recon": recon}


# ---------------------------------------------------------------------------
# train_any
# ---------------------------------------------------------------------------


def train_any(
    name: str,
    X: torch.Tensor,
    *,
    F: int = 32,
    steps: int = 5,
    lr: float = 1e-3,
    optimizer: str = "sgd",
    **build_kwargs,
) -> dict:
    """Construct + briefly train a registered SAE variant; return metrics.

    Returns a dict with keys:
        ``name``, ``loss_initial``, ``loss_final``, ``loss_reduced``,
        ``recon_r2``, ``n_params``.
    """
    if name not in SAEModelRegistry:
        raise KeyError(f"Unknown SAE variant {name!r}; known: {list(SAEModelRegistry)}")

    D = X.shape[-1]
    model = SAEModelRegistry[name](D=D, F=F, **build_kwargs)

    opt_cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[optimizer]
    opt = opt_cls(model.parameters(), lr=lr)

    out0 = _call_loss(model, X, name)
    loss0 = float(out0["loss"].detach())

    for _ in range(steps):
        out = _call_loss(model, X, name)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()

    out_final = _call_loss(model, X, name)
    loss_final = float(out_final["loss"].detach())

    recon = out_final["recon"].detach()
    var = X.var().item()
    res = ((recon - X) ** 2).mean().item()
    r2 = 1.0 - res / max(var, 1e-12)

    return {
        "name": name,
        "model": model,
        "loss_initial": loss0,
        "loss_final": loss_final,
        "loss_reduced": loss_final < loss0,
        "recon_r2": r2,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------------------
# compose_pipeline
# ---------------------------------------------------------------------------


def _extract_recon(model: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    out = _call_loss(model, x, name)
    return out["recon"].detach()


def compose_pipeline(
    variants: Sequence[str],
    X: torch.Tensor,
    *,
    F: int = 32,
    steps: int = 2,
    **build_kwargs,
) -> dict:
    """Chain SAEs sequentially — each consumes the previous reconstruction.

    Trains each variant briefly on the residual stream it receives, then
    forwards the reconstruction to the next variant.

    Returns ``{"models": [...], "per_stage_r2": [...], "final_recon": Tensor}``.
    """
    models: List[nn.Module] = []
    per_stage_r2: List[float] = []
    current = X
    for name in variants:
        result = train_any(name, current, F=F, steps=steps, **build_kwargs)
        models.append(result["model"])
        per_stage_r2.append(result["recon_r2"])
        current = _extract_recon(result["model"], current, name)
    return {
        "models": models,
        "variants": list(variants),
        "per_stage_r2": per_stage_r2,
        "final_recon": current,
    }


__all__ = [
    "SAEModelRegistry",
    "train_any",
    "compose_pipeline",
]
