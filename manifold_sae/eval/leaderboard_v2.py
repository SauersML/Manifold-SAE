"""Leaderboard v2 — unified ranking across ALL trained SAE variants.

Extends ``manifold_sae.eval.registry`` with auto-discovery + loaders for
every checkpoint shipped under ``runs/``. Runs the full ``Harness`` plus
``SteeringBench`` on each loadable checkpoint, then ranks via a weighted
composite:

    composite = 0.40 * val_r2
              + 0.30 * steering_composite
              + 0.20 * hsv_coherence
              + 0.10 * (1 - dead_atom_fraction)

Each component is clamped to ``[0, 1]`` before weighting so a single
catastrophic axis cannot make the score negative or dominate via blowup.

Public API
----------
``scan_checkpoints(root)`` → list of ``CheckpointEntry`` for every
``*.pt`` under ``root``.

``load_checkpoint(path, d_in, device)`` → ``(SAEWrapper, load_meta)`` or
``(None, reason_str)`` on failure.

``run_leaderboard(root, data_path, output_dir, ...)`` → composes
everything: scan → load → Harness + SteeringBench → composite → write
``leaderboard.json``, ``leaderboard.md``, ``leaderboard_radar.png``.

Side-channel
------------
This module deliberately re-uses (does not modify) the existing
``manifold_sae.eval.harness.Harness`` and ``steering_bench.SteeringBench``
implementations. New variant loaders live here so we never touch
``registry.py`` (other agents may be editing it).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .harness import (
    Harness,
    HarnessLabels,
    HarnessResult,
    SAEWrapper,
    collect_activations,
)
from .registry import (
    L1Wrapper,
    ManifoldFourierWrapper,
    TopKWrapper,
    _L1SAE,
    _ManifoldFourierSAE,
    _TopKSAE,
)
from .run import prepare
from .steering_bench import BenchResult, SteeringBench


log = logging.getLogger("leaderboard_v2")


# ---------------------------------------------------------------------------
# Extra wrappers for variants not in registry.py
# ---------------------------------------------------------------------------


class _GenericLinearWrapper(SAEWrapper):
    """Wrapper for any SAE-like module exposing ``.encode(x) -> z`` (tensor
    or tuple whose first element is the dense activation) and a way to
    decode either via ``.decode(z)`` or ``.decode_from_acts`` / matmul on
    a stored decoder.
    """

    def __init__(
        self,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        name: str,
        n_features: int,
        input_dim: int,
        firing_threshold: float = 1e-3,
    ) -> None:
        self._enc = encode_fn
        self._dec = decode_fn
        self.name = name
        self.n_features = int(n_features)
        self.input_dim = int(input_dim)
        self.firing_threshold = firing_threshold

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._enc(x)

    def decode_from_activations(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._dec(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self._enc(x)
            return self._dec(z)


# ---------- variant-specific loaders ----------


def _load_adaptive_k(path: str, d_in: int, device: str) -> SAEWrapper:
    sd = torch.load(path, map_location=device, weights_only=True)
    F = sd["W_e"].shape[1]
    from manifold_sae.adaptive_k import AdaptiveKSAE

    model = AdaptiveKSAE(input_dim=d_in, F=F)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    def enc(x):
        recon, z_sparse, k_pred = model(x)
        return z_sparse

    def dec(z):
        return z @ model.W_d + model.b_d

    return _GenericLinearWrapper(
        enc, dec, name="adaptive_k", n_features=F, input_dim=d_in
    )


def _load_wasserstein(path: str, d_in: int, device: str) -> SAEWrapper:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]
    sd = blob["model"]
    from manifold_sae.wasserstein_sae import WassersteinSAE

    model = WassersteinSAE(
        F=int(cfg["F"]),
        M=int(cfg["M"]),
        D=int(cfg["D"]),
        eps=float(cfg.get("eps", 0.01)),
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    def enc(x):
        return model.encode(x)

    def dec(pi):
        recon, _ = model.decode(pi)
        return recon

    return _GenericLinearWrapper(
        enc, dec, name="wasserstein", n_features=int(cfg["F"]), input_dim=d_in,
    )


def _load_das_sae(path: str, d_in: int, device: str) -> SAEWrapper:
    sd = torch.load(path, map_location=device, weights_only=True)
    F = sd["W_enc"].shape[0]
    from manifold_sae.das_sae import DASSAE, DASSAEConfig

    model = DASSAE(DASSAEConfig(input_dim=d_in, n_features=F))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    def enc(x):
        z, _ = model.encode(x)
        return z

    def dec(z):
        return model.decode(z)

    return _GenericLinearWrapper(
        enc, dec, name="das_sae", n_features=F, input_dim=d_in
    )


def _load_sheaf(path: str, d_in: int, device: str) -> SAEWrapper:
    """Sheaf checkpoint stores 3 layer-specific SAEs of different widths.
    For the leaderboard we evaluate the first SAE only (its in/out dim
    must match d_in)."""
    blob = torch.load(path, map_location=device, weights_only=False)
    dims = blob["dims"]
    if dims[0] != d_in:
        raise RuntimeError(
            f"sheaf SAE[0] expects input dim {dims[0]}, harness data has {d_in}"
        )
    sd = blob["saes"]

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W_enc = torch.nn.Parameter(sd["0.W_enc"].clone())
            self.b_enc = torch.nn.Parameter(sd["0.b_enc"].clone())
            self.W_dec = torch.nn.Parameter(sd["0.W_dec"].clone())
            self.b_dec = torch.nn.Parameter(sd["0.b_dec"].clone())

        def encode(self, x):
            return torch.relu((x - self.b_dec) @ self.W_enc.t() + self.b_enc)

        def decode(self, z):
            return z @ self.W_dec.t() + self.b_dec

    m = _Tiny().to(device).eval()
    F = sd["0.b_enc"].shape[0]
    return _GenericLinearWrapper(
        m.encode, m.decode, name="sheaf_l0", n_features=F, input_dim=d_in
    )


# ---------------------------------------------------------------------------
# Auto-discovery + dispatch
# ---------------------------------------------------------------------------


@dataclass
class CheckpointEntry:
    name: str          # human-readable variant tag
    path: str
    variant: str       # registry key
    size_bytes: int
    mtime: float


def _kind_for(path: str) -> str | None:
    p = path.lower()
    name = os.path.basename(p)
    if "model_topk" in name:
        return "topk"
    if "model_l1" in name:
        return "l1"
    if "model_manifold" in name:
        return "manifold"
    if "adaptive_k" in p:
        return "adaptive_k"
    if "wasserstein" in p:
        return "wasserstein"
    if "das_sae" in p or "das-sae" in p:
        return "das_sae"
    if "sheaf" in p:
        return "sheaf"
    if "crosscoder" in p:
        return "crosscoder"  # not yet loadable here
    if "equivariant" in p:
        return "equivariant"
    if "matryoshka" in p:
        return "matryoshka"
    if "skip" in p or "transcoder" in p:
        return "transcoder"
    if "hyperbolic" in p:
        return "hyperbolic"
    if "identifiable" in p or "ivae" in p:
        return "ivae"
    if "crm" in p:
        return "crm"
    return None


_LOADERS: dict[str, Callable[[str, int, str], SAEWrapper]] = {
    "topk": lambda p, d, dev: TopKWrapper(
        _load_topk_module(p, d, dev), name="topk"
    ),
    "l1": lambda p, d, dev: L1Wrapper(
        _load_l1_module(p, d, dev), name="l1"
    ),
    "manifold": lambda p, d, dev: ManifoldFourierWrapper(
        _load_manifold_module(p, d, dev), name="manifold"
    ),
    "adaptive_k": _load_adaptive_k,
    "wasserstein": _load_wasserstein,
    "das_sae": _load_das_sae,
    "sheaf": _load_sheaf,
}


def _load_topk_module(path, d_in, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    F = sd["W_e"].shape[1]
    # heuristic top_k = 32 (matches scripts/train_sae_comparison.py default)
    m = _TopKSAE(d_in, F, top_k=32)
    m.load_state_dict(sd)
    return m.to(device).eval()


def _load_l1_module(path, d_in, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    F = sd["W_e"].shape[1]
    m = _L1SAE(d_in, F)
    m.load_state_dict(sd)
    return m.to(device).eval()


def _load_manifold_module(path, d_in, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    F = sd["W_gate"].shape[1]
    # M_F derived from D_k second dim (basis_dim = 2*M_F + 1)
    basis_dim = sd["D_k"].shape[1]
    M_F = (basis_dim - 1) // 2
    m = _ManifoldFourierSAE(d_in, F, M_F=M_F)
    m.load_state_dict(sd)
    return m.to(device).eval()


def scan_checkpoints(root: Path) -> list[CheckpointEntry]:
    out: list[CheckpointEntry] = []
    for p in sorted(Path(root).rglob("*.pt")):
        sp = str(p)
        if "__pycache__" in sp:
            continue
        kind = _kind_for(sp)
        if kind is None:
            continue
        st = p.stat()
        out.append(CheckpointEntry(
            name=f"{kind}::{p.name}",
            path=sp,
            variant=kind,
            size_bytes=st.st_size,
            mtime=st.st_mtime,
        ))
    return out


def load_checkpoint(entry: CheckpointEntry, d_in: int, device: str) -> tuple[SAEWrapper | None, str]:
    loader = _LOADERS.get(entry.variant)
    if loader is None:
        return None, f"no loader registered for variant={entry.variant}"
    try:
        w = loader(entry.path, d_in, device)
        # Sanity: rename wrapper to include the file stem so two checkpoints
        # of the same variant don't collide.
        w.name = f"{entry.variant}__{Path(entry.path).stem}"
        return w, "ok"
    except Exception as e:
        return None, f"load failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def composite_score(
    val_r2: float,
    steering_composite: float,
    hsv_coherence: float,
    dead_rate: float,
) -> float:
    return (
        0.40 * _clamp01(val_r2)
        + 0.30 * _clamp01(steering_composite)
        + 0.20 * _clamp01(hsv_coherence)
        + 0.10 * _clamp01(1.0 - (dead_rate if dead_rate is not None else 1.0))
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class VariantRow:
    variant: str
    name: str
    file: str
    F: int
    val_r2: float
    mean_K: float           # L0 mean active per row
    dead_rate: float
    hsv_coherence: float
    manifold_dim: float
    probe_hsv: float
    steering_composite: float
    steering_linear_push: float
    steering_anchor_swap: float
    steering_magnitude: float
    steering_compositional: float
    composite_score: float
    training_time_s: float | None = None
    n_active: int = 0
    failure: str | None = None


def _row_from_results(
    entry: CheckpointEntry,
    wrapper: SAEWrapper,
    harness_res: HarnessResult,
    bench_res: BenchResult,
) -> VariantRow:
    m = harness_res.metrics
    summary = bench_res.summary()
    val_r2 = float(m.get("val_r2", float("nan")))
    hsv = float(m.get("hsv_coherence", {}).get("mean_top20_coherence", float("nan")))
    dead = float(m.get("dead_atom_fraction", 1.0))
    mfd = float(m.get("manifold_dim", {}).get("mean_effective_rank", float("nan")))
    probe_hsv = m.get("probes", {}).get("hsv", {})
    probe_hsv_val = float(probe_hsv.get("r2_mean", probe_hsv.get("r2", float("nan"))))
    steer_comp = float(summary.get("composite", float("nan")))
    return VariantRow(
        variant=entry.variant,
        name=wrapper.name,
        file=entry.path,
        F=int(wrapper.n_features),
        val_r2=val_r2,
        mean_K=float(m.get("sparsity", {}).get("L0", float("nan"))),
        dead_rate=dead,
        hsv_coherence=hsv,
        manifold_dim=mfd,
        probe_hsv=probe_hsv_val,
        steering_composite=steer_comp,
        steering_linear_push=float(summary.get("linear_push_r2", float("nan"))),
        steering_anchor_swap=float(summary.get("anchor_swap_r2", float("nan"))),
        steering_magnitude=float(summary.get("magnitude_scaling_r2", float("nan"))),
        steering_compositional=float(summary.get("compositional_r2", float("nan"))),
        composite_score=composite_score(val_r2, steer_comp, hsv, dead),
        n_active=int(m.get("n_active_atoms", 0)),
    )


def run_leaderboard(
    root: Path,
    data_path: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    max_val_rows: int = 1024,
    ablation_subset: int = 16,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("scanning %s for checkpoints", root)
    entries = scan_checkpoints(Path(root))
    log.info("found %d candidate checkpoints", len(entries))

    # Prepare data once.
    _, X_val, labels, D = prepare(
        Path(root), Path(data_path), device=device, max_val_rows=max_val_rows
    )
    hsv_labels = labels.color_hsv[labels.row_color_idx]
    name_labels = labels.row_color_idx

    rows: list[VariantRow] = []
    skipped: list[dict[str, str]] = []

    for entry in entries:
        t0 = time.time()
        wrapper, status = load_checkpoint(entry, d_in=D, device=device)
        if wrapper is None:
            log.warning("skip %s: %s", entry.path, status)
            skipped.append({"path": entry.path, "variant": entry.variant,
                            "reason": status})
            continue
        log.info("scoring %s (F=%d)", wrapper.name, wrapper.n_features)
        try:
            h = Harness(wrapper, X_val, labels=labels,
                        ablation_subset=ablation_subset)
            hres = h.run()
        except Exception as e:
            log.exception("harness failed on %s", wrapper.name)
            skipped.append({"path": entry.path, "variant": entry.variant,
                            "reason": f"harness: {type(e).__name__}: {e}"})
            continue
        try:
            bench = SteeringBench(
                wrapper, X_val, hsv_labels=hsv_labels, name_labels=name_labels
            )
            bres = bench.run()
        except Exception as e:
            log.exception("steering bench failed on %s", wrapper.name)
            # Still emit row with steering NaN.
            class _Empty:
                def summary(self):
                    return {}
            bres = BenchResult(model_name=wrapper.name)
        row = _row_from_results(entry, wrapper, hres, bres)
        row.training_time_s = None  # unknown unless we sidecar the trainer
        rows.append(row)
        log.info(" -> R²=%.3f steer=%.3f comp=%.3f",
                 row.val_r2, row.steering_composite, row.composite_score)

    # Sort by composite descending.
    rows.sort(key=lambda r: -r.composite_score)

    # ---- write JSON ----
    payload = {
        "rows": [asdict(r) for r in rows],
        "skipped": skipped,
        "weights": {"val_r2": 0.4, "steering_composite": 0.3,
                    "hsv_coherence": 0.2, "alive_atoms": 0.1},
        "n_checkpoints_found": len(entries),
        "n_scored": len(rows),
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(payload, indent=2, default=float))

    # ---- write markdown ----
    _render_markdown(rows, output_dir / "leaderboard.md", skipped=skipped)

    # ---- radar plot top 5 ----
    try:
        _render_radar(rows[:5], output_dir / "leaderboard_radar.png")
    except Exception as e:
        log.warning("radar plot failed: %s", e)

    return payload


def _render_markdown(rows: list[VariantRow], path: Path,
                     skipped: list[dict[str, str]] | None = None) -> None:
    lines = ["# Full SAE Leaderboard v2\n\n"]
    lines.append("composite = 0.40·R² + 0.30·steering_composite + 0.20·HSV_coh"
                 " + 0.10·(1 − dead_rate); each term clamped to [0,1].\n\n")
    lines.append("| Rank | Variant | F | R² | mean_K | dead | HSV_coh | steer_comp | composite |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r.variant} ({Path(r.file).stem}) | {r.F} "
            f"| {r.val_r2:.3f} | {r.mean_K:.1f} | {r.dead_rate:.3f} "
            f"| {r.hsv_coherence:.3f} | {r.steering_composite:.3f} "
            f"| **{r.composite_score:.3f}** |\n"
        )
    lines.append("\n## Per-protocol steering R²\n\n")
    lines.append("| Variant | linear_push | anchor_swap | magnitude | compositional |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in rows:
        lines.append(
            f"| {r.variant} | {r.steering_linear_push:.3f} "
            f"| {r.steering_anchor_swap:.3f} | {r.steering_magnitude:.3f} "
            f"| {r.steering_compositional:.3f} |\n"
        )
    if skipped:
        lines.append("\n## Skipped\n\n")
        for s in skipped:
            lines.append(f"- `{s['path']}` ({s['variant']}): {s['reason']}\n")
    path.write_text("".join(lines))


def _render_radar(rows: list[VariantRow], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = [("R²", lambda r: r.val_r2),
            ("HSV_coh", lambda r: r.hsv_coherence),
            ("probe_hsv", lambda r: r.probe_hsv),
            ("steer", lambda r: r.steering_composite),
            ("alive", lambda r: 1.0 - r.dead_rate),
            ("compose", lambda r: r.steering_compositional)]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    for r in rows:
        vals = [_clamp01(fn(r)) for _, fn in axes]
        vals += vals[:1]
        ax.plot(angles, vals, label=f"{r.variant}", lw=2)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a for a, _ in axes])
    ax.set_ylim(0, 1)
    ax.set_title("Top-5 SAE variants — composite radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "CheckpointEntry",
    "VariantRow",
    "scan_checkpoints",
    "load_checkpoint",
    "composite_score",
    "run_leaderboard",
]
