"""Comparative atlas generation across models and concepts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .concepts import label_prompts
from .gauge import fit_gauge
from .ingest import harvest_activations


@dataclass(frozen=True, slots=True)
class AtlasRow:
    """One model/concept atlas record."""

    model: str
    concept: str
    layer: int
    n_prompts: int
    width: int
    d: int
    mean_r2: float
    min_bic_d: int


def run_atlas(
    models: Sequence[str],
    concepts: Sequence[str],
    prompts: Sequence[str],
    *,
    layer: int,
    out_dir: str | Path,
    batch_size: int = 4,
    trust_remote_code: bool = False,
) -> list[AtlasRow]:
    """Harvest, fit, and compare ``N models x M concepts``.

    Produces ``atlas.csv`` and ``atlas_r2.png`` in ``out_dir``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[AtlasRow] = []
    for model in models:
        harvest = harvest_activations(
            model,
            prompts,
            layer,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
        )
        np.save(out / f"{_safe(model)}_L{layer}.npy", harvest.activations)
        for concept in concepts:
            labels = label_prompts(tuple(prompts), concept)
            fit = fit_gauge(harvest.activations, labels, targets=[concept])
            fit.save(out / f"{_safe(model)}_{concept}_gauge.npz")
            rows.append(
                AtlasRow(
                    model=model,
                    concept=concept,
                    layer=layer,
                    n_prompts=len(prompts),
                    width=int(harvest.activations.shape[-1]),
                    d=fit.d,
                    mean_r2=float(np.mean(list(fit.r2.values()))),
                    min_bic_d=min(fit.bic_by_d, key=fit.bic_by_d.__getitem__) if fit.bic_by_d else fit.d,
                )
            )
    write_atlas_table(rows, out / "atlas.csv")
    plot_atlas(rows, out / "atlas_r2.png")
    return rows


def write_atlas_table(rows: Sequence[AtlasRow], path: str | Path) -> Path:
    """Write atlas rows as CSV."""
    p = Path(path)
    with p.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AtlasRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return p


def plot_atlas(rows: Sequence[AtlasRow], path: str | Path) -> Path:
    """Render a simple comparative R-squared bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise NotImplementedError("plot_atlas requires matplotlib") from exc
    labels = [f"{r.model}\n{r.concept}" for r in rows]
    vals = [r.mean_r2 for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.4), 3.5))
    ax.bar(labels, vals)
    ax.set_ylabel("mean target R2")
    ax.set_ylim(bottom=min(0.0, min(vals, default=0.0)), top=max(1.0, max(vals, default=1.0)))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=140)
    return out


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
