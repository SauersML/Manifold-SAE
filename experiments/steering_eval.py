"""Steering benchmark: Manifold-SAE vs Linear-SAE vs diff-of-means.

Hypothesis: Manifold-SAE wins on tasks with curved geometry (days-of-week shift,
month shift, age progression) because intrinsic-position steering moves *along* the
learned manifold rather than teleporting in ambient space.

For each (steering method x alpha) cell we sample N trials. Each trial picks a
source-category example and a different target category; we apply the steering
operator and measure (1) success rate via a nearest-neighbour probe on harvested
activations, and (2) a side-effect score quantifying how much the rest of the
representation moved.

Outputs:
- ``<output_dir>/steering_results.json`` -- the full table of metrics.
- ``<output_dir>/steering_results.png`` -- bar chart (only if matplotlib is installed).
- A printed winner declaration based on best-alpha success rate per method.

Assumptions:
- Manifold-SAE and Linear-SAE checkpoints already exist (Phase 3/4 deliverables).
- The activations file is a torch-saved dict with at least ``activations`` (N, D) and
  ``labels`` (length N, category strings). The harvest script (Phase 4) produces this.

Run from repo root:
    python -m experiments.steering_eval --task days-of-week-shift \\
        --manifold-sae-checkpoint runs/msae.pt --linear-sae-checkpoint runs/lsae.pt \\
        --activations runs/activations.pt --output-dir runs/steering/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# -----------------------------------------------------------------------------
# Task metadata: label vocab + how to pick a source/target pair.
# -----------------------------------------------------------------------------

TASKS: dict[str, dict[str, Any]] = {
    "days-of-week-shift": {
        "categories": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        "periodic": True,
    },
    "month-shift": {
        "categories": [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ],
        "periodic": True,
    },
    "age-progression": {
        "categories": ["child", "teenager", "adult", "elderly"],
        "periodic": False,
    },
}


# -----------------------------------------------------------------------------
# Data loading.
# -----------------------------------------------------------------------------


@dataclass
class ActivationBundle:
    activations: torch.Tensor  # (N, D)
    labels: list[str]  # length N
    by_label: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> ActivationBundle:
        """Load from a torch-saved dict with keys ``activations`` (N, D) and ``labels`` (N,).

        This is the exact format the harvest pipeline (``experiments.llm_activations.harvest``)
        emits — no other format is supported.
        """
        blob = torch.load(str(path), map_location="cpu")
        if not isinstance(blob, dict) or "activations" not in blob or "labels" not in blob:
            raise ValueError(
                f"activations file {path} must be a torch dict with keys "
                "'activations' (N, D) and 'labels' (length N)"
            )
        bundle = cls(activations=blob["activations"].float(), labels=[str(x) for x in blob["labels"]])
        bundle._index_by_label()
        return bundle

    def _index_by_label(self) -> None:
        self.by_label.clear()
        for i, lab in enumerate(self.labels):
            self.by_label.setdefault(lab, []).append(i)
        self.by_label = {k: torch.tensor(v, dtype=torch.long) for k, v in self.by_label.items()}


# -----------------------------------------------------------------------------
# Metrics.
# -----------------------------------------------------------------------------


def nearest_neighbor_label(
    steered: torch.Tensor, bundle: ActivationBundle, exclude_idx: int | None = None
) -> str:
    """Return the label of the nearest activation in the bundle."""
    acts = bundle.activations
    if exclude_idx is not None:
        # Boost distance for the excluded row so it never wins.
        diff = acts - steered.unsqueeze(0)
        dists = (diff * diff).sum(dim=-1)
        dists[exclude_idx] = float("inf")
    else:
        diff = acts - steered.unsqueeze(0)
        dists = (diff * diff).sum(dim=-1)
    j = int(torch.argmin(dists).item())
    return bundle.labels[j]


def side_effect_score(
    original: torch.Tensor,
    steered: torch.Tensor,
    bundle: ActivationBundle,
    source_label: str,
    target_label: str,
) -> float:
    """How much did rows *outside* the source/target categories shift in centroid?

    We compute the change in cosine distance from steered-vs-original to the centroid
    of every non-{source,target} category. Smaller is better.
    """
    other_labels = [l for l in bundle.by_label if l not in (source_label, target_label)]
    if not other_labels:
        return 0.0
    deltas = []
    for lab in other_labels:
        idx = bundle.by_label[lab]
        centroid = bundle.activations.index_select(0, idx).mean(dim=0)
        before = _cos_dist(original, centroid)
        after = _cos_dist(steered, centroid)
        deltas.append(abs(after - before))
    return float(sum(deltas) / len(deltas))


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float(torch.dot(a.flatten(), b.flatten()).item())
    denom = float((a.norm() * b.norm()).item()) + 1e-12
    return 1.0 - num / denom


# -----------------------------------------------------------------------------
# Trial selection.
# -----------------------------------------------------------------------------


def sample_trials(
    bundle: ActivationBundle,
    task_categories: list[str],
    n_trials: int,
    rng: torch.Generator,
) -> list[tuple[int, str, str]]:
    """Return a list of (example_idx, source_label, target_label) tuples."""
    available = [c for c in task_categories if c in bundle.by_label]
    if len(available) < 2:
        raise ValueError(
            f"need at least two task categories present in activations; found {available}"
        )
    trials: list[tuple[int, str, str]] = []
    for _ in range(n_trials):
        # Pick source uniformly from available; target = any other available.
        s = int(torch.randint(0, len(available), (1,), generator=rng).item())
        src = available[s]
        remaining = [c for c in available if c != src]
        t = int(torch.randint(0, len(remaining), (1,), generator=rng).item())
        tgt = remaining[t]
        idx_pool = bundle.by_label[src]
        pick = int(torch.randint(0, idx_pool.shape[0], (1,), generator=rng).item())
        example_idx = int(idx_pool[pick].item())
        trials.append((example_idx, src, tgt))
    return trials


def position_delta_for(
    source_label: str, target_label: str, task_categories: list[str], periodic: bool
) -> float:
    """Translate (source, target) category labels into a delta on [0, 1] coordinates."""
    n = len(task_categories)
    s = task_categories.index(source_label)
    t = task_categories.index(target_label)
    if periodic:
        # Shortest-arc delta on the circle.
        raw = (t - s) / n
        if raw > 0.5:
            raw -= 1.0
        elif raw < -0.5:
            raw += 1.0
        return raw
    return (t - s) / max(n - 1, 1)


# -----------------------------------------------------------------------------
# Feature selection: which SAE feature do we steer?
#
# Cheap heuristic: pick the feature whose amplitude best correlates with category
# index across the bundle. A real implementation would call out to a feature-
# attribution routine; this is sufficient for a first-pass benchmark.
# -----------------------------------------------------------------------------


def select_manifold_feature(sae, bundle: ActivationBundle, task_categories: list[str]) -> int:
    sae.eval()
    with torch.no_grad():
        out = sae(bundle.activations)
    return _best_feature_for_categories(out.amplitudes, bundle.labels, task_categories)


def select_linear_feature(sae, bundle: ActivationBundle, task_categories: list[str]) -> int:
    sae.eval()
    with torch.no_grad():
        amps = sae.encode(bundle.activations)
    return _best_feature_for_categories(amps, bundle.labels, task_categories)


def _best_feature_for_categories(
    amplitudes: torch.Tensor, labels: list[str], task_categories: list[str]
) -> int:
    # Score each feature by the spread of per-category mean activations.
    cat_means = []
    for c in task_categories:
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx:
            continue
        cat_means.append(amplitudes[torch.tensor(idx, dtype=torch.long)].mean(dim=0))
    if not cat_means:
        return 0
    stacked = torch.stack(cat_means, dim=0)  # (n_cats, F)
    spread = stacked.var(dim=0)
    return int(torch.argmax(spread).item())


# -----------------------------------------------------------------------------
# The benchmark driver.
# -----------------------------------------------------------------------------


@dataclass
class CellResult:
    method: str
    alpha: float
    success_rate: float
    side_effect_mean: float
    n_trials: int


def run_benchmark(cfg: Config) -> dict[str, Any]:
    task = TASKS[cfg.task]
    task_categories: list[str] = task["categories"]
    periodic: bool = task["periodic"]

    print(f"[steering-eval] task={cfg.task} periodic={periodic} categories={task_categories}")
    bundle = ActivationBundle.from_path(Path(cfg.activations))
    print(f"[steering-eval] loaded {bundle.activations.shape[0]} activations dim={bundle.activations.shape[1]}")

    from manifold_sae import steering

    msae = _load_torch_module(cfg.manifold_sae_checkpoint, kind="manifold")
    lsae = _load_torch_module(cfg.linear_sae_checkpoint, kind="linear")

    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
    trials = sample_trials(bundle, task_categories, cfg.n_trials, rng)

    m_feat = select_manifold_feature(msae, bundle, task_categories)
    l_feat = select_linear_feature(lsae, bundle, task_categories)
    print(f"[steering-eval] manifold steering feature = {m_feat}; linear steering feature = {l_feat}")

    results: list[CellResult] = []
    alphas = list(cfg.alpha_grid)

    for method in ("manifold", "linear", "diff-means"):
        for alpha in alphas:
            hits, side_effects = 0, []
            for example_idx, src, tgt in trials:
                x = bundle.activations[example_idx].unsqueeze(0)
                if method == "manifold":
                    delta = position_delta_for(src, tgt, task_categories, periodic) * alpha
                    steered = steering.steer_manifold(
                        msae, x, m_feat, delta, cyclic=periodic
                    ).squeeze(0)
                elif method == "linear":
                    steered = steering.steer_linear(lsae, x, l_feat, alpha).squeeze(0)
                else:  # diff-means
                    steered = steering.steer_baseline_diff_means(
                        bundle.activations,
                        labels_source=[src],
                        labels_target=[tgt],
                        x=x.squeeze(0),
                        alpha=alpha,
                        all_labels=bundle.labels,
                    )

                pred = nearest_neighbor_label(steered, bundle, exclude_idx=example_idx)
                success = pred == tgt
                hits += int(success)
                side_effects.append(side_effect_score(x.squeeze(0), steered, bundle, src, tgt))
            cell = CellResult(
                method=method,
                alpha=alpha,
                success_rate=hits / max(len(trials), 1),
                side_effect_mean=float(sum(side_effects) / max(len(side_effects), 1)),
                n_trials=len(trials),
            )
            results.append(cell)
            print(
                f"[steering-eval] {method:>10s} alpha={alpha:+.3f} "
                f"success={cell.success_rate:.3f} side_effect={cell.side_effect_mean:.4f}"
            )

    summary = {
        "task": cfg.task,
        "n_trials": cfg.n_trials,
        "alphas": alphas,
        "results": [cell.__dict__ for cell in results],
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "steering_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[steering-eval] wrote {json_path}")

    _plot_results(results, alphas, out_dir / "steering_results.png")
    _declare_winner(results)
    return summary


def _load_torch_module(path: str | None, kind: str):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{kind}-sae checkpoint not found at {p}")
    obj = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    return obj


def _plot_results(results: list[CellResult], alphas: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({c.method for c in results})
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        cells = [c for c in results if c.method == m]
        xs = [alphas.index(c.alpha) + i * width for c in cells]
        ys = [c.success_rate for c in cells]
        ax.bar(xs, ys, width=width, label=m)
    ax.set_xticks([j + width * (len(methods) - 1) / 2 for j in range(len(alphas))])
    ax.set_xticklabels([f"{a:+.2f}" for a in alphas])
    ax.set_xlabel("alpha")
    ax.set_ylabel("success rate")
    ax.set_title("Steering success: manifold vs linear vs diff-of-means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"[steering-eval] wrote {path}")


def _declare_winner(results: list[CellResult]) -> None:
    if not results:
        print("[steering-eval] no results to score")
        return
    best_per_method: dict[str, CellResult] = {}
    for c in results:
        cur = best_per_method.get(c.method)
        if cur is None or c.success_rate > cur.success_rate:
            best_per_method[c.method] = c
    ordered = sorted(best_per_method.values(), key=lambda c: c.success_rate, reverse=True)
    print("\n=== winner declaration ===")
    for c in ordered:
        print(
            f"  {c.method:>10s}: best success={c.success_rate:.3f} "
            f"(alpha={c.alpha:+.3f}, side_effect={c.side_effect_mean:.4f})"
        )
    winner = ordered[0]
    print(f"--> winner: {winner.method} at alpha={winner.alpha:+.3f} "
          f"(success={winner.success_rate:.3f})")


# -----------------------------------------------------------------------------
# Config dataclass and default entrypoint.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Steering-benchmark configuration. Edit ``DEFAULT_CONFIG`` to override."""

    task: str = "days-of-week-shift"  # one of TASKS.keys()
    manifold_sae_checkpoint: str = "runs/sae_days_v1/sae.pt"
    linear_sae_checkpoint: str = "runs/lsae_days_v1/sae.pt"
    activations: str = "runs/llama31_8b_l28.pt"
    n_trials: int = 100
    alpha_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    output_dir: str = "runs/steering"
    seed: int = 0


DEFAULT_CONFIG = Config()


def main(cfg: Config = DEFAULT_CONFIG) -> int:
    if cfg.task not in TASKS:
        raise ValueError(f"unknown task {cfg.task!r}; choices: {sorted(TASKS)}")
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
