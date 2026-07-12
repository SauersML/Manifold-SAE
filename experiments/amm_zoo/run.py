"""Warm, resumable AMM-zoo benchmark driver.

One worker owns an entire ``(seed, sigma)`` group. It generates the corpus once,
warms dependency/runtime caches once, and then evaluates every requested arm.
Each arm result is checkpointed inside the worker before the next arm starts.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ARM_NAMES = ("topk_sae", "bsf_vanilla", "bsf_grassmann", "sasa", "ours")
TOPOLOGIES = ("circle", "arc", "helix", "torus", "sphere", "mobius", "linear")


def _write_json(path: Path, payload: Any, *, indent: int | None = None) -> None:
    """Atomic JSON checkpoint in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=indent, default=float))
    os.replace(temporary, path)


def _key(cell: dict[str, Any]) -> tuple[int, float, float, str]:
    return (
        int(cell["seed"]),
        float(cell["sigma_frac"]),
        float(cell["coherence"]),
        str(cell["arm"]),
    )


def _warm_dependencies(arms: list[str], threads: int) -> dict[str, Any]:
    """Pay lazy import/kernel initialization once before any measured arm."""
    started = time.perf_counter()
    import torch

    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter = torch.nn.Parameter(torch.ones(32, device=device))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    (parameter.square().sum()).backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    provenance: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_device": str(device),
    }
    if "ours" in arms:
        import gamfit

        if not hasattr(gamfit, "sae_manifold_fit"):
            raise RuntimeError(f"gamfit at {gamfit.__file__} has no sae_manifold_fit")
        provenance.update(
            gamfit_version=getattr(gamfit, "__version__", "unknown"),
            gamfit_path=str(gamfit.__file__),
        )
    provenance["warm_s"] = round(time.perf_counter() - started, 6)
    return provenance


def _worker_impl(spec_path: str) -> None:
    from amm import generate_amm
    from arms import run_arm
    from metrics import matched_geodesic_permutation_test, score_arm

    spec = json.loads(Path(spec_path).read_text())
    output_path = Path(spec["out_path"])
    if output_path.exists():
        group = json.loads(output_path.read_text())
        if group.get("fingerprint") != spec["fingerprint"]:
            raise RuntimeError(f"worker checkpoint fingerprint mismatch at {output_path}")
    else:
        group = {"fingerprint": spec["fingerprint"], "cells": []}

    existing = {_key(cell): cell for cell in group["cells"]}
    warm = _warm_dependencies(spec["arms"], int(spec["threads"]))
    generated_at = time.perf_counter()
    dataset = generate_amm(
        seed=spec["seed"],
        sigma_frac=spec["sigma_frac"],
        coherence=spec["coherence"],
        n_train=spec["n_train"],
        n_test=spec["n_test"],
    )
    generate_s = round(time.perf_counter() - generated_at, 6)

    for arm in spec["arms"]:
        cell_key = (spec["seed"], spec["sigma_frac"], spec["coherence"], arm)
        if existing.get(cell_key, {}).get("report") is not None:
            continue
        arm_started = time.perf_counter()
        timing: dict[str, Any] = {
            "warm_s": warm["warm_s"],
            "generate_s": generate_s,
        }
        try:
            recovered = run_arm(
                dataset,
                arm,
                steps=spec["steps"],
                batch_size=spec["batch_size"],
                manifold_iters=spec["manifold_iters"],
                manifold_oos_batch=spec["manifold_oos_batch"],
                seed=spec["seed"],
                timing=timing,
            )
            scored_at = time.perf_counter()
            report = score_arm(dataset, recovered, "test", seed=spec["seed"])
            timing["score_s"] = round(time.perf_counter() - scored_at, 6)

            null_at = time.perf_counter()
            rng = np.random.default_rng(spec["seed"] + 991)
            recovered_by_name = {factor.name: factor for factor in recovered}
            for factor_report in report["per_factor"]:
                if factor_report["true_topology"] == "linear":
                    continue
                recovered_factor = recovered_by_name[factor_report["recovered"]]
                rows, true_coord = dataset.true_intrinsic(
                    "test", factor_report["true_factor"]
                )
                recovered_coord = recovered_factor.coord[rows]
                valid = np.all(np.isfinite(recovered_coord), axis=1)
                if valid.sum() < 8:
                    factor_report["geodesic_null_p"] = 1.0
                    factor_report["structure_recovered"] = False
                    continue
                observed, p_value = matched_geodesic_permutation_test(
                    true_coord[valid],
                    factor_report["true_topology"],
                    recovered_coord[valid],
                    recovered_factor.topology,
                    recovered_factor.meta,
                    rng,
                    b_perm=spec["b_perm"],
                )
                factor_report["geodesic_null_observed"] = observed
                factor_report["geodesic_null_p"] = p_value
                factor_report["structure_recovered"] = bool(p_value < 0.05)
            timing["null_s"] = round(time.perf_counter() - null_at, 6)
            timing["total_arm_s"] = round(time.perf_counter() - arm_started, 6)
            cell = {
                "seed": spec["seed"],
                "sigma_frac": spec["sigma_frac"],
                "coherence": spec["coherence"],
                "arm": arm,
                "min_principal_angle_deg": dataset.min_principal_angle_deg,
                "signal_rms": dataset.signal_rms,
                "runtime": timing,
                "provenance": warm,
                "report": report,
            }
        except Exception as error:
            timing["total_arm_s"] = round(time.perf_counter() - arm_started, 6)
            cell = {
                "seed": spec["seed"],
                "sigma_frac": spec["sigma_frac"],
                "coherence": spec["coherence"],
                "arm": arm,
                "status": "FAILED",
                "runtime": timing,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-6000:],
            }

        existing[cell_key] = cell
        group["cells"] = list(existing.values())
        _write_json(output_path, group)
        overall = cell.get("report", {}).get("overall", {})
        print(
            f"[{arm}] R2={overall.get('mean_contribution_r2')} "
            f"topoID={overall.get('topology_id_accuracy')} "
            f"wall={timing['total_arm_s']:.2f}s status={cell.get('status', 'OK')}",
            flush=True,
        )


def _worker(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    profile_path = spec.get("profile_path")
    if profile_path is None:
        _worker_impl(spec_path)
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        _worker_impl(spec_path)
    finally:
        profiler.disable()
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile_path)


def _fingerprint(cfg: dict[str, Any]) -> str:
    encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _run_group(
    seed: int,
    sigma: float,
    arms: list[str],
    cfg: dict[str, Any],
    scratch: Path,
    profile_dir: Path | None,
) -> list[dict[str, Any]]:
    """Run/retry one warm worker, retaining successful arm checkpoints."""
    scratch.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(cfg)
    tag = f"s{seed}_n{sigma:g}_c{cfg['coherence']:g}_{fingerprint}"
    spec_path = scratch / f"{tag}.spec.json"
    output_path = scratch / f"{tag}.out.json"
    requested = list(arms)
    last_diagnostic = ""

    for attempt in range(cfg["retries"] + 1):
        if output_path.exists():
            checkpoint = json.loads(output_path.read_text())
            cells = {_key(cell): cell for cell in checkpoint.get("cells", [])}
        else:
            cells = {}
        pending = [
            arm
            for arm in requested
            if cells.get((seed, sigma, cfg["coherence"], arm), {}).get("report") is None
        ]
        if not pending:
            return [cells[(seed, sigma, cfg["coherence"], arm)] for arm in requested]

        spec = {
            "fingerprint": fingerprint,
            "seed": seed,
            "sigma_frac": sigma,
            "coherence": cfg["coherence"],
            "arms": pending,
            "n_train": cfg["n_train"],
            "n_test": cfg["n_test"],
            "steps": cfg["steps"],
            "batch_size": cfg["batch_size"],
            "manifold_iters": cfg["manifold_iters"],
            "manifold_oos_batch": cfg["manifold_oos_batch"],
            "b_perm": cfg["b_perm"],
            "threads": cfg["threads"],
            "out_path": str(output_path),
            "profile_path": None
            if profile_dir is None
            else str(profile_dir / f"{tag}.attempt{attempt + 1}.prof"),
        }
        _write_json(spec_path, spec)
        command = [sys.executable, os.path.abspath(__file__), "--worker", str(spec_path)]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=cfg["group_timeout"],
            )
            last_diagnostic = (process.stdout + "\n" + process.stderr)[-6000:]
        except subprocess.TimeoutExpired as error:
            last_diagnostic = f"worker timed out after {time.perf_counter() - started:.1f}s: {error}"
        print(
            f"[group seed={seed} sigma={sigma:g}] attempt {attempt + 1} "
            f"pending={pending} wall={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    if output_path.exists():
        cells = {_key(cell): cell for cell in json.loads(output_path.read_text()).get("cells", [])}
    else:
        cells = {}
    result = []
    for arm in requested:
        key = (seed, sigma, cfg["coherence"], arm)
        result.append(
            cells.get(
                key,
                {
                    "seed": seed,
                    "sigma_frac": sigma,
                    "coherence": cfg["coherence"],
                    "arm": arm,
                    "status": "FAILED",
                    "error": last_diagnostic,
                },
            )
        )
    return result


def drive(
    cfg: dict[str, Any],
    *,
    out: Path,
    scratch: Path,
    fresh: bool,
    profile: bool,
    figures: bool,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / cfg["results_name"]
    if results_path.exists() and not fresh:
        master = json.loads(results_path.read_text())
        if master.get("config") != cfg:
            raise RuntimeError(
                f"existing {results_path} has a different config; pass --fresh to replace it"
            )
    else:
        master = {"config": cfg, "cells": []}
        _write_json(results_path, master, indent=2)

    cells = {_key(cell): cell for cell in master["cells"]}
    profile_dir = out / "profiles" if profile else None
    total = len(cfg["seeds"]) * len(cfg["sigmas"]) * len(cfg["arms"])
    completed = sum(cell.get("report") is not None for cell in cells.values())
    print(f"resume: {completed}/{total} successful cells already checkpointed", flush=True)

    for seed in cfg["seeds"]:
        for sigma in cfg["sigmas"]:
            pending = [
                arm
                for arm in cfg["arms"]
                if cells.get((seed, sigma, cfg["coherence"], arm), {}).get("report") is None
            ]
            if not pending:
                continue
            group_cells = _run_group(seed, sigma, pending, cfg, scratch, profile_dir)
            for cell in group_cells:
                cells[_key(cell)] = cell
                master["cells"] = list(cells.values())
                _write_json(results_path, master, indent=2)
                overall = cell.get("report", {}).get("overall", {})
                print(
                    f"[{len([c for c in cells.values() if c.get('report') is not None])}/{total}] "
                    f"{cell['arm']} seed={seed} sigma={sigma:g}: "
                    f"R2={overall.get('mean_contribution_r2')} "
                    f"topoID={overall.get('topology_id_accuracy')} "
                    f"status={cell.get('status', 'OK')}",
                    flush=True,
                )

    if figures:
        make_figures(master, out)
        _write_report(master, out)
    return master


def _agg(master: dict[str, Any]):
    from collections import defaultdict

    r2 = defaultdict(lambda: defaultdict(list))
    topology_id = defaultdict(lambda: defaultdict(list))
    for cell in master["cells"]:
        report = cell.get("report")
        if report is None:
            continue
        key = (cell["arm"], cell["sigma_frac"])
        for topology, values in report["by_topology"].items():
            if values.get("mean_contribution_r2") is not None:
                r2[key][topology].append(values["mean_contribution_r2"])
            topology_id[key][topology].append(values["topology_id_accuracy"])
    return r2, topology_id


def make_figures(master: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = master["config"]["arms"]
    sigmas = sorted(master["config"]["sigmas"])
    r2, topology_id = _agg(master)
    colors = {
        "topk_sae": "#8C8C8C",
        "bsf_vanilla": "#4C78A8",
        "bsf_grassmann": "#F58518",
        "sasa": "#B279A2",
        "ours": "#2A9D6F",
    }

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), sharey=True)
    for axis, topology in zip(axes.flat, TOPOLOGIES):
        for arm in arms:
            means = []
            errors = []
            for sigma in sigmas:
                values = np.asarray(r2[(arm, sigma)][topology], dtype=float)
                means.append(float(values.mean()) if values.size else np.nan)
                errors.append(
                    float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
                )
            axis.errorbar(
                sigmas,
                means,
                yerr=errors,
                marker="o",
                linewidth=2.2 if arm == "ours" else 1.5,
                markersize=5,
                capsize=2,
                color=colors[arm],
                label=arm,
                alpha=1.0 if arm == "ours" else 0.86,
            )
        axis.axhline(0.0, color="#BBBBBB", linewidth=0.8)
        axis.set_title(topology.title())
        axis.set_xlabel("noise / signal RMS")
        axis.grid(alpha=0.18)
    axes[0, 0].set_ylabel("held-out contribution R²")
    axes[1, 0].set_ylabel("held-out contribution R²")
    legend_axis = axes.flat[-1]
    legend_axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", frameon=False, fontsize=11)
    figure.suptitle("Manifold Zoo · additive factor recovery across every topology", fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(out / "r2_vs_sigma.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    heat = np.full((len(arms), len(TOPOLOGIES)), np.nan)
    for row, arm in enumerate(arms):
        for col, topology in enumerate(TOPOLOGIES):
            values = [
                value
                for sigma in sigmas
                for value in topology_id[(arm, sigma)][topology]
            ]
            if values:
                heat[row, col] = float(np.mean(values))
    figure, axis = plt.subplots(figsize=(10.5, 3.6))
    image = axis.imshow(heat, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(TOPOLOGIES)), [t.title() for t in TOPOLOGIES])
    axis.set_yticks(range(len(arms)), arms)
    for row in range(len(arms)):
        for col in range(len(TOPOLOGIES)):
            if np.isfinite(heat[row, col]):
                axis.text(
                    col,
                    row,
                    f"{heat[row, col]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if heat[row, col] > 0.62 else "#16352A",
                    fontweight="bold",
                )
    axis.set_title("Topology identification · held-out mean over seeds and noise")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.025, label="accuracy")
    figure.tight_layout()
    figure.savefig(out / "topology_id.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(master: dict[str, Any], out: Path) -> None:
    r2, topology_id = _agg(master)
    sigmas = sorted(master["config"]["sigmas"])
    arms = master["config"]["arms"]
    lines = [
        "# AMM zoo — production Manifold-SAE benchmark\n",
        f"Config: {json.dumps(master['config'])}\n",
        "## Contribution R² (held-out mean over seeds)\n",
        "| arm | topology | " + " | ".join(f"σ={sigma}" for sigma in sigmas) + " |",
        "|---|---|" + "---|" * len(sigmas),
    ]
    for arm in arms:
        for topology in TOPOLOGIES:
            values = [
                f"{np.mean(r2[(arm, sigma)][topology]):.3f}"
                if r2[(arm, sigma)][topology]
                else "-"
                for sigma in sigmas
            ]
            lines.append(f"| {arm} | {topology} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "\n## Topology-ID accuracy (mean over seeds and noise)\n",
            "| arm | " + " | ".join(TOPOLOGIES) + " |",
            "|---|" + "---|" * len(TOPOLOGIES),
        ]
    )
    for arm in arms:
        values = []
        for topology in TOPOLOGIES:
            observed = [
                value
                for sigma in sigmas
                for value in topology_id[(arm, sigma)][topology]
            ]
            values.append(f"{np.mean(observed):.2f}" if observed else "-")
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def _config(full: bool, arms: list[str]) -> dict[str, Any]:
    if full:
        return {
            "benchmark_revision": "production_msae_v1",
            "seeds": [0, 1, 2, 3, 4],
            "sigmas": [0.02, 0.05, 0.1, 0.2],
            "coherence": 0.0,
            "arms": arms,
            "n_train": 200_000,
            "n_test": 50_000,
            "steps": 4000,
            "batch_size": 8192,
            "manifold_iters": 50,
            "manifold_oos_batch": 10_000,
            "b_perm": 500,
            "geodesic_sample": 256,
            "null_sample": 128,
            "threads": 8,
            "group_timeout": 21_600,
            "retries": 2,
            "results_name": "results.json",
        }
    return {
        "benchmark_revision": "production_msae_v1",
        "seeds": [0],
        "sigmas": [0.05, 0.2],
        "coherence": 0.0,
        "arms": arms,
        "n_train": 8000,
        "n_test": 2500,
        "steps": 600,
        "batch_size": 4096,
        "manifold_iters": 8,
        "manifold_oos_batch": 2500,
        "b_perm": 120,
        "geodesic_sample": 256,
        "null_sample": 128,
        "threads": 4,
        "group_timeout": 3600,
        "retries": 2,
        "results_name": "results_quick.json",
    }


def _parse_arms(value: str) -> list[str]:
    arms = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [arm for arm in arms if arm not in ARM_NAMES]
    if not arms or unknown:
        raise argparse.ArgumentTypeError(
            f"arms must be a non-empty comma list from {ARM_NAMES}; unknown={unknown}"
        )
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--full", action="store_true")
    size.add_argument("--quick", action="store_true")
    parser.add_argument("--arms", type=_parse_arms, default=list(ARM_NAMES))
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(tempfile.gettempdir()) / "amm_zoo_work",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        _worker(args.worker)
        return
    if not (args.full or args.quick):
        parser.error("choose exactly one of --quick or --full")
    drive(
        _config(args.full, args.arms),
        out=args.out.resolve(),
        scratch=args.scratch.resolve(),
        fresh=args.fresh,
        profile=args.profile,
        figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()
