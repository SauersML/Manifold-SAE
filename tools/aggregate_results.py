#!/usr/bin/env python3
"""Aggregate results across all experiment runs into one markdown summary.

Scans a directory tree (usually /home/.../runs or /content/runs) for
known result files and produces a single overview table covering every
experiment. Pulls together what's otherwise scattered across:

  runs/<sweep_name>/results.json          (llm_sweep)
  runs/<sweep_name>/eval_F{F}.json        (llm_sweep per-F eval cache)
  runs/<probe_name>/results.json          (llm_probe)
  runs/<scenario>/<scenario>/summary.json (realistic_scaling)

Usage:
  python tools/aggregate_results.py <runs_dir>
  python tools/aggregate_results.py <runs_dir> > REPORT.md
  python tools/aggregate_results.py --remote <host>:<repo_root>/runs

The --remote variant rsyncs the JSONs locally first (skips the huge
checkpoints), then aggregates. Useful when the cluster ran jobs and
you want a local report without manually downloading anything.

This is read-only — never modifies the runs dir. Intentionally tolerant
of partial/missing data so it can run mid-sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders for each experiment's result schema. Each returns rows of dicts
# with a shared schema for the master table.
# ---------------------------------------------------------------------------


def _safe_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_llm_sweep(run_dir: Path) -> list[dict]:
    """A llm_sweep run drops results.json with per-F entries plus
    eval_F{F}.json caches. Prefer results.json; fall back to caches.
    """
    rows: list[dict] = []
    results = _safe_json(run_dir / "results.json")
    if results and isinstance(results, list):
        for r in results:
            rows.append({
                "experiment": "llm_sweep",
                "run": run_dir.name,
                "F": r.get("F"),
                "top_k": r.get("top_k"),
                "vanilla_expl": r.get("vanilla_explained"),
                "curve_expl": r.get("curve_explained"),
                "locked_expl": r.get("curve_locked_explained"),
                "vanilla_alive": r.get("vanilla_alive"),
                "curve_alive": r.get("curve_alive"),
                "delta_expl": ((r.get("curve_explained") or 0) -
                               (r.get("vanilla_explained") or 0)),
            })
        return rows
    # Fall back to per-F caches.
    for cache in sorted(run_dir.glob("eval_F*.json")):
        d = _safe_json(cache)
        if not d:
            continue
        r = d.get("result", {})
        rows.append({
            "experiment": "llm_sweep",
            "run": run_dir.name,
            "F": r.get("F"),
            "top_k": r.get("top_k"),
            "vanilla_expl": r.get("vanilla_explained"),
            "curve_expl": r.get("curve_explained"),
            "locked_expl": r.get("curve_locked_explained"),
            "vanilla_alive": r.get("vanilla_alive"),
            "curve_alive": r.get("curve_alive"),
            "delta_expl": ((r.get("curve_explained") or 0) -
                           (r.get("vanilla_explained") or 0)),
        })
    return rows


def load_llm_probe(run_dir: Path) -> list[dict]:
    """Multi-concept manifold probe. Emit one row per (concept, layer)."""
    rows: list[dict] = []
    results = _safe_json(run_dir / "results.json")
    if not results:
        return rows
    phase1 = results.get("phase1", {})
    phase2 = results.get("phase2", {})
    for concept, per_layer in phase1.items():
        for L_str, d in per_layer.items():
            L = int(L_str)
            best_rho = d.get("best_abs_rho_top8", 0)
            p2 = phase2.get(f"{concept}_L{L}", {})
            v = p2.get("vanilla") or {}
            cp = p2.get("curve_position") or {}
            rows.append({
                "experiment": "llm_probe",
                "run": run_dir.name,
                "concept": concept,
                "layer": L,
                "phase1_rho": best_rho,
                "vanilla_best_rho": v.get("best"),
                "vanilla_n_above_strong": v.get("n_atoms_above_strong"),
                "vanilla_n_above_moderate": v.get("n_atoms_above_moderate"),
                "curve_best_rho_pos": cp.get("best"),
                "curve_n_above_strong": cp.get("n_atoms_above_strong"),
                "curve_n_above_moderate": cp.get("n_atoms_above_moderate"),
            })
    return rows


def load_realistic_scaling(run_dir: Path) -> list[dict]:
    """realistic_scaling drops one summary.json per top-level dir, but
    also per-scenario subdirs. We look for summary.json at top.
    """
    rows: list[dict] = []
    summary = _safe_json(run_dir / "summary.json")
    if not summary or not isinstance(summary, dict):
        return rows
    for scenario_name, r in summary.items():
        v = r.get("vanilla", {}) or {}
        c = r.get("curve", {}) or {}
        rows.append({
            "experiment": "realistic_scaling",
            "run": run_dir.name,
            "scenario": scenario_name,
            "vanilla_expl": v.get("explained"),
            "curve_expl": c.get("explained"),
            "delta_expl": (c.get("explained") or 0) - (v.get("explained") or 0),
            "vanilla_chamfer": (v.get("chamfer", {}) or {}).get("mean"),
            "curve_chamfer": (c.get("chamfer", {}) or {}).get("mean"),
        })
    return rows


# ---------------------------------------------------------------------------
# Markdown table rendering
# ---------------------------------------------------------------------------


def render_table(rows: list[dict], columns: list[str], title: str) -> str:
    if not rows:
        return f"\n### {title}\n\n_no rows found_\n"
    lines = [f"\n### {title}\n", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:+.3f}" if c.startswith("delta") else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Remote fetch (optional)
# ---------------------------------------------------------------------------


def fetch_remote(remote: str) -> Path:
    """rsync just the JSON files (not the checkpoints) to a temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="agg-"))
    print(f"[fetch] rsync {remote} -> {tmp}", file=sys.stderr)
    cmd = [
        "rsync", "-a",
        "--include=*/", "--include=*.json", "--exclude=*",
        f"{remote}/", str(tmp) + "/",
    ]
    subprocess.run(cmd, check=True)
    return tmp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default="runs",
                        help="local runs/ tree to scan")
    parser.add_argument("--remote", default=None,
                        help="remote dir (e.g. node2:/home/.../runs) to rsync first")
    args = parser.parse_args()

    root = fetch_remote(args.remote) if args.remote else Path(args.path)
    if not root.exists():
        print(f"[aggregate] no runs at {root}", file=sys.stderr)
        return 1

    sweep_rows: list[dict] = []
    probe_rows: list[dict] = []
    scaling_rows: list[dict] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        # Each kind of experiment fingerprints itself by which JSON it
        # drops. Try each loader; the wrong ones return [].
        sweep_rows.extend(load_llm_sweep(entry))
        probe_rows.extend(load_llm_probe(entry))
        scaling_rows.extend(load_realistic_scaling(entry))

    out = [f"# Manifold-SAE runs aggregate\n\nRoot: `{root}`\n"]
    out.append(render_table(
        sweep_rows,
        ["run", "F", "top_k", "vanilla_expl", "curve_expl", "locked_expl",
         "vanilla_alive", "curve_alive", "delta_expl"],
        f"llm_sweep ({len(sweep_rows)} rows)",
    ))
    out.append(render_table(
        probe_rows,
        ["run", "concept", "layer", "phase1_rho",
         "vanilla_best_rho", "vanilla_n_above_moderate",
         "curve_best_rho_pos", "curve_n_above_moderate"],
        f"llm_probe ({len(probe_rows)} rows)",
    ))
    out.append(render_table(
        scaling_rows,
        ["run", "scenario", "vanilla_expl", "curve_expl", "delta_expl",
         "vanilla_chamfer", "curve_chamfer"],
        f"realistic_scaling ({len(scaling_rows)} rows)",
    ))

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
