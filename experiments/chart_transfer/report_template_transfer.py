"""Render the template-transfer report from template_transfer.json.

The report makes all-fold held-out EV the primary metric and treats coordinate
consistency as secondary evidence. It does not rerun chart fits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_JSON = HERE / "template_out" / "template_transfer.json"
DEFAULT_OUT = HERE / "TEMPLATE_TRANSFER_REPORT.md"


def _round(x: float | None, ndigits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{ndigits}f}"


def _folds(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        fold for fold in result.get("loto", [])
        if fold.get("chart_status") == "CONVERGED"
    ]


def _values(folds: list[dict[str, Any]], key: str) -> list[float]:
    return [float(fold[key]) for fold in folds if key in fold and fold[key] is not None]


def _summary(result: dict[str, Any]) -> dict[str, float | int | None]:
    folds = _folds(result)
    coord = _values(folds, "coord_consistency_circ_corr")
    chart = _values(folds, "chart_ev_eval")
    linear1 = _values(folds, "linear1_ev_eval")
    linear2 = _values(folds, "linear2_ev_eval")
    adjacency = _values(folds, "adjacency_eval_templates")
    clean = [
        fold for fold in folds
        if float(fold.get("coord_consistency_circ_corr", -1.0)) > 0.8
    ]
    clean_chart = _values(clean, "chart_ev_eval")
    clean_linear1 = _values(clean, "linear1_ev_eval")
    clean_linear2 = _values(clean, "linear2_ev_eval")
    return {
        "n_folds": len(folds),
        "chart_ev": mean(chart) if chart else None,
        "linear1_ev": mean(linear1) if linear1 else None,
        "linear2_ev": mean(linear2) if linear2 else None,
        "chart_minus_linear1": (mean(chart) - mean(linear1)) if chart and linear1 else None,
        "chart_minus_linear2": (mean(chart) - mean(linear2)) if chart and linear2 else None,
        "coord_mean": mean(coord) if coord else None,
        "coord_median": median(coord) if coord else None,
        "coord_frac_over_0_8": mean([x > 0.8 for x in coord]) if coord else None,
        "adjacency": mean(adjacency) if adjacency else None,
        "clean_n": len(clean),
        "clean_chart_ev": mean(clean_chart) if clean_chart else None,
        "clean_linear1_ev": mean(clean_linear1) if clean_linear1 else None,
        "clean_linear2_ev": mean(clean_linear2) if clean_linear2 else None,
    }


def _ev_verdict(name: str, s: dict[str, float | int | None]) -> str:
    chart = s["chart_ev"]
    linear2 = s["linear2_ev"]
    if chart is None or linear2 is None:
        return "blocked"
    if chart > linear2:
        return "chart beats 2D linear on all-fold EV"
    if name == "weekday":
        return "EV-null/fragile: 2D linear beats chart"
    return "EV-null: chart does not establish a robust all-fold win"


def render(data: dict[str, Any]) -> str:
    sets = data["sets"]
    summaries = {name: _summary(result) for name, result in sets.items()}

    lines = [
        "# Template-transfer report",
        "",
        "**Primary metric:** all-fold held-out-template EV from",
        "`template_out/template_transfer.json`. Coordinate consistency is reported as",
        "secondary evidence only; it is not allowed to override the EV result.",
        "",
        "## Verdict",
        "",
        "The all-fold EV result is **null** for a robust transferable circle claim.",
        "Weekday is explicitly fragile: its coordinate median is high, but the",
        "2-coordinate linear baseline has higher all-fold EV. Month is the stronger",
        "case, but it is best described as month/null-robust rather than a general",
        "weekday-and-month robust-circle result.",
        "",
        "| set | folds | chart EV | linear-1 EV | linear-2 EV | chart-linear1 | chart-linear2 | EV verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in sorted(summaries):
        s = summaries[name]
        lines.append(
            f"| {name} | {s['n_folds']} | {_round(s['chart_ev'])} | "
            f"{_round(s['linear1_ev'])} | {_round(s['linear2_ev'])} | "
            f"{_round(s['chart_minus_linear1'])} | {_round(s['chart_minus_linear2'])} | "
            f"{_ev_verdict(name, s)} |"
        )

    lines.extend([
        "",
        "## Coordinate Readout",
        "",
        "Coordinate consistency is useful as an interpretability diagnostic, but the",
        "weekday result is not a robust reconstruction-transfer win.",
        "",
        "| set | mean circ-corr | median circ-corr | frac folds > 0.8 | unseen adjacency |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in sorted(summaries):
        s = summaries[name]
        lines.append(
            f"| {name} | {_round(s['coord_mean'])} | {_round(s['coord_median'])} | "
            f"{_round(s['coord_frac_over_0_8'], 2)} | {_round(s['adjacency'])} |"
        )

    lines.extend([
        "",
        "## Clean-Fold Slice",
        "",
        "The clean-fold slice keeps folds with coordinate consistency > 0.8. It is",
        "reported for diagnosis, not as the headline metric.",
        "",
        "| set | clean folds | chart EV | linear-1 EV | linear-2 EV |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in sorted(summaries):
        s = summaries[name]
        lines.append(
            f"| {name} | {s['clean_n']} | {_round(s['clean_chart_ev'])} | "
            f"{_round(s['clean_linear1_ev'])} | {_round(s['clean_linear2_ev'])} |"
        )

    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 experiments/chart_transfer/report_template_transfer.py",
        "```",
        "",
        "This command renders the markdown report from the existing JSON. Rerunning",
        "`template_transfer.py` requires the richer harvest cache in",
        "`experiments/chart_transfer/template_out/harvest_more/`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = json.loads(args.json.read_text())
    args.out.write_text(render(data))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
