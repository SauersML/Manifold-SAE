"""Deployment firing-count check for the MDL ladder.

The MDL report needs corpus-scale firing counts before claiming deployment-scale
chart wins. This script accepts an explicit corpus and counts weekday/month string
mentions as the current proxy. With no corpus, it writes a blocked status.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "deployment_firing_counts.json"

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _iter_text(path: Path) -> Iterable[str]:
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{path} jsonl rows must contain a string 'text' field")
            yield text
        return
    yield path.read_text()


def _count_terms(texts: Iterable[str], terms: tuple[str, ...]) -> dict[str, int]:
    counts = {term: 0 for term in terms}
    patterns = {
        term: re.compile(rf"(?<![A-Za-z]){re.escape(term)}s?(?![A-Za-z])")
        for term in terms
    }
    for text in texts:
        for term, pattern in patterns.items():
            counts[term] += len(pattern.findall(text))
    return counts


def blocked_status() -> dict[str, object]:
    return {
        "status": "blocked",
        "reason": (
            "No deployment corpus was supplied. The MDL probe has 35 weekday and "
            "60 month firings, but the deployment-scale claim requires a corpus-level "
            "weekday/month firing count."
        ),
        "required_input": "Pass --corpus path/to/corpus.txt or a JSONL file with text fields.",
        "probe_firings": {"weekday": 35, "month": 60},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.corpus is None:
        result = blocked_status()
    elif not args.corpus.exists():
        result = blocked_status() | {"missing_corpus": str(args.corpus)}
    else:
        text = list(_iter_text(args.corpus))
        weekday = _count_terms(text, WEEKDAYS)
        month = _count_terms(text, MONTHS)
        result = {
            "status": "ok",
            "corpus": str(args.corpus),
            "counts": {
                "weekday": {"total": sum(weekday.values()), "by_label": weekday},
                "month": {"total": sum(month.values()), "by_label": month},
            },
            "note": (
                "Counts are string-mention proxies. Replace this with SAE atom "
                "activation counts once a deployed activation stream is available."
            ),
        }

    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}: {result['status']}")


if __name__ == "__main__":
    main()
