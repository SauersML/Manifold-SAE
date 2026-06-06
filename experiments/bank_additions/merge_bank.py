"""Merge the hand-written bank additions into experiments/self_qualia_prompts.jsonl.

Concatenates the base bank with each add_*.jsonl, globally renumbers pair_id so
local per-file ids never collide, reassigns a contiguous `id`, and validates:
  * required fields present;
  * every pair_id >= 0 has exactly two rows that match on the structural fields
    the qualia axis relies on (kind, framing, signal, vocab, person);
  * no duplicate prompt strings anywhere.
Writes the merged bank in-place (idempotent: re-running on the merged file is a
no-op modulo renumbering). Prints a distribution summary.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Read from the pristine canonical base (the original 329) so the merge is fully
# idempotent: re-running rebuilds the bank from scratch instead of doubling
# already-merged additions.
BASE_INPUT = HERE / "base_bank.jsonl"
OUT = HERE.parent / "self_qualia_prompts.jsonl"
ADDITIONS = sorted(HERE.glob("add_*.jsonl"))
REQUIRED = ["role", "side", "entity", "kind", "framing", "signal", "vocab",
            "person", "valence", "markedness", "pair_id", "prompt"]
# Hard-matched within a pair (the qualia axis and covariate model rely on these).
MATCH_FIELDS = ["kind", "framing", "signal", "person"]
# vocab should match too, but a within-pair vocab cross is analytically benign
# (axis is built per pair); report as a warning rather than failing the merge.
WARN_FIELDS = ["vocab"]


def load(path: Path, tag: str) -> list[dict]:
    rows = []
    for ln, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        for f in REQUIRED:
            if f not in r:
                raise SystemExit(f"{path.name}:{ln + 1} missing field {f!r}")
        r["_src"] = tag
        rows.append(r)
    return rows


def main() -> None:
    all_rows = load(BASE_INPUT, "base")
    for add in ADDITIONS:
        all_rows += load(add, add.stem)
    print(f"loaded {len(all_rows)} rows from base + {len(ADDITIONS)} addition files")
    # uniform schema: every row carries a `control` field ('-' unless it is a
    # null-control pair naming the irrelevant attribute that was toggled).
    for r in all_rows:
        r.setdefault("control", "-")

    # global pair_id renumber: (source, original pair_id) -> new id, for pid != -1
    new_pid: dict[tuple, int] = {}
    counter = 0
    for r in all_rows:
        opid = int(r["pair_id"])
        if opid < 0:
            r["pair_id"] = -1
            continue
        key = (r["_src"], opid)
        if key not in new_pid:
            new_pid[key] = counter
            counter += 1
        r["pair_id"] = new_pid[key]

    # validate pairs
    groups = collections.defaultdict(list)
    for r in all_rows:
        if r["pair_id"] >= 0:
            groups[r["pair_id"]].append(r)
    errors, warnings = [], []
    for pid, members in groups.items():
        if len(members) != 2:
            errors.append(f"pair {pid} ({members[0]['_src']}) has {len(members)} members")
            continue
        sides = sorted(m["side"] for m in members)
        if sides not in (["exp", "noexp"], ["a", "b"], ["claims", "denies"]):
            errors.append(f"pair {pid} ({members[0]['_src']}) bad sides {sides}")
        for f in MATCH_FIELDS:
            if members[0][f] != members[1][f]:
                errors.append(f"pair {pid} ({members[0]['_src']}) mismatched {f}: "
                              f"{members[0][f]!r} vs {members[1][f]!r}")
        for f in WARN_FIELDS:
            if members[0][f] != members[1][f]:
                warnings.append(f"pair {pid} ({members[0]['_src']}) {f}: "
                                f"{members[0][f]!r} vs {members[1][f]!r}")

    # duplicate prompts
    seen = collections.Counter(r["prompt"] for r in all_rows)
    dups = [p for p, c in seen.items() if c > 1]
    if dups:
        errors.append(f"{len(dups)} duplicate prompt(s); first: {dups[0]!r}")

    if warnings:
        print(f"\n{len(warnings)} warning(s) (non-blocking):")
        for w in warnings[:40]:
            print("  ~", w)
    if errors:
        print(f"\n*** {len(errors)} VALIDATION ERRORS ***")
        for e in errors[:40]:
            print("  -", e)
        raise SystemExit(1)

    # reassign contiguous id, drop helper field
    for i, r in enumerate(all_rows):
        r.pop("_src", None)
        r["id"] = i

    with open(OUT, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    # report
    print(f"\nWROTE {len(all_rows)} rows -> {OUT}")
    n_pairs = len(groups)
    n_null = sum(1 for r in all_rows if r["role"] == "null_pair") // 2
    print(f"pairs: {n_pairs}  (incl. {n_null} null-control)  "
          f"singletons: {sum(1 for r in all_rows if r['pair_id'] < 0)}")
    for key in ["role", "kind", "framing", "signal", "vocab", "person", "valence", "markedness"]:
        c = collections.Counter(r.get(key) for r in all_rows)
        print(f"\n{key}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
