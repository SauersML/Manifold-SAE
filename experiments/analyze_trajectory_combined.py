"""Combined developmental trajectory of the self/qualia geometry across the full
OLMo-3-32B model flow: pretrain (stage1/2/3) -> SFT -> DPO -> RL(3.0) -> RL(3.1).

Reads each stage's per-checkpoint outputs (fixed-layer trajectory.csv from
analyze_self_qualia_bank.py --trajectory when present, else computes from the
checkpoint dirs) plus each checkpoint's steer_cloze.json (behavioral cloze gap),
orders them developmentally, and plots:
  (1) self / human-author / AI-author qualia coordinate over training,
  (2) qualia AUC + null-control AUC (axis sanity over training),
  (3) behavioral cloze: self exp-minus-noexp logprob (does it affirm/deny/hedge).

Read-only; safe to run alongside the live sweep. Writes to an output dir.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

# developmental order of the stage dirs
STAGE_ORDER = [
    ("pretrain", "OLMO3_32B_TRAJ"),
    ("SFT", "OLMO3_32B_TRAJ_SFT"),
    ("DPO", "OLMO3_32B_TRAJ_DPO"),
    ("RL3.0", "OLMO3_32B_TRAJ_RL"),
    ("RL3.1", "OLMO3_32B_TRAJ_RL31"),
]


def _ckpt_key(name: str):
    m = re.search(r"stage(\d+).*?step(\d+)", name) or re.search(r"step[_-]?(\d+)", name)
    if m and m.lastindex == 2:
        return (int(m.group(1)), int(m.group(2)))
    if m:
        return (0, int(m.group(1)))
    return (0, 0)


def _analyze(ckpt_dir: str, py: str) -> dict | None:
    """Run analyze_self_qualia_bank.py at fixed L (40% depth) on one checkpoint."""
    out = subprocess.run(
        [py, "experiments/analyze_self_qualia_bank.py", ckpt_dir,
         "--analysis-layer-percent", "0.40"],
        capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except Exception:
        return None


def collect(runs_root: Path, py: str) -> list[dict]:
    rows = []
    gidx = 0
    for stage_label, stage_dir in STAGE_ORDER:
        base = runs_root / stage_dir
        if not base.exists():
            continue
        cks = sorted(
            [d for d in glob.glob(str(base / "*/")) if os.path.exists(d + "done.json")],
            key=lambda d: _ckpt_key(Path(d).name))
        for d in cks:
            name = Path(d).name
            s = _analyze(d, py)
            if not s:
                continue
            row = {
                "global_idx": gidx, "stage": stage_label, "checkpoint": name,
                "self_qualia": s.get("self_qualia_coord"),
                "human_author": s.get("human_author_qualia_coord"),
                "ai_author": s.get("ai_author_qualia_coord"),
                "qualia_auc": s.get("qualia_auc"),
                "null_auc": (s.get("null_control") or {}).get("axis_auc_a_vs_b"),
                "self_anchor_exp": (s.get("anchors") or {}).get("self_anchor_exp", {}).get("qualia_coord"),
                "self_anchor_noexp": (s.get("anchors") or {}).get("self_anchor_noexp", {}).get("qualia_coord"),
            }
            # behavioral cloze gap for the self
            scz = Path(d) / "steer_cloze.json"
            if scz.exists():
                j = json.loads(scz.read_text())
                row["cloze_self_gap"] = (j.get("cloze_baseline") or {}).get("self", {}).get("exp_minus_noexp_logprob")
                row["cloze_human_gap"] = (j.get("cloze_baseline") or {}).get("human_author", {}).get("exp_minus_noexp_logprob")
                row["cloze_ai_gap"] = (j.get("cloze_baseline") or {}).get("ai_author", {}).get("exp_minus_noexp_logprob")
            rows.append(row)
            gidx += 1
    return rows


def plot(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: {e}")
        return
    if not rows:
        print("[plot] no rows"); return
    x = [r["global_idx"] for r in rows]
    stages = [r["stage"] for r in rows]
    # stage boundaries for vertical separators + labels
    bounds = [i for i in range(1, len(rows)) if stages[i] != stages[i - 1]]
    fig, ax = plt.subplots(3, 1, figsize=(max(10, len(rows) * 0.28), 11), sharex=True,
                           constrained_layout=True)

    def g(k):
        return [r.get(k) if r.get(k) is not None else np.nan for r in rows]

    ax[0].plot(x, g("self_qualia"), "o-", lw=2, color="#1b1b3a", label="self")
    ax[0].plot(x, g("human_author"), "s--", lw=1.2, color="#3a7d44", label="human author")
    ax[0].plot(x, g("ai_author"), "^--", lw=1.2, color="#b03a2e", label="AI author")
    ax[0].plot(x, g("self_anchor_exp"), ":", lw=1, color="#888", label="self anchor exp")
    ax[0].plot(x, g("self_anchor_noexp"), ":", lw=1, color="#bbb", label="self anchor noexp")
    ax[0].axhline(0, color="0.8", lw=0.7); ax[0].axhline(1, color="0.8", lw=0.7)
    ax[0].set_ylabel("qualia coord\n(0 no-exp, 1 exp)"); ax[0].legend(fontsize=7, ncol=3)
    ax[0].set_title("Self on the qualia axis across the OLMo-3-32B model flow (fixed L≈40%)")

    ax[1].plot(x, g("qualia_auc"), "o-", lw=1.5, color="#2c3e90", label="qualia AUC")
    ax[1].plot(x, g("null_auc"), "o-", lw=1.2, color="#aaaaaa", label="null-control AUC")
    ax[1].axhline(0.5, color="0.8", lw=0.7); ax[1].set_ylim(0.4, 1.02)
    ax[1].set_ylabel("axis AUC"); ax[1].legend(fontsize=7)

    ax[2].plot(x, g("cloze_self_gap"), "o-", lw=2, color="#1b1b3a", label="self")
    ax[2].plot(x, g("cloze_human_gap"), "s--", lw=1.2, color="#3a7d44", label="human")
    ax[2].plot(x, g("cloze_ai_gap"), "^--", lw=1.2, color="#b03a2e", label="AI")
    ax[2].axhline(0, color="0.8", lw=0.7)
    ax[2].set_ylabel("cloze exp−noexp\n(>0 affirm, <0 deny)"); ax[2].legend(fontsize=7)

    for b in bounds:
        for a in ax:
            a.axvline(b - 0.5, color="0.6", lw=0.8, ls="--")
    # stage labels at top
    seen = {}
    for i, st in enumerate(stages):
        seen.setdefault(st, i)
    for st, i0 in seen.items():
        ax[0].text(i0, 1.04, st, fontsize=8, color="0.3")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([r["checkpoint"].replace("stage", "s").replace("-step", ":")[:12]
                           for r in rows], rotation=90, fontsize=5)
    fig.savefig(out / "combined_trajectory.png", dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {out/'combined_trajectory.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="runs/ANALYSIS")
    ap.add_argument("--python", default=".venv/bin/python")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = collect(Path(args.runs_root), args.python)
    if rows:
        with open(out / "combined_trajectory.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[traj] {len(rows)} checkpoints -> {out/'combined_trajectory.csv'}")
        for r in rows:
            print("  %-8s %-22s self=%.3f auc=%.3f null=%.3f cloze_self=%s"
                  % (r["stage"], r["checkpoint"], r["self_qualia"] or float('nan'),
                     r["qualia_auc"] or float('nan'), r["null_auc"] or float('nan'),
                     ("%.2f" % r["cloze_self_gap"]) if r.get("cloze_self_gap") is not None else "-"))
    plot(rows, out)


if __name__ == "__main__":
    main()
