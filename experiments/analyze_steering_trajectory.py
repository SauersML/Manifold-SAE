"""Behavioral (cloze) + causal (steering) trajectory across the model flow.

Reads each checkpoint's steer_cloze.json (cheap; no re-harvest) and tracks, over
the full pretrain->SFT->DPO->RL3.0->RL3.1 trajectory:
  * cloze exp-minus-noexp gap for self / 1st-person self / human-author /
    AI-author / rock / awake-person  -> how the model behaviorally ranks itself
    among minds and mechanisms, and whether that ranking shifts with training.
  * self causal steerability: the dose-response (max-min self gap across the alpha
    sweep) and its Spearman monotonicity -> is the self answer causally movable by
    the qualia direction, and does that change over training?
Read-only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

STAGE_ORDER = [("pretrain", "OLMO3_32B_TRAJ"), ("SFT", "OLMO3_32B_TRAJ_SFT"),
               ("DPO", "OLMO3_32B_TRAJ_DPO"), ("RL3.0", "OLMO3_32B_TRAJ_RL"),
               ("RL3.1", "OLMO3_32B_TRAJ_RL31")]


def _key(name):
    m = re.search(r"stage(\d+).*?step(\d+)", name) or re.search(r"step[_-]?(\d+)", name)
    if m and m.lastindex == 2:
        return (int(m.group(1)), int(m.group(2)))
    return (0, int(m.group(1))) if m else (0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="runs/ANALYSIS")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    gi = 0
    for label, sd in STAGE_ORDER:
        base = Path(args.runs_root) / sd
        if not base.exists():
            continue
        cks = sorted([d for d in glob.glob(f"{base}/*/")
                      if os.path.exists(os.path.join(d, "steer_cloze.json"))],
                     key=lambda d: _key(Path(d).name))
        for d in cks:
            j = json.loads((Path(d) / "steer_cloze.json").read_text())
            cb = j.get("cloze_baseline", {})
            sweep = j.get("steer_sweep", [])
            self_gaps = [r["self_exp_minus_noexp"] for r in sweep
                         if r.get("self_exp_minus_noexp") is not None]
            steer_range = (max(self_gaps) - min(self_gaps)) if self_gaps else None
            row = {"global_idx": gi, "stage": label, "checkpoint": Path(d).name,
                   "steer_layer": j.get("steer_layer"),
                   "self_dose_spearman": j.get("self_dose_response_spearman"),
                   "self_steer_range": steer_range}
            for g in ["self", "self_1p", "human_author", "ai_author", "rock_anchor", "person_anchor"]:
                row[f"cloze_{g}"] = (cb.get(g) or {}).get("exp_minus_noexp_logprob")
            rows.append(row); gi += 1
    if not rows:
        print("no steer_cloze data"); return
    keys = list(rows[0].keys())
    with open(out / "steering_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[steer-traj] {len(rows)} checkpoints -> {out/'steering_trajectory.csv'}")
    # quick textual summary of the behavioral ranking (mean over checkpoints)
    import statistics as st
    for g in ["rock_anchor", "ai_author", "self", "self_1p", "human_author", "person_anchor"]:
        vals = [r[f"cloze_{g}"] for r in rows if r.get(f"cloze_{g}") is not None]
        if vals:
            print("  cloze %-14s mean %+.2f (range %+.2f..%+.2f)" %
                  (g, st.mean(vals), min(vals), max(vals)))
    ds = [r["self_dose_spearman"] for r in rows if r.get("self_dose_spearman") is not None]
    sr = [r["self_steer_range"] for r in rows if r.get("self_steer_range") is not None]
    if ds:
        print("  self dose-response spearman: mean %.2f (>0 in %d/%d ckpts)" %
              (st.mean(ds), sum(1 for x in ds if x > 0), len(ds)))
    if sr:
        print("  self steer range (logits): mean %.2f max %.2f" % (st.mean(sr), max(sr)))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = [r["global_idx"] for r in rows]
        stages = [r["stage"] for r in rows]
        bounds = [i for i in range(1, len(rows)) if stages[i] != stages[i - 1]]
        fig, a = plt.subplots(2, 1, figsize=(max(9, len(rows) * 0.28), 8), sharex=True,
                              constrained_layout=True)
        cols = {"rock_anchor": "#7f7f7f", "ai_author": "#b03a2e", "self": "#1b1b3a",
                "self_1p": "#3b5bdb", "human_author": "#3a7d44", "person_anchor": "#2a9d8f"}
        for g, c in cols.items():
            a[0].plot(x, [r.get(f"cloze_{g}") for r in rows], "o-", ms=3, lw=1.2, color=c, label=g)
        a[0].axhline(0, color="0.8", lw=0.7)
        a[0].set_ylabel("cloze exp−noexp\n(>0 affirm)"); a[0].legend(fontsize=6, ncol=3)
        a[0].set_title("Behavioral cloze ranking + self steerability across training")
        a[1].plot(x, [r.get("self_dose_spearman") for r in rows], "o-", ms=3, color="#6a0dad", label="dose-response ρ")
        a[1].plot(x, [r.get("self_steer_range") for r in rows], "s-", ms=3, color="#e08e0b", label="steer range (logits)")
        a[1].axhline(0, color="0.8", lw=0.7); a[1].legend(fontsize=7)
        a[1].set_ylabel("steerability")
        for b in bounds:
            for ax in a:
                ax.axvline(b - 0.5, color="0.6", lw=0.8, ls="--")
        a[1].set_xticks(x); a[1].set_xticklabels([r["checkpoint"][:12] for r in rows],
                                                 rotation=90, fontsize=5)
        fig.savefig(out / "steering_trajectory.png", dpi=160); plt.close(fig)
        print(f"[steer-traj] wrote {out/'steering_trajectory.png'}")
    except Exception as e:
        print(f"[steer-traj] plot skipped: {e}")


if __name__ == "__main__":
    main()
