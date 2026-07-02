"""Extend the weekday/month harvest with MORE template families for a powered
chart-transfer test. Reuses curved_feature_probes' model-loading + hook idioms
(resumable, per-set checkpoint, subprocess-safe under the box's OOM reaper).

Writes harvest_{weekday,month}.npz into chart_transfer/template_out/harvest_more/
with ~14 diverse template sentences per set (varied syntactic frame, tense, and
target position). Same model (Qwen2.5-0.5B), same layers, same readout as the probe.

Run:  python harvest_templates.py           # orchestrates retried subprocesses
      python harvest_templates.py --once     # one (kill-prone) harvest pass
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = Path(os.environ.get("HARVEST_MORE_OUT", HERE / "template_out" / "harvest_more"))

_spec = importlib.util.spec_from_file_location("cfp", EXP / "curved_feature_probes.py")
cfp = importlib.util.module_from_spec(_spec)
sys.modules["cfp"] = cfp
_spec.loader.exec_module(cfp)

WEEKDAYS = cfp.WEEKDAYS
MONTHS = cfp.MONTHS

# 14 diverse template families each: statement / question / subordinate clause /
# different tense / target at different sentence positions.
WEEKDAY_TEMPLATES = [
    "I will see you on {x}.",
    "The meeting is scheduled for {x}.",
    "She was born on a {x}.",
    "We always rest on {x}.",
    "By {x}, everything was ready.",
    "{x} is my favorite day of the week.",
    "Do you think {x} works for the call?",
    "Every {x} the market opens early.",
    "He promised to finish it before {x}.",
    "On {x} the whole team gathers.",
    "Nothing good ever happens on a {x}.",
    "Let's postpone the trip until {x}.",
    "The results were announced last {x}.",
    "Come rain or shine, {x} is cleaning day.",
]
MONTH_TEMPLATES = [
    "It happened in {x}.",
    "We got married in {x}.",
    "The festival is held every {x}.",
    "By {x} the snow had melted.",
    "Her birthday is in {x}.",
    "{x} is the busiest month for us.",
    "Will the harvest be ready by {x}?",
    "Every {x} the town holds a fair.",
    "He was hired back in {x}.",
    "In {x} the days grow noticeably longer.",
    "The lease expires at the end of {x}.",
    "They plan to travel again next {x}.",
    "The report is due the first of {x}.",
    "Nothing blooms here until {x}.",
]


def _sets():
    return {
        "weekday": {"labels": WEEKDAYS, "order": list(range(len(WEEKDAYS))),
                    "cyclic": True, "templates": WEEKDAY_TEMPLATES},
        "month": {"labels": MONTHS, "order": list(range(len(MONTHS))),
                  "cyclic": True, "templates": MONTH_TEMPLATES},
    }


def once():
    sys.excepthook = sys.__excepthook__
    cfg = cfp.HarvestConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    cfp.harvest(cfg, _sets(), OUT)     # resumable, per-set checkpoint


def main():
    if "--once" in sys.argv:
        return once()
    OUT.mkdir(parents=True, exist_ok=True)
    target = ["weekday", "month"]

    def cached():
        return [s for s in target if (OUT / f"harvest_{s}.npz").exists()]

    for attempt in range(1, 6):
        if len(cached()) == len(target):
            break
        print(f"[harvest_more] attempt {attempt}, cached={cached()}", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--once"], env=os.environ)
    print(f"[harvest_more] final cached={cached()} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
