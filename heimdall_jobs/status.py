#!/usr/bin/env python3
"""List Heimdall jobs filtered to manifold-sae.

Reads the same local config / env vars as submit.py for the API URL.
Prints a compact status table by default; --detailed pulls log tails too.

Usage:
  python3 heimdall_jobs/status.py                    # all manifold-sae jobs
  python3 heimdall_jobs/status.py --jobs <id> <id>   # specific job IDs
  python3 heimdall_jobs/status.py --detailed         # include log tails
  python3 heimdall_jobs/status.py --watch 10         # refresh every 10s
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "manifold-sae" / "heimdall.json"


def load_api() -> str:
    if "HEIMDALL_API" in os.environ and os.environ["HEIMDALL_API"]:
        return os.environ["HEIMDALL_API"]
    if DEFAULT_CONFIG_PATH.exists():
        try:
            cfg = json.loads(DEFAULT_CONFIG_PATH.read_text())
            if "api" in cfg:
                return str(cfg["api"])
        except json.JSONDecodeError:
            pass
    print("[status] HEIMDALL_API not set and no local config — pass --api", file=sys.stderr)
    sys.exit(2)


def fetch(url: str) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"[status] fetch {url} failed: {exc}", file=sys.stderr)
        return None


def list_jobs(api: str, jobs_arg: list[str] | None) -> list[dict]:
    if jobs_arg:
        out = []
        for jid in jobs_arg:
            d = fetch(f"{api}/jobs/{jid}")
            if d:
                out.append(d)
        return out
    # No filter — list all then keep manifold-sae ones
    base = api.rstrip("/")
    parent = base.rsplit("/", 1)[0]  # strip trailing /jobs
    listing = fetch(f"{parent}/jobs") or fetch(api)
    if isinstance(listing, dict) and "jobs" in listing:
        listing = listing["jobs"]
    if not isinstance(listing, list):
        return []
    return [j for j in listing
            if "manifold-sae" in (j.get("spec", {}).get("name") or "")
            or "manifold-sae" in (j.get("spec", {}).get("tags") or [])]


def fmt_time(ts: str | None) -> str:
    if not ts:
        return "—"
    return ts.replace("T", " ").split(".")[0]


def print_table(jobs: list[dict], detailed: bool, api: str) -> None:
    if not jobs:
        print("[status] no jobs")
        return
    # Sort newest first by submitted_at.
    jobs = sorted(jobs, key=lambda j: j.get("submitted_at") or "", reverse=True)
    print(f"{'job':14} {'status':12} {'node':6} {'name':40} {'started':20} {'finished':20}")
    print("-" * 120)
    for j in jobs:
        jid = j.get("id", "?")[:14]
        st = j.get("status", "?")[:12]
        node = (j.get("node") or "?")[:6]
        name = (j.get("spec", {}).get("name") or "")[:40]
        started = fmt_time(j.get("started_at"))[:20]
        finished = fmt_time(j.get("finished_at"))[:20]
        print(f"{jid:14} {st:12} {node:6} {name:40} {started:20} {finished:20}")
        if detailed:
            log_data = fetch(f"{api}/jobs/{j.get('id')}/logs")
            if log_data and isinstance(log_data, dict) and "log" in log_data:
                tail = "\n".join(log_data["log"].splitlines()[-5:])
                print(f"  log tail:")
                for line in tail.splitlines():
                    print(f"    {line}")
                print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=None, help="Heimdall API URL")
    parser.add_argument("--jobs", nargs="+", default=None, help="specific job IDs")
    parser.add_argument("--detailed", action="store_true", help="include log tails")
    parser.add_argument("--watch", type=int, default=0,
                        help="refresh interval in seconds (0 = one-shot)")
    args = parser.parse_args()
    api = args.api or load_api()
    while True:
        if args.watch:
            # ANSI clear-screen + home cursor; portable, no os.system.
            sys.stdout.write("\033[2J\033[H")
            print(f"=== {time.strftime('%H:%M:%S')} ===")
        jobs = list_jobs(api, args.jobs)
        print_table(jobs, args.detailed, api)
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
