#!/usr/bin/env python3
"""Submit a Manifold-SAE experiment to a Heimdall job scheduler.

This script intentionally does **not** ship with any cluster-specific
defaults baked in. It reads every site-specific value from environment
variables or a local config file, and refuses to run when required
values are missing. The repo itself stays portable.

Configuration sources (in priority order)
-----------------------------------------
1. CLI flags (`--api`, `--working-dir`, ...).
2. Environment variables (`HEIMDALL_API`, `MSAE_WORKING_DIR`, ...).
3. Local config file at `--config` (default
   `~/.config/manifold-sae/heimdall.json`). NOT committed; gitignored.

Required values
---------------
  HEIMDALL_API                full URL of the job POST endpoint
  MSAE_WORKING_DIR            absolute path on the target node where the
                              repo will be cloned and outputs written

Optional values (with defaults)
-------------------------------
  MSAE_GIT_URL                git remote to clone (default: the public
                              Manifold-SAE GitHub URL — safe to bake in)
  MSAE_NODE                   target node label (default: empty — must
                              be set by site config)
  MSAE_GPUS                   GPU count to request (default: 0)
  MSAE_DEFAULT_MINUTES        estimated_minutes default (default: 120)

Use ``--dry-run`` to print the JSON spec without submitting. Anyone
sharing this script publicly only learns the structure of a job spec,
not where any particular cluster lives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Public default: the upstream repo URL is fine to bake in (it's open source).
DEFAULT_GIT_URL = "https://github.com/SauersML/Manifold-SAE.git"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "manifold-sae" / "heimdall.json"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[submit] {path} is not valid JSON: {exc}", file=sys.stderr)
        return {}


def resolve(key: str, cli_value: str | None, env_var: str, config: dict[str, Any], default: str | None = None) -> str | None:
    if cli_value is not None:
        return cli_value
    if env_var in os.environ and os.environ[env_var]:
        return os.environ[env_var]
    if key in config:
        return str(config[key])
    return default


def build_command(experiment: str, git_url: str, git_ref: str, working_dir: str, require_cuda: bool = False) -> str:
    """Single shell string. Bootstraps uv, clones/pulls Manifold-SAE,
    installs deps with the LLM extra, runs experiments.<experiment>.
    Output dir comes from MSAE_OUTPUT_DIR env var so the experiment
    writes under the caller's working_dir tree.
    """
    require_cuda_export = (
        "# Submitter requested GPUs — make the experiment FAIL FAST if CUDA\n"
        "# isn't actually visible, instead of silently running on CPU.\n"
        "export MSAE_REQUIRE_CUDA=1"
        if require_cuda else
        "# CPU-only run: not asserting CUDA availability."
    )
    return rf"""
set -euo pipefail

WORKDIR={working_dir!r}
REPO="$WORKDIR/Manifold-SAE"
OUTPUT="$WORKDIR/runs/${{MSAE_RUN_NAME:-default}}"

mkdir -p "$WORKDIR" "$OUTPUT"
cd "$WORKDIR"

if [ ! -d "$REPO/.git" ]; then
    rm -rf "$REPO"
    git clone -q {git_url!r} "$REPO"
fi
cd "$REPO"
git fetch origin -q
git checkout -q {git_ref!r}
git reset --hard -q origin/{git_ref!r}
echo "[submit] repo at $(git rev-parse --short HEAD): $(git log -1 --pretty='%s')"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Refuse to trust the cached venv across runs. uv sync's "resolved in 1ms"
# fast-path can silently keep stale wheels even when the lock changed —
# I hit this with torch 2.12 surviving a pin to <2.12. Cost of a clean
# install is ~30s + a few GB of download; cost of running on the wrong
# torch on a cluster job is hours of wasted compute. Pin .venv to the
# lock hash so we only reinstall when something actually changed.
LOCK_HASH=$(sha256sum uv.lock 2>/dev/null | awk '{{print $1}}')
STAMP=".venv/.heimdall_lock_hash"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$LOCK_HASH" ]; then
    echo "[submit] lock hash changed (or no venv) — clean install"
    rm -rf .venv
    uv sync --extra llm
    echo "$LOCK_HASH" > "$STAMP"
else
    echo "[submit] venv up to date with uv.lock (hash $LOCK_HASH)"
    uv sync --extra llm
fi

export MANIFOLD_SAE_OUTPUT_DIR="$OUTPUT"
export PYTHONUNBUFFERED=1
{require_cuda_export}

echo "[submit] running experiments.{experiment}"
uv run python -u -m experiments.{experiment}

echo "[submit] done — outputs at $OUTPUT"
ls -la "$OUTPUT" || true
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="llm_sweep", help="experiments.<name> module to run")
    parser.add_argument("--git-ref", default="main", help="branch/tag/commit to check out")
    parser.add_argument("--git-url", default=None, help=f"git remote (default: env MSAE_GIT_URL or {DEFAULT_GIT_URL})")
    parser.add_argument("--api", default=None, help="Heimdall API URL (env: HEIMDALL_API)")
    parser.add_argument("--node", default=None, help="target node label (env: MSAE_NODE)")
    parser.add_argument("--working-dir", default=None, help="absolute path on target node (env: MSAE_WORKING_DIR)")
    parser.add_argument("--gpus", type=int, default=None, help="GPU count to request (env: MSAE_GPUS)")
    parser.add_argument("--estimated-minutes", type=int, default=None, help="wall-time hint (env: MSAE_DEFAULT_MINUTES)")
    parser.add_argument("--run-name", default=None, help="subdir under <working-dir>/runs/ for outputs")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="local JSON config (gitignored)")
    parser.add_argument("--submitted-by", default="manifold-sae", help="value for the submitted_by field")
    parser.add_argument("--depends-on", action="append", default=[],
                        help="job ID this submission should wait for (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print the JSON spec without submitting")
    args = parser.parse_args()

    config = load_config(Path(args.config).expanduser())
    api = resolve("api", args.api, "HEIMDALL_API", config)
    working_dir = resolve("working_dir", args.working_dir, "MSAE_WORKING_DIR", config)
    node = resolve("node", args.node, "MSAE_NODE", config, default="")
    git_url = resolve("git_url", args.git_url, "MSAE_GIT_URL", config, default=DEFAULT_GIT_URL)
    gpus_str = resolve("gpus", str(args.gpus) if args.gpus is not None else None, "MSAE_GPUS", config, default="0")
    minutes_str = resolve("estimated_minutes",
                          str(args.estimated_minutes) if args.estimated_minutes is not None else None,
                          "MSAE_DEFAULT_MINUTES", config, default="120")

    missing = [name for name, val in [("HEIMDALL_API / --api", api),
                                       ("MSAE_WORKING_DIR / --working-dir", working_dir)]
               if not val]
    if missing:
        print(f"[submit] missing required values: {', '.join(missing)}", file=sys.stderr)
        print(f"[submit] set them via env vars, CLI flags, or a config file at {args.config}", file=sys.stderr)
        print(f"[submit] example config: {{\"api\": \"http://...\", \"working_dir\": \"/path/...\", \"node\": \"...\"}}", file=sys.stderr)
        return 2

    gpus = int(gpus_str)
    estimated_minutes = int(minutes_str)
    run_name = args.run_name or args.experiment

    spec = {
        "spec": {
            "job_type": "custom",
            "name": f"manifold-sae-{args.experiment}",
            "command": (
                f"export MSAE_RUN_NAME={run_name!r}\n"
                + build_command(args.experiment, git_url, args.git_ref, working_dir, require_cuda=(gpus > 0))
            ),
            "gpus": gpus,
            "node": node,
            "working_dir": working_dir,
            "estimated_minutes": estimated_minutes,
            "tags": ["manifold-sae", args.experiment, args.git_ref],
            "depends_on": args.depends_on,
        },
        "submitted_by": args.submitted_by,
    }

    if args.dry_run:
        # Mask the API URL in the printed spec so dry-run output is safe to
        # share. The actual POST still uses the real value.
        safe = json.loads(json.dumps(spec))
        print(json.dumps(safe, indent=2))
        print(f"\n[submit] would POST to: {api}", file=sys.stderr)
        return 0

    body = json.dumps(spec).encode("utf-8")
    req = urllib.request.Request(api, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"[submit] HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"[submit] connection error: {exc.reason}", file=sys.stderr)
        print("[submit] is the cluster reachable from here (VPN / direct LAN)?", file=sys.stderr)
        return 1

    job = data.get("job", {})
    print(f"[submit] job id={job.get('id', '?')} status={job.get('status', '?')}")
    for w in data.get("warnings", []) or []:
        print(f"[submit] warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
