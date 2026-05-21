#!/bin/bash
# Pull just the JSONs + PNGs from a cluster run dir into a local mirror.
# Skips the heavy .pt checkpoints (typically GBs).
#
# Usage:
#   heimdall_jobs/fetch_results.sh <run-name>
#   heimdall_jobs/fetch_results.sh llm_sweep
#   heimdall_jobs/fetch_results.sh llm_sweep_L18
#
# Reads SSH host + working dir from ~/.config/manifold-sae/heimdall.json
# (or env: MSAE_FETCH_HOST, MSAE_WORKING_DIR). Default host is `node2`.

set -euo pipefail

RUN="${1:?usage: fetch_results.sh <run-name>}"

CONFIG="$HOME/.config/manifold-sae/heimdall.json"
if [ -f "$CONFIG" ]; then
    WORKING_DIR="${MSAE_WORKING_DIR:-$(python3 -c "import json; print(json.load(open('$CONFIG'))['working_dir'])")}"
    NODE="${MSAE_NODE:-$(python3 -c "import json; print(json.load(open('$CONFIG')).get('node', 'node2'))")}"
else
    WORKING_DIR="${MSAE_WORKING_DIR:?MSAE_WORKING_DIR not set and no config file}"
    NODE="${MSAE_FETCH_HOST:-node2}"
fi

REMOTE="$WORKING_DIR/runs/$RUN"
LOCAL="runs_cluster/$RUN"

mkdir -p "$LOCAL"
echo "[fetch] $NODE:$REMOTE -> $LOCAL"

# Only JSON + PNG + .md, skip the multi-GB checkpoints.
rsync -av --include='*/' \
    --include='*.json' --include='*.png' --include='*.md' \
    --exclude='*' \
    "$NODE:$REMOTE/" "$LOCAL/"

echo
echo "[fetch] done. fetched files:"
find "$LOCAL" -type f \( -name '*.json' -o -name '*.png' -o -name '*.md' \) | head -20
echo

# On macOS open the figures in Preview so they're visible without
# manual file navigation. Skip if --no-open is passed or we're headless.
if [ "$(uname)" = "Darwin" ] && [ "${2:-}" != "--no-open" ]; then
    pngs=$(find "$LOCAL" -type f -name '*.png' | head -10)
    if [ -n "$pngs" ]; then
        echo "[fetch] opening figures in Preview…"
        echo "$pngs" | xargs open -a Preview
    fi
fi

echo "[fetch] to aggregate this run:"
echo "  python3 tools/aggregate_results.py runs_cluster/"
