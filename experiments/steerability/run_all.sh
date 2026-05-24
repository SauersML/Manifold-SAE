#!/usr/bin/env bash
# One-shot runner: poll cogito until ready, then harvest hex + normal-color,
# then run the steering benchmark, then summarize via metrics.py.
#
# Reads endpoint from COGITO_API_BASE (NEVER hardcoded).
# Bails on any step's failure with a clear error.
#
# Usage:
#   COGITO_API_BASE=http://<host>:8000 bash run_all.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

: "${COGITO_API_BASE:?Set COGITO_API_BASE before running (e.g. http://<host>:8000)}"
STATUS_URL="${COGITO_API_BASE%/}/v1/status"

PY="${PYTHON:-python3}"

echo "[run_all] polling $STATUS_URL until cogito is ready (max 20 min)"
deadline=$(( $(date +%s) + 1200 ))
ready=0
while (( $(date +%s) < deadline )); do
    if curl -fsS --max-time 8 "$STATUS_URL" >/dev/null 2>&1; then
        echo "[run_all] cogito is reachable at $STATUS_URL"
        ready=1
        break
    fi
    echo "[run_all] not ready, sleeping 30s..."
    sleep 30
done
if (( ready != 1 )); then
    echo "[run_all][error] cogito did not become ready within 20 min" >&2
    exit 1
fi

echo "[run_all] ===== STEP 1/4: harvest_hex.py ====="
"$PY" harvest_hex.py || { echo "[run_all][error] harvest_hex.py failed" >&2; exit 2; }

echo "[run_all] ===== STEP 2/4: harvest_normal_color.py ====="
"$PY" harvest_normal_color.py || { echo "[run_all][error] harvest_normal_color.py failed" >&2; exit 3; }

echo "[run_all] ===== STEP 3/4: cogito_steering_bench.py ====="
"$PY" cogito_steering_bench.py || { echo "[run_all][error] cogito_steering_bench.py failed" >&2; exit 4; }

echo "[run_all] ===== STEP 4/4: metrics.py summary ====="
"$PY" metrics.py cogito_intervention_results.jsonl || { echo "[run_all][error] metrics.py failed" >&2; exit 5; }

echo "[run_all] ALL STEPS COMPLETE"
