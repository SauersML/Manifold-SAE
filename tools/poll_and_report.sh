#!/bin/bash
# Poll Heimdall for completion, fetch results, regenerate the aggregate
# markdown report.
#
#   bash tools/poll_and_report.sh
#   bash tools/poll_and_report.sh --interval 60
#
# Runs once unless --watch passed.
set -euo pipefail

INTERVAL=0
WATCH=0
for arg in "$@"; do
    case "$arg" in
        --watch) WATCH=1 ;;
        --interval) shift; INTERVAL="$1" ;;
    esac
    shift
done

iter() {
    echo "=== $(date +%H:%M:%S) ==="
    # Fetch every run dir mentioned in active jobs
    for run in llm_sweep llm_sweep_L18 llm_sweep_L18_F4096_topk32 \
               llm_sweep_q15b_L18 llm_sweep_q15b_L18_F1024 \
               llm_sweep_L4_F64 llm_sweep_L8_F64 llm_sweep_L16_F64 llm_sweep_L20_F64 \
               llm_sweep_q15b_L4_F64 llm_sweep_q15b_L8_F64 \
               llm_sweep_q15b_L12_F64 llm_sweep_q15b_L16_F64 llm_sweep_q15b_L20_F64 \
               llm_sweep_L12_F128_R1 llm_sweep_L12_F128_R4 llm_sweep_L12_F128_R8 \
               llm_sweep_L12_F128_K4 llm_sweep_L12_F128_K24 \
               llm_sweep_L12_F128_binary_amp \
               llm_probe llm_probe_L18 \
               realistic_scaling realistic_scaling_v2 \
               cyclic_concepts_q15b_L18 continuous_recovery \
               steering_F256; do
        MSAE_NODE=node2 heimdall_jobs/fetch_results.sh "$run" --no-open 2>/dev/null \
            | grep -E '^runs_cluster' | wc -l | tr -d ' ' \
            | xargs -I{} echo "  $run: {} files"
    done
    # Status table
    python3 heimdall_jobs/status.py 2>&1 | head -25
    # Aggregate
    python3 tools/aggregate_results.py runs_cluster > runs/AGGREGATE.md 2>/dev/null || true
    if [ -f runs/AGGREGATE.md ]; then
        wc -l runs/AGGREGATE.md
    fi
}

if [ $WATCH -eq 1 ] && [ "$INTERVAL" != "0" ]; then
    while true; do
        iter
        sleep "$INTERVAL"
    done
else
    iter
fi
