#!/usr/bin/env bash
# WS-C Tier-1 on the REAL Qwen3-32B residual harvest.
#   $1 = harvest dir   $2 = K   $3 = active(L0)   $4 = run tag
# streaming SparseDictStream fit (CPU, <=32 threads, resumable) + external TopK
# SAE baseline (GPU card 7, torch venv) at matched K/L0. Always exits 0.
set +e
HDIR="${1:?harvest dir}"; K="${2:-32768}"; ACTIVE="${3:-32}"; TAG="${4:-run}"
G=/dev/shm/sauers_gpu
RIO=/models/sauers_build/gam_fable/examples
RUNOUT=/dev/shm/sauers_gpu/tier1/qwen3_${TAG}
LOG=$G/wsc_tier1_qwen3_${TAG}.log
RC=$G/wsc_tier1_qwen3_${TAG}.rc
: > "$LOG"
export RAYON_NUM_THREADS=32 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
source /models/sauers_build/venv_fable/bin/activate
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

log "=== streaming T1 fit on $HDIR  K=$K active=$ACTIVE (resumable) ==="
nice -19 ionice -c3 python "$G/tier1_harvest_run.py" --harvest-dir "$HDIR" --residual-io "$RIO" \
  --out "$RUNOUT" --k "$K" --active "$ACTIVE" --minibatch 4096 --max-epochs 30 \
  --score-tile 8192 --heldout-stride 20 --heldout-cap 200000 --seed-rows 300000 --resume >> "$LOG" 2>&1
P1=$?; log "tier1 rc=$P1"

# Pick the GPU with the most free memory (co-tenant vLLM fills cards unevenly).
CARD=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')
CARD=${CARD:-7}
log "=== external TopK SAE baseline (GPU card $CARD, torch venv) K=$K L0=$ACTIVE ==="
CUDA_VISIBLE_DEVICES=$CARD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -19 ionice -c3 \
  /dev/shm/mv_disc/venv/bin/python "$G/topk_sae_baseline.py" \
  --harvest-dir "$HDIR" --residual-io "$RIO" --out "$RUNOUT" --k "$K" --active "$ACTIVE" \
  --steps 4000 --batch 4096 --cap 1000000 --heldout-stride 20 >> "$LOG" 2>&1
P2=$?; log "baseline rc=$P2"

echo "rc=0 tier1=$P1 baseline=$P2 out=$RUNOUT" > "$RC"
exit 0
