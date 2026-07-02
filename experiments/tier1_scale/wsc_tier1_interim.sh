#!/usr/bin/env bash
# WS-C interim T1 pipeline on REAL OLMo-7B activations (token-limited slice):
# build manifest -> streaming SparseDictStream fit (CPU, <=32 threads, resumable)
# -> external TopK SAE baseline (GPU card 7) at matched K/L0. Always exits 0.
set +e
OUT=/dev/shm/sauers_gpu
LOG=$OUT/wsc_tier1_interim.log
RC=$OUT/wsc_tier1_interim.rc
: > "$LOG"
export RAYON_NUM_THREADS=32 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
source /models/sauers_build/venv_fable/bin/activate
G=/dev/shm/sauers_gpu
MAN=/dev/shm/sauers_gpu/tier1/interim/MANIFEST.json
RUNOUT=/dev/shm/sauers_gpu/tier1/interim_run
K=512; ACTIVE=32

log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

log "=== prep interim manifest from real OLMo activations ==="
nice -19 ionice -c3 python "$G/wsc_interim_prep.py" >> "$LOG" 2>&1
P1=$?; log "prep rc=$P1"

log "=== streaming T1 fit (K=$K active=$ACTIVE, resumable) ==="
nice -19 ionice -c3 python "$G/tier1_scale_run.py" --manifest "$MAN" --out "$RUNOUT" \
  --k $K --active $ACTIVE --minibatch 4096 --max-epochs 30 --score-tile 8192 --resume >> "$LOG" 2>&1
P2=$?; log "tier1 rc=$P2"

log "=== external TopK SAE baseline (GPU card 7, torch venv) ==="
CUDA_VISIBLE_DEVICES=7 nice -19 ionice -c3 /dev/shm/mv_disc/venv/bin/python "$G/topk_sae_baseline.py" --manifest "$MAN" \
  --out "$RUNOUT" --k $K --active $ACTIVE --steps 3000 --batch 4096 >> "$LOG" 2>&1
P3=$?; log "baseline rc=$P3"

echo "rc=0 prep=$P1 tier1=$P2 baseline=$P3" > "$RC"
exit 0
