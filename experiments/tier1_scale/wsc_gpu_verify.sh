#!/usr/bin/env bash
# WS-C item 4 launcher: run GPU engagement/decline verify on idle card 7,
# on the decline-fix venv_fable. Always exits 0; rc+log to /dev/shm/sauers_gpu.
set +e
OUT=/dev/shm/sauers_gpu
LOG=$OUT/wsc_gpu_verify.log
RC=$OUT/wsc_gpu_verify.rc
export CUDA_VISIBLE_DEVICES=7
export RAYON_NUM_THREADS=32 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
source /models/sauers_build/venv_fable/bin/activate
nice -19 ionice -c3 python /dev/shm/sauers_gpu/wsc_gpu_verify.py > "$LOG" 2>&1
echo "rc=$?" > "$RC"
exit 0
