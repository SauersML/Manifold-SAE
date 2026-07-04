#!/usr/bin/env bash
#SBATCH --job-name=premise_date_harvest
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
set -uo pipefail
ROOT=/projects/standard/hsiehph/sauer354
export ROOT
export HF_HOME=$ROOT/hf_cache
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PREMISE_OUT=$ROOT/premise_out
mkdir -p $PREMISE_OUT
PY=$ROOT/saevenv_head/bin/python
NVLIB=$ROOT/saevenv_head/lib/python3.11/site-packages/nvidia/cu13/lib
TORCHLIB=$($PY -c "import torch,os;print(os.path.dirname(torch.__file__)+'/lib')")
export LD_LIBRARY_PATH=$ROOT/cudart_shim:$NVLIB:$TORCHLIB:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd $ROOT/premise_run
export HD_MODEL=$ROOT/models/qwen3-8b HD_LAYER=18 HD_RANK=8 HD_DEVICE=cuda:0 HD_NDAYS=28
timeout 6000 $PY harvest_date.py || echo "harvest rc=$?"
echo "=== DONE ==="
ls -la $PREMISE_OUT/harvest_cache_dayofmonth_L18.npz $PREMISE_OUT/harvest_cache_jointdate_L18.npz 2>/dev/null
