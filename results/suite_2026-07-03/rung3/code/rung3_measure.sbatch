#!/usr/bin/env bash
#SBATCH --job-name=rung3_measure
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --requeue
set -e
ROOT=/projects/standard/hsiehph/sauer354
export HF_HOME=$ROOT/hf_cache
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
PY=$ROOT/saevenv/bin/python
NVLIB=$ROOT/saevenv/lib/python3.11/site-packages/nvidia/cu13/lib
TORCHLIB=$($PY -c "import torch,os;print(os.path.dirname(torch.__file__)+'/lib')")
export LD_LIBRARY_PATH=$ROOT/cudart_shim:$NVLIB:$TORCHLIB:$LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export R3_STAGE=measure R3_OUT=$ROOT/rung3_out R3_DEVICE=cuda:0 R3_LAYER=${R3_LAYER:-18} R3_CORPUS=${R3_CORPUS:-both}
cd $ROOT/Manifold-SAE/experiments
echo "=== rung3 measure (splice-KL + loss-recovered) ==="
$PY rung3_splice.py
echo "=== measure DONE rc=$? ==="
ls -la $R3_OUT
