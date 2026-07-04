#!/usr/bin/env bash
#SBATCH --job-name=rung3_harvest
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:00:00
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
export R3_STAGE=harvest R3_OUT=$ROOT/rung3_out R3_DEVICE=cuda:0
export R3_CORPUS=${R3_CORPUS:-both} R3_LAYER=${R3_LAYER:-18} R3_MAXTOK=${R3_MAXTOK:-64}
mkdir -p $R3_OUT
cd $ROOT/Manifold-SAE/experiments
echo "=== rung3 harvest layer $R3_LAYER corpus $R3_CORPUS ==="
$PY rung3_splice.py
echo "=== harvest DONE rc=$? ==="
ls -la $R3_OUT
