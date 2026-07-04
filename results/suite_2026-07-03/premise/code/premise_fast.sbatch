#!/usr/bin/env bash
#SBATCH --job-name=premise_fast
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
set -uo pipefail
ROOT=/projects/standard/hsiehph/sauer354
export ROOT
export PREMISE_OUT=$ROOT/premise_out
mkdir -p $PREMISE_OUT
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
# fast, bounded fit path: structure + outer-rho search OFF (K=1 fixed), tight per-fit alarm.
export PREMISE_NITER=${PREMISE_NITER:-30}
export PREMISE_FIT_ALARM=${PREMISE_FIT_ALARM:-150}
export PREMISE_RDIM=${PREMISE_RDIM:-8}
PY=$ROOT/venv_head_atlas/bin/python
cd $ROOT/premise_run
echo "=== EXP1 premise_deviance (all 7, uniform fast protocol) ==="
timeout 4000 $PY premise_deviance.py || echo "deviance rc=$?"
echo "=== EXP2 slow_feature_atlas ==="
timeout 1500 $PY slow_feature_atlas.py || echo "atlas rc=$?"
echo "=== DONE rc=$? ==="
ls -la $PREMISE_OUT
