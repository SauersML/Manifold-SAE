#!/usr/bin/env bash
#SBATCH --job-name=premise_cpu
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
ROOT=/projects/standard/hsiehph/sauer354
export ROOT
export PREMISE_OUT=$ROOT/premise_out
mkdir -p $PREMISE_OUT
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY=$ROOT/venv_head_atlas/bin/python
cd $ROOT/premise_run
echo "=== gamfit version ==="
$PY -c "import gamfit;print(getattr(gamfit,'__version__','?'))" || true
echo "=== EXP1 premise_deviance (held-out paired deviance) ==="
timeout 16000 $PY premise_deviance.py || echo "deviance rc=$?"
echo "=== EXP2 slow_feature_atlas (context-mean charts) ==="
timeout 2000 $PY slow_feature_atlas.py || echo "atlas rc=$?"
echo "=== DONE rc=$? ==="
ls -la $PREMISE_OUT
