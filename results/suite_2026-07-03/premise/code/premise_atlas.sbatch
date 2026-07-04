#!/usr/bin/env bash
#SBATCH --job-name=premise_atlas
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
ROOT=/projects/standard/hsiehph/sauer354
export ROOT
export PREMISE_OUT=$ROOT/premise_out
mkdir -p $PREMISE_OUT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export ATLAS_FIT_ALARM=200
PY=$ROOT/venv_head_atlas/bin/python
cd $ROOT/premise_run
timeout 2500 $PY slow_feature_atlas.py || echo "atlas rc=$?"
echo "=== DONE atlas ==="
ls -la $PREMISE_OUT/atlas_*.json $PREMISE_OUT/slow_feature_atlas.json 2>/dev/null
