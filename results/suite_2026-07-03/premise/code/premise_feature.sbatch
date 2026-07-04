#!/usr/bin/env bash
#SBATCH --job-name=premise_feat
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --output=%x_%A.out
set -uo pipefail
ROOT=/projects/standard/hsiehph/sauer354
export ROOT
export PREMISE_OUT=$ROOT/premise_out
mkdir -p $PREMISE_OUT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# struct search OFF (K=1 fixed) but outer-rho/REML smoothness search ON (regularizes the
# circle so it does not overfit held-out rows -> keeps the Gaussian surrogate flat).
export PREMISE_RHO_SEARCH=1
export PREMISE_NITER=${PREMISE_NITER:-40}
export PREMISE_FIT_ALARM=${PREMISE_FIT_ALARM:-600}
export PREMISE_RDIM=${PREMISE_RDIM:-8}
# PREMISE_ONLY is passed via sbatch --export
echo "=== premise feature=$PREMISE_ONLY rho_search=$PREMISE_RHO_SEARCH niter=$PREMISE_NITER ==="
PY=$ROOT/venv_head_atlas/bin/python
cd $ROOT/premise_run
timeout 10000 $PY premise_deviance.py || echo "rc=$?"
echo "=== DONE feature=$PREMISE_ONLY ==="
ls -la $PREMISE_OUT/result_${PREMISE_ONLY}.json 2>/dev/null
