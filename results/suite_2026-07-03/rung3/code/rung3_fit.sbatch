#!/usr/bin/env bash
#SBATCH --job-name=rung3_fit
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
set -e
ROOT=/projects/standard/hsiehph/sauer354
unset PYTHONPATH
# OMP/OpenBLAS threaded for the numpy SVD + block lane; RAYON=1 so the gamfit curved fit
# does NOT hit the #2074 thread-oversubscription slowdown/deadlock.
# atlas wheel (3cbb04b76) has the #2074 oversubscription fix, so RAYON can run multi-core;
# the earlier "hang" was serial-BLAS SVD grind (now replaced by a Gram-matrix eig).
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=8
PY=$ROOT/venv_head_atlas/bin/python
export R3_STAGE=fit R3_OUT=$ROOT/rung3_out
export R3_K=${R3_K:-128} R3_TOPK=${R3_TOPK:-32} R3_PCA_KS=${R3_PCA_KS:-1,8,64}
export R3_MAXFIT_TOK=${R3_MAXFIT_TOK:-0} R3_CURVED_RDIM=${R3_CURVED_RDIM:-48}
export R3_MAXBIRTHS=${R3_MAXBIRTHS:-16} R3_FITNITER=${R3_FITNITER:-40} R3_BACKFIT=${R3_BACKFIT:-3}
export R3_CURVED_BUDGET=${R3_CURVED_BUDGET:-150}
cd $ROOT/Manifold-SAE/experiments
echo "=== rung3 fit rdim=$R3_RDIM max_births=$R3_MAXBIRTHS (newest gamfit) ==="
$PY -c "import gamfit;print('gamfit',gamfit.__version__)"
$PY rung3_splice.py
echo "=== fit DONE rc=$? ==="
ls -la $R3_OUT
