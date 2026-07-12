#!/bin/bash
#SBATCH --job-name=amm-all-zoo
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --time=24:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/amm_zoo/amm-all-zoo-%j.out

set -euo pipefail

ROOT=/projects/standard/hsiehph/sauer354
GAM_SOURCE="$ROOT/gam"
EXPERIMENT="$ROOT/Manifold-SAE/experiments/amm_zoo"
OUT="$ROOT/amm_zoo/all_topologies"

source "$ROOT/gamfit_current_manifest.sh"
SOURCE_COMMIT=$(cd "$GAM_SOURCE" && git rev-parse HEAD)
if [[ "$GAMFIT_MSI_COMMIT" != "$SOURCE_COMMIT" ]]; then
  echo "manifest wheel $GAMFIT_MSI_COMMIT != newest source $SOURCE_COMMIT" >&2
  exit 87
fi

"$GAMFIT_MSI_PYTHON" - <<'PY'
import gamfit
import torch
print("warm dependency check")
print("gamfit", gamfit.__version__, gamfit.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("AMM full run requires its requested CUDA allocation")
if not hasattr(gamfit, "sae_manifold_fit"):
    raise SystemExit("current gamfit wheel has no sae_manifold_fit")
PY

mkdir -p "$OUT"
cd "$EXPERIMENT"
"$GAMFIT_MSI_PYTHON" run.py \
  --full \
  --profile \
  --out "$OUT" \
  --scratch "${SLURM_TMPDIR}/amm_zoo"
