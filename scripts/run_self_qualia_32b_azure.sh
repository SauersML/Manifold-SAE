#!/usr/bin/env bash
# Run the FULL self/qualia experiment on OLMo-3-32B (base + instruct) on an
# Azure A100-80GB VM. 32B is ~64GB in bf16 -> fits one 80GB A100 for inference
# at small batch. Harvests the full 760-prompt bank for BOTH models and BOTH
# poolings, so base and instruct are finally matched on the rich bank.
#
# Run from the repo root on the VM:
#   bash scripts/run_self_qualia_32b_azure.sh
set -euo pipefail

ROOT="$(pwd)"
BASE_MODEL="allenai/Olmo-3-1125-32B"
INSTRUCT_MODEL="allenai/Olmo-3.1-32B-Instruct"
HF_HOME="$ROOT/.scratch/hf"
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$ROOT/.scratch/uvcache" TMPDIR="$ROOT/.scratch/tmp"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
LOG="$ROOT/runs/OLMO3_32B_SELF_QUALIA_RUN.log"
mkdir -p "$(dirname "$LOG")" "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

harvest () {  # $1=model  $2=tag(base|instruct)  $3=pooling(last_token|mean_pool)
  local model="$1" tag="$2" pool="$3"
  local out="$ROOT/runs/OLMO3_32B_${tag^^}_SELF_QUALIA_${pool^^}"
  echo "### harvest $tag $pool -> $out $(date)"
  rm -rf "$out"
  uv run python -m experiments.self_qualia_olmo \
    --model "$model" --revision main --out-dir "$out" \
    --device cuda --dtype bfloat16 --batch-size 2 \
    --pooling "$pool" --analysis-layer-percent 0.70
}

{
  echo "### HOST $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
  echo "### sync"; uv sync --extra llm; uv pip install hf_transfer >/dev/null 2>&1 || true
  uv run python -c "import torch;print('cuda',torch.cuda.is_available())"
  harvest "$BASE_MODEL"     base     last_token
  harvest "$BASE_MODEL"     base     mean_pool
  harvest "$INSTRUCT_MODEL" instruct last_token
  harvest "$INSTRUCT_MODEL" instruct mean_pool
  echo "### summaries"
  for d in runs/OLMO3_32B_*_SELF_QUALIA_*; do
    [ -f "$d/summary.json" ] && uv run python -c "import json,sys;s=json.load(open('$d/summary.json'));b=s['best_layer_metrics'];print('$d',(s['n_prompts'],s['n_layers'],s['hidden_dim']),'bestL',s['best_layer'],'kAUC',round(b['kind_auc'],3),'qAUC',round(b['qualia_auc'],3),'self',(round(b['self_kind_coord'],2),round(b['self_qualia_coord'],2)))"
  done
  echo "ALLDONE $(date)"
} >"$LOG" 2>&1
