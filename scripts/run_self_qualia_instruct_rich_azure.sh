#!/usr/bin/env bash
# Run the rich OLMo self/qualia experiment on an Azure GPU VM.
#
# Intended location: repository root on the Azure VM.
# Example:
#   bash scripts/run_self_qualia_instruct_rich_azure.sh

set -euo pipefail

ROOT="$(pwd)"
MODEL="allenai/Olmo-3-7B-Instruct"
REVISION="main"
HF_HOME="$ROOT/.scratch/hf"
UV_CACHE_DIR="$ROOT/.scratch/uvcache"
TMPDIR="$ROOT/.scratch/tmp"
LAST_OUT="$ROOT/runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST"
MEAN_OUT="$ROOT/runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_MEAN"
LOG="$ROOT/runs/OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_RUN.log"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR
export TMPDIR
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$(dirname "$LOG")" "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd "$ROOT"

{
  echo "### HOST $(hostname) $(date)"
  echo "### GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "### DISK"
  df -h "$HF_HOME" "$ROOT" /
  echo "### install uv if needed"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv --version
  echo "### sync"
  uv sync --extra llm
  echo "### versions"
  uv run python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
  echo "### prompt bank"
  uv run python - <<'PY'
from collections import Counter
from experiments.self_qualia_olmo import build_prompt_bank, CARRIERS
items = build_prompt_bank()
print("carriers", len(CARRIERS))
print("prompts", len(items))
print("items", len(set(x.item_id for x in items)))
print("roles", Counter(x.role for x in items))
print("qualia_pairs", len(set(x.pair_id for x in items if x.pair_id)))
PY

  rm -rf "$LAST_OUT" "$MEAN_OUT"

  echo "### run last_token"
  uv run python -m experiments.self_qualia_olmo \
    --model "$MODEL" \
    --revision "$REVISION" \
    --out-dir "$LAST_OUT" \
    --device cuda \
    --dtype bfloat16 \
    --batch-size 4 \
    --pooling last_token \
    --analysis-layer-percent 0.70
  echo "### plot last_token"
  uv run python -m experiments.plot_self_qualia_olmo --run-dir "$LAST_OUT"

  echo "### run mean_pool"
  uv run python -m experiments.self_qualia_olmo \
    --model "$MODEL" \
    --revision "$REVISION" \
    --out-dir "$MEAN_OUT" \
    --device cuda \
    --dtype bfloat16 \
    --batch-size 4 \
    --pooling mean_pool \
    --analysis-layer-percent 0.70
  echo "### plot mean_pool"
  uv run python -m experiments.plot_self_qualia_olmo --run-dir "$MEAN_OUT"

  echo "### summaries"
  uv run python - <<'PY'
import json
from pathlib import Path
for name in ["OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_LAST", "OLMO3_7B_INSTRUCT_SELF_QUALIA_RICH_MEAN"]:
    p = Path("runs") / name
    s = json.load(open(p / "summary.json"))
    m = json.load(open(p / "run_meta.json"))
    b = s["best_layer_metrics"]
    print(name, m["pooling"], "shape", (s["n_prompts"], s["n_layers"], s["hidden_dim"]))
    print("  best_layer", s["best_layer"], "kind_auc", b["kind_auc"], "qualia_auc", b["qualia_auc"],
          "axis_cos", b["axis_cosine_kind_qualia"])
    print("  self", (b["self_kind_coord"], b["self_qualia_coord"]))
    print("  human", (b["human_author_kind_coord"], b["human_author_qualia_coord"]))
    print("  ai", (b["ai_author_kind_coord"], b["ai_author_qualia_coord"]))
PY

  echo "### artifacts"
  find "$LAST_OUT" "$MEAN_OUT" -maxdepth 2 -type f -printf "%p %s bytes\n" | sort
  echo "ALLDONE $(date)"
} >"$LOG" 2>&1
