#!/usr/bin/env bash
# Harvest the self/qualia prompt bank ACROSS OLMo-3-32B base-training checkpoints
# (last token, all 64 layers) + steering/cloze probes, on the Azure A100 VM.
#
# Thin wrapper around the resilient/resumable/pipelined Python driver
# (experiments/run_self_qualia_trajectory.py): per-checkpoint isolated HF cache,
# atomic done.json markers (re-run to resume), and NEXT-checkpoint prefetch
# overlapped with the current checkpoint's steering+cloze compute. At most two
# 64GB checkpoints live on the NVMe at once.
#
# No `set -e`: the Python driver already catches per-checkpoint failures and
# continues the sweep.
cd /home/azuser/Manifold-SAE || exit 9
export HF_HOME=/mnt/nvme/hf TMPDIR=/mnt/nvme/tmp
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /mnt/nvme/hf /mnt/nvme/tmp /mnt/nvme/hf_traj runs/OLMO3_32B_TRAJ
PY="$PWD/.venv/bin/python"

"$PY" -m experiments.run_self_qualia_trajectory \
  --model allenai/Olmo-3-1125-32B \
  --prompts-file experiments/self_qualia_prompts.jsonl \
  --out-parent runs/OLMO3_32B_TRAJ \
  --cache-root /mnt/nvme/hf_traj \
  --device cuda --dtype bfloat16 --batch-size 16 \
  --steer-layer-percent 0.40

# After the sweep, build the trajectory CSV + plot across all checkpoints:
"$PY" experiments/analyze_self_qualia_bank.py runs/OLMO3_32B_TRAJ --trajectory \
  --analysis-layer-percent 0.40
