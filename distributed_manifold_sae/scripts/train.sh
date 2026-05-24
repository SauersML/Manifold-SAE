#!/usr/bin/env bash
# Launch distributed Manifold-SAE training on 4 GPUs.
set -euo pipefail

CONFIG="${1:-distributed_manifold_sae/configs/k1m_circle_cogito.yaml}"
NPROC="${NPROC:-4}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/k1m_circle_cogito}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m distributed_manifold_sae.train \
    --config "${CONFIG}" \
    --ckpt-dir "${CKPT_DIR}" \
    --epochs 10
