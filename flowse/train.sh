#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-config/train.yaml}"
num_gpus="${NUM_GPUS:-1}"
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-29525}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${num_gpus}" \
  --master_addr="${master_addr}" \
  --master_port="${master_port}" \
  train.py \
  -conf "${config_path}"
