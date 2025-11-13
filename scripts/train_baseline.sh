#!/bin/bash
set -euo pipefail

# ======== User Inputs ==========
NUM_WORKER=${1:-1}
NODE_ID=${2:-0}
NUM_TRAINERS=${3:-2}
CONFIG_FILE=${4:-"experiments/hcstvg2.yaml"}
CUDA_VISIBLE_DEVICES=${5:-"0,3"}
# ===============================

# Make sure operate in the correct directory
cd /home/arthur/stvg_petl/TA-STVG || exit 1
# export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# Limit OpenMP threads for determinism/performance
# export OMP_NUM_THREADS=1

echo "==========================================="
echo " Running Distributed Test "
echo "-------------------------------------------"
echo " NUM_WORKER:      $NUM_WORKER"
echo " NODE_ID:         $NODE_ID"
echo " NUM_TRAINERS:      $NUM_TRAINERS"
echo " CONFIG_FILE:     $CONFIG_FILE"
echo " CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# distributed training launch
# https://docs.pytorch.org/docs/stable/elastic/run.html
# torchrun = python3 -m torch.distributed.run
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  torchrun \
  --standalone \
  --nnodes=$NUM_WORKER \
  --nproc-per-node=$NUM_TRAINERS \
  scripts/train_net.py \
  --config-file $CONFIG_FILE \

# Single GPU training (for debugging)
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
# python3 scripts/train_net.py \
#   --config-file $CONFIG_FILE \
#   OUTPUT_DIR $OUTPUT_DIR \
#   TENSORBOARD_DIR $OUTPUT_DIR