#!/bin/bash
set -euo pipefail

# ======== User Inputs ==========
N_GPUS=${1:-2}
CONFIG_FILE=${2:-"experiments/hcstvg2.yaml"}
MODEL_WEIGHT=${3:-"checkpoints/TASTVG_HCSTVG2.pth"}
OUTPUT_DIR=${4:-"output/hcstvg2"}
CUDA_VISIBLE_DEVICES=${5:-"1,3"}
# ===============================

# Make sure operate in the correct directory
cd /home/arthur/stvg_petl/TA-STVG || exit 1
# export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# Limit OpenMP threads for determinism/performance
export OMP_NUM_THREADS=1

echo "==========================================="
echo " Running Distributed Test "
echo "-------------------------------------------"
echo " Number of GPUs       : $N_GPUS"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Config File          : $CONFIG_FILE"
echo " Model Weights        : $MODEL_WEIGHT"
echo " Output Directory     : $OUTPUT_DIR"
echo "==========================================="

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  python3 -m torch.distributed.run \
  --nproc_per_node="$N_GPUS" \
  scripts/test_net.py \
  --config-file "$CONFIG_FILE" \
  MODEL.WEIGHT "$MODEL_WEIGHT" \
  OUTPUT_DIR "$OUTPUT_DIR"
