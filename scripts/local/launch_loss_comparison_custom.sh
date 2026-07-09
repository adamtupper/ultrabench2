#!/bin/bash

set -e

# Comparison run using the custom DiceCELoss implementation.
# Pair with launch_loss_comparison_monai.sh to validate equivalence.

export HF_HOME=/raid/aradhachandran/thyroid/code/UltraBench2o/hf_data
export CUDA_VISIBLE_DEVICES=2

MODEL="urfm"
HEAD="upsampler"
MODE="head_only"
TASK="psfhs_segmentation"
SEED=42

OUTPUT_DIR="/raid/aradhachandran/thyroid/code/UltraBench2o/logs/loss_comparison/custom"

IMAGE_SIZE=256
BATCH_SIZE=32
ACCUM_GRAD=1
MAX_STEPS=10000
VAL_CHECK_INTERVAL=100
PATIENCE=null
DATA_LOADER_WORKERS=8

python main_launch.py -m \
    system.output_dir=${OUTPUT_DIR} \
    system.data_loader_workers=${DATA_LOADER_WORKERS} \
    model=${MODEL} \
    head=${HEAD} \
    task=${TASK} \
    mode=${MODE} \
    data.image_size=${IMAGE_SIZE} \
    optimization.seed=${SEED} \
    optimization.max_steps=${MAX_STEPS} \
    optimization.val_check_interval=${VAL_CHECK_INTERVAL} \
    optimization.patience=${PATIENCE} \
    optimization.batch_size=${BATCH_SIZE} \
    optimization.accum_grad=${ACCUM_GRAD}

echo "Custom DiceCELoss run completed!"
