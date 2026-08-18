#!/bin/bash
#SBATCH --job-name=classification-gbcu-finetuning-sweep
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=3:00:00
#SBATCH --array=0-23%1
#
# Sweep learning rate, weight decay, and augmentation strategy for
# head-only vs. full fine-tuning on GBCU classification with UltraFedFM, to
# validate the backbone fine-tuning LR fix (capped max LR + linear warmup +
# per-layer decay). Launches individual jobs for each combo.
#
# Usage: sbatch scripts/slurm/run_classification_gbcu_finetuning_sweep.sh

set -e

SEED=42
METHODS=(head_only full_finetuning)
LRS=(1e-3 1e-4)
WEIGHT_DECAYS=(1e-2 5e-2 1e-1)
AUGMENTATIONS=(null general)

declare -a combos
for method in "${METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
        for wd in "${WEIGHT_DECAYS[@]}"; do
            for aug in "${AUGMENTATIONS[@]}"; do
                combos+=("$method $lr $wd $aug")
            done
        done
    done
done

source $project/ultrabench2/scripts/slurm/setup_node.sh

cd $SLURM_TMPDIR

read -r METHOD LR WEIGHT_DECAY AUGMENTATION <<< "${combos[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT_DIR="$project/models/ultrabench"
OUTPUT_DIR="$scratch/results/ultrabench/testing"
MODEL="ultrafedfm"
HEAD="linear_classifier"
TASK="gbcu_classification"
BATCH_SIZE=32
ACCUM_GRAD=1

python ultrabench2/src/ultrabench/training/main_launch.py \
    system.output_dir=${OUTPUT_DIR} \
    system.checkpoint_dir=${CHECKPOINT_DIR} \
    system.save_last_checkpoint=false \
    system.data_loader_workers=${SLURM_CPUS_PER_TASK} \
    system.enable_progress_bar=false \
    model=${MODEL} \
    head=${HEAD} \
    task=${TASK} \
    mode=${METHOD} \
    mode.optimizer.lr=${LR} \
    mode.optimizer.weight_decay=${WEIGHT_DECAY} \
    data.augmentation_strategy=${AUGMENTATION} \
    optimization.seed=${SEED} \
    optimization.patience=null \
    optimization.batch_size=${BATCH_SIZE} \
    optimization.accum_grad=${ACCUM_GRAD}
