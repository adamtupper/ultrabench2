#!/bin/bash
#
# Submit the run scripts that still have outstanding runs, one sbatch per
# script. Each script enqueues its own combos and chains successor jobs until
# its queue is empty, so this only has to be run once.
#
# The selection below comes from the 2026-09-06 audit of
# /project/ultrabench2/20260906_results/runs: 778 of the 3360 runs in the
# 16 models x 21 tasks x 2 methods x 5 seeds grid had no test_metrics.csv.
# Scripts whose grid is already complete are listed but commented out, so the
# file stays a record of what was skipped and why.
#
# Every script's combos are a superset of what is outstanding for that model.
# The overlap is small (9 runs for urfm_large, see below) and a redundant run
# just adds another timestamped directory next to the existing one, which the
# "keep the latest" rule resolves.
#
# The combos are the only thing a queue reads at creation time, so a stale
# ledger from an earlier sweep would override the selection below. Clear any
# leftovers first: rm -rf $scratch/results/ultrabench/_queue/<experiment>
#
# Usage: scripts/slurm/launch_jobs.sh          # submit
#        scripts/slurm/launch_jobs.sh --dry-run
#
# Submit from the repository root: the --output paths in the run scripts are
# relative to the submission directory.

set -e

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Classification, seeds 45 and 46 outstanding for the whole grid (44 runs each).
SCRIPTS=(
    run_classification_deblurmim_convmae.sh
    run_classification_deblurmim_mae.sh
    run_classification_dinov3_convnext_base.sh
    run_classification_dinov3_convnext_large.sh
    run_classification_dinov3_vitb16.sh
    run_classification_dinov3_vitl16.sh
    run_classification_echocare.sh
    run_classification_openus.sh
    run_classification_ultrafedfm.sh
    run_classification_ultrasam.sh
    run_classification_urfm_base.sh
    run_classification_usfm.sh

    # Same, plus the three camus_classification head_only runs missing from
    # seeds 42-44, which the script appends to its combos (47 runs).
    run_classification_samus.sh

    # The mirror image: seeds 45 and 46 are done, seeds 42-44 are not. The
    # script now enqueues 42-44, which is exactly what is missing for usfmae
    # (66 runs) and 9 runs more than is missing for urfm_large (57 of 66).
    run_classification_usfmae.sh
    run_classification_urfm_large.sh

    # Segmentation, seeds 45 and 46 outstanding for the whole grid (40 each).
    # Every other segmentation model already has all five seeds.
    run_segmentation_echocare.sh
    run_segmentation_samus.sh

    # Complete as of the audit, nothing to submit:
    #   run_classification_tinyusfm.sh
    #   run_segmentation_deblurmim_convmae.sh
    #   run_segmentation_deblurmim_mae.sh
    #   run_segmentation_dinov3_convnext_base.sh
    #   run_segmentation_dinov3_convnext_large.sh
    #   run_segmentation_dinov3_vitb16.sh
    #   run_segmentation_dinov3_vitl16.sh
    #   run_segmentation_openus.sh
    #   run_segmentation_tinyusfm.sh
    #   run_segmentation_ultrafedfm.sh
    #   run_segmentation_ultrasam.sh
    #   run_segmentation_urfm_base.sh
    #   run_segmentation_urfm_large.sh
    #   run_segmentation_usfm.sh
    #   run_segmentation_usfmae.sh
)

for script in "${SCRIPTS[@]}"; do
    if [[ ! -f "$HERE/$script" ]]; then
        echo "[launch_jobs] no such script: $HERE/$script" >&2
        exit 1
    fi
    if (( DRY_RUN )); then
        echo "sbatch $HERE/$script"
    else
        sbatch "$HERE/$script"
    fi
done

echo "[launch_jobs] ${#SCRIPTS[@]} script(s) $( (( DRY_RUN )) && echo listed || echo submitted )"
echo "[launch_jobs] progress: scripts/slurm/queue_status.sh <experiment>, e.g. classification_samus"
