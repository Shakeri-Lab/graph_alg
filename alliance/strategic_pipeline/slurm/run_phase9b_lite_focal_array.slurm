#!/bin/bash
# Phase 9B-lite — score each (focal, score_year) pair in the manifest.
#
# Prerequisite (run once locally):
#   python -m strategic_pipeline.aggregate.phase9b_lite_score_realized_focals build-manifest
#
# Then submit (size N to manifest length):
#   N=$(($(wc -l < intermediate/phase9b_lite_focals_to_score.csv) - 1))
#   sbatch --array=1-${N}%50 strategic_pipeline/slurm/run_phase9b_lite_focal_array.slurm
#
#SBATCH --job-name=p9b_lite_score
#SBATCH --account=cdt_computing
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/strategic/logs/p9b_%A_%a.out
#SBATCH --error=outputs/strategic/logs/p9b_%A_%a.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

python -m strategic_pipeline.aggregate.phase9b_lite_score_realized_focals \
    score-task \
    --array-index "$SLURM_ARRAY_TASK_ID"
