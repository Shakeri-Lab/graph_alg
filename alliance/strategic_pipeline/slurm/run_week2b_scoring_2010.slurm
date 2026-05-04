#!/bin/bash
# Week 2B — historical scoring at t=2010 for the out-of-time backtest
# of 2011 realized L2 ties (no-leakage version: rankings produced from
# data window [2006, 2010] only).
#
# Prerequisite:
#   python -m strategic_pipeline.aggregate.week2_personalization precompute --year 2010
#
# Submit in two batches under the QOS submit cap:
#   sbatch --array=1-4000%50 strategic_pipeline/slurm/run_week2b_scoring_2010.slurm
#   sbatch --array=4001-7626%50 strategic_pipeline/slurm/run_week2b_scoring_2010.slurm
#
# Per-focal parquets land in
# outputs/strategic/aggregate/week2_personalization_2010/<cusip>.parquet
#
#SBATCH --job-name=w2b_score10
#SBATCH --account=cdt_computing
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/strategic/logs/w2b10_%A_%a.out
#SBATCH --error=outputs/strategic/logs/w2b10_%A_%a.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

python -m strategic_pipeline.aggregate.week2_personalization score \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --year 2010
