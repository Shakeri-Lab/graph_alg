#!/bin/bash
# Week 2A — score every Compustat-matched focal firm against its full
# L2 candidate pool, dumping one parquet per focal for the
# personalization / null-baseline diagnostic.
#
# Prerequisite: run the candidate-feature pre-compute once first:
#   python -m strategic_pipeline.aggregate.week2_personalization precompute --year 2017
#
# Submit in two batches under the QOS submit cap:
#   sbatch --array=1-4000%50 strategic_pipeline/slurm/run_week2_personalization_array.slurm
#   sbatch --array=4001-7626%50 strategic_pipeline/slurm/run_week2_personalization_array.slurm
#
#SBATCH --job-name=w2_perso
#SBATCH --account=cdt_computing
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/strategic/logs/w2_%A_%a.out
#SBATCH --error=outputs/strategic/logs/w2_%A_%a.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

: "${YEAR:=2017}"

python -m strategic_pipeline.aggregate.week2_personalization score \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --year "$YEAR"
