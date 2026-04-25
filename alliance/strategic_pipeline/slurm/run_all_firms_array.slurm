#!/bin/bash
# Slurm array job over all ~7,626 Compustat-matched CUSIPs.
# The CUSIP list is auto-built on first run and stored at:
#   intermediate/compustat_firm_list.csv
#
# Before submission, create the CUSIP list once:
#   python -m strategic_pipeline.run_firm --array-index 1 --question all
# This writes the list as a side-effect if it doesn't exist.
#
# Then submit (throttles at 50 concurrent, 7626 total):
#   sbatch strategic_pipeline/slurm/run_all_firms_array.slurm
#
#SBATCH --job-name=alliance_strat_arr
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --array=1-7626%50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/strategic/logs/arr_%A_%a.out
#SBATCH --error=outputs/strategic/logs/arr_%A_%a.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

: "${YEAR:=2017}"
: "${QUESTION:=all}"

python -m strategic_pipeline.run_firm \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --year "$YEAR" \
    --question "$QUESTION"
