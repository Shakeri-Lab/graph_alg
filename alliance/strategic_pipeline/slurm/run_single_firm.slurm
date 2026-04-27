#!/bin/bash
# Single-firm strategic pipeline run.
# Usage:
#   sbatch --export=ALL,CUSIP=747525,YEAR=2005 strategic_pipeline/slurm/run_single_firm.slurm
#
#SBATCH --job-name=alliance_strat
#SBATCH --account=cdt_computing
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/strategic/logs/single_%j.out
#SBATCH --error=outputs/strategic/logs/single_%j.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

: "${CUSIP:?CUSIP must be set (e.g. --export=ALL,CUSIP=747525)}"
: "${YEAR:=2017}"
: "${QUESTION:=all}"

echo "[$(date)] Strategic pipeline: cusip=$CUSIP year=$YEAR question=$QUESTION"
python -m strategic_pipeline.run_firm \
    --cusip "$CUSIP" \
    --year "$YEAR" \
    --question "$QUESTION"
echo "[$(date)] done"
