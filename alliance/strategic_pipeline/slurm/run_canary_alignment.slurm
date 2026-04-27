#!/bin/bash
# Canary regeneration of alignment_commercialization for ~30 firms
# spanning R&D × hub-presence × portfolio-density.
#
# Reads cusips from intermediate/canary_cusips.csv (one per line, no
# header).  Submit with the array sized to match the canary CSV:
#   sbatch --array=1-21%10 strategic_pipeline/slurm/run_canary_alignment.slurm
#
#SBATCH --job-name=align_canary
#SBATCH --account=cdt_computing
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/strategic/logs/canary_%A_%a.out
#SBATCH --error=outputs/strategic/logs/canary_%A_%a.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

CUSIP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" intermediate/canary_cusips.csv)
echo "[canary] task ${SLURM_ARRAY_TASK_ID}: cusip ${CUSIP}"

python -m strategic_pipeline.run_firm \
    --cusip "$CUSIP" \
    --year 2017 \
    --question alignment \
    --goal commercialization
