#!/bin/bash
# Retry pass for firms that failed in the main array (typically
# ArrowMemoryError during load_all under concurrent node contention).
# Doubled memory (--mem=32G), lower concurrency (25), reads from
# intermediate/retry_firm_list.csv.
#
# Submit:
#   sbatch --array=1-481%25 strategic_pipeline/slurm/retry_failed.slurm
#
#SBATCH --job-name=alliance_retry
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=outputs/strategic/logs/retry_%A_%a.out
#SBATCH --error=outputs/strategic/logs/retry_%A_%a.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

# Look up CUSIP from the retry list (SLURM_ARRAY_TASK_ID is 1-based)
RETRY_LIST=/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance/intermediate/retry_firm_list.csv
CUSIP=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR == n {print $1}' "$RETRY_LIST")

if [[ -z "$CUSIP" ]]; then
    echo "ERROR: no CUSIP for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

: "${YEAR:=2017}"
: "${QUESTION:=all}"

echo "[$(date)] Retry: cusip=$CUSIP year=$YEAR question=$QUESTION"
python -m strategic_pipeline.run_firm \
    --cusip "$CUSIP" \
    --year "$YEAR" \
    --question "$QUESTION"
echo "[$(date)] done"
