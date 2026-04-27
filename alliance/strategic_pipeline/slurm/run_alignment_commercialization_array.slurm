#!/bin/bash
# Regenerate per-firm alignment_commercialization.md reports under the
# persistence re-ranker (Hankel-DMD persistence-vs-acquisition asymmetry).
# Faster than --question all because we skip timing + stress.
#
# Submit in two batches to stay under the QOS submit cap:
#   sbatch --array=1-4000%50 strategic_pipeline/slurm/run_alignment_commercialization_array.slurm
#   sbatch --array=4001-7626%50 strategic_pipeline/slurm/run_alignment_commercialization_array.slurm
#
#SBATCH --job-name=align_comm_arr
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=outputs/strategic/logs/align_%A_%a.out
#SBATCH --error=outputs/strategic/logs/align_%A_%a.err

set -e

cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
mkdir -p outputs/strategic/logs

module purge
module load miniforge/24.11.3-py3.12

: "${YEAR:=2017}"

python -m strategic_pipeline.run_firm \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --year "$YEAR" \
    --question alignment \
    --goal commercialization
