#!/bin/bash
#SBATCH --job-name=jones-nas-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err
#
# Fast end-to-end sanity run on a compute node (~a couple of minutes).
# Validates parse -> search -> Pareto -> knee -> final retrain -> test MRE
# before committing to the full submit_nas.slurm job.
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="$HOME/.conda/envs/nsga2/bin/python"
WORKDIR="/scratch/cb5330/Jones_polynomial"
cd "$WORKDIR"

echo "host=$(hostname)  started=$(date)"
srun "$PY" run_nas.py \
    --data "$WORKDIR/jones+volume.txt" \
    --outdir "$WORKDIR/results_smoke" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
    --smoke
echo "finished=$(date)"
