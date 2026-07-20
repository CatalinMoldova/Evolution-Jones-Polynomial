#!/bin/bash
#SBATCH --job-name=jones-j2j3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=nas_%j.out
#SBATCH --error=nas_%j.err
#
# NSGA-II NAS on the colored-Jones dataset (data_j2+j3, 1,419 knots).
# Same matched budget as the J2 phase run (README 1.2/1.3): pop 40 / gens 40 /
# full data / gap 0.01 / 4 search seeds / 7 final seeds / 500 final epochs.
#
# Submit one run per feature set (fresh outdir per input width -- the score
# cache is keyed by architecture only):
#   sbatch submit_nas_j2j3.slurm j2      # -> results_j2j3_j2/
#   sbatch submit_nas_j2j3.slurm j3      # -> results_j2j3_j3/
#   sbatch submit_nas_j2j3.slurm j2j3    # -> results_j2j3_j2j3/
set -euo pipefail

FS="${1:?usage: sbatch submit_nas_j2j3.slurm <j2|j3|j2j3>}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="$HOME/.conda/envs/nsga2/bin/python"
WORKDIR="/scratch/cb5330/Jones_polynomial"
cd "$WORKDIR"

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  feature_set=$FS  started=$(date)"

srun "$PY" run_nas.py \
    --feature-set "$FS" \
    --outdir "$WORKDIR/results_j2j3_$FS" \
    --pop 40 --gens 40 \
    --epochs 80 --subsample 0 --patience 15 \
    --search-seeds 4 \
    --log-target \
    --gap-constraint 0.01 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}" \
    --device cpu \
    --final-epochs 500 --final-seeds 7 --final-patience 50

echo "finished=$(date)"
