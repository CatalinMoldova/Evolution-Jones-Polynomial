#!/bin/bash
#SBATCH --job-name=jones-nas
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=nas_%j.out
#SBATCH --error=nas_%j.err
#
# NSGA-II neural architecture search: Jones polynomial -> hyperbolic volume.
#
# Runs the WHOLE search on a compute node.  Do NOT run this on the login node:
# each candidate trains a PyTorch MLP, and torch defaults to one thread per
# core -- on a login node that exhausts the per-user process limit.  The env
# vars below pin every math library to a single thread so the parallelism comes
# purely from the pool of worker processes (one candidate per CPU).
#
# Submit:   sbatch submit_nas.slurm
# Watch:    squeue -u $USER ;  tail -f nas_<jobid>.out
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="$HOME/.conda/envs/nsga2/bin/python"
WORKDIR="/scratch/cb5330/Jones_polynomial"
cd "$WORKDIR"

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  started=$(date)"

srun "$PY" run_nas.py \
    --data "$WORKDIR/jones+volume.txt" \
    --outdir "$WORKDIR/results" \
    --pop 40 --gens 40 \
    --epochs 80 --subsample 0 --patience 15 \
    --search-seeds 4 \
    --log-target \
    --gap-constraint 0.01 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}" \
    --device cpu \
    --final-epochs 500 --final-seeds 7 --final-patience 50

echo "finished=$(date)"
