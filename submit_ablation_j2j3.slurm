#!/bin/bash
#SBATCH --job-name=jones-abl-j2j3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=abl_%j.out
#SBATCH --error=abl_%j.err
#
# Random-feature control at the fixed j2j3 knee (README §3 step 4).
# 3 variants x 7 seeds x 500 epochs on a [27,27] MLP -- minutes.
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY="$HOME/.conda/envs/nsga2/bin/python"
cd /scratch/cb5330/Jones_polynomial

echo "host=$(hostname)  started=$(date)"
srun "$PY" control_ablation_j2j3.py
echo "finished=$(date)"
