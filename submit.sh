#!/bin/bash
# Usage: bash submit.sh [slurm_account]
# Default: CS-GY-6923 Fall 2026 Cloud Bursting account (DL course).
# Chains setup -> gpu -> cpu. Rerunning resumes: finished steps are skipped.
# Burst requires --partition on the sbatch CLI (script #SBATCH alone is not enough).
set -euo pipefail
cd "$(dirname "$0")"
account=${1:-cs_gy_6923-2026fa}
mkdir -p logs
# Burst: pass partition/mem on CLI; --export=NONE avoids user_env_retrieval_failed holds.
setup=$(sbatch --parsable --account="$account" --partition=n2c48m24 --mem=16G --export=NONE hpc/setup.sbatch)
gpu=$(sbatch --parsable --account="$account" --partition=g2-standard-12 --mem=40G --export=NONE --dependency=afterok:"$setup" hpc/gpu.sbatch)
cpu=$(sbatch --parsable --account="$account" --partition=n2c48m24 --mem=20G --export=NONE --dependency=afterok:"$gpu" hpc/cpu.sbatch)
echo "submitted: account=$account setup=$setup gpu=$gpu cpu=$cpu"
echo "progress:  squeue -u \$USER    |    tail -f logs/cellmatch-*.out"
echo "result:    $(pwd)/submission.csv"
