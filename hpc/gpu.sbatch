#!/bin/bash
#SBATCH --job-name=cellmatch-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
# All training and every prediction. Resubmitting resumes: finished models and caches are skipped.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" run_hpc.py gpu
