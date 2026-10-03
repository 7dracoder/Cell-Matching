#!/bin/bash
#SBATCH --job-name=cellmatch-gpu
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --comment="gpu_mps=yes"
#SBATCH --output=logs/%x-%j.out
# All training and every prediction. Resubmitting resumes: finished models and caches are skipped.
# Class cloud-burst partition: 1× L4. Preemptible — caches on disk so resubmits resume.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" run_hpc.py gpu
