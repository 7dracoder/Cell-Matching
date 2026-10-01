#!/bin/bash
#SBATCH --job-name=cellmatch-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Search on cached held-out predictions, then submission.csv. CPU only, so no GPU sits idle.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
"$PYTHON" run_hpc.py cpu
