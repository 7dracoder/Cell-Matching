#!/bin/bash
# Usage: bash submit.sh <slurm_account>      (list your accounts with: my_slurm_accounts)
# Chains setup -> gpu -> cpu. Rerunning resumes: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
account=${1:?usage: bash submit.sh <slurm_account>   (run my_slurm_accounts to see yours)}
mkdir -p logs
setup=$(sbatch --parsable --account="$account" hpc/setup.sbatch)
gpu=$(sbatch --parsable --account="$account" --dependency=afterok:"$setup" hpc/gpu.sbatch)
cpu=$(sbatch --parsable --account="$account" --dependency=afterok:"$gpu" hpc/cpu.sbatch)
echo "submitted: setup=$setup gpu=$gpu cpu=$cpu"
echo "progress:  squeue -u \$USER    |    tail -f logs/cellmatch-*.out"
echo "result:    $(pwd)/submission.csv"
