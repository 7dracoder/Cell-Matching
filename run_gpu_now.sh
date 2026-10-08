#!/bin/bash
set -euo pipefail
cd /scratch/ts5789/cellmatch
mkdir -p logs
ACCOUNT=cs_gy_6923-2026fa
git pull --ff-only origin main || true
if [ ! -x env/bin/python ]; then
  setup=$(sbatch --parsable -A "$ACCOUNT" --export=NONE -p n2c48m24 --mem=16G hpc/setup.sbatch)
  echo setup=$setup
  dep="--dependency=afterok:$setup"
else
  dep=
fi
gpu=$(sbatch --parsable -A "$ACCOUNT" --export=NONE -p g2-standard-12 --mem=40G --gres=gpu:1 $dep hpc/gpu.sbatch)
cpu=$(sbatch --parsable -A "$ACCOUNT" --export=NONE -p n2c48m24 --mem=20G --dependency=afterok:$gpu hpc/cpu.sbatch)
echo submitted: gpu=$gpu cpu=$cpu
squeue -u "$USER"
