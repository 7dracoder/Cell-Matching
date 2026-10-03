#!/bin/bash
#SBATCH --job-name=cellmatch-setup
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
# Installs on a compute node: Torch login nodes cap memory at 2 GB and run a different OS image.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  curl -fsSL -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash miniforge.sh -b -p "$CONDA_DIR"
  rm miniforge.sh
fi
if [ ! -x "$PYTHON" ]; then
  "$CONDA_DIR/bin/conda" create -y -p "$ENV_DIR" python=3.12
fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install "cellpose>=4.2" tifffile scikit-image scipy pandas opencv-python-headless

"$PYTHON" - <<'EOF'
import zipfile
from pathlib import Path
if not Path("Project_2_Dataset/training/train_ground_truth.csv").exists():
    zipfile.ZipFile("testingpj-2.zip").extractall(".")
print("dataset:", sorted(p.name for p in Path("Project_2_Dataset/training").glob("subject_*")),
      len(list(Path("Project_2_Dataset/hidden_test").glob("*/*"))), "test regions")
from cellpose import models
print("cellpose weights:", models.cache_model_path("cpsam_v2"))
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
EOF
echo "setup done"
