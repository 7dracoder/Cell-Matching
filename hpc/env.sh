# Sourced by every job. Everything lives inside the project directory on /scratch
# (home has a 30,000-file limit on Torch).
export WORK_DIR="$SLURM_SUBMIT_DIR"
export CONDA_DIR="$WORK_DIR/miniforge"
export ENV_DIR="$WORK_DIR/env"
export CELLPOSE_LOCAL_MODELS_PATH="$WORK_DIR/cellpose_weights"
export PIP_CACHE_DIR="$WORK_DIR/.pip_cache"
export PYTHONUNBUFFERED=1
export PYTHON="$ENV_DIR/bin/python"
