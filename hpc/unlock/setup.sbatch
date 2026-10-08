#!/bin/bash
#SBATCH --job-name=unlock-setup
#SBATCH --output=logs/unlock-setup-%j.out
# sbatch <account/partition/resource flags> hpc/unlock/setup.sbatch <singularity|venv> [--no-cellpose]
# Setup Stage of the unlock Job_Chain (Req 1.3-1.7). Account, partition, memory, CPUs and
# time come from the sbatch command line (hpc_unlock/chain.py), not from #SBATCH lines.
#   1. module purge, cd to the Project_Folder, input check (every missing path, Req 1.6).
#   2. singularity: setup_overlay.sh inside singularity exec --overlay <overlay>:rw <sif>.
#      venv: reuse env/ if its core pins match, else reuse or create env_unlock/.
#   3. One pip install per pin (PIP_FAILED <package> on failure), torch every run,
#      cellpose + cpsam_v2 weights unless --no-cellpose, research/data/hpc/setup/setup.json.
# Runs alone: it is the only job that mounts the overlay :rw.
set -Eeuo pipefail
# --export=NONE starts a non-login shell, so load the module function if it is missing.
if ! type module >/dev/null 2>&1; then
  for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    if [ -r "$init" ]; then
      set +u
      source "$init"
      set -u
      break
    fi
  done
fi
if type module >/dev/null 2>&1; then
  module purge
else
  echo "MODULE_PURGE_SKIPPED no module command on this node"
fi
cd "$SLURM_SUBMIT_DIR"

usage() {
  echo "usage: sbatch [flags] hpc/unlock/setup.sbatch <singularity|venv> [--no-cellpose]" >&2
  exit 2
}
[ "$#" -ge 1 ] || usage
MODE="$1"
shift
CELLPOSE=1
for arg in "$@"; do
  case "$arg" in
    --no-cellpose) CELLPOSE=0 ;;
    *) usage ;;
  esac
done
case "$MODE" in singularity|venv) ;; *) usage ;; esac

source hpc/unlock/common.sh
WORK="$UNLOCK_WORK"
mkdir -p "$WORK/logs" "$WORK/research/data/hpc/setup" "$UNLOCK_MODELS"
export PIP_CACHE_DIR="$WORK/.pip_cache" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export CELLPOSE_LOCAL_MODELS_PATH="$UNLOCK_MODELS"
# A failed setup must not leave an old setup.json behind (the submit cell checks it).
rm -f "$WORK/research/data/hpc/setup/setup.json"
echo "SETUP_START mode=$MODE cellpose=$CELLPOSE work=$WORK job=${SLURM_JOB_ID:-none}"

# ---------------------------------------------------------------- input check
# The check unpickles the research/data caches, so it needs Python >= 3.8 with numpy.
# Use the first python that can run it; otherwise run it inside the new env after the
# install (still before any computation Stage, which depends on setup with afterok).
CHECK_PY=""
CANDIDATES=()
if [ "$MODE" = venv ]; then
  if [ -r "$UNLOCK_ENV_PATH_FILE" ]; then
    CANDIDATES+=("$(set +u; source "$UNLOCK_ENV_PATH_FILE"; echo "${ENV_DIR:-}")/bin/python")
  fi
  CANDIDATES+=("$WORK/env/bin/python" "$WORK/env_unlock/bin/python")
fi
CANDIDATES+=(python3)
for py in "${CANDIDATES[@]}"; do
  if unlock_python_can_check "$py"; then
    CHECK_PY="$py"
    break
  fi
done
CHECK_IN_ENV=0
if [ -n "$CHECK_PY" ]; then
  unlock_input_check "$CHECK_PY" "$MODE" || unlock_die "input check failed (INPUT_FAILED lines above)"
else
  echo "INPUT_CHECK_DEFERRED no host python with numpy; the check runs in the new env"
  CHECK_IN_ENV=1
fi

# ---------------------------------------------------------------- singularity
if [ "$MODE" = singularity ]; then
  SIF="$(unlock_find_sif)"
  MISSING=0
  if [ ! -r "$UNLOCK_OVERLAY" ]; then
    echo "INPUT_FAILED $UNLOCK_OVERLAY: missing" >&2
    MISSING=1
  fi
  if [ -z "$SIF" ] || [ ! -r "$SIF" ]; then
    echo "INPUT_FAILED $UNLOCK_CONTAINER/*.sif: missing" >&2
    MISSING=1
  fi
  [ "$MISSING" = 0 ] || unlock_die "container files missing (copy them from greene-dtn, notebook Step 1)"
  SING="$(unlock_singularity)"
  INSTALLER="$UNLOCK_CONTAINER/$(unlock_miniconda_installer)"
  unlock_fetch_miniconda "$INSTALLER"
  ARGS=(exec)
  if [ -n "${UNLOCK_BIND:-}" ]; then
    ARGS+=(--bind "$UNLOCK_BIND")
  fi
  ARGS+=(--overlay "$UNLOCK_OVERLAY:rw" "$SIF")
  echo "SIF $(basename "$SIF")"
  "$SING" "${ARGS[@]}" /bin/bash "$UNLOCK_DIR/setup_overlay.sh" \
    "$CELLPOSE" "$INSTALLER" "$(basename "$SIF")" "$CHECK_IN_ENV"
  echo "SETUP_DONE singularity"
  exit 0
fi

# ---------------------------------------------------------------- venv
SLURM_SUBMIT_DIR="$WORK"
set +u
source "$WORK/hpc/env.sh"
set -u
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "MINIFORGE_INSTALL -> $CONDA_DIR"
  curl -fsSL --retry 3 -o "$WORK/miniforge.sh.partial" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  mv "$WORK/miniforge.sh.partial" "$WORK/miniforge.sh"
  bash "$WORK/miniforge.sh" -b -u -p "$CONDA_DIR"
  rm -f "$WORK/miniforge.sh"
fi
# The old pipeline's env/ is reused only if its core pins match; it is never recreated.
if unlock_env_matches "$WORK/env/bin/python"; then
  ENV_DIR="$WORK/env"
  echo "ENV_REUSE $ENV_DIR"
else
  ENV_DIR="$WORK/env_unlock"
  if unlock_env_matches "$ENV_DIR/bin/python"; then
    echo "ENV_REUSE $ENV_DIR"
  else
    echo "ENV_CREATE $ENV_DIR (core pins missing or different)"
    rm -rf "$ENV_DIR"
    "$CONDA_DIR/bin/conda" create -y -p "$ENV_DIR" python=3.12 pip
  fi
fi
unlock_install_pins "$ENV_DIR/bin/python" "$UNLOCK_PINS" "$CELLPOSE" "$UNLOCK_MODELS"
if [ "$CHECK_IN_ENV" = 1 ]; then
  unlock_input_check "$ENV_DIR/bin/python" venv
fi
unlock_write_env_path "$ENV_DIR"
unlock_write_setup_json "$ENV_DIR/bin/python" venv "$ENV_DIR" "" "$CELLPOSE"
echo "SETUP_DONE venv $ENV_DIR"
