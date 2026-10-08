#!/bin/bash
# setup_overlay.sh <cellpose:0|1> <miniconda installer path> <sif name> <input check:0|1>
# Runs INSIDE the container with the overlay mounted read-write, started by setup.sbatch:
#   singularity exec --overlay <overlay>:rw <sif> /bin/bash hpc/unlock/setup_overlay.sh ...
# 1. Pinned Miniconda in /ext3/miniconda3 (if missing), /ext3/env.sh, pip + ipykernel.
# 2. Reuse conda env "unlock" only if every core pin matches (reuse_env), else recreate it.
# 3. One pip install per pin, torch from default PyPI, cellpose + cpsam_v2 weights.
# 4. Input check inside the env if the host python could not run it (check=1).
# 5. research/data/hpc/setup/setup.json (no CUDA check: CPU node).
set -Eeuo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 <cellpose:0|1> <miniconda installer> <sif name> <check:0|1>" >&2
  exit 2
fi
CELLPOSE="$1"
INSTALLER="$2"
SIF_NAME="$3"
CHECK="$4"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [ ! -d /ext3 ] || [ ! -w /ext3 ]; then
  unlock_die "/ext3 is not writable: run inside singularity exec --overlay <overlay>:rw"
fi
CONDA="$UNLOCK_CONDA_ROOT/bin/conda"
# Keep envs and package cache inside the overlay, never in $HOME (file-count limit).
export CONDA_ENVS_DIRS="$UNLOCK_CONDA_ROOT/envs" CONDA_PKGS_DIRS="$UNLOCK_CONDA_ROOT/pkgs"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes PYTHONNOUSERSITE=1
FRESH=0
if [ ! -x "$CONDA" ]; then
  [ -s "$INSTALLER" ] || unlock_die "MISSING $INSTALLER"
  echo "MINICONDA_INSTALL $(basename "$INSTALLER") -> $UNLOCK_CONDA_ROOT"
  bash "$INSTALLER" -b -u -p "$UNLOCK_CONDA_ROOT"
  FRESH=1
fi
for channel in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
  "$CONDA" tos accept --override-channels --channel "$channel" >/dev/null 2>&1 || true
done
cat > /ext3/env.sh.tmp <<'EOF'
#!/bin/bash
# Written by hpc/unlock/setup_overlay.sh. Sourced by run_in_env.sh and the OOD kernel.
unset -f which
export CONDA_ENVS_DIRS=/ext3/miniconda3/envs CONDA_PKGS_DIRS=/ext3/miniconda3/pkgs
export PYTHONNOUSERSITE=1
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate unlock
EOF
mv /ext3/env.sh.tmp /ext3/env.sh
if [ "$FRESH" = 1 ]; then
  "$CONDA" install -y pip ipykernel
fi
PY="$UNLOCK_CONDA_ENV/bin/python"
if unlock_env_matches "$PY"; then
  echo "ENV_REUSE $UNLOCK_CONDA_ENV"
else
  echo "ENV_CREATE $UNLOCK_CONDA_ENV (core pins missing or different)"
  if [ -d "$UNLOCK_CONDA_ENV" ]; then
    "$CONDA" remove -y -n unlock --all
    rm -rf "$UNLOCK_CONDA_ENV"
  fi
  "$CONDA" create -y -n unlock python=3.12 pip ipykernel
fi
unlock_install_pins "$PY" "$UNLOCK_PINS" "$CELLPOSE" "$UNLOCK_MODELS"
if [ "$CHECK" = 1 ]; then
  unlock_input_check "$PY" singularity
fi
unlock_write_setup_json "$PY" singularity "$UNLOCK_CONDA_ENV" "$SIF_NAME" "$CELLPOSE"
echo "OVERLAY_ENV_READY $UNLOCK_CONDA_ENV"
