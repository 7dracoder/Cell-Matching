#!/bin/bash
#SBATCH --job-name=unlock-stage
#SBATCH --output=logs/unlock-stage-%j.out
# sbatch <account/partition/resource flags> hpc/unlock/stage.sbatch <singularity|venv> <stage> --run <fp> [args...]
# One CPU Stage of the unlock Job_Chain (Req 3.1, 3.3). Account, partition, memory, CPUs
# and time come from the sbatch command line (hpc_unlock/chain.py), not from #SBATCH lines.
# Runs `python run_unlock.py <stage> --run <fp> [args...]` through run_in_env.sh with
# gpu=0 (no --nv). The job's exit code is run_unlock.py's, so afterok dependencies hold.
# Thread counts are left to the Stage (it sets them per worker pool).
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
  echo "usage: sbatch [flags] hpc/unlock/stage.sbatch <singularity|venv> <stage> --run <fp> [args...]" >&2
  exit 2
}
[ "$#" -ge 4 ] && [ "$3" = "--run" ] || usage
MODE="$1"
shift
export PYTHONUNBUFFERED=1

echo "STAGE_START stage=$1 mode=$MODE job=${SLURM_JOB_ID:-none} host=$(hostname)"
set +e
bash hpc/unlock/run_in_env.sh "$MODE" 0 -- python run_unlock.py "$@"
RC=$?
set -e
echo "STAGE_EXIT stage=$1 rc=$RC"
exit "$RC"
