#!/bin/bash
#SBATCH --job-name=unlock-gpu-stage
#SBATCH --output=logs/unlock-gpu-stage-%j.out
# sbatch <account/partition/gres/resource flags> hpc/unlock/gpu_stage.sbatch <singularity|venv> <stage> --run <fp> [args...]
# One GPU Stage of the unlock Job_Chain (Req 3.1, 3.3, 3.4). Account, partition, GPU and
# resources come from the sbatch command line (hpc_unlock/chain.py), not from #SBATCH lines.
# Same as stage.sbatch, but run_in_env.sh gets gpu=1 (singularity --nv), and an nvidia-smi
# logger writes one line of GPU use per minute into the Stage log until the job exits.
# The job's exit code is run_unlock.py's, so afterok dependencies hold.
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
  echo "usage: sbatch [flags] hpc/unlock/gpu_stage.sbatch <singularity|venv> <stage> --run <fp> [args...]" >&2
  exit 2
}
[ "$#" -ge 4 ] && [ "$3" = "--run" ] || usage
MODE="$1"
shift
export PYTHONUNBUFFERED=1

# GPU use logger (Req 3.4): inherits the job's stdout, i.e. the Stage log.
NVSMI_PID=""
stop_gpu_log() {
  if [ -n "$NVSMI_PID" ]; then
    kill "$NVSMI_PID" 2>/dev/null || true
    wait "$NVSMI_PID" 2>/dev/null || true
    NVSMI_PID=""
  fi
}
# Keep the exit status of the Stage: capture it before the cleanup runs.
trap 'RC_TRAP=$?; stop_gpu_log; exit "$RC_TRAP"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 &
  NVSMI_PID=$!
else
  echo "GPU_LOG_SKIPPED nvidia-smi not found on PATH"
fi

echo "STAGE_START stage=$1 mode=$MODE job=${SLURM_JOB_ID:-none} host=$(hostname)"
set +e
bash hpc/unlock/run_in_env.sh "$MODE" 1 -- python run_unlock.py "$@"
RC=$?
set -e
echo "STAGE_EXIT stage=$1 rc=$RC"
exit "$RC"
