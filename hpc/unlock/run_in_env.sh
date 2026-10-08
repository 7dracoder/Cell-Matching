#!/bin/bash
# run_in_env.sh <singularity|venv> <gpu:0|1> -- <command> [args...]
# Runs one command in the environment built by setup.sbatch, from the Project_Folder root.
#   singularity: singularity exec [--nv] --overlay <overlay>:ro <sif> /bin/bash -c
#                "source /ext3/env.sh; cd <WORK>; <command>"
#                The overlay is mounted :ro, so several Stage jobs can hold it at once;
#                --nv (host GPU driver) only when gpu=1.
#   venv:        source hpc/env.sh and hpc/unlock/.env_path, then <command>.
# Optional: UNLOCK_BIND=<src[:dst],...> adds a singularity --bind (used by the OOD kernel).
set -euo pipefail

usage() {
  echo "usage: $0 <singularity|venv> <0|1> -- <command> [args...]" >&2
  exit 2
}
[ "$#" -ge 4 ] || usage
MODE="$1"
GPU="$2"
[ "$3" = "--" ] || usage
shift 3
case "$GPU" in 0|1) ;; *) usage ;; esac

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
WORK="$UNLOCK_WORK"
export CELLPOSE_LOCAL_MODELS_PATH="$UNLOCK_MODELS" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
  PIP_CACHE_DIR="$WORK/.pip_cache"

case "$MODE" in
  singularity)
    SIF="$(unlock_find_sif)"
    [ -n "$SIF" ] || unlock_die "MISSING hpc/unlock/container/*.sif"
    [ -r "$UNLOCK_OVERLAY" ] || unlock_die "MISSING ${UNLOCK_OVERLAY#"$WORK"/}"
    SING="$(unlock_singularity)"
    ARGS=(exec)
    if [ "$GPU" = 1 ]; then
      ARGS+=(--nv)
    fi
    if [ -n "${UNLOCK_BIND:-}" ]; then
      ARGS+=(--bind "$UNLOCK_BIND")
    fi
    ARGS+=(--overlay "$UNLOCK_OVERLAY:ro" "$SIF")
    # CELLPOSE_LOCAL_MODELS_PATH etc. reach the container through singularity's default
    # environment pass-through; the command words are shell-quoted one by one.
    CMD="$(printf '%q ' "$@")"
    INNER="source /ext3/env.sh; cd $(printf '%q' "$WORK"); ${CMD% }"
    exec "$SING" "${ARGS[@]}" /bin/bash -c "$INNER"
    ;;
  venv)
    [ -r "$UNLOCK_ENV_PATH_FILE" ] || unlock_die "MISSING hpc/unlock/.env_path (run setup first)"
    cd "$WORK"
    SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$WORK}"
    set +u
    source "$WORK/hpc/env.sh"
    source "$UNLOCK_ENV_PATH_FILE"
    set -u
    [ -x "$ENV_DIR/bin/python" ] || unlock_die "MISSING $ENV_DIR/bin/python (run setup first)"
    export ENV_DIR PYTHON="$ENV_DIR/bin/python" PATH="$ENV_DIR/bin:$PATH" \
      CELLPOSE_LOCAL_MODELS_PATH="$UNLOCK_MODELS"
    exec "$@"
    ;;
  *)
    usage
    ;;
esac
