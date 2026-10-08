#!/bin/bash
# Shared helpers for the unlock env scripts (Req 1.3-1.7). Sourced, not executed, by
# run_in_env.sh, setup.sbatch and setup_overlay.sh (the last one inside the container).
# Every path is derived from this file's location: no NetID, no account name.
set -euo pipefail

UNLOCK_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNLOCK_DIR="$UNLOCK_WORK/hpc/unlock"
UNLOCK_PINS="$UNLOCK_DIR/requirements-unlock.txt"
UNLOCK_CONTAINER="$UNLOCK_DIR/container"
UNLOCK_OVERLAY="$UNLOCK_CONTAINER/overlay-15GB-500K.ext3"
UNLOCK_MODELS="$UNLOCK_DIR/models"
UNLOCK_ENV_PATH_FILE="$UNLOCK_DIR/.env_path"
UNLOCK_CUDA11_SIF="cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
UNLOCK_CONDA_ROOT="/ext3/miniconda3"
UNLOCK_CONDA_ENV="$UNLOCK_CONDA_ROOT/envs/unlock"
# Used when the pins file has no "# MINICONDA_INSTALLER=" line. SHA-256 as published
# on https://repo.anaconda.com/miniconda/ for this exact file.
UNLOCK_MINICONDA_DEFAULT="Miniconda3-py312_26.7.1-1-Linux-x86_64.sh"
UNLOCK_MINICONDA_DEFAULT_SHA256="b27f60ab63e77eeab50a5417c989120f767e863df32400190d4c7262369f8695"
UNLOCK_PYPI="https://pypi.org/simple"
UNLOCK_CURRENT_PIN=""

unlock_die() {
  echo "ERROR $*" >&2
  exit 1
}

# Same rule as hpc_unlock.paths.find_sif: a .sif whose name contains cuda12 (highest
# version string), else the CUDA 11.8 image, else nothing.
unlock_find_sif() {
  local dir="${1:-$UNLOCK_CONTAINER}" best
  [ -d "$dir" ] || return 0
  best="$(ls -1d "$dir"/*.sif 2>/dev/null | grep -i 'cuda12[^/]*$' | LC_ALL=C sort -V \
    | tail -n 1 || true)"
  if [ -n "$best" ]; then
    echo "$best"
  elif [ -f "$dir/$UNLOCK_CUDA11_SIF" ]; then
    echo "$dir/$UNLOCK_CUDA11_SIF"
  fi
}

unlock_singularity() {
  command -v singularity || command -v apptainer || unlock_die "singularity not found on PATH"
}

# Installer file name from the "# MINICONDA_INSTALLER=<file>" line of the pins file.
unlock_miniconda_installer() {
  local pins="${1:-$UNLOCK_PINS}" name
  name="$(sed -n 's/^#[[:space:]]*MINICONDA_INSTALLER=\([^[:space:]]*\).*/\1/p' "$pins" 2>/dev/null \
    | head -n 1 || true)"
  echo "${name:-$UNLOCK_MINICONDA_DEFAULT}"
}

# Downloads the installer once (cached next to the container files) and checks the
# published SHA-256 when the file is the default pinned installer.
unlock_fetch_miniconda() {
  local dest="$1" name url sum
  name="$(basename "$dest")"
  url="https://repo.anaconda.com/miniconda/$name"
  if [ ! -s "$dest" ]; then
    echo "MINICONDA_DOWNLOAD $url"
    mkdir -p "$(dirname "$dest")"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL --retry 3 -o "$dest.partial" "$url"
    else
      wget -q -O "$dest.partial" "$url"
    fi
    mv "$dest.partial" "$dest"
  fi
  if [ "$name" = "$UNLOCK_MINICONDA_DEFAULT" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
      sum="$(sha256sum "$dest" | cut -d' ' -f1)"
    else
      sum="$(shasum -a 256 "$dest" | cut -d' ' -f1)"
    fi
    if [ "$sum" != "$UNLOCK_MINICONDA_DEFAULT_SHA256" ]; then
      rm -f "$dest"
      unlock_die "checksum mismatch for $name (got $sum); removed, resubmit setup"
    fi
  fi
}

# One requirement spec per line ("pkg==ver" or a bare "pkg"); comments and blanks dropped.
unlock_pins() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | tr -d '[:space:]')"
    if [ -n "$line" ]; then
      printf '%s\n' "$line"
    fi
  done < "${1:-$UNLOCK_PINS}"
}

unlock_pin_name() {
  local spec="$1"
  printf '%s\n' "${spec%%[=<>!~;[]*}" | tr '[:upper:]_' '[:lower:]-'
}

# True if <python> exists and has every core pin at its pinned version
# (decision rule hpc_unlock.inputs.reuse_env, Property 7). Prints the installed versions.
unlock_env_matches() {
  local py="$1" pins
  [ -x "$py" ] || return 1
  pins="$(cd "$UNLOCK_WORK" && PYTHONNOUSERSITE=1 "$py" -m hpc_unlock.inputs pins < /dev/null)" \
    || return 1
  echo "ENV_PINS $py $pins"
  (cd "$UNLOCK_WORK" && PYTHONNOUSERSITE=1 "$py" -m hpc_unlock.inputs reuse < /dev/null) \
    >/dev/null
}
# True if <python> can run the full input check: Python >= 3.8 with numpy (the check
# unpickles the .pkl caches and opens the .npz files).
unlock_python_can_check() {
  local py="$1"
  command -v "$py" >/dev/null 2>&1 || return 1
  "$py" -c 'import sys, numpy; sys.exit(0 if sys.version_info >= (3, 8) else 1)' \
    >/dev/null 2>&1 < /dev/null
}
# unlock_input_check <python> <mode>: run_unlock.py check (Req 1.6); lists every missing or
# unreadable input as INPUT_FAILED and returns non-zero if there is any.
unlock_input_check() {
  local py="$1" mode="$2"
  echo "INPUT_CHECK python=$py mode=$mode"
  (cd "$UNLOCK_WORK" && PYTHONNOUSERSITE=1 "$py" run_unlock.py check --env-mode "$mode" < /dev/null)
}
# unlock_write_setup_json <python> <mode> <env path> <sif name or ""> <cellpose:0|1>
# Writes research/data/hpc/setup/setup.json atomically. Imports torch for its version and
# build CUDA version only: no torch.cuda call, the setup job runs on a CPU node.
unlock_write_setup_json() {
  (cd "$UNLOCK_WORK" && UNLOCK_JSON_MODE="$2" UNLOCK_JSON_ENV="$3" UNLOCK_JSON_SIF="$4" \
    UNLOCK_JSON_CELLPOSE="$5" UNLOCK_JSON_OVERLAY="$UNLOCK_OVERLAY" \
    UNLOCK_JSON_MODELS="$UNLOCK_MODELS" PYTHONNOUSERSITE=1 "$1" - <<'EOF'
import json, os, platform, sys, time
from pathlib import Path
import torch
from hpc_unlock.inputs import installed_versions, read_pins
env = os.environ
mode = env["UNLOCK_JSON_MODE"]
info = {
    "env_mode": mode,
    "env_path": env["UNLOCK_JSON_ENV"],
    "python": sys.executable,
    "python_version": platform.python_version(),
    "overlay": env["UNLOCK_JSON_OVERLAY"] if mode == "singularity" else None,
    "sif": env["UNLOCK_JSON_SIF"] or None,
    "cellpose_installed": env["UNLOCK_JSON_CELLPOSE"] == "1",
    "models_dir": env["UNLOCK_JSON_MODELS"],
    "pins": read_pins(),
    "versions": installed_versions(),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "slurm_job_id": env.get("SLURM_JOB_ID"),
    "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
out = Path("research/data/hpc/setup/setup.json")
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, out)
print(f"SETUP_JSON {out} torch={torch.__version__} cuda={torch.version.cuda}")
EOF
  )
}

unlock_write_env_path() {
  printf 'export ENV_DIR=%q\n' "$1" > "$UNLOCK_ENV_PATH_FILE.tmp"
  mv "$UNLOCK_ENV_PATH_FILE.tmp" "$UNLOCK_ENV_PATH_FILE"
}

# ERR trap: names the package whose install failed (Req 1.7). Needs set -E in the caller.
unlock_on_err() {
  if [ -n "${UNLOCK_CURRENT_PIN:-}" ]; then
    echo "PIN_FAILED $UNLOCK_CURRENT_PIN" >&2
    UNLOCK_CURRENT_PIN=""
  fi
}

# unlock_pip <python> <pins file> <spec> [pip args...]: one pip install per pin, with the
# pins file as constraints so a later install cannot move an earlier pin.
unlock_pip() {
  local py="$1" pins="$2" spec="$3"
  shift 3
  UNLOCK_CURRENT_PIN="$(unlock_pin_name "$spec")"
  echo "PIN_INSTALL $spec"
  "$py" -m pip install --no-input --disable-pip-version-check -c "$pins" "$@" "$spec" < /dev/null
  UNLOCK_CURRENT_PIN=""
}

# unlock_install_pins <python> <pins file> <cellpose:0|1> <models dir>
# Every pin except torch/cellpose, then torch from the default PyPI index (every run,
# Req 1.4), then cellpose and the cached cpsam_v2 weights unless disabled (Req 1.5).
unlock_install_pins() {
  local py="$1" pins="$2" cellpose="$3" models="$4"
  local spec name torch_spec="" cellpose_spec="" i
  local specs=()
  set -E
  trap unlock_on_err ERR
  unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
  export PYTHONNOUSERSITE=1
  while IFS= read -r spec; do
    specs+=("$spec")
  done < <(unlock_pins "$pins")
  [ "${#specs[@]}" -gt 0 ] || unlock_die "no pins in $pins"
  for i in "${!specs[@]}"; do
    spec="${specs[$i]}"
    name="$(unlock_pin_name "$spec")"
    case "$name" in
      torch) torch_spec="$spec" ;;
      cellpose) cellpose_spec="$spec" ;;
      *) unlock_pip "$py" "$pins" "$spec" ;;
    esac
  done
  [ -n "$torch_spec" ] || unlock_die "no torch pin in $pins"
  unlock_pip "$py" "$pins" "$torch_spec" --index-url "$UNLOCK_PYPI"
  if [ "$cellpose" = 1 ]; then
    [ -n "$cellpose_spec" ] || unlock_die "no cellpose pin in $pins"
    unlock_pip "$py" "$pins" "$cellpose_spec"
    mkdir -p "$models"
    UNLOCK_CURRENT_PIN="cpsam_v2"
    CELLPOSE_LOCAL_MODELS_PATH="$models" "$py" -c \
      'from cellpose import models; print("CPSAM_WEIGHTS", models.cache_model_path("cpsam_v2"))' < /dev/null
    UNLOCK_CURRENT_PIN=""
  else
    echo "CELLPOSE_SKIPPED --no-cellpose"
  fi
  trap - ERR
}
