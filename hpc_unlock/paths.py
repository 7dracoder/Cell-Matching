"""Project-relative paths for the unlock job chain (Req 1.1).

Every path is derived from the location of this file, so the project folder
runs unchanged from /scratch/$USER/cellmatch or any other upload location.
Imports only the standard library, so a plain Python kernel can use it.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Project_2_Dataset"
RDATA = ROOT / "research" / "data"
HPC = RDATA / "hpc"
HPC_SMOKE = RDATA / "hpc_smoke"
LOGS = ROOT / "logs"
CONTAINER = ROOT / "hpc" / "unlock" / "container"
MODELS = ROOT / "hpc" / "unlock" / "models"

CUDA11_SIF = "cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"

for _p in (ROOT, ROOT / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def work_dir(notebook_dir: str | Path | None = None,
             env: Mapping[str, str] | None = None) -> Path:
    """Directory the notebook submits jobs from.

    The notebook's own directory if it contains ``hpc_unlock/``, else
    ``$CELLMATCH_DIR``, else ``/scratch/$USER/cellmatch``.
    """
    env = os.environ if env is None else env
    nb = Path.cwd() if notebook_dir is None else Path(notebook_dir)
    if (nb / "hpc_unlock").is_dir():
        return nb.resolve()
    if env.get("CELLMATCH_DIR"):
        return Path(env["CELLMATCH_DIR"])
    return Path("/scratch") / env.get("USER", "") / "cellmatch"


def _version_key(name: str) -> tuple:
    """Natural sort key: digit runs compare as integers (cuda12.10 > cuda12.9)."""
    return tuple((0, int(t)) if t.isdigit() else (1, t)
                 for t in re.findall(r"\d+|\D+", name.lower()))


def find_sif(container: str | Path | None = None) -> Path | None:
    """Pick the Singularity image in ``container``.

    A name containing ``cuda12`` first (highest version string), else the
    CUDA 11.8 image, else ``None``.
    """
    d = CONTAINER if container is None else Path(container)
    if not d.is_dir():
        return None
    sifs = [p for p in d.iterdir() if p.is_file() and p.suffix == ".sif"]
    cuda12 = [p for p in sifs if "cuda12" in p.name.lower()]
    if cuda12:
        return max(cuda12, key=lambda p: _version_key(p.name))
    fallback = d / CUDA11_SIF
    return fallback if fallback.is_file() else None


def available_cores() -> int:
    """Cores available to this process (affinity mask where supported)."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)  # macOS has no sched_getaffinity


def worker_count(env: Mapping[str, str] | None = None,
                 cores: int | None = None) -> int:
    """``SLURM_CPUS_PER_TASK`` if set, else available cores; clamped to [1, cores] (Req 3.2)."""
    env = os.environ if env is None else env
    cores = available_cores() if cores is None else max(1, int(cores))
    raw = env.get("SLURM_CPUS_PER_TASK", "").strip()
    try:
        n = int(raw) if raw else cores
    except ValueError:
        n = cores
    return min(max(n, 1), cores)
