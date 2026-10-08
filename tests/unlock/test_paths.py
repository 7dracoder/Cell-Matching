"""Unit tests for hpc_unlock.paths (Req 1.1, 3.2)."""
from __future__ import annotations

import sys
from pathlib import Path

from hpc_unlock import paths


def test_root_is_project_folder():
    assert paths.ROOT == Path(paths.__file__).resolve().parents[1]
    assert (paths.ROOT / "hpc_unlock" / "paths.py").is_file()
    assert paths.HPC == paths.ROOT / "research" / "data" / "hpc"
    assert paths.CONTAINER == paths.ROOT / "hpc" / "unlock" / "container"
    assert str(paths.ROOT) in sys.path
    assert str(paths.ROOT / "research") in sys.path


def test_paths_imports_only_stdlib():
    src = Path(paths.__file__).read_text()
    imported = {line.split()[1].split(".")[0] for line in src.splitlines()
                if line.startswith(("import ", "from "))}
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def test_work_dir_order(project_tree, tmp_path):
    env = {"CELLMATCH_DIR": "/x/cm", "USER": "someone"}
    assert paths.work_dir(project_tree, env) == project_tree.resolve()
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert paths.work_dir(other, env) == Path("/x/cm")
    assert paths.work_dir(other, {"USER": "someone"}) == Path("/scratch/someone/cellmatch")


def test_find_sif_prefers_highest_cuda12(project_tree):
    c = project_tree / "hpc" / "unlock" / "container"
    assert paths.find_sif(c) is None
    (c / paths.CUDA11_SIF).touch()
    (c / "overlay-15GB-500K.ext3").touch()
    assert paths.find_sif(c) == c / paths.CUDA11_SIF
    for n in ("cuda12.1.1-cudnn8.9-devel-ubuntu22.04.sif",
              "cuda12.9.0-cudnn9-devel-ubuntu22.04.sif",
              "cuda12.10.0-cudnn9-devel-ubuntu22.04.sif"):
        (c / n).touch()
    assert paths.find_sif(c).name == "cuda12.10.0-cudnn9-devel-ubuntu22.04.sif"


def test_worker_count_clamps():
    assert paths.worker_count({}, cores=8) == 8
    assert paths.worker_count({"SLURM_CPUS_PER_TASK": "4"}, cores=8) == 4
    assert paths.worker_count({"SLURM_CPUS_PER_TASK": "48"}, cores=8) == 8
    assert paths.worker_count({"SLURM_CPUS_PER_TASK": "0"}, cores=8) == 1
    assert 1 <= paths.worker_count({}) <= paths.available_cores()
