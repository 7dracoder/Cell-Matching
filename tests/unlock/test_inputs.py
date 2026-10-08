"""Unit tests for hpc_unlock.inputs (Req 1.2, 1.3, 1.6)."""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hpc_unlock import inputs, paths
from hpc_unlock.config import UnlockConfig

TRAIN_IDS = ("subject_aaa__region_111", "subject_aaa__region_222")
TEST_IDS = ("subject_bbb__region_333",)


def _write_csv(path: Path, ids) -> None:
    lines = ["sample_id,invivo_instances,exvivo_instances,match_pairs"]
    lines += [f"{s},{{}},{{}},[]" for s in ids]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def full_tree(project_tree: Path) -> Path:
    """A temporary project tree that contains every required input."""
    data = project_tree / "Project_2_Dataset"
    _write_csv(data / "training" / "train_ground_truth.csv", TRAIN_IDS)
    _write_csv(data / "sample_submission.csv", TEST_IDS)
    for split, ids in (("training", TRAIN_IDS), ("hidden_test", TEST_IDS)):
        for sid in ids:
            d = data / split / Path(*sid.split("__"))
            d.mkdir(parents=True)
            for m in ("invivo", "exvivo"):
                (d / f"{m}.tif").write_bytes(b"II*\x00tiff")
    rdata = project_tree / "research" / "data"
    for name in inputs.RDATA_CACHES:
        p = rdata / name
        if name.endswith(".pkl"):
            p.write_bytes(pickle.dumps({"a": np.arange(3)}))
        elif name.endswith(".npz"):
            np.savez(p, x=np.zeros(2))
        else:
            p.write_text("sample_id\n")
    for name in inputs.BASELINE_CSVS:
        (project_tree / name).write_text("sample_id\n")
    c = project_tree / "hpc" / "unlock" / "container"
    (c / inputs.OVERLAY_NAME).write_bytes(b"")          # content is not inspected
    (c / paths.CUDA11_SIF).write_bytes(b"")
    return project_tree


SING = UnlockConfig()
VENV = UnlockConfig(env_mode="venv")


# ---------------------------------------------------------------- pins file
def test_pins_file_matches_design_and_local_venv():
    pins = inputs.read_pins()
    assert pins["torch"] == "2.14.0"
    assert pins["cellpose"] == "4.2.1.1"
    assert set(inputs.CORE_PACKAGES) <= set(pins)
    assert {"scikit-image", "torch", "cellpose"} <= set(pins)
    installer = inputs.miniconda_installer()
    assert installer.startswith("Miniconda3-py312_")
    assert installer.endswith("-Linux-x86_64.sh")
    # Every pin is the version installed in the local .venv
    installed = inputs.installed_versions()
    assert installed == pins


def test_read_pins_rejects_unpinned_and_missing_core(tmp_path):
    p = tmp_path / "req.txt"
    p.write_text("numpy\n")
    with pytest.raises(ValueError, match="unpinned"):
        inputs.read_pins(p)
    p.write_text("numpy==1.0  # comment\n")
    with pytest.raises(ValueError, match="core pin"):
        inputs.read_pins(p)


# ---------------------------------------------------------------- reuse_env
def test_reuse_env_exact_match_only():
    pins = inputs.read_pins()
    core = {n: pins[n] for n in inputs.CORE_PACKAGES}
    assert inputs.reuse_env(core)
    assert inputs.reuse_env({**core, "torch": "0.0.1", "extra": None})
    for n in inputs.CORE_PACKAGES:
        assert not inputs.reuse_env({k: v for k, v in core.items() if k != n})
        assert not inputs.reuse_env({**core, n: None})
        assert not inputs.reuse_env({**core, n: core[n] + ".post1"})
    assert not inputs.reuse_env({})


# ---------------------------------------------------------- required inputs
def test_required_inputs_lists_everything(full_tree):
    req = inputs.required_inputs(SING, full_tree)
    rel = {p.relative_to(full_tree).as_posix() for p in req}
    assert "Project_2_Dataset/training/train_ground_truth.csv" in rel
    assert "Project_2_Dataset/sample_submission.csv" in rel
    assert "Project_2_Dataset/training/subject_aaa/region_111/exvivo.tif" in rel
    assert "Project_2_Dataset/hidden_test/subject_bbb/region_333/invivo.tif" in rel
    assert {f"research/data/{n}" for n in inputs.RDATA_CACHES} <= rel
    assert set(inputs.BASELINE_CSVS) <= rel
    assert f"hpc/unlock/container/{inputs.OVERLAY_NAME}" in rel
    assert f"hpc/unlock/container/{paths.CUDA11_SIF}" in rel
    assert len(req) == len(set(req)) == 2 + 2 * 3 + len(inputs.RDATA_CACHES) + 2 + 2


def test_design_caches_are_required():
    design = {"lab.pkl", "vote_cands.pkl", "cp_pose_train.pkl", "reg_window_vote.pkl",
              "reg_hough.pkl", "heldout_labels.npz", "test_cp_base.npz", "submission.csv"}
    assert design <= set(inputs.RDATA_CACHES)


def test_venv_mode_skips_container_files(full_tree):
    req = inputs.required_inputs(VENV, full_tree)
    assert not any(p.suffix in (".ext3", ".sif") for p in req)
    assert inputs.check_inputs(VENV, full_tree) == []


def test_complete_tree_passes(full_tree):
    assert inputs.check_inputs(SING, full_tree) == []


def test_missing_sif_reported_as_glob(full_tree):
    c = full_tree / "hpc" / "unlock" / "container"
    (c / paths.CUDA11_SIF).unlink()
    assert inputs.check_inputs(SING, full_tree) == [c / "*.sif"]
    named = UnlockConfig(sif_name="cuda12.4-x.sif")
    assert inputs.check_inputs(named, full_tree) == [c / "cuda12.4-x.sif"]


def test_reports_every_failure_not_only_first(full_tree):
    rdata = full_tree / "research" / "data"
    tif = full_tree / "Project_2_Dataset/hidden_test/subject_bbb/region_333/exvivo.tif"
    gt = full_tree / "Project_2_Dataset/training/train_ground_truth.csv"
    broken_pkl = rdata / "lab.pkl"
    broken_npz = rdata / "test_cp_base.npz"
    empty = full_tree / "submission_v7_grow15.csv"
    tif.unlink()
    gt.unlink()                                   # region dirs still found on disk
    broken_pkl.write_bytes(b"not a pickle")
    broken_npz.write_bytes(b"not a zip")
    empty.write_bytes(b"")
    (full_tree / "hpc/unlock/container" / inputs.OVERLAY_NAME).unlink()
    failed = inputs.input_failures(SING, full_tree)
    got = {p: r for p, r in failed}
    expected = {gt, tif, broken_pkl, broken_npz, empty,
                full_tree / "hpc/unlock/container" / inputs.OVERLAY_NAME}
    assert set(got) == expected
    assert got[tif] == "missing"
    assert got[broken_pkl].startswith("unreadable")
    assert got[broken_npz].startswith("unreadable")
    assert inputs.check_inputs(SING, full_tree) == [p for p, _ in failed]


def test_region_listed_in_csv_but_absent_on_disk(full_tree):
    _write_csv(full_tree / "Project_2_Dataset/sample_submission.csv",
               TEST_IDS + ("subject_bbb__region_444",))
    d = full_tree / "Project_2_Dataset/hidden_test/subject_bbb/region_444"
    assert set(inputs.check_inputs(VENV, full_tree)) == {d / "invivo.tif", d / "exvivo.tif"}


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="permission bits are not enforced")
def test_unreadable_file_reported(full_tree):
    p = full_tree / "research" / "data" / "reg_hough.pkl"
    sif = full_tree / "hpc/unlock/container" / paths.CUDA11_SIF
    p.chmod(0)
    sif.chmod(0)
    try:
        assert set(inputs.check_inputs(SING, full_tree)) == {p, sif}
    finally:
        p.chmod(0o644)
        sif.chmod(0o644)


# ---------------------------------------------------------------------- CLI
def _cli(*args, cwd=paths.ROOT):
    return subprocess.run([sys.executable, "-m", "hpc_unlock.inputs", *args],
                          cwd=cwd, capture_output=True, text=True)


def test_cli_pins_prints_installed_versions_as_json():
    r = _cli("pins")
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert list(got) == list(inputs.read_pins())
    assert got == inputs.installed_versions()


def test_cli_check_exit_codes(full_tree):
    r = _cli("check", "--root", str(full_tree))
    assert r.returncode == 0, r.stderr
    (full_tree / "research/data/lab.pkl").unlink()
    (full_tree / "submission_v10_cpgate.csv").unlink()
    r = _cli("check", "--root", str(full_tree))
    assert r.returncode == 1
    assert r.stderr.count("INPUT_FAILED") == 2
    assert "lab.pkl" in r.stderr and "submission_v10_cpgate.csv" in r.stderr


def test_inputs_top_level_imports_only_stdlib():
    src = Path(inputs.__file__).read_text()
    top = {line.split()[1].split(".")[0] for line in src.splitlines()
           if line.startswith(("import ", "from "))}
    assert top <= set(sys.stdlib_module_names) | {"__future__", ""}
