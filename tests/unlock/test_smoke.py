"""Unit tests for hpc_unlock/smoke.py helpers (Req 13.1-13.5).

The full local smoke run is not executed here (task 14.2 runs it); the
runner is exercised with a fake ``run_one`` in temporary directories.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hpc_unlock import paths, smoke
from hpc_unlock.smoke import FAIL, PASS, Result


# ---------------------------------------------------------------- lines / exit code
def test_result_line_format():
    assert smoke.result_line(PASS, "prep") == "PASS prep"
    assert smoke.result_line(FAIL, "joint", "exit 1") == "FAIL joint (exit 1)"
    with pytest.raises(ValueError):
        smoke.result_line("SKIP", "prep")


def test_first_failure_and_exit_code():
    ok = [Result("prep", PASS), Result("gpu_scan", PASS)]
    assert smoke.first_failure(ok) is None and smoke.exit_code(ok) == 0
    assert smoke.summary_line(ok, 3.0).startswith("SMOKE_PASSED")
    bad = ok + [Result("joint", FAIL, "exit 1"), Result("pairs", FAIL, "not run")]
    assert smoke.first_failure(bad) == "joint"
    assert smoke.exit_code(bad) == 1
    assert "first_failing_stage=joint" in smoke.summary_line(bad, 3.0)


def test_budget_result():
    assert smoke.budget_result(299.9).ok
    r = smoke.budget_result(300.1)
    assert not r.ok and r.stage == smoke.BUDGET


# ---------------------------------------------------------------- isolation hashes
def _tree(tmp_path: Path):
    root, hpc = tmp_path / "root", tmp_path / "root" / "research" / "data" / "hpc"
    (hpc / "fp").mkdir(parents=True)
    (hpc / "fp" / "prep.done").write_text("x")
    (root / "submission_v10_cpgate.csv").write_text("a")
    (root / "submission (1).csv").write_text("b")
    (root / "other.csv").write_text("c")
    return root, hpc


def test_snapshot_covers_guarded_files(tmp_path):
    root, hpc = _tree(tmp_path)
    snap = smoke.snapshot(root, hpc)
    assert snap[str(hpc / "fp" / "prep.done")] is not None
    assert snap[str(root / "submission_v10_cpgate.csv")] is not None
    assert snap[str(root / "submission (1).csv")] is not None
    assert snap[str(root / "hpc_unlock_report.json")] is None      # absent, still listed
    assert str(root / "other.csv") not in snap


def test_unchanged_tree_passes(tmp_path):
    root, hpc = _tree(tmp_path)
    before = smoke.snapshot(root, hpc)
    assert smoke.diff_snapshots(before, smoke.snapshot(root, hpc)) == []
    assert smoke.isolation_result(before, smoke.snapshot(root, hpc)).ok


@pytest.mark.parametrize("mutate,kind", [
    (lambda r, h: (r / "submission_v10_cpgate.csv").write_text("changed"), "modified"),
    (lambda r, h: (r / "hpc_unlock_report.md").write_text("new"), "created"),
    (lambda r, h: (r / "submission_v13_joint_cons.csv").write_text("new"), "created"),
    (lambda r, h: (h / "fp" / "prep.done").unlink(), "removed"),
    (lambda r, h: (h / "fp2" / "x.pkl").parent.mkdir() or (h / "fp2" / "x.pkl").write_text("1"),
     "created"),
])
def test_mutation_is_detected(tmp_path, mutate, kind):
    root, hpc = _tree(tmp_path)
    before = smoke.snapshot(root, hpc)
    mutate(root, hpc)
    changes = smoke.diff_snapshots(before, smoke.snapshot(root, hpc))
    assert len(changes) == 1 and changes[0].startswith(kind)
    r = smoke.isolation_result(before, smoke.snapshot(root, hpc))
    assert not r.ok and r.stage == smoke.ISOLATION


def test_creating_empty_hpc_dir_is_detected(tmp_path):
    root, hpc = tmp_path / "root", tmp_path / "root" / "research" / "data" / "hpc"
    root.mkdir()
    before = smoke.snapshot(root, hpc)
    hpc.mkdir(parents=True)
    assert smoke.diff_snapshots(before, smoke.snapshot(root, hpc)) == [f"created {hpc}{os.sep}"]


# ---------------------------------------------------------------- env / wipe
def test_stage_env_strips_slurm_and_pins_threads():
    base = {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "48", "PATH": "/bin"}
    env = smoke.stage_env("prep", 4, base)
    assert not any(k.startswith("SLURM_") for k in env) and env["PATH"] == "/bin"
    assert all(env[v] == "1" for v in smoke.THREAD_VARS)
    assert smoke.stage_env("gpu_scan", 4, base)["OMP_NUM_THREADS"] == "4"


def test_safe_wipe_only_inside_base(tmp_path):
    base = tmp_path / "hpc_smoke"
    (base / "fp").mkdir(parents=True)
    (base / "fp" / "a.pkl").write_text("x")
    smoke.safe_wipe(base / "fp", base)
    assert not (base / "fp").exists()
    other = tmp_path / "hpc"
    other.mkdir()
    for bad in (base, other, base / ".." / "hpc"):
        with pytest.raises(ValueError):
            smoke.safe_wipe(bad, base)
    assert other.is_dir() and base.is_dir()


# ---------------------------------------------------------------- per-Stage checks
def test_stage_checks():
    held = {"a": {"subject": "m1"}, "b": {"subject": "m2"}}
    assert "heldout=2 mice=2" in smoke.check_prep({"heldout": held, "test": {}})
    with pytest.raises(AssertionError):
        smoke.check_prep({"heldout": {"a": {"subject": "m1"}, "b": {"subject": "m1"}}})
    grid = {"angles": [0] * 5, "scales": [1] * 3, "stretches": [(1.0, None)], "smoke": True}
    assert "5x3x1" in smoke.check_gpu_scan({"grid": grid})
    with pytest.raises(AssertionError):
        smoke.check_gpu_scan({"grid": {**grid, "angles": [0] * 6}})
    assert smoke.check_validate({"baseline": {"reproduced": None}}) == \
        "Baseline tolerance n/a (smoke)"
    with pytest.raises(AssertionError):
        smoke.check_validate({"baseline": {"reproduced": True}})
    assert smoke.check_selftrain_gpu({"jobs": {"m1": {"status": "skipped_smoke"}}}) == \
        "SKIPPED (no fine-tune in smoke)"
    with pytest.raises(AssertionError):
        smoke.check_selftrain_gpu({"jobs": {"m1": {"status": "done"}}})


def test_check_assemble(tmp_path):
    base, run_dir = tmp_path / "hpc_smoke", tmp_path / "hpc_smoke" / "fp"
    run_dir.mkdir(parents=True)
    ck = {"report": {"baseline_reproduction": {"status": "n/a (smoke)"}}}
    with pytest.raises(AssertionError, match="not written"):
        smoke.check_assemble(ck, run_dir, base)
    (base / smoke.SMOKE_CSV).write_text("x")
    assert "Format_Checker ok" in smoke.check_assemble(ck, run_dir, base)
    (run_dir / "submission_v13_joint_cons.csv").write_text("x")
    with pytest.raises(AssertionError, match="submission CSVs"):
        smoke.check_assemble(ck, run_dir, base)


# ---------------------------------------------------------------- runner (fake Stages)
@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    root = tmp_path / "root"
    hpc_smoke = root / "research" / "data" / "hpc_smoke"
    hpc = root / "research" / "data" / "hpc"
    (hpc / "old").mkdir(parents=True)
    (hpc / "old" / "prep.done").write_text("keep")
    (root / "submission_v10_cpgate.csv").write_text("baseline")
    monkeypatch.setattr(paths, "ROOT", root)
    monkeypatch.setattr(paths, "HPC", hpc)
    monkeypatch.setattr(paths, "HPC_SMOKE", hpc_smoke)
    return root


def _fake_run_one(fail_at=None, side_effect=None):
    calls = []

    def run_one(name, fp, run_dir, smoke_base, timeout, workers, log_dir):
        calls.append(name)
        assert workers <= 4 and timeout <= smoke.BUDGET_S
        assert run_dir.parent == smoke_base and (run_dir / "run_config.json").is_file()
        if side_effect:
            side_effect(name)
        return Result(name, FAIL if name == fail_at else PASS, "exit 1" if name == fail_at else "")
    return run_one, calls


def test_run_all_pass(fake_paths, monkeypatch):
    run_one, calls = _fake_run_one()
    monkeypatch.setattr(smoke, "run_one", run_one)
    lines: list[str] = []
    assert smoke.run(out=lines.append) == 0
    assert calls[0] == "prep" and calls[-1] == "assemble" and "setup" not in calls
    assert "selftrain_gpu" in calls
    assert lines[-3].startswith("PASS isolation") and lines[-2].startswith("PASS budget")
    assert lines[-1].startswith("SMOKE_PASSED")
    assert (paths.HPC / "old" / "prep.done").read_text() == "keep"


def test_run_names_first_failing_stage(fake_paths, monkeypatch):
    run_one, calls = _fake_run_one(fail_at="joint")
    monkeypatch.setattr(smoke, "run_one", run_one)
    lines: list[str] = []
    assert smoke.run(out=lines.append) == 1
    assert calls[-1] == "joint"                                   # later Stages not run
    assert "FAIL joint (exit 1)" in lines
    assert any(ln.startswith("FAIL pairs (not run: Stage joint failed") for ln in lines)
    assert lines[-1].startswith("SMOKE_FAILED first_failing_stage=joint")


def test_run_fails_isolation_on_full_run_write(fake_paths, monkeypatch):
    def write_full(name):
        if name == "validate":
            (paths.ROOT / "hpc_unlock_report.json").write_text("{}")
    run_one, _ = _fake_run_one(side_effect=write_full)
    monkeypatch.setattr(smoke, "run_one", run_one)
    lines: list[str] = []
    assert smoke.run(out=lines.append) == 1
    assert any(ln.startswith("FAIL isolation (created ") for ln in lines)
    assert lines[-1].startswith("SMOKE_FAILED first_failing_stage=isolation")


def test_main_rejects_budget_over_300():
    with pytest.raises(SystemExit):
        smoke.main(["--budget", "301"])
