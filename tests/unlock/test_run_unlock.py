"""Unit tests for the run_unlock.py driver (Req 1.6, 13.6, 3.2, 4.8)."""
from __future__ import annotations

import pickle
import sys
import types

import numpy as np
import pytest

import run_unlock
from hpc_unlock import checkpoint, inputs, paths, stage
from hpc_unlock.config import UnlockConfig


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temporary HPC dirs plus fake Stages ``fakestage`` and ``fakedep``."""
    hpc, smoke = tmp_path / "hpc", tmp_path / "hpc_smoke"
    monkeypatch.setattr(paths, "HPC", hpc)
    monkeypatch.setattr(paths, "HPC_SMOKE", smoke)
    monkeypatch.setattr(stage, "STAGES",
                        stage.STAGES + ["fakestage", "fakedep", "exitstage", "notwritten"])
    calls = []

    def compute(cfg, ctx):
        calls.append(ctx)
        return {"sigma": cfg.sigma, "workers": ctx.workers, "smoke": ctx.smoke}

    def compute_dep(cfg, ctx):
        return {"from_fake": ctx.load("fakestage")}

    def compute_exit(cfg, ctx):
        raise SystemExit(1)

    for name, fn in (("fakestage", compute), ("fakedep", compute_dep),
                     ("exitstage", compute_exit)):
        mod = types.ModuleType(f"hpc_unlock.{name}")
        mod.compute = fn
        monkeypatch.setitem(sys.modules, f"hpc_unlock.{name}", mod)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return types.SimpleNamespace(hpc=hpc, smoke=smoke, calls=calls, mp=monkeypatch)


def _save(cfg: UnlockConfig, base) -> str:
    fp = cfg.fingerprint()
    cfg.save(base / fp)
    return fp


def test_outside_slurm_exits_2_before_compute(env, capsys):
    fp = _save(UnlockConfig(), env.hpc)
    assert run_unlock.main(["fakestage", "--run", fp]) == 2
    err = capsys.readouterr().err
    assert "start this Stage through the Job_Chain or the HPC_Notebook" in err
    assert env.calls == []
    assert not checkpoint.marker_path(env.hpc / fp, "fakestage").exists()


def test_invalid_config_exits_nonzero_before_compute(env, capsys):
    env.mp.setenv("SLURM_JOB_ID", "123")
    fp = _save(UnlockConfig(sigma=3.0, scan_k=0), env.hpc)
    assert run_unlock.main(["fakestage", "--run", fp]) == 1
    out = capsys.readouterr().out
    assert "CONFIG_INVALID" in out and "sigma=3.0" in out and "scan_k=0" in out
    assert env.calls == []
    assert not checkpoint.marker_path(env.hpc / fp, "fakestage").exists()


def test_stage_runs_through_run_stage_and_reuses(env, capsys):
    env.mp.setenv("SLURM_JOB_ID", "123")
    env.mp.setenv("SLURM_CPUS_PER_TASK", "1")
    fp = _save(UnlockConfig(), env.hpc)
    assert run_unlock.main(["fakestage", "--run", fp]) == 0
    out = capsys.readouterr().out
    assert f"STAGE_START fakestage fingerprint={fp} workers=1" in out
    assert "STAGE_END fakestage" in out and "elapsed=" in out
    assert len(env.calls) == 1 and env.calls[0].run_dir == env.hpc / fp
    assert checkpoint.marker_path(env.hpc / fp, "fakestage").exists()

    assert run_unlock.main(["fakestage", "--run", fp]) == 0
    assert "REUSE fakestage" in capsys.readouterr().out
    assert len(env.calls) == 1

    # A later Stage reads the earlier Checkpoint through ctx.load
    assert run_unlock.main(["fakedep", "--run", fp]) == 0
    dep = checkpoint.load(checkpoint.checkpoint_path(env.hpc / fp, "fakedep"))
    assert dep["from_fake"]["sigma"] == 2.5


def test_missing_dependency_fails(env, capsys):
    env.mp.setenv("SLURM_JOB_ID", "123")
    fp = _save(UnlockConfig(), env.hpc)
    assert run_unlock.main(["fakedep", "--run", fp]) == 1
    assert "DEPENDENCY_MISSING" in capsys.readouterr().out
    assert not checkpoint.marker_path(env.hpc / fp, "fakedep").exists()


def test_smoke_allowed_without_slurm(env):
    assert run_unlock.main(["fakestage", "--smoke"]) == 0
    fp = UnlockConfig().fingerprint()
    assert checkpoint.marker_path(env.smoke / fp, "fakestage").exists()
    assert env.calls[0].smoke is True and env.calls[0].workers <= 4
    assert not env.hpc.exists()          # no full-run directory touched


def test_stage_system_exit_propagates_without_marker(env):
    env.mp.setenv("SLURM_JOB_ID", "123")
    fp = _save(UnlockConfig(), env.hpc)
    assert run_unlock.main(["exitstage", "--run", fp]) == 1
    assert not checkpoint.marker_path(env.hpc / fp, "exitstage").exists()


@pytest.mark.parametrize("argv", [["nosuchstage", "--smoke"], ["setup", "--smoke"],
                                  ["notwritten", "--smoke"]])
def test_unknown_rejected_or_unavailable_stage_exits_2(env, argv, capsys):
    assert run_unlock.main(argv) == 2
    cap = capsys.readouterr()
    assert argv[0] in cap.out + cap.err


def test_full_stage_requires_run(env, capsys):
    env.mp.setenv("SLURM_JOB_ID", "123")
    assert run_unlock.main(["fakestage"]) == 2
    assert "--run" in capsys.readouterr().err


def test_check_lists_every_failing_path(project_tree, capsys):
    assert run_unlock.main(["check", "--root", str(project_tree), "--env-mode", "venv"]) == 1
    err = capsys.readouterr().err
    failing = [ln for ln in err.splitlines() if ln.startswith("INPUT_FAILED")]
    expected = inputs.required_inputs(UnlockConfig(env_mode="venv"), project_tree)
    assert len(failing) == len(expected)
    for p in expected:
        assert str(p) in err


def test_check_passes_when_inputs_present(project_tree, capsys):
    data, rdata = project_tree / "Project_2_Dataset", project_tree / "research" / "data"
    for _, rel in inputs.SPLIT_CSVS:
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text("sample_id\n")
    for name in inputs.RDATA_CACHES:
        p = rdata / name
        if p.suffix == ".pkl":
            p.write_bytes(pickle.dumps({}))
        elif p.suffix == ".npz":
            np.savez(p, a=np.zeros(1))
        else:
            p.write_text("x\n")
    for name in inputs.BASELINE_CSVS:
        (project_tree / name).write_text("x\n")
    assert run_unlock.main(["check", "--root", str(project_tree), "--env-mode", "venv"]) == 0
    assert "INPUT_CHECK_OK" in capsys.readouterr().out
