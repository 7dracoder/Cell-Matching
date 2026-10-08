"""Setup, environment and Stage scripts (tasks 3.1, 3.2; Req 1.3-1.7, 3.1, 3.3, 3.4).

Static checks on the bash scripts plus dry runs of ``run_in_env.sh``, ``stage.sbatch``
and ``gpu_stage.sbatch`` in a copied project tree, with fake ``singularity``,
``nvidia-smi`` and ``module`` commands on PATH.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UNLOCK = REPO / "hpc" / "unlock"
STAGE_SCRIPTS = ("stage.sbatch", "gpu_stage.sbatch")
SCRIPTS = [UNLOCK / n for n in ("run_in_env.sh", "setup.sbatch", "setup_overlay.sh", "common.sh",
                                *STAGE_SCRIPTS)]
OVERLAY = "overlay-15GB-500K.ext3"
CUDA11 = "cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"


def _text(name: str) -> str:
    return (UNLOCK / name).read_text(encoding="utf-8")


def _code(name: str) -> str:
    """Script text without comment lines (so comments cannot satisfy a check)."""
    return "\n".join(l for l in _text(name).splitlines() if not l.lstrip().startswith("#"))


# ------------------------------------------------------------------ static checks
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script: Path):
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "set -" in script.read_text() and "pipefail" in script.read_text()


def test_setup_purges_modules_then_cds_to_submit_dir():
    code = _code("setup.sbatch")
    assert "module purge" in code
    assert 'cd "$SLURM_SUBMIT_DIR"' in code
    assert code.index("module purge") < code.index('cd "$SLURM_SUBMIT_DIR"')


def test_setup_mounts_overlay_rw_and_stages_mount_ro():
    setup = _code("setup.sbatch")
    assert '--overlay "$UNLOCK_OVERLAY:rw"' in setup
    assert "setup_overlay.sh" in setup
    run = _code("run_in_env.sh")
    assert '--overlay "$UNLOCK_OVERLAY:ro"' in run
    assert ":rw" not in run


def test_no_account_or_partition_in_scripts():
    for p in SCRIPTS + [UNLOCK / "kernel" / "kernel.json"]:
        text = p.read_text()
        assert "cs_gy_" not in text, p
        sbatch_lines = [l for l in text.splitlines() if l.startswith("#SBATCH")]
        for line in sbatch_lines:
            assert re.match(r"#SBATCH --(job-name|output)=", line), line


def test_pip_failure_names_package_and_torch_from_default_pypi():
    common = _code("common.sh")
    assert 'echo "PIN_FAILED $UNLOCK_CURRENT_PIN"' in common
    assert "trap unlock_on_err ERR" in common
    assert 'https://pypi.org/simple' in common
    assert "cache_model_path(\"cpsam_v2\")" in common
    assert "torch.cuda" not in common  # no CUDA check on the CPU setup node


@pytest.mark.parametrize("name,gpu", [("stage.sbatch", "0"), ("gpu_stage.sbatch", "1")])
def test_stage_scripts_purge_cd_and_gpu_flag(name: str, gpu: str):
    code = _code(name)
    assert code.index("module purge") < code.index('cd "$SLURM_SUBMIT_DIR"')
    assert f'run_in_env.sh "$MODE" {gpu} -- python run_unlock.py "$@"' in code
    assert "export PYTHONUNBUFFERED=1" in code
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert var not in code  # Stages size their own pools (Req 3.1)


def test_only_gpu_stage_logs_nvidia_smi_and_kills_it_on_exit():
    gpu = _code("gpu_stage.sbatch")
    assert ("nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used "
            "--format=csv,noheader -l 60 &") in gpu
    assert re.search(r"^trap .*stop_gpu_log.* EXIT$", gpu, re.M)
    assert 'kill "$NVSMI_PID"' in gpu
    cpu = _code("stage.sbatch")
    assert "nvidia-smi" not in cpu and "trap" not in cpu


def test_kernel_json_runs_ipykernel_in_container_read_only():
    spec = json.loads((UNLOCK / "kernel" / "kernel.json").read_text())
    argv = spec["argv"]
    assert argv[-1] == "{connection_file}"
    joined = " ".join(argv)
    assert "run_in_env.sh\" singularity 0 --" in joined
    assert "python -m ipykernel_launcher -f" in joined
    assert spec["language"] == "python"


# ------------------------------------------------------------------ dry runs
@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Project tree with copies of the scripts, fake container files and fake singularity."""
    root = (tmp_path / "cellmatch").resolve()
    (root / "hpc" / "unlock" / "container").mkdir(parents=True)
    for name in ("run_in_env.sh", "common.sh", "requirements-unlock.txt", *STAGE_SCRIPTS):
        shutil.copy2(UNLOCK / name, root / "hpc" / "unlock" / name)
    shutil.copy2(REPO / "hpc" / "env.sh", root / "hpc" / "env.sh")
    container = root / "hpc" / "unlock" / "container"
    for name in (OVERLAY, CUDA11, "cuda12.2.2-cudnn8.9-devel-ubuntu22.04.3.sif",
                 "cuda12.10.0-cudnn9-devel-ubuntu24.04.sif"):
        (container / name).write_bytes(b"x")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    # singularity: prints its arguments; waits for the fake nvidia-smi when asked to, so
    # the logger line is in the output; exits with $FAKE_EXIT.
    _fake(fakebin / "singularity",
          'if [ -n "${NVSMI_PIDFILE:-}" ]; then\n'
          '  for _ in $(seq 100); do [ -s "$NVSMI_PIDFILE" ] && break; sleep 0.05; done\n'
          'fi\n'
          'echo "FAKE_PWD $PWD" >&2\n'
          'echo FAKE_SINGULARITY\nprintf \'%s\\n\' "$@"\nexit "${FAKE_EXIT:-0}"\n')
    # nvidia-smi: records its pid and blocks like `-l 60` (exec: the pid is the sleep).
    _fake(fakebin / "nvidia-smi",
          'echo "FAKE_NVSMI $*"\necho $$ > "${NVSMI_PIDFILE:-/dev/null}"\nexec sleep 300\n')
    _fake(fakebin / "module", 'echo "FAKE_MODULE $*"\n')
    return root


def _fake(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["PATH"] = f"{root.parent / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(["bash", str(root / "hpc" / "unlock" / "run_in_env.sh"), *args],
                          capture_output=True, text=True, env=env, cwd=root.parent, timeout=60)


@pytest.mark.parametrize("gpu", ["0", "1"])
def test_singularity_command_line(tree: Path, gpu: str):
    res = _run(tree, "singularity", gpu, "--", "python", "run_unlock.py", "prep", "--run", "abc123")
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert lines[0] == "FAKE_SINGULARITY"
    c = tree / "hpc" / "unlock" / "container"
    expected = (["exec"] + (["--nv"] if gpu == "1" else [])
                + ["--overlay", f"{c / OVERLAY}:ro",
                   str(c / "cuda12.10.0-cudnn9-devel-ubuntu24.04.sif"),
                   "/bin/bash", "-c",
                   f"source /ext3/env.sh; cd {tree}; python run_unlock.py prep --run abc123"])
    assert lines[1:] == expected


def test_singularity_falls_back_to_cuda11_and_quotes_args(tree: Path):
    c = tree / "hpc" / "unlock" / "container"
    for p in c.glob("cuda12*.sif"):
        p.unlink()
    res = _run(tree, "singularity", "0", "--", "echo", "a b")
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert lines[4] == str(c / CUDA11)
    assert lines[-1] == f"source /ext3/env.sh; cd {tree}; echo a\\ b"


def test_singularity_missing_overlay_fails(tree: Path):
    (tree / "hpc" / "unlock" / "container" / OVERLAY).unlink()
    res = _run(tree, "singularity", "0", "--", "true")
    assert res.returncode != 0
    assert "FAKE_SINGULARITY" not in res.stdout
    assert OVERLAY in res.stderr


def test_venv_mode_sources_env_sh_and_env_path(tree: Path):
    env_dir = tree / "env_unlock"
    (env_dir / "bin").mkdir(parents=True)
    py = env_dir / "bin" / "python"
    py.write_text("#!/bin/bash\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    (tree / "hpc" / "unlock" / ".env_path").write_text(f"export ENV_DIR={env_dir}\n")
    res = _run(tree, "venv", "0", "--", "bash", "-c",
               'echo "$CONDA_DIR|$ENV_DIR|$PYTHON|$PWD|$CELLPOSE_LOCAL_MODELS_PATH"')
    assert res.returncode == 0, res.stderr
    assert "FAKE_SINGULARITY" not in res.stdout
    conda, env, python, pwd, models = res.stdout.strip().split("|")
    assert conda == str(tree / "miniforge")          # set by hpc/env.sh
    assert env == str(env_dir)                       # overridden by hpc/unlock/.env_path
    assert python == str(py)
    assert pwd == str(tree)
    assert models == str(tree / "hpc" / "unlock" / "models")


def test_venv_mode_without_setup_fails(tree: Path):
    res = _run(tree, "venv", "0", "--", "true")
    assert res.returncode != 0
    assert ".env_path" in res.stderr


@pytest.mark.parametrize("args", [["singularity", "2", "--", "true"],
                                  ["docker", "0", "--", "true"],
                                  ["venv", "0", "true", "x"]])
def test_usage_errors(tree: Path, args):
    res = _run(tree, *args)
    assert res.returncode == 2
    assert "usage" in res.stderr


# ------------------------------------------------------------------ Stage dry runs
def _sbatch(root: Path, script: str, *args: str, **extra: str) -> subprocess.CompletedProcess:
    """Runs a Stage script like Slurm does: from another directory, SLURM_SUBMIT_DIR set."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["PATH"] = f"{root.parent / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["SLURM_SUBMIT_DIR"] = str(root)
    env["SLURM_JOB_ID"] = "4242"
    env.pop("PYTHONUNBUFFERED", None)
    env.update(extra)
    return subprocess.run(["bash", str(root / "hpc" / "unlock" / script), *args],
                          capture_output=True, text=True, env=env, cwd=root.parent, timeout=30)


def _singularity_args(stdout: str) -> list[str]:
    lines = stdout.splitlines()
    start = lines.index("FAKE_SINGULARITY") + 1
    end = next(i for i, l in enumerate(lines) if l.startswith("STAGE_EXIT"))
    return lines[start:end]


@pytest.mark.parametrize("script,gpu", [("stage.sbatch", False), ("gpu_stage.sbatch", True)])
def test_stage_dry_run_command_and_arg_pass_through(tree: Path, tmp_path: Path, script: str,
                                                    gpu: bool):
    pidfile = tmp_path / "nvsmi.pid"
    extra = {"NVSMI_PIDFILE": str(pidfile)} if gpu else {}
    res = _sbatch(tree, script, "singularity", "assemble", "--run", "abc123", "--report-only",
                  **extra)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert out.index("FAKE_MODULE purge") < out.index("STAGE_START stage=assemble")
    assert f"FAKE_PWD {tree}" in res.stderr  # cd "$SLURM_SUBMIT_DIR" before running
    c = tree / "hpc" / "unlock" / "container"
    assert _singularity_args(out) == (
        ["exec"] + (["--nv"] if gpu else [])
        + ["--overlay", f"{c / OVERLAY}:ro", str(c / "cuda12.10.0-cudnn9-devel-ubuntu24.04.sif"),
           "/bin/bash", "-c",
           f"source /ext3/env.sh; cd {tree}; python run_unlock.py assemble --run abc123 --report-only"])
    assert "STAGE_EXIT stage=assemble rc=0" in out
    if gpu:
        assert ("FAKE_NVSMI --query-gpu=timestamp,utilization.gpu,memory.used "
                "--format=csv,noheader -l 60") in out
        pid = int(pidfile.read_text())
        with pytest.raises(ProcessLookupError):  # the EXIT trap killed the logger
            os.kill(pid, 0)
    else:
        assert "FAKE_NVSMI" not in out


@pytest.mark.parametrize("script", STAGE_SCRIPTS)
def test_stage_exit_code_is_inner_command_exit_code(tree: Path, tmp_path: Path, script: str):
    res = _sbatch(tree, script, "singularity", "prep", "--run", "abc123",
                  FAKE_EXIT="3", NVSMI_PIDFILE=str(tmp_path / "nvsmi.pid"))
    assert res.returncode == 3, res.stdout + res.stderr
    assert "STAGE_EXIT stage=prep rc=3" in res.stdout


@pytest.mark.parametrize("script", STAGE_SCRIPTS)
@pytest.mark.parametrize("args", [["singularity", "prep"], ["singularity", "prep", "-r", "x"]])
def test_stage_usage_errors(tree: Path, script: str, args):
    res = _sbatch(tree, script, *args)
    assert res.returncode == 2
    assert "usage" in res.stderr
    assert "FAKE_SINGULARITY" not in res.stdout
