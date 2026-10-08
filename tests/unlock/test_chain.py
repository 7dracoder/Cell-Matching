"""Unit tests for hpc_unlock/chain.py with a fake sbatch / squeue / sacct runner."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hpc_unlock import chain
from hpc_unlock.config import UnlockConfig

GPU_PARTS = ("g2-standard-12", "c12m85-a100-1")
MODES = ("singularity", "venv")


class FakeSlurm:
    """Records every call; sbatch hands out job IDs 1001, 1002, ...

    ``fail_at``: 0-based sbatch call index that returns an error.
    ``queue``: job id -> "STATE|reason" for squeue (absent = left the queue).
    ``acct``: job id -> "STATE|ExitCode" for sacct.
    """

    def __init__(self, fail_at=None, queue=None, acct=None, first_id=1001):
        self.fail_at = fail_at
        self.queue = dict(queue or {})
        self.acct = dict(acct or {})
        self.next_id = first_id
        self.calls: list[SimpleNamespace] = []

    def sbatch_calls(self):
        return [c for c in self.calls if c.argv[0] == "sbatch"]

    def __call__(self, argv, **kw):
        self.calls.append(SimpleNamespace(argv=list(argv), **kw))
        tool = argv[0]
        if tool == "sbatch":
            if self.fail_at is not None and len(self.sbatch_calls()) - 1 == self.fail_at:
                return SimpleNamespace(returncode=1, stdout="",
                                       stderr="sbatch: error: Invalid account or partition")
            jid = str(self.next_id)
            self.next_id += 1
            return SimpleNamespace(returncode=0, stdout=f"{jid}\n", stderr="")
        jid = argv[argv.index("-j") + 1]
        if tool == "squeue":
            if jid in self.queue:
                return SimpleNamespace(returncode=0, stdout=self.queue[jid] + "\n", stderr="")
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="slurm_load_jobs error: Invalid job id specified")
        if tool == "sacct":
            return SimpleNamespace(returncode=0, stdout=self.acct.get(jid, "") + "\n", stderr="")
        raise AssertionError(f"unexpected command {argv}")


def _opt(argv, name):
    vals = [a.split("=", 1)[1] for a in argv if a.startswith(name + "=")]
    assert len(vals) <= 1, argv
    return vals[0] if vals else None


def _script_args(argv):
    i = next(i for i, a in enumerate(argv) if a.startswith("hpc/unlock/"))
    return argv[i], argv[i + 1:]


def _setup_done(root: Path, job_id="900", with_json=True):
    d = root / "research" / "data" / "hpc" / "setup"
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps({"job_id": job_id, "partition": "n2c48m24",
                                            "log": f"logs/unlock-setup-{job_id}.out"}))
    if with_json:
        (d / "setup.json").write_text("{}")


# -------------------------------------------------------------- sbatch_argv
@pytest.mark.parametrize("gpu", GPU_PARTS)
@pytest.mark.parametrize("mode", MODES)
def test_sbatch_argv_cpu_and_gpu_stage(gpu, mode):
    cfg = UnlockConfig(gpu_partition=gpu, env_mode=mode)
    fp = cfg.fingerprint()

    cpu = chain.sbatch_argv(cfg, "pose_search", "77")
    res = cfg.stage_resources("pose_search")
    assert cpu == ["sbatch", "--parsable", "--account=cs_gy_6923-2026fa",
                   "--partition=n2c48m24", f"--mem={res['mem']}",
                   f"--cpus-per-task={res['cpus']}", f"--time={res['time']}",
                   "--export=NONE", "--requeue", "--job-name=unlock-pose_search",
                   "--output=logs/unlock-pose_search-%j.out",
                   "--error=logs/unlock-pose_search-%j.out",
                   "--dependency=afterok:77",
                   "hpc/unlock/stage.sbatch", mode, "pose_search", "--run", fp]
    assert not any(a.startswith("--gres") for a in cpu)

    g = chain.sbatch_argv(cfg, "gpu_scan", None)
    gres = cfg.stage_resources("gpu_scan")
    assert _opt(g, "--partition") == gpu
    assert _opt(g, "--mem") == gres["mem"] == ("64G" if gpu == "c12m85-a100-1" else "40G")
    assert g.count("--gres=gpu:1") == 1
    assert _opt(g, "--dependency") is None
    assert "--export=NONE" in g and "--requeue" in g
    script, args = _script_args(g)
    assert script == "hpc/unlock/gpu_stage.sbatch"
    assert args == [mode, "gpu_scan", "--run", fp]


@pytest.mark.parametrize("disable", [False, True])
def test_sbatch_argv_setup(disable):
    cfg = UnlockConfig(env_mode="venv", disable_selftrain=disable)
    argv = chain.sbatch_argv(cfg, "setup", None)
    assert _opt(argv, "--partition") == "n2c48m24"
    assert _opt(argv, "--mem") == cfg.stage_resources("setup")["mem"]
    assert not any(a.startswith("--gres") for a in argv)
    script, args = _script_args(argv)
    assert script == "hpc/unlock/setup.sbatch"
    assert args == (["venv", "--no-cellpose"] if disable else ["venv"])


# --------------------------------------------------------------------- plan
def test_plan_with_and_without_selftrain():
    on = chain.plan(UnlockConfig(), "setup")
    assert on == chain.STAGES
    off = chain.plan(UnlockConfig(disable_selftrain=True), "prep")
    assert off == ["prep", "gpu_scan", "pose_search", "joint", "pairs", "verifier",
                   "validate", "assemble"]
    assert chain.plan(UnlockConfig(), "selftrain_gpu") == ["selftrain_gpu", "selftrain_pairs",
                                                           "assemble"]
    with pytest.raises(ValueError):
        chain.plan(UnlockConfig(disable_selftrain=True), "selftrain_prep")
    with pytest.raises(ValueError):
        chain.plan(UnlockConfig(), "nope")


def test_reexports_stage_constants():
    from hpc_unlock import stage
    assert chain.STAGES is stage.STAGES
    assert chain.GPU_STAGES is stage.GPU_STAGES and chain.SELFTRAIN is stage.SELFTRAIN


# ------------------------------------------------------------------- submit
@pytest.mark.parametrize("disable", [False, True])
def test_submit_dependency_chain_and_jobs_json(project_tree, capsys, disable):
    _setup_done(project_tree)
    fake = FakeSlurm(acct={"900": "COMPLETED|0:0"})
    cfg = UnlockConfig(gpu_partition="c12m85-a100-1", disable_selftrain=disable)
    rows = chain.submit(cfg, start="prep", runner=fake, root=project_tree)

    expected = chain.plan(cfg, "prep")
    assert [r[0] for r in rows] == expected
    calls = fake.sbatch_calls()
    assert len(calls) == len(expected)
    assert _opt(calls[0].argv, "--dependency") is None          # setup COMPLETED
    for prev, cur in zip(rows, calls[1:]):
        assert _opt(cur.argv, "--dependency") == f"afterok:{prev[2]}"
    for (stage, part, _), c in zip(rows, calls):
        assert _opt(c.argv, "--partition") == part
        assert part == ("c12m85-a100-1" if stage in chain.GPU_STAGES else "n2c48m24")
        assert c.cwd == str(project_tree)
    if disable:
        assert _opt(calls[-1].argv, "--dependency") == f"afterok:{dict((r[0], r[2]) for r in rows)['validate']}"

    run_dir = project_tree / "research" / "data" / "hpc" / cfg.fingerprint()
    assert UnlockConfig.load(run_dir) == cfg
    jobs = json.loads((run_dir / "jobs.json").read_text())["jobs"]
    assert [(j["stage"], j["partition"], j["job_id"]) for j in jobs] == rows
    assert jobs[0]["log"] == f"logs/unlock-prep-{rows[0][2]}.out"
    out = capsys.readouterr().out
    for stage, part, jid in rows:
        assert f"{stage:<16} {part:<16} {jid}" in out


def test_submit_removes_slurm_env(project_tree):
    _setup_done(project_tree)
    fake = FakeSlurm(acct={"900": "COMPLETED|0:0"})
    env = {"PATH": "/usr/bin", "HOME": "/h", "SLURM_JOB_ID": "5", "SLURM_CPUS_PER_TASK": "4"}
    chain.submit(UnlockConfig(), runner=fake, root=project_tree, env=env)
    assert fake.calls
    for c in fake.calls:
        assert not any(k.startswith("SLURM_") for k in c.env)
        assert c.env["PATH"] == "/usr/bin"


@pytest.mark.parametrize("fail_at", [0, 3])
def test_submit_stops_on_first_sbatch_error(project_tree, capsys, fail_at):
    _setup_done(project_tree)
    fake = FakeSlurm(fail_at=fail_at, acct={"900": "COMPLETED|0:0"})
    cfg = UnlockConfig()
    with pytest.raises(chain.SubmitError) as ei:
        chain.submit(cfg, runner=fake, root=project_tree)
    assert len(fake.sbatch_calls()) == fail_at + 1               # nothing after the failure
    assert len(ei.value.submitted) == fail_at
    assert "Invalid account or partition" in capsys.readouterr().err
    jobs_path = project_tree / "research" / "data" / "hpc" / cfg.fingerprint() / "jobs.json"
    if fail_at == 0:
        assert not jobs_path.exists()
    else:
        assert len(json.loads(jobs_path.read_text())["jobs"]) == fail_at


def test_submit_setup_records_job(project_tree, capsys):
    fake = FakeSlurm()
    cfg = UnlockConfig(disable_selftrain=True)
    row = chain.submit_setup(cfg, runner=fake, root=project_tree)
    assert row == ("setup", "n2c48m24", "1001")
    assert len(fake.sbatch_calls()) == 1
    job = json.loads((project_tree / "research/data/hpc/setup/job.json").read_text())
    assert job["job_id"] == "1001" and job["log"] == "logs/unlock-setup-1001.out"
    assert "setup            n2c48m24         1001" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["PENDING|Priority", "RUNNING|None"])
def test_submit_waits_on_running_setup(project_tree, state):
    _setup_done(project_tree, with_json=False)
    fake = FakeSlurm(queue={"900": state})
    chain.submit(UnlockConfig(), runner=fake, root=project_tree)
    assert _opt(fake.sbatch_calls()[0].argv, "--dependency") == "afterok:900"


@pytest.mark.parametrize("acct,with_json", [("FAILED|1:0", True), ("CANCELLED by 1|0:0", True),
                                            ("COMPLETED|0:0", False), ("", True)])
def test_submit_refuses_when_setup_not_ready(project_tree, capsys, acct, with_json):
    _setup_done(project_tree, with_json=with_json)
    fake = FakeSlurm(acct={"900": acct})
    with pytest.raises(chain.SetupNotReady) as ei:
        chain.submit(UnlockConfig(), runner=fake, root=project_tree)
    assert fake.sbatch_calls() == []
    assert "logs/unlock-setup-900.out" in str(ei.value)
    assert "logs/unlock-setup-900.out" in capsys.readouterr().err


def test_submit_refuses_without_setup_job(project_tree):
    fake = FakeSlurm()
    with pytest.raises(chain.SetupNotReady):
        chain.submit(UnlockConfig(), runner=fake, root=project_tree)
    assert fake.sbatch_calls() == []


def test_submit_from_setup_chains_on_setup(project_tree):
    fake = FakeSlurm()
    rows = chain.submit(UnlockConfig(), start="setup", runner=fake, root=project_tree)
    assert rows[0] == ("setup", "n2c48m24", "1001")
    assert _opt(fake.sbatch_calls()[1].argv, "--dependency") == "afterok:1001"


def test_submit_invalid_config_submits_nothing(project_tree):
    fake = FakeSlurm()
    with pytest.raises(chain.ConfigError, match="bogus"):
        chain.submit(UnlockConfig(gpu_partition="bogus"), runner=fake, root=project_tree)
    assert fake.calls == []


# ------------------------------------------------------------------ monitor
def _jobs(root: Path, cfg: UnlockConfig, rows):
    run_dir = root / "research" / "data" / "hpc" / cfg.fingerprint()
    chain._write_jobs(cfg, run_dir, rows)
    return run_dir


@pytest.mark.parametrize("acct", ["FAILED|1:0", "TIMEOUT|0:0", "CANCELLED by 123|0:0",
                                  "OUT_OF_MEMORY|0:125", "COMPLETED|2:0"])
def test_monitor_reports_failed_state(project_tree, acct):
    cfg = UnlockConfig()
    _jobs(project_tree, cfg, [("prep", "n2c48m24", "11"), ("gpu_scan", "g2-standard-12", "12"),
                              ("pose_search", "n2c48m24", "13")])
    fake = FakeSlurm(queue={"13": f"PENDING|{chain.NEVER_SATISFIED}"},
                     acct={"11": "COMPLETED|0:0", "12": acct})
    view = chain.monitor(cfg, runner=fake, root=project_tree)
    text = view.render()
    assert [j.failed for j in view.jobs] == [False, True, False]
    assert view.failures == [("gpu_scan", "logs/unlock-gpu_scan-12.out",
                              f"python run_unlock.py submit --from gpu_scan --run {cfg.fingerprint()}")]
    assert "FAILED gpu_scan" in text and "logs/unlock-gpu_scan-12.out" in text
    assert chain.resubmit_command(cfg, "gpu_scan") in text
    assert "scancel 13" in text
    # squeue is asked first, sacct only for jobs that left the queue
    tools13 = [c.argv[0] for c in fake.calls if "13" in c.argv]
    assert tools13 == ["squeue"]


def test_monitor_completed_job_is_not_failed(project_tree):
    cfg = UnlockConfig()
    _jobs(project_tree, cfg, [("prep", "n2c48m24", "11")])
    view = chain.monitor(cfg, runner=FakeSlurm(acct={"11": "COMPLETED|0:0"}), root=project_tree)
    assert view.failures == [] and view.jobs[0].state == "COMPLETED"
    assert "FAILED" not in view.render()


def test_monitor_log_tails_and_done_list(project_tree):
    cfg = UnlockConfig()
    run_dir = _jobs(project_tree, cfg, [("prep", "n2c48m24", "11"),
                                        ("gpu_scan", "g2-standard-12", "12"),
                                        ("pose_search", "n2c48m24", "13")])
    logs = project_tree / "logs"
    (logs / "unlock-prep-11.out").write_text("".join(f"line{i}\n" for i in range(100)))
    (logs / "unlock-gpu_scan-12.out").write_text("a\nb\nc\n")
    (run_dir / "prep.done").write_text("{}")
    (run_dir / "gpu_scan.done").write_text("{}")
    fake = FakeSlurm(queue={"12": "RUNNING|None", "13": "PENDING|Dependency"},
                     acct={"11": "COMPLETED|0:0"})
    view = chain.monitor(cfg, runner=fake, root=project_tree)
    prep, gpu, pose = view.jobs
    assert prep.log_tail == [f"line{i}" for i in range(60, 100)]
    assert gpu.log_tail == ["a", "b", "c"]
    assert pose.log_tail is None
    assert view.done == ["gpu_scan.done", "prep.done"]
    text = view.render()
    assert "logs/unlock-pose_search-13.out: log not created yet" in text
    assert "line59" not in text and "line60" in text
    assert "Done-markers: gpu_scan.done, prep.done" in text
    assert view.never_satisfied == []


def test_monitor_includes_setup_job(project_tree):
    _setup_done(project_tree, job_id="900")
    cfg = UnlockConfig()
    view = chain.monitor(cfg, runner=FakeSlurm(acct={"900": "FAILED|1:0"}), root=project_tree)
    assert view.jobs[0].stage == "setup"
    assert view.failures[0][2] == f"python run_unlock.py submit --setup --run {cfg.fingerprint()}"
    assert "jobs.json" in view.render()                          # no chain jobs yet


def test_monitor_baseline_failure_hint(project_tree):
    cfg = UnlockConfig()
    _jobs(project_tree, cfg, [("validate", "n2c48m24", "21"), ("assemble", "n2c48m24", "22")])
    (project_tree / "logs" / "unlock-validate-21.out").write_text(
        "BASELINE_REPRODUCTION_FAILED f1=0.401 full=0.48\n")
    fake = FakeSlurm(queue={"22": f"PENDING|{chain.NEVER_SATISFIED}"},
                     acct={"21": "FAILED|1:0"})
    text = chain.monitor(cfg, runner=fake, root=project_tree).render()
    assert "BASELINE_REPRODUCTION_FAILED" in text
    assert "--report-only" in text
    assert "scancel 22" in text


# ---------------------------------------------------------------------- main
def test_main_submit_and_monitor(project_tree, capsys):
    _setup_done(project_tree)
    cfg = UnlockConfig(disable_selftrain=True)
    cfg.save(project_tree / "research" / "data" / "hpc" / cfg.fingerprint())
    fake = FakeSlurm(acct={"900": "COMPLETED|0:0"})
    fp = cfg.fingerprint()
    assert chain.main(["submit", "--run", fp, "--from", "validate"],
                      runner=fake, root=project_tree) == 0
    assert [_script_args(c.argv)[1][1] for c in fake.sbatch_calls()] == ["validate", "assemble"]
    assert chain.main(["monitor", "--run", fp], runner=fake, root=project_tree) == 0
    assert "validate" in capsys.readouterr().out

    failing = FakeSlurm(fail_at=0, acct={"900": "COMPLETED|0:0"})
    assert chain.main(["submit", "--run", fp], runner=failing, root=project_tree) == 1
    assert chain.main(["submit", "--from", "nope"], runner=failing, root=project_tree) == 2
    assert chain.main(["submit", "--run", "0000000000"], runner=failing, root=project_tree) == 1
