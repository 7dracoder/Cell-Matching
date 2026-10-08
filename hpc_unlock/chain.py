"""SLURM Job_Chain: Stage plan, sbatch commands, submit and monitor (Req 2, 3, 12.1-12.2).

Notebook usage (plain Python kernel, standard library only)::

    from hpc_unlock import chain
    chain.submit_setup(cfg)            # Step 2: setup alone (it mounts the overlay :rw)
    chain.submit(cfg, start="prep")    # Step 3: the chain, gated on the setup job
    print(chain.monitor(cfg))          # Step 4: states, log tails, done-markers, failures

Command line (through ``run_unlock.py``)::

    python run_unlock.py submit [--run FP] [--from STAGE] [--setup]
    python run_unlock.py monitor [--run FP]

Every subprocess call goes through an injectable ``runner`` (default
``subprocess.run``) so the tests can fake ``sbatch`` / ``squeue`` / ``sacct``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from . import paths
from .config import UnlockConfig
from .stage import GPU_STAGES, NOT_DRIVEN, SELFTRAIN, STAGES

__all__ = ["STAGES", "GPU_STAGES", "SELFTRAIN", "NOT_DRIVEN", "ChainError", "ConfigError",
           "SetupNotReady", "SubmitError", "JobStatus", "MonitorView", "plan", "partition",
           "log_path", "sbatch_argv", "resubmit_command", "report_only_command",
           "setup_state", "submit_setup", "submit", "monitor", "main"]

Runner = Callable[..., Any]

SETUP_SCRIPT = "hpc/unlock/setup.sbatch"
CPU_SCRIPT = "hpc/unlock/stage.sbatch"
GPU_SCRIPT = "hpc/unlock/gpu_stage.sbatch"
JOBS_NAME = "jobs.json"
SETUP_DIR = "setup"
SETUP_JOB_NAME = "job.json"
SETUP_JSON_NAME = "setup.json"
VALIDATE_FAILURE_NAME = "validate_failure.json"
BASELINE_FAILED_TAG = "BASELINE_REPRODUCTION_FAILED"
NEVER_SATISFIED = "DependencyNeverSatisfied"
TAIL_LINES = 40

# squeue / sacct states of a job that has not finished yet (afterok still pending).
ACTIVE_STATES = frozenset({"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED",
                           "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SUSPENDED",
                           "SIGNALING", "STAGE_OUT"})
FAILED_STATES = frozenset({"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                           "BOOT_FAIL", "DEADLINE"})


class ChainError(RuntimeError):
    """Base class for refusals and submission errors."""


class ConfigError(ChainError):
    """The config is invalid; nothing was submitted (Req 2.3)."""


class SetupNotReady(ChainError):
    """The setup job is missing, failed, or finished without ``setup.json``."""


class SubmitError(ChainError):
    """An ``sbatch`` call failed; later Stages were not submitted (Req 2.9)."""

    def __init__(self, msg: str, submitted: list[tuple[str, str, str]] | None = None,
                 stderr: str = ""):
        super().__init__(msg)
        self.submitted = list(submitted or [])
        self.stderr = stderr


# ----------------------------------------------------------------- locations
def _root(root: str | Path | None) -> Path:
    return paths.ROOT if root is None else Path(root)


def _hpc(root: str | Path | None) -> Path:
    return paths.HPC if root is None else Path(root) / "research" / "data" / "hpc"


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------- plan
def plan(cfg: UnlockConfig, start: str = "setup") -> list[str]:
    """Stages from ``start`` to ``assemble`` in chain order.

    Self-training Stages are dropped when ``cfg.disable_selftrain`` is set, so
    ``assemble`` then follows ``validate`` (Req 12.2).
    """
    if start not in STAGES:
        raise ValueError(f"unknown Stage {start!r}; expected one of {', '.join(STAGES)}")
    if cfg.disable_selftrain and start in SELFTRAIN:
        raise ValueError(f"Stage {start!r} is a self-training Stage, but self-training "
                         f"is disabled in the config")
    stages = STAGES[STAGES.index(start):]
    if cfg.disable_selftrain:
        stages = [s for s in stages if s not in SELFTRAIN]
    return stages


def partition(cfg: UnlockConfig, stage: str) -> str:
    """GPU Stages on the configured GPU partition, every other Stage on the CPU one."""
    return cfg.gpu_partition if stage in GPU_STAGES else cfg.cpu_partition


def log_path(stage: str, job_id: str | None = None) -> str:
    """Stage log, relative to the submit directory (``%j`` until the job ID is known)."""
    return f"logs/unlock-{stage}-{job_id if job_id is not None else '%j'}.out"


def sbatch_argv(cfg: UnlockConfig, stage: str, dep: str | None) -> list[str]:
    """The full ``sbatch`` command for one Stage (pure).

    ``--export=NONE`` means the env mode and run fingerprint travel as script
    arguments; the env mode is always the first one.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown Stage {stage!r}")
    res = cfg.stage_resources(stage)
    out = log_path(stage)
    argv = ["sbatch", "--parsable",
            f"--account={cfg.account}",
            f"--partition={partition(cfg, stage)}",
            f"--mem={res['mem']}",
            f"--cpus-per-task={res['cpus']}",
            f"--time={res['time']}",
            "--export=NONE",
            "--requeue",
            f"--job-name=unlock-{stage}",
            f"--output={out}",
            f"--error={out}"]
    if stage in GPU_STAGES:
        argv.append("--gres=gpu:1")
    if dep:
        argv.append(f"--dependency=afterok:{dep}")
    if stage == "setup":
        argv += [SETUP_SCRIPT, cfg.env_mode]
        if cfg.disable_selftrain:
            argv.append("--no-cellpose")
    else:
        script = GPU_SCRIPT if stage in GPU_STAGES else CPU_SCRIPT
        argv += [script, cfg.env_mode, stage, "--run", cfg.fingerprint()]
    return argv


def resubmit_command(cfg: UnlockConfig, stage: str) -> str:
    """Command that resubmits the Job_Chain from ``stage`` (Req 2.8)."""
    if stage == "setup":
        return f"python run_unlock.py submit --setup --run {cfg.fingerprint()}"
    return f"python run_unlock.py submit --from {stage} --run {cfg.fingerprint()}"


def report_only_command(cfg: UnlockConfig) -> str:
    """sbatch command for ``assemble --report-only`` after a Baseline reproduction failure."""
    return " ".join(sbatch_argv(cfg, "assemble", None) + ["--report-only"])


# ------------------------------------------------------------- subprocesses
def _clean_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment without ``SLURM_*`` (a notebook inside a job must not leak them)."""
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if not k.startswith("SLURM_")}


def _run(runner: Runner, argv: list[str], root: Path, env: Mapping[str, str] | None):
    return runner(argv, capture_output=True, text=True, cwd=str(root),
                  env=_clean_env(env), check=False)


def _query(runner: Runner, argv: list[str], root: Path,
           env: Mapping[str, str] | None) -> str | None:
    """stdout of a read-only query, or ``None`` on error / missing tool."""
    try:
        r = _run(runner, argv, root, env)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout or ""


def _last_line(text: str | None) -> str | None:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _job_state(runner: Runner, job_id: str, root: Path,
               env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{"state", "reason", "exit_code", "source"}``: ``squeue`` first, then ``sacct``."""
    line = _last_line(_query(runner, ["squeue", "-h", "-j", str(job_id), "-o", "%T|%r"],
                             root, env))
    if line:
        state, _, reason = line.partition("|")
        return {"state": state.strip().upper(), "reason": reason.strip(),
                "exit_code": "", "source": "squeue"}
    line = _last_line(_query(runner, ["sacct", "-n", "-X", "-P", "-j", str(job_id),
                                      "-o", "State,ExitCode"], root, env))
    if line:
        parts = [p.strip() for p in line.split("|")]
        state = parts[0].upper() if parts else ""
        exit_code = parts[1] if len(parts) > 1 else ""
        return {"state": state, "reason": "", "exit_code": exit_code, "source": "sacct"}
    return {"state": "UNKNOWN", "reason": "", "exit_code": "", "source": "none"}


def _exit_nonzero(exit_code: str) -> bool:
    """``sacct`` ExitCode ``<code>:<signal>``; non-zero if either part is."""
    if not exit_code:
        return False
    for part in exit_code.split(":"):
        part = part.strip()
        if part and part != "0":
            return True
    return False


def is_failed(state: str, exit_code: str = "") -> bool:
    """FAILED, TIMEOUT, CANCELLED*, OUT_OF_MEMORY (and node/boot failures) or non-zero exit."""
    s = (state or "").upper()
    if s.startswith("CANCELLED") or s in FAILED_STATES:
        return True
    return _exit_nonzero(exit_code)


# ---------------------------------------------------------------- setup job
def _setup_dir(root: str | Path | None) -> Path:
    return _hpc(root) / SETUP_DIR


def setup_state(cfg: UnlockConfig | None = None, runner: Runner = subprocess.run,
                root: str | Path | None = None,
                env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The recorded setup job and its state.

    ``{"job": job.json or None, "state", "exit_code", "setup_json": bool,
    "log": path or None, "dependency": job ID / None, "ready": bool, "message"}``.
    ``ready`` means the chain may be submitted; ``dependency`` is the setup job
    ID to wait on (``afterok``), or ``None`` once setup has completed.
    """
    rt = _root(root)
    sdir = _setup_dir(root)
    job = _read_json(sdir / SETUP_JOB_NAME)
    has_json = (sdir / SETUP_JSON_NAME).is_file()
    out: dict[str, Any] = {"job": job, "state": None, "exit_code": "",
                           "setup_json": has_json, "log": None, "dependency": None,
                           "ready": False, "message": ""}
    if not isinstance(job, dict) or not job.get("job_id"):
        out["message"] = (f"no setup job recorded in {_rel(sdir / SETUP_JOB_NAME, rt)}; "
                          f"submit setup first (notebook Step 2, chain.submit_setup(cfg))")
        return out
    jid = str(job["job_id"])
    log = job.get("log") or log_path("setup", jid)
    out["log"] = log
    st = _job_state(runner, jid, rt, env)
    out["state"], out["exit_code"] = st["state"], st["exit_code"]
    if st["state"] in ACTIVE_STATES and not _exit_nonzero(st["exit_code"]):
        out.update(ready=True, dependency=jid,
                   message=f"setup job {jid} is {st['state']}; the chain waits with "
                           f"--dependency=afterok:{jid}")
    elif st["state"] == "COMPLETED" and not _exit_nonzero(st["exit_code"]) and has_json:
        out.update(ready=True, message=f"setup job {jid} COMPLETED; setup.json present")
    elif st["state"] == "COMPLETED" and not _exit_nonzero(st["exit_code"]):
        out["message"] = (f"setup job {jid} COMPLETED but "
                          f"{_rel(sdir / SETUP_JSON_NAME, rt)} is missing; "
                          f"see the setup log {log}")
    else:
        code = f" exit {st['exit_code']}" if st["exit_code"] else ""
        out["message"] = (f"setup job {jid} is {st['state']}{code}; the chain is not "
                          f"submitted. See the setup log {log} and resubmit setup "
                          f"({resubmit_command(cfg, 'setup') if cfg else 'submit --setup'})")
    return out


# -------------------------------------------------------------------- submit
def _validate(cfg: UnlockConfig) -> None:
    errors = list(cfg.validate())
    if errors:
        for e in errors:
            print(f"CONFIG_INVALID {e}", file=sys.stderr, flush=True)
        raise ConfigError("invalid config, nothing submitted: " + "; ".join(errors))


def _sbatch(cfg: UnlockConfig, stage: str, dep: str | None, runner: Runner, root: Path,
            env: Mapping[str, str] | None,
            submitted: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    argv = sbatch_argv(cfg, stage, dep)
    try:
        r = _run(runner, argv, root, env)
    except (OSError, subprocess.SubprocessError) as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"SBATCH_FAILED {stage}: {msg}", file=sys.stderr, flush=True)
        raise SubmitError(f"sbatch failed for Stage {stage!r}: {msg}", submitted, msg) from e
    stderr = (r.stderr or "").strip()
    job_id = (r.stdout or "").strip().split(";")[0].strip()
    if r.returncode != 0 or not job_id:
        print(f"SBATCH_FAILED {stage} (exit {r.returncode}); later Stages not submitted",
              file=sys.stderr, flush=True)
        if stderr:
            print(stderr, file=sys.stderr, flush=True)
        raise SubmitError(f"sbatch failed for Stage {stage!r} (exit {r.returncode}): "
                          f"{stderr or 'no job ID returned'}", submitted, stderr)
    row = (stage, partition(cfg, stage), job_id)
    print(f"{stage:<16} {row[1]:<16} {job_id}", flush=True)
    return row


def submit_setup(cfg: UnlockConfig, runner: Runner = subprocess.run,
                 root: str | Path | None = None,
                 env: Mapping[str, str] | None = None) -> tuple[str, str, str]:
    """Submit only the setup Stage (the single ``:rw`` overlay writer); record its job ID."""
    _validate(cfg)
    rt = _root(root)
    (rt / "logs").mkdir(parents=True, exist_ok=True)
    row = _sbatch(cfg, "setup", None, runner, rt, env, [])
    _write_json_atomic(_setup_dir(root) / SETUP_JOB_NAME,
                       {"job_id": row[2], "partition": row[1], "log": log_path("setup", row[2]),
                        "env_mode": cfg.env_mode, "disable_selftrain": cfg.disable_selftrain,
                        "submitted": _now()})
    return row


def _write_jobs(cfg: UnlockConfig, run_dir: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Merge the new submissions into ``jobs.json`` (older entries for other Stages kept)."""
    path = run_dir / JOBS_NAME
    old = _read_json(path)
    jobs = {j["stage"]: j for j in (old or {}).get("jobs", []) if isinstance(j, dict)
            and j.get("stage") in STAGES}
    for stage, part, jid in rows:
        jobs[stage] = {"stage": stage, "partition": part, "job_id": jid,
                       "log": log_path(stage, jid)}
    ordered = sorted(jobs.values(), key=lambda j: STAGES.index(j["stage"]))
    _write_json_atomic(path, {"fingerprint": cfg.fingerprint(), "updated": _now(),
                              "jobs": ordered})
    return path


def submit(cfg: UnlockConfig, start: str = "prep", runner: Runner = subprocess.run,
           root: str | Path | None = None,
           env: Mapping[str, str] | None = None) -> list[tuple[str, str, str]]:
    """Submit the chain from ``start``; return ``[(stage, partition, job_id), ...]``.

    The first Stage waits on the setup job while it is pending or running and
    has no dependency once setup has COMPLETED with ``setup.json``; any other
    setup state refuses (``SetupNotReady``). ``start="setup"`` submits setup
    first and chains the rest on it. Each later Stage depends ``afterok`` on
    the previous one. The first ``sbatch`` error stops the submission
    (``SubmitError``); ``jobs.json`` still records what was submitted.
    """
    _validate(cfg)
    stages = plan(cfg, start)
    rt = _root(root)
    run_dir = cfg.run_dir(_hpc(root))
    rows: list[tuple[str, str, str]] = []

    if stages[0] == "setup":
        rows.append(submit_setup(cfg, runner=runner, root=root, env=env))
        dep: str | None = rows[0][2]
        stages = stages[1:]
    else:
        st = setup_state(cfg, runner=runner, root=root, env=env)
        if not st["ready"]:
            print(f"SETUP_NOT_READY {st['message']}", file=sys.stderr, flush=True)
            raise SetupNotReady(st["message"])
        dep = st["dependency"]

    cfg.save(run_dir)
    (rt / "logs").mkdir(parents=True, exist_ok=True)
    chain_rows: list[tuple[str, str, str]] = []
    try:
        for stage in stages:
            row = _sbatch(cfg, stage, dep, runner, rt, env, rows + chain_rows)
            chain_rows.append(row)
            dep = row[2]
    finally:
        if chain_rows:
            _write_jobs(cfg, run_dir, chain_rows)
    return rows + chain_rows


# ------------------------------------------------------------------- monitor
@dataclass
class JobStatus:
    stage: str
    job_id: str
    partition: str
    state: str
    exit_code: str
    reason: str
    source: str
    log: str
    log_tail: list[str] | None          # None = log not created yet
    failed: bool


@dataclass(repr=False)
class MonitorView:
    fingerprint: str
    run_dir: str
    jobs: list[JobStatus] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # stage, log, cmd
    never_satisfied: list[tuple[str, str]] = field(default_factory=list)  # stage, job id
    hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        out = [f"Run {self.fingerprint}  ({self.run_dir})"]
        out += self.notes
        if self.jobs:
            out.append(f"{'stage':<16} {'job_id':<10} {'partition':<16} state")
            for j in self.jobs:
                extra = f" exit {j.exit_code}" if j.exit_code else ""
                extra += f" ({j.reason})" if j.reason and j.reason not in ("None", "(null)") else ""
                out.append(f"{j.stage:<16} {j.job_id:<10} {j.partition:<16} {j.state}{extra}")
        out.append("Done-markers: " + (", ".join(self.done) if self.done else "(none)"))
        for j in self.jobs:
            if j.log_tail is None:
                out.append(f"--- {j.stage} [{j.job_id}] {j.log}: log not created yet")
            else:
                out.append(f"--- {j.stage} [{j.job_id}] {j.log} (last {TAIL_LINES} lines)")
                out += j.log_tail
        for stage, log, cmd in self.failures:
            out.append(f"FAILED {stage}: log {log}; resubmit from this Stage with: {cmd}")
        if self.never_satisfied:
            names = ", ".join(f"{s} [{jid}]" for s, jid in self.never_satisfied)
            ids = " ".join(jid for _, jid in self.never_satisfied)
            out.append(f"Pending with {NEVER_SATISFIED}: {names}. They will never start; "
                       f"cancel them before resubmitting: scancel {ids}")
        out += self.hints
        return "\n".join(out)

    __str__ = render
    __repr__ = render


def _tail(path: Path, n: int = TAIL_LINES) -> list[str] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [ln.rstrip("\n") for ln in deque(fh, maxlen=n)]
    except OSError:
        return None


def _log_has(path: Path, needle: str) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return any(needle in ln for ln in fh)
    except OSError:
        return False


def monitor(cfg: UnlockConfig, runner: Runner = subprocess.run,
            root: str | Path | None = None,
            env: Mapping[str, str] | None = None) -> MonitorView:
    """State of every recorded job, log tails, done-markers and failure advice (Req 2.6, 2.8)."""
    rt = _root(root)
    run_dir = cfg.run_dir(_hpc(root))
    view = MonitorView(fingerprint=cfg.fingerprint(), run_dir=_rel(run_dir, rt))

    entries: list[dict] = []
    sjob = _read_json(_setup_dir(root) / SETUP_JOB_NAME)
    if isinstance(sjob, dict) and sjob.get("job_id"):
        entries.append({"stage": "setup", "job_id": str(sjob["job_id"]),
                        "partition": sjob.get("partition", cfg.cpu_partition),
                        "log": sjob.get("log") or log_path("setup", sjob["job_id"])})
    jobs = _read_json(run_dir / JOBS_NAME)
    if isinstance(jobs, dict):
        entries += [j for j in jobs.get("jobs", []) if isinstance(j, dict) and j.get("job_id")]
    else:
        view.notes.append(f"No chain jobs recorded ({_rel(run_dir / JOBS_NAME, rt)} missing); "
                          f"run chain.submit(cfg) first.")

    for e in entries:
        stage, jid = str(e["stage"]), str(e["job_id"])
        log = str(e.get("log") or log_path(stage, jid))
        st = _job_state(runner, jid, rt, env)
        failed = is_failed(st["state"], st["exit_code"])
        view.jobs.append(JobStatus(stage=stage, job_id=jid, partition=str(e.get("partition", "")),
                                   state=st["state"], exit_code=st["exit_code"],
                                   reason=st["reason"], source=st["source"], log=log,
                                   log_tail=_tail(rt / log), failed=failed))
        if failed:
            view.failures.append((stage, log, resubmit_command(cfg, stage)))
            if stage == "validate" and ((run_dir / VALIDATE_FAILURE_NAME).is_file()
                                        or _log_has(rt / log, BASELINE_FAILED_TAG)):
                view.hints.append(
                    f"{BASELINE_FAILED_TAG}: the Validator did not reproduce the Baseline "
                    f"(see {_rel(run_dir / VALIDATE_FAILURE_NAME, rt)}); no CSV will be "
                    f"written. Write the Run_Report with assemble --report-only: "
                    f"{report_only_command(cfg)}")
        elif NEVER_SATISFIED in st["reason"]:
            view.never_satisfied.append((stage, jid))

    if run_dir.is_dir():
        view.done = sorted(p.name for p in run_dir.glob("*.done"))
    return view


# ---------------------------------------------------------------------- CLI
def _cfg_for(run: str | None, root: str | Path | None) -> UnlockConfig:
    if not run:
        return UnlockConfig()
    return UnlockConfig.load(_hpc(root) / run)


def main(argv: list[str] | None = None, runner: Runner = subprocess.run,
         root: str | Path | None = None) -> int:
    """``submit [--run FP] [--from STAGE] [--setup]`` | ``monitor [--run FP]``; exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("submit", "monitor"):
        print("usage: chain {submit [--run FP] [--from STAGE] [--setup] | monitor [--run FP]}",
              file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"run_unlock.py {cmd}")
    ap.add_argument("--run", default=None, metavar="FP")
    if cmd == "submit":
        ap.add_argument("--from", dest="start", default="prep", metavar="STAGE")
        ap.add_argument("--setup", action="store_true", help="submit only the setup Stage")
    args = ap.parse_args(rest)
    try:
        cfg = _cfg_for(args.run, root)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"RUN_CONFIG_ERROR cannot load run {args.run!r}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    if cmd == "monitor":
        print(monitor(cfg, runner=runner, root=root).render())
        return 0
    if not args.setup and args.start not in STAGES:
        print(f"UNKNOWN_STAGE {args.start!r}; expected one of {', '.join(STAGES)}",
              file=sys.stderr)
        return 2
    try:
        if args.setup:
            submit_setup(cfg, runner=runner, root=root)
        else:
            submit(cfg, start=args.start, runner=runner, root=root)
    except ValueError as e:                     # plan(): self-training Stage while disabled
        print(f"INVALID_START {e}", file=sys.stderr)
        return 2
    except ChainError as e:                     # details already printed
        print(f"SUBMIT_FAILED {e}", file=sys.stderr)
        return 1
    return 0
