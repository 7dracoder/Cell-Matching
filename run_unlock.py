#!/usr/bin/env python
"""Driver for the hpc-registration-unlock Job_Chain.

Usage::

    python run_unlock.py <stage> --run <fp>          # one Stage, inside a SLURM job
    python run_unlock.py <stage> --smoke [--run <fp>]  # one Stage, local smoke dir
    python run_unlock.py assemble --run <fp> --report-only  # Run_Report only, no CSV
    python run_unlock.py check [--env-mode M] [--sif-name N] [--root R] [--run <fp>]
    python run_unlock.py submit [--run <fp>] [--from <stage>] [--setup]
    python run_unlock.py monitor [--run <fp>]
    python run_unlock.py smoke

A Stage is the module ``hpc_unlock.<stage>`` (see ``hpc_unlock/stage.py`` for
the contract). It runs through ``checkpoint.run_stage`` in
``research/data/hpc/<fp>/`` (``research/data/hpc_smoke/<fp>/`` with
``--smoke``), using the ``run_config.json`` stored there.

Exit codes: 0 ok, 1 Stage failed / invalid config / missing inputs,
2 usage error, unknown or unavailable Stage, or a full-scale Stage started
outside a SLURM allocation (Req 13.6).
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hpc_unlock import checkpoint, paths, stage  # noqa: E402
from hpc_unlock.config import RUN_CONFIG_NAME, UnlockConfig  # noqa: E402

COMMANDS = ("check", "submit", "monitor", "smoke")
REPORT_ONLY_STAGE = "assemble"
log = checkpoint.log_line


def _err(tag: str, detail: str) -> None:
    print(f"{tag} {detail}".rstrip(), file=sys.stderr, flush=True)


# ------------------------------------------------------------------ config
class RunConfigError(RuntimeError):
    """``run_config.json`` missing, unreadable or not matching ``--run``."""


def resolve_run(run: str | None, smoke: bool) -> tuple[UnlockConfig, Path]:
    """``(cfg, run_dir)`` for a Stage run.

    Full-scale: ``paths.HPC/<run>/run_config.json`` must exist. Smoke:
    ``paths.HPC_SMOKE/<fp>``; without ``--run`` the default config is used,
    and a full-run config is copied (read-only use) into the smoke dir.
    Raises ``RunConfigError`` with a readable message.
    """
    base = paths.HPC_SMOKE if smoke else paths.HPC
    if run is None:
        if not smoke:
            raise RunConfigError("--run <fingerprint> is required for a full-scale Stage")
        cfg = UnlockConfig()
        run_dir = base / cfg.fingerprint()
        if (run_dir / RUN_CONFIG_NAME).is_file():
            cfg = _load(run_dir)
        else:
            cfg.save(run_dir)
        return cfg, run_dir

    run_dir = base / run
    src = run_dir
    if smoke and not (run_dir / RUN_CONFIG_NAME).is_file() \
            and (paths.HPC / run / RUN_CONFIG_NAME).is_file():
        src = paths.HPC / run                       # read only; never written
    if not (src / RUN_CONFIG_NAME).is_file():
        raise RunConfigError(f"no {RUN_CONFIG_NAME} in {run_dir}; submit the chain "
                             f"from the HPC_Notebook so it writes one")
    cfg = _load(src)
    if cfg.fingerprint() != run:
        raise RunConfigError(f"{src / RUN_CONFIG_NAME}: config fingerprint "
                             f"{cfg.fingerprint()!r} does not match --run {run!r}")
    if src != run_dir:
        cfg.save(run_dir)
    return cfg, run_dir


def _load(run_dir: Path) -> UnlockConfig:
    try:
        return UnlockConfig.load(run_dir)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RunConfigError(f"cannot load {run_dir / RUN_CONFIG_NAME}: "
                             f"{type(e).__name__}: {e}") from e


# ------------------------------------------------------------------ stages
def run_stage_cmd(argv: list[str]) -> int:
    name = argv[0]
    ap = argparse.ArgumentParser(prog=f"run_unlock.py {name}")
    ap.add_argument("--run", default=None, metavar="FP",
                    help="run fingerprint (directory under research/data/hpc)")
    ap.add_argument("--smoke", action="store_true",
                    help="local smoke run, outputs under research/data/hpc_smoke")
    ap.add_argument("--report-only", action="store_true",
                    help="assemble only: write the Run_Report after a Baseline "
                         "reproduction failure, no CSV, no done-marker")
    args = ap.parse_args(argv[1:])

    if args.report_only and name != REPORT_ONLY_STAGE:
        _err("USAGE", f"--report-only applies only to the {REPORT_ONLY_STAGE} Stage, "
                      f"not {name!r}")
        return stage.EXIT_USAGE
    if name not in stage.STAGES:
        _err("UNKNOWN_STAGE", f"{name!r}; expected one of {', '.join(stage.STAGES)} "
                              f"or a command ({', '.join(COMMANDS)})")
        return stage.EXIT_USAGE
    if name in stage.NOT_DRIVEN:
        _err("NOT_A_DRIVER_STAGE", f"{name}: runs as hpc/unlock/setup.sbatch "
                                   f"(notebook Step 2 / 'run_unlock.py submit --setup'), "
                                   f"not through run_unlock.py")
        return stage.EXIT_USAGE
    if not args.smoke and not stage.in_slurm():
        _err("OUTSIDE_SLURM", f"{name}: {stage.OUTSIDE_SLURM_MSG}")
        return stage.EXIT_USAGE

    try:
        cfg, run_dir = resolve_run(args.run, args.smoke)
    except RunConfigError as e:
        _err("RUN_CONFIG_ERROR", f"{name}: {e}")
        return stage.EXIT_FAILED if args.run else stage.EXIT_USAGE
    options = {"report_only": True} if args.report_only else {}
    return stage.execute(name, cfg, run_dir, smoke=args.smoke, options=options)


# ---------------------------------------------------------------- commands
def check_cmd(argv: list[str]) -> int:
    """Input check (Req 1.6): lists every missing/unreadable path, exit 1 if any."""
    ap = argparse.ArgumentParser(prog="run_unlock.py check")
    ap.add_argument("--env-mode", default=None, choices=("singularity", "venv"))
    ap.add_argument("--sif-name", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run", default=None, metavar="FP",
                    help="take env mode and .sif name from this run's config")
    args = ap.parse_args(argv)
    env_mode, sif = args.env_mode, args.sif_name
    if args.run:
        try:
            cfg = _load(paths.HPC / args.run)
        except RunConfigError as e:
            _err("RUN_CONFIG_ERROR", str(e))
            return stage.EXIT_FAILED
        env_mode = env_mode or cfg.env_mode
        sif = sif or cfg.sif_name
    from hpc_unlock import inputs
    fwd = ["check", "--env-mode", env_mode or "singularity"]
    if sif:
        fwd += ["--sif-name", sif]
    if args.root:
        fwd += ["--root", args.root]
    return inputs.main(fwd)


def _import_optional(modname: str, task: str):
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name == modname:
            _err("NOT_AVAILABLE", f"{modname} is not implemented yet ({task}); "
                                  f"use the HPC_Notebook cells instead")
            return None
        raise


def _chain_cfg(run: str | None) -> UnlockConfig:
    return _load(paths.HPC / run) if run else UnlockConfig()


def submit_cmd(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="run_unlock.py submit")
    ap.add_argument("--run", default=None, metavar="FP")
    ap.add_argument("--from", dest="start", default="prep", metavar="STAGE")
    ap.add_argument("--setup", action="store_true", help="submit only the setup Stage")
    args = ap.parse_args(argv)
    chain = _import_optional("hpc_unlock.chain", "task 3.3")
    if chain is None:
        return stage.EXIT_FAILED
    if hasattr(chain, "main"):
        return int(chain.main(["submit", *argv]) or 0)
    try:
        cfg = _chain_cfg(args.run)
    except RunConfigError as e:
        _err("RUN_CONFIG_ERROR", str(e))
        return stage.EXIT_FAILED
    errors = cfg.validate()
    if errors:
        for err in errors:
            _err("CONFIG_INVALID", err)
        return stage.EXIT_FAILED
    if args.start not in stage.STAGES:
        _err("UNKNOWN_STAGE", repr(args.start))
        return stage.EXIT_USAGE
    try:
        if args.setup:
            rows = [chain.submit_setup(cfg)]
        else:
            rows = chain.submit(cfg, start=args.start)
    except Exception as e:  # noqa: BLE001 - sbatch error already shown by chain
        _err("SUBMIT_FAILED", f"{type(e).__name__}: {e}")
        return stage.EXIT_FAILED
    for row in rows or []:
        print("SUBMITTED " + " ".join(str(x) for x in row))
    return stage.EXIT_OK


def monitor_cmd(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="run_unlock.py monitor")
    ap.add_argument("--run", default=None, metavar="FP")
    args = ap.parse_args(argv)
    chain = _import_optional("hpc_unlock.chain", "task 3.3")
    if chain is None:
        return stage.EXIT_FAILED
    if hasattr(chain, "main"):
        return int(chain.main(["monitor", *argv]) or 0)
    try:
        view = chain.monitor(_chain_cfg(args.run))
    except RunConfigError as e:
        _err("RUN_CONFIG_ERROR", str(e))
        return stage.EXIT_FAILED
    render = getattr(view, "render", None)
    print(render() if callable(render) else view)
    return stage.EXIT_OK


def smoke_cmd(argv: list[str]) -> int:
    smoke = _import_optional("hpc_unlock.smoke", "task 14.1")
    if smoke is None:
        return stage.EXIT_FAILED
    return int((smoke.main(argv) if argv else smoke.main()) or 0)


HANDLERS = {"check": check_cmd, "submit": submit_cmd,
            "monitor": monitor_cmd, "smoke": smoke_cmd}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Stages: " + ", ".join(s for s in stage.STAGES if s not in stage.NOT_DRIVEN))
        return stage.EXIT_OK if argv else stage.EXIT_USAGE
    cmd = argv[0]
    if cmd in HANDLERS:
        return HANDLERS[cmd](argv[1:])
    return run_stage_cmd(argv)


if __name__ == "__main__":
    sys.exit(main())
