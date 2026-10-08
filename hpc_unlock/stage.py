"""Stage names and the Stage contract shared by run_unlock.py, chain.py and smoke.py.

Every Stage module ``hpc_unlock.<name>`` implements::

    def compute(cfg: UnlockConfig, ctx: StageContext) -> object
        # returns the Checkpoint object; run_unlock.py saves it via checkpoint.run_stage

and may define:

    NEEDS_GPU = True                    # GPU Stage; it checks GPU visibility itself
    def validate_config(cfg) -> list[str]   # Stage-specific config errors (all of them)

``ctx.load(dep)`` returns the Checkpoint of an earlier Stage from the same run
directory, and fails unless that Stage has a valid done-marker.

Imports only the standard library, so the notebook and chain.py can use it.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from . import checkpoint, paths

STAGES = ["setup", "prep", "gpu_scan", "pose_search", "joint", "pairs", "verifier",
          "validate", "selftrain_prep", "selftrain_gpu", "selftrain_pairs", "assemble"]
GPU_STAGES = frozenset({"gpu_scan", "selftrain_gpu"})
SELFTRAIN = frozenset({"selftrain_prep", "selftrain_gpu", "selftrain_pairs"})

# Stages that run_unlock.py does not compute (setup runs in setup.sbatch).
NOT_DRIVEN = frozenset({"setup"})

OUTSIDE_SLURM_MSG = ("start this Stage through the Job_Chain or the HPC_Notebook "
                     "(no SLURM_JOB_ID: full-scale Stages run only inside a SLURM "
                     "job allocation; use --smoke for a local smoke run)")

SMOKE_MAX_WORKERS = 4

# Exit codes
EXIT_OK = 0
EXIT_FAILED = 1     # Stage failed / invalid config / missing dependency
EXIT_USAGE = 2      # usage error, outside SLURM, unknown or unavailable Stage


class DependencyError(RuntimeError):
    """An earlier Stage's Checkpoint is missing, unmarked or fails to load."""


class StageUnavailable(ImportError):
    """The Stage module does not exist (not written yet) or has no ``compute``."""


@dataclass
class StageContext:
    """What a Stage's ``compute(cfg, ctx)`` gets besides the config."""
    name: str
    run_dir: Path
    smoke: bool
    workers: int
    options: dict = field(default_factory=dict)   # e.g. {"report_only": True} (assemble)
    _cache: dict = field(default_factory=dict, repr=False)

    def load(self, dep_name: str) -> Any:
        """Checkpoint object of the earlier Stage ``dep_name`` in this run directory.

        Requires the done-marker, a matching SHA-256 and a successful load.
        """
        if dep_name in self._cache:
            return self._cache[dep_name]
        root = Path(self.run_dir)
        ckpt = checkpoint.checkpoint_path(root, dep_name)
        try:
            rec = checkpoint.read_marker(root, dep_name)
        except FileNotFoundError:
            raise DependencyError(f"Stage {dep_name!r} has no done-marker in {root}; "
                                  f"run it before {self.name!r}") from None
        except Exception as e:  # noqa: BLE001
            raise DependencyError(f"Stage {dep_name!r} marker unreadable: {e}") from e
        if not ckpt.is_file():
            raise DependencyError(f"Stage {dep_name!r} Checkpoint missing: {ckpt}")
        if checkpoint.sha256_file(ckpt) != rec["sha256"]:
            raise DependencyError(f"Stage {dep_name!r} Checkpoint does not match its "
                                  f"done-marker: {ckpt}")
        try:
            obj = checkpoint.load(ckpt)
        except Exception as e:  # noqa: BLE001
            raise DependencyError(f"Stage {dep_name!r} Checkpoint fails to load: "
                                  f"{type(e).__name__}: {e}") from e
        self._cache[dep_name] = obj
        return obj

    def path(self, filename: str) -> Path:
        """A side-output path inside the run directory."""
        return Path(self.run_dir) / filename


def stage_workers(smoke: bool, env=None, cores: int | None = None) -> int:
    """Worker count (Req 3.2); at most 4 in smoke mode (Req 13.1)."""
    n = paths.worker_count(env, cores)
    return min(n, SMOKE_MAX_WORKERS) if smoke else n


def in_slurm(env=None) -> bool:
    env = os.environ if env is None else env
    return bool(str(env.get("SLURM_JOB_ID", "")).strip())


def import_stage(name: str) -> ModuleType:
    """``hpc_unlock.<name>``.

    Raises ``StageUnavailable`` if the module does not exist or has no
    ``compute``; any other import error (e.g. a missing package inside the
    Stage) propagates unchanged.
    """
    modname = f"{__package__}.{name}"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name == modname:
            raise StageUnavailable(f"Stage module {modname} is not available "
                                   f"(hpc_unlock/{name}.py not found)") from None
        raise
    if not callable(getattr(mod, "compute", None)):
        raise StageUnavailable(f"Stage module {modname} defines no compute(cfg, ctx)")
    return mod


def execute(name: str, cfg, run_dir: str | Path, smoke: bool = False,
            workers: int | None = None, log: Callable[..., str] = checkpoint.log_line,
            options: dict | None = None) -> int:
    """Validate, import and run one Stage via ``checkpoint.run_stage``; return an exit code.

    ``options`` reach the Stage as ``ctx.options``. With ``options["report_only"]``
    the Stage's ``compute`` runs directly: no Checkpoint, no done-marker and no
    reuse of an earlier marker (``assemble --report-only``).

    Does not check the SLURM allocation; ``run_unlock.py`` does that first.
    """
    options = dict(options or {})
    run_dir = Path(run_dir)
    fp = cfg.fingerprint()
    workers = stage_workers(smoke) if workers is None else int(workers)

    errors = list(cfg.validate())
    if errors:
        for err in errors:
            log("CONFIG_INVALID", f"{name}: {err}")
        log("STAGE_ABORTED", f"{name}: invalid config, nothing computed")
        return EXIT_FAILED
    try:
        mod = import_stage(name)
    except StageUnavailable as e:
        log("STAGE_UNAVAILABLE", f"{name}: {e}")
        return EXIT_USAGE
    except Exception as e:  # noqa: BLE001 - e.g. torch missing in the env
        traceback.print_exc(file=sys.stderr)
        log("STAGE_IMPORT_FAILED", f"{name}: {type(e).__name__}: {e}")
        return EXIT_FAILED
    extra = getattr(mod, "validate_config", None)
    if callable(extra):
        errors = list(extra(cfg))
        if errors:
            for err in errors:
                log("CONFIG_INVALID", f"{name}: {err}")
            log("STAGE_ABORTED", f"{name}: invalid config, nothing computed")
            return EXIT_FAILED

    ctx = StageContext(name=name, run_dir=run_dir, smoke=smoke, workers=workers,
                       options=options)
    opt = f" options={sorted(k for k, v in options.items() if v)}" if options else ""
    log("STAGE_START", f"{name} fingerprint={fp} workers={workers} smoke={smoke} "
                       f"gpu={bool(getattr(mod, 'NEEDS_GPU', False))} run_dir={run_dir}{opt}")
    t0 = time.monotonic()
    status, code = "ok", EXIT_OK
    try:
        if options.get("report_only"):
            run_dir.mkdir(parents=True, exist_ok=True)
            mod.compute(cfg, ctx)
            log("NO_MARKER", f"{name}: report-only run, no Checkpoint or done-marker")
        else:
            checkpoint.run_stage(name, lambda: mod.compute(cfg, ctx), run_dir, fp)
    except SystemExit as e:  # a Stage may exit itself (e.g. NO_GPU_VISIBLE -> 1)
        c = e.code
        code = c if isinstance(c, int) else (EXIT_OK if c is None else EXIT_FAILED)
        status = "ok" if code == EXIT_OK else "failed"
    except DependencyError as e:
        log("DEPENDENCY_MISSING", f"{name}: {e}")
        status, code = "failed", EXIT_FAILED
    except Exception:  # noqa: BLE001 - log, report, exit non-zero
        traceback.print_exc(file=sys.stderr)
        status, code = "failed", EXIT_FAILED
    log("STAGE_END", f"{name} fingerprint={fp} workers={workers} status={status} "
                     f"elapsed={time.monotonic() - t0:.1f}s")
    return code
