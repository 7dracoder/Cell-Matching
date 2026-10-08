"""Local smoke test: ``python run_unlock.py smoke`` (Req 13.1-13.5).

Runs every driver Stage of the chain (``chain.plan(cfg, "prep")``) once, in
order, as ``run_unlock.py <stage> --smoke --run <fp>`` subprocesses, so it uses
exactly the per-Stage smoke mechanism of the driver:

* 2 held-out regions from 2 different mice (``prep.smoke_heldout``),
  ``workers = min(4, cores)`` (``stage.stage_workers``), BLAS / torch threads
  pinned so a Stage uses at most 4 cores (13.1);
* ``gpu_scan`` on CPU torch with the 5 x 3 x 1 grid (13.2);
* no Cellpose fine-tuning: ``selftrain_gpu`` is reported
  ``SKIPPED (no fine-tune in smoke)`` (13.3);
* outputs only under ``research/data/hpc_smoke/``: the run directory
  ``hpc_smoke/<fp>/`` (wiped first, so no earlier marker is reused) and
  ``hpc_smoke/smoke_candidate.csv`` written by ``assemble`` and checked by the
  Format_Checker (13.5).

After each Stage a few cheap checks read its Checkpoint (region count, grid
size, ``n/a (smoke)`` Baseline tolerance, skipped fine-tune, smoke CSV). The
SHA-256 of every file under ``research/data/hpc/``, the root Run_Report files
and every root ``submission*.csv`` is taken before and after; any difference
fails the ``isolation`` check. The whole run has a 300 s wall-clock budget.

Output: one ``PASS <stage> ...`` / ``FAIL <stage> ...`` line per Stage and
check, then ``SMOKE_PASSED`` (exit 0) or ``SMOKE_FAILED first_failing_stage=<stage>``
(exit 1) (13.4). Stage output goes to ``research/data/hpc_smoke/logs/<stage>.log``;
the tail is printed for a failing Stage.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from . import checkpoint, paths, report
from .config import UnlockConfig

BUDGET_S = 300.0
MAX_HELDOUT = 2
MAX_GRID = (5, 3, 1)                    # angles x scales x stretches (13.2)
SMOKE_CSV = "smoke_candidate.csv"
REPORT_FILES = (report.JSON_NAME, report.MD_NAME)
SUBMISSION_GLOB = "submission*.csv"
SELFTRAIN_SKIP = "SKIPPED (no fine-tune in smoke)"
TAIL_LINES = 30
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
ISOLATION = "isolation"
BUDGET = "budget"

PASS, FAIL = "PASS", "FAIL"


@dataclass
class Result:
    stage: str
    status: str              # PASS | FAIL
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def line(self) -> str:
        return result_line(self.status, self.stage, self.detail)


# ---------------------------------------------------------------- pure helpers
def result_line(status: str, stage: str, detail: str = "") -> str:
    """``PASS <stage>`` / ``FAIL <stage>``, optionally followed by ``(detail)``."""
    if status not in (PASS, FAIL):
        raise ValueError(f"status must be {PASS} or {FAIL}, not {status!r}")
    return f"{status} {stage}" + (f" ({detail})" if detail else "")


def first_failure(results: Iterable[Result]) -> str | None:
    """Name of the first failing Stage or check, or ``None``."""
    return next((r.stage for r in results if not r.ok), None)


def exit_code(results: Sequence[Result]) -> int:
    return 0 if first_failure(results) is None else 1


def summary_line(results: Sequence[Result], elapsed: float) -> str:
    bad = first_failure(results)
    if bad is None:
        return f"SMOKE_PASSED stages={len(results)} elapsed={elapsed:.1f}s"
    return f"SMOKE_FAILED first_failing_stage={bad} elapsed={elapsed:.1f}s"


def guarded_files(root: str | Path | None = None, hpc: str | Path | None = None) -> list[Path]:
    """Files the smoke test must not create or modify (13.5).

    Every file under ``hpc`` (full-run Checkpoints, markers, logs), the root
    Run_Report files (listed even when absent, so creating them is caught) and
    every root ``submission*.csv``.
    """
    root = Path(paths.ROOT if root is None else root)
    hpc = Path(paths.HPC if hpc is None else hpc)
    out = sorted(p for p in hpc.rglob("*") if p.is_file()) if hpc.is_dir() else []
    out += [root / n for n in REPORT_FILES]
    out += sorted(p for p in root.glob(SUBMISSION_GLOB) if p.is_file())
    return out


def snapshot(root: str | Path | None = None, hpc: str | Path | None = None) -> dict[str, str | None]:
    """``{path: sha256}`` of the guarded files; ``None`` for an absent file.

    The ``hpc`` directory itself is an entry (``"dir"`` / ``None``), so creating
    an empty ``research/data/hpc/`` is caught too.
    """
    hpc = Path(paths.HPC if hpc is None else hpc)
    snap: dict[str, str | None] = {f"{hpc}{os.sep}": "dir" if hpc.is_dir() else None}
    snap.update({str(p): (checkpoint.sha256_file(p) if p.is_file() else None)
                 for p in guarded_files(root, hpc)})
    return snap


def diff_snapshots(before: Mapping[str, str | None],
                   after: Mapping[str, str | None]) -> list[str]:
    """``created`` / ``modified`` / ``removed`` lines, sorted by path."""
    out = []
    for p in sorted(set(before) | set(after)):
        b, a = before.get(p), after.get(p)
        if b == a:
            continue
        kind = "created" if b is None else "removed" if a is None else "modified"
        out.append(f"{kind} {p}")
    return out


def isolation_result(before: Mapping, after: Mapping) -> Result:
    changes = diff_snapshots(before, after)
    if not changes:
        return Result(ISOLATION, PASS, f"{len(after)} guarded files unchanged")
    more = f" (+{len(changes) - 3} more)" if len(changes) > 3 else ""
    return Result(ISOLATION, FAIL, "; ".join(changes[:3]) + more)


def budget_result(elapsed: float, budget: float = BUDGET_S) -> Result:
    status = PASS if elapsed <= budget else FAIL
    return Result(BUDGET, status, f"elapsed={elapsed:.1f}s limit={budget:.0f}s")


def stage_env(name: str, workers: int, base: Mapping[str, str] | None = None) -> dict:
    """Subprocess env: no ``SLURM_*``; at most ``workers`` cores per Stage (13.1).

    Pool Stages fork ``workers`` processes, so their BLAS / torch threads are
    pinned to 1; ``gpu_scan`` has no pool and gets ``workers`` torch threads.
    """
    env = {k: v for k, v in dict(os.environ if base is None else base).items()
           if not k.startswith("SLURM_")}
    threads = str(workers if name == "gpu_scan" else 1)
    env.update({v: threads for v in THREAD_VARS})
    env["PYTHONUNBUFFERED"] = "1"
    return env


def safe_wipe(target: Path, base: Path) -> None:
    """``rm -r target`` only if it lies strictly inside ``base``."""
    target, base = Path(target).resolve(), Path(base).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"refusing to delete {target}: not inside {base}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def _tail(path: Path, n: int = TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


# ---------------------------------------------------------------- per-Stage checks
def _ck(run_dir: Path, name: str):
    return checkpoint.load(checkpoint.checkpoint_path(run_dir, name))


def check_prep(ck: Mapping) -> str:
    held = ck["heldout"]
    mice = {str(r.get("subject")) for r in held.values()}
    if len(held) > MAX_HELDOUT or len(mice) != len(held) or len(held) < 2:
        raise AssertionError(f"expected {MAX_HELDOUT} held-out regions from "
                             f"{MAX_HELDOUT} mice, got {len(held)} from {sorted(mice)}")
    return f"heldout={len(held)} mice={len(mice)} test={len(ck.get('test') or {})}"


def check_gpu_scan(ck: Mapping) -> str:
    g = ck["grid"]
    dims = (len(g["angles"]), len(g["scales"]), len(g["stretches"]))
    if not g.get("smoke") or any(d > m for d, m in zip(dims, MAX_GRID)):
        raise AssertionError(f"smoke grid {dims} (smoke={g.get('smoke')}) exceeds {MAX_GRID}")
    dev = (ck.get("torch") or {}).get("device")
    return f"CPU torch grid={dims[0]}x{dims[1]}x{dims[2]} device={dev}"


def check_validate(ck: Mapping) -> str:
    if ck["baseline"].get("reproduced") is not None:
        raise AssertionError("Baseline tolerance applied in smoke "
                             f"(reproduced={ck['baseline'].get('reproduced')!r})")
    return "Baseline tolerance n/a (smoke)"


def check_selftrain_gpu(ck: Mapping) -> str:
    jobs = ck.get("jobs") or {}
    bad = [j for j, v in jobs.items() if v.get("status") != "skipped_smoke"]
    if bad:
        raise AssertionError(f"fine-tune ran in smoke for jobs {bad}")
    return SELFTRAIN_SKIP


def check_assemble(ck: Mapping, run_dir: Path, smoke_base: Path) -> str:
    csv = smoke_base / SMOKE_CSV
    if not csv.is_file():
        raise AssertionError(f"{csv} not written")
    extra = sorted(p.name for p in run_dir.glob("submission*.csv"))
    if extra:
        raise AssertionError(f"smoke wrote submission CSVs: {extra}")
    rep = ck.get("report") or {}
    st = (rep.get("baseline_reproduction") or {}).get("status")
    if st != "n/a (smoke)":
        raise AssertionError(f"Run_Report Baseline reproduction {st!r} != 'n/a (smoke)'")
    shown = csv.relative_to(paths.ROOT) if csv.is_relative_to(paths.ROOT) else csv
    return f"{shown} Format_Checker ok; Baseline tolerance n/a (smoke)"


CHECKS: dict[str, Callable[..., str]] = {
    "prep": lambda ck, rd, sb: check_prep(ck),
    "gpu_scan": lambda ck, rd, sb: check_gpu_scan(ck),
    "validate": lambda ck, rd, sb: check_validate(ck),
    "selftrain_gpu": lambda ck, rd, sb: check_selftrain_gpu(ck),
    "assemble": check_assemble,
}


# ---------------------------------------------------------------- runner
def run_one(name: str, fp: str, run_dir: Path, smoke_base: Path, timeout: float,
            workers: int, log_dir: Path) -> Result:
    """One Stage subprocess plus its check; a ``Result``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    cmd = [sys.executable, str(paths.ROOT / "run_unlock.py"), name, "--smoke", "--run", fp]
    t0 = time.monotonic()
    try:
        with open(log_path, "w") as log:
            rc = subprocess.run(cmd, cwd=str(paths.ROOT), env=stage_env(name, workers),
                                stdout=log, stderr=subprocess.STDOUT,
                                timeout=max(1.0, timeout)).returncode
    except subprocess.TimeoutExpired:
        return Result(name, FAIL, f"wall-clock budget exceeded after "
                                  f"{time.monotonic() - t0:.1f}s; log {log_path}")
    dt = time.monotonic() - t0
    if rc != 0:
        print(_tail(log_path), flush=True)
        return Result(name, FAIL, f"exit {rc} after {dt:.1f}s; log {log_path}")
    detail = f"{dt:.1f}s"
    check = CHECKS.get(name)
    if check is not None:
        try:
            detail += "; " + check(_ck(run_dir, name), run_dir, smoke_base)
        except Exception as e:  # noqa: BLE001 - a failed check fails the Stage
            return Result(name, FAIL, f"check failed: {type(e).__name__}: {e}")
    return Result(name, PASS, detail)


def run(budget: float = BUDGET_S, out=print) -> int:
    t0 = time.monotonic()
    cfg = UnlockConfig()
    fp = cfg.fingerprint()
    smoke_base = paths.HPC_SMOKE
    run_dir = smoke_base / fp
    from . import chain, stage
    workers = stage.stage_workers(True)
    stages = [s for s in chain.plan(cfg, "prep") if s not in stage.NOT_DRIVEN]
    out(f"SMOKE_START fingerprint={fp} workers={workers} budget={budget:.0f}s "
        f"run_dir={run_dir} stages={','.join(stages)}")

    before = snapshot()
    for stale in (run_dir, smoke_base / SMOKE_CSV):
        if stale.exists():
            safe_wipe(stale, smoke_base)
    cfg.save(run_dir)

    results: list[Result] = []
    failed: str | None = None
    for name in stages:
        if failed is not None:
            r = Result(name, FAIL, f"not run: Stage {failed} failed")
        else:
            left = budget - (time.monotonic() - t0)
            if left <= 0:
                r = Result(name, FAIL, "not run: wall-clock budget exhausted")
            else:
                r = run_one(name, fp, run_dir, smoke_base, left, workers,
                            smoke_base / "logs")
            if not r.ok:
                failed = name
        results.append(r)
        out(r.line())

    results.append(isolation_result(before, snapshot()))
    out(results[-1].line())
    elapsed = time.monotonic() - t0
    results.append(budget_result(elapsed, budget))
    out(results[-1].line())
    out(summary_line(results, elapsed))
    return exit_code(results)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_unlock.py smoke",
                                 description="Local smoke test (<= 2 regions, <= 4 cores, "
                                             "<= 300 s, no fine-tune).")
    ap.add_argument("--budget", type=float, default=BUDGET_S,
                    help=f"wall-clock budget in seconds (default {BUDGET_S:.0f})")
    args = ap.parse_args(list(argv or []))
    if args.budget <= 0 or args.budget > BUDGET_S:
        ap.error(f"--budget must be in (0, {BUDGET_S:.0f}]")
    return run(args.budget)


if __name__ == "__main__":
    sys.exit(main())
