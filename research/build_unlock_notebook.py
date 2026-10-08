"""Write CellMatch_HPC_Unlock.ipynb (run locally: .venv/bin/python research/build_unlock_notebook.py).

The notebook drives the HPC registration-unlock Job_Chain on NYU Burst from a
plain Python 3 kernel. Code cells use only the standard library plus
``hpc_unlock.{paths,config,chain}`` (and an optional ``IPython.display``
inside a ``try``). Nothing heavy runs in the notebook; every Stage is an
``sbatch`` job. Spec: .kiro/specs/hpc-registration-unlock (task 15.1, Req 2, 12.1-12.2).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "CellMatch_HPC_Unlock.ipynb"

# --------------------------------------------------------------------- intro
INTRO = r"""# CellMatch HPC registration unlock (NYU Burst)

This notebook submits and watches a CPU + GPU SLURM job chain on NYU Cloud Bursting:

```
setup (CPU, alone)
  -> prep (CPU) -> gpu_scan (GPU dense pose scan) -> pose_search (CPU) -> joint (CPU)
  -> pairs (CPU) -> verifier (CPU) -> validate (CPU)
  -> [selftrain_prep (CPU) -> selftrain_gpu (GPU, Cellpose-SAM self-training) -> selftrain_pairs (CPU re-pairing)]
  -> assemble (CPU: candidate CSVs + Run_Report)
```

The bracketed self-training Stages run by default; set `DISABLE_SELFTRAIN = True` in Cell 1 to skip them (9 chain Stages instead of 12 including setup).

**Nothing heavy runs in this notebook.** Every cell only calls `sbatch` / `squeue` / `sacct` or reads small files, so a plain Python 3 kernel on a login or small CPU session is enough.

**Honest outlook.** A public score above 0.65 is the target, not a guarantee. Held-out gains are estimates. A candidate CSV (`submission_v13_*.csv`) is written only if its held-out full score beats the current Baseline v10 (`submission_v10_cpgate.csv`, held-out 0.5186). v10 (public 0.48893) stays the safe final submission either way.

Run the cells **in order**. Each code cell has a numbered instruction cell above it that says what to run and what output to expect."""

# ------------------------------------------------------------------- Step 0
MD0 = r"""## Step 0: Upload and container files

**Cell 0: locate the project, print the `scp` commands, check inputs.**

Before running it:

1. Upload the whole project folder to `/scratch/$USER/cellmatch` on Burst (OOD *Files* upload, or `rsync -av` from your machine). `research/data/` is about 800 MB and includes files over GitHub's 100 MB limit, so do not use git for it.
2. Open this notebook from that folder in Burst OOD Jupyter (a login or small CPU session is fine). `sbatch`, `squeue` and `sacct` must be on the `PATH` of the kernel.
3. Burst cannot see Greene's `/scratch/work/public`, so copy the two container files once from a **Burst login shell** (`ssh greene`, then `ssh burst`). Cell 0 prints these commands with your real project path filled in:

```bash
scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz "$WORK/hpc/unlock/container/" && gunzip "$WORK/hpc/unlock/container/overlay-15GB-500K.ext3.gz"
ssh greene-dtn ls /scratch/work/public/singularity/ | grep -i cuda
scp greene-dtn:/scratch/work/public/singularity/<sif> "$WORK/hpc/unlock/container/"
```

For `<sif>` use a CUDA 12 image if the `ls | grep` lists one (a name containing `cuda12`), else `cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`. Only `ENV_MODE = "singularity"` needs these files.

**Expected output:** `WORK=/scratch/<you>/cellmatch`, `hpc_unlock/ present: True`, `sbatch: /usr/bin/sbatch` (or similar), the `scp` commands, which container files are present, and then the input check: `INPUT_CHECK_OK <N> inputs`, or one `INPUT_FAILED <path>: <reason>` line per missing file and `INPUT_CHECK_FAILED`.

The input check fully loads the `.pkl` / `.npz` caches, which needs numpy. If this kernel's Python has no numpy, those caches may be reported as unreadable; that is fine, because the setup job runs the same check again inside the pinned environment. Missing TIFFs, CSVs or container files are real problems: fix them before Step 2."""

CODE0 = r'''# Cell 0: locate WORK, print the scp commands, check container files and inputs
import os
import shutil
import subprocess
import sys
from pathlib import Path

CHECK_ENV_MODE = "singularity"   # keep equal to ENV_MODE in Cell 1 ("singularity" or "venv")

# Bootstrap with the standard library so hpc_unlock can be imported from any cwd:
# the notebook's folder if it holds hpc_unlock/, else $CELLMATCH_DIR, else /scratch/$USER/cellmatch.
_here = Path.cwd()
_cands = [_here, _here.parent]
if os.environ.get("CELLMATCH_DIR"):
    _cands.append(Path(os.environ["CELLMATCH_DIR"]))
_cands.append(Path("/scratch") / os.environ.get("USER", "") / "cellmatch")
_base = next((c for c in _cands if (c / "hpc_unlock").is_dir()), None)
if _base is None:
    raise RuntimeError("hpc_unlock/ not found in " + ", ".join(map(str, _cands))
                       + ". Upload the project to /scratch/$USER/cellmatch (or set CELLMATCH_DIR) "
                       "and open this notebook from that folder.")
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

from hpc_unlock import paths

WORK = paths.work_dir(_base)
print("WORK =", WORK)
print("hpc_unlock/ present:", (WORK / "hpc_unlock").is_dir())
if paths.ROOT.resolve() != Path(WORK).resolve():
    print(f"WARNING: hpc_unlock was imported from {paths.ROOT}, not WORK; jobs are submitted from {paths.ROOT}")

for tool in ("sbatch", "squeue", "sacct"):
    where = shutil.which(tool)
    print(f"{tool}: {where or 'NOT FOUND'}")
if not shutil.which("sbatch"):
    print("sbatch is not available in this kernel. Open the notebook from Burst OOD Jupyter "
          "(or run `python run_unlock.py submit` from a Burst login shell); Steps 2-3 cannot submit.")

CONTAINER = Path(WORK) / "hpc" / "unlock" / "container"
CONTAINER.mkdir(parents=True, exist_ok=True)
OVERLAY = CONTAINER / "overlay-15GB-500K.ext3"
print("\nRun these once from a Burst login shell (ssh greene, then ssh burst):")
print(f'scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz "{CONTAINER}/" '
      f'&& gunzip "{CONTAINER}/overlay-15GB-500K.ext3.gz"')
print("ssh greene-dtn ls /scratch/work/public/singularity/ | grep -i cuda")
print(f'scp greene-dtn:/scratch/work/public/singularity/<cuda12 image from the ls above> "{CONTAINER}/"')
print(f'#   or, if no cuda12 image is listed:\n'
      f'scp greene-dtn:/scratch/work/public/singularity/{paths.CUDA11_SIF} "{CONTAINER}/"')

sif = paths.find_sif(CONTAINER)
print("\nContainer files in", CONTAINER)
print("  overlay:", "present" if OVERLAY.is_file() else "MISSING", f"({OVERLAY.name})")
print("  .sif   :", sif.name if sif else "MISSING (*.sif)")
if OVERLAY.with_name(OVERLAY.name + ".gz").is_file():
    print("  note: overlay is still gzipped; run the gunzip command above")

print(f"\nInput check: python run_unlock.py check --env-mode {CHECK_ENV_MODE}")
r = subprocess.run([sys.executable, "run_unlock.py", "check", "--env-mode", CHECK_ENV_MODE],
                   cwd=str(WORK), capture_output=True, text=True, timeout=1800)
print(r.stdout.strip())
if r.stderr.strip():
    print(r.stderr.strip())
print("input check exit code:", r.returncode)'''

# ------------------------------------------------------------------- Step 1
MD1 = r"""## Step 1: Config

**Cell 1: set the run configuration and validate it.**

Edit the constants at the top of the cell, then run it:

- `ENV_MODE`: `"singularity"` (default; Singularity `.sif` + ext3 overlay) or `"venv"` (miniforge env from `hpc/env.sh`).
- `GPU_PARTITION`: `"g2-standard-12"` (L4, default) or `"c12m85-a100-1"` (A100).
- `DISABLE_SELFTRAIN`: `False` (default) runs the three self-training Stages; `True` skips them.
- `SIGMA` (Soft_Score σ, [1.5, 2.5] px, default 2.5), `SCAN_K` (GPU scan peaks kept, integer in [1, 500], default 50), `MATCH_RADIUS` and `SEED_RADIUS` (self-training pseudo-labels, [1, 20] px, default 6).
- `RESOURCES`: optional per-Stage overrides merged onto the defaults, e.g. `{"prep": {"cpus": 16, "mem": "20G"}}` if the scheduler rejects a request.

The account (`cs_gy_6923-2026fa`) and CPU partition (`n2c48m24`) come from `UnlockConfig` defaults.

**Expected output:** `config OK, run <fingerprint>, env singularity, sif <name>`, the run directory `research/data/hpc/<fingerprint>`, and the Stage plan (stage / partition / cpus / mem / time). If anything is invalid you get one `CONFIG ERROR ...` line per invalid field instead, `CONFIG_OK = False`, and Cells 2, 3 and 6 refuse to submit anything until you fix it and re-run this cell."""

CODE1 = r'''# Cell 1: run configuration
from hpc_unlock import chain, paths
from hpc_unlock.config import UnlockConfig, default_resources

ENV_MODE = "singularity"          # "singularity" (default) or "venv"
GPU_PARTITION = "g2-standard-12"  # "g2-standard-12" (L4) or "c12m85-a100-1" (A100)
DISABLE_SELFTRAIN = False         # True skips selftrain_prep / selftrain_gpu / selftrain_pairs
SIGMA = 2.5                       # Soft_Score sigma, [1.5, 2.5] px
SCAN_K = 50                       # GPU scan peaks kept per region, integer in [1, 500]
MATCH_RADIUS = 6.0                # self-training match radius, [1, 20] px
SEED_RADIUS = 6.0                 # self-training seed disk radius, [1, 20] px
RESOURCES = {}                    # optional overrides, e.g. {"prep": {"cpus": 16, "mem": "20G"}}

_res = default_resources()
_res_errors = []
for _stage, _over in RESOURCES.items():
    if _stage not in _res:
        _res_errors.append(f"RESOURCES[{_stage!r}]: unknown Stage; expected one of {', '.join(_res)}")
    elif not isinstance(_over, dict):
        _res_errors.append(f"RESOURCES[{_stage!r}] must be a dict like {{'cpus': 16, 'mem': '20G'}}")
    else:
        _res[_stage] = {**_res[_stage], **_over}

cfg = UnlockConfig(env_mode=ENV_MODE, gpu_partition=GPU_PARTITION,
                   disable_selftrain=DISABLE_SELFTRAIN, sigma=SIGMA, scan_k=SCAN_K,
                   match_radius=MATCH_RADIUS, seed_radius=SEED_RADIUS, resources=_res)
CONFIG_ERRORS = list(cfg.validate()) + _res_errors
CONFIG_OK = not CONFIG_ERRORS

if CONFIG_OK:
    _sif = cfg.sif_name or (paths.find_sif().name if paths.find_sif() else "MISSING")
    print(f"config OK, run {cfg.fingerprint()}, env {cfg.env_mode}, "
          f"sif {_sif if cfg.env_mode == 'singularity' else 'n/a (venv)'}")
    print("account:", cfg.account, "| cpu partition:", cfg.cpu_partition,
          "| gpu partition:", cfg.gpu_partition)
    print("run dir:", cfg.run_dir())
    print("self-training:", "DISABLED" if cfg.disable_selftrain else "enabled")
    print(f"\n{'stage':<16} {'partition':<16} {'cpus':>4} {'mem':>5}  time")
    for _s in ["setup"] + chain.plan(cfg, "prep"):
        _r = cfg.stage_resources(_s)
        print(f"{_s:<16} {chain.partition(cfg, _s):<16} {_r['cpus']:>4} {_r['mem']:>5}  {_r['time']}")
else:
    for _e in CONFIG_ERRORS:
        print("CONFIG ERROR", _e)
    print("CONFIG_OK = False: nothing will be submitted. Fix the values above and re-run Cell 1.")'''

# ------------------------------------------------------------------- Step 2
MD2 = r"""## Step 2: Setup job

**Cell 2: submit the setup job (it runs alone).**

Setup checks every input, then builds the environment. In singularity mode it mounts the overlay read-write, installs Miniconda and the pinned packages (numpy, scipy, opencv, scikit-learn, torch, cellpose unless self-training is disabled) into `/ext3`, and caches the Cellpose-SAM weights. The first run takes roughly 20–40 minutes; later runs reuse the env and take a few minutes. Setup must finish before chain jobs mount the overlay read-only, because only one job can hold the overlay read-write. Do not submit setup while chain jobs or the optional container kernel are using the overlay.

If a setup job is already pending or running, the cell does not submit a second one and only shows its state.

**Expected output:** `setup            n2c48m24         <job id>`, then a status line such as `setup job <id> is PENDING; the chain waits with --dependency=afterok:<id>`. Re-run Cell 4 (or this cell) to watch it; once it has `COMPLETED`, this cell also prints `setup.json` (env path, pin versions, `torch.version.cuda`)."""

CODE2 = r'''# Cell 2: submit setup alone
import json

if "cfg" not in globals() or "CONFIG_OK" not in globals():
    raise RuntimeError("Run Cell 0 and Cell 1 first.")

if not CONFIG_OK:
    print("Config has errors (Cell 1); setup NOT submitted:")
    for _e in CONFIG_ERRORS:
        print("  ", _e)
else:
    _st = chain.setup_state(cfg)
    if _st["state"] in chain.ACTIVE_STATES:
        print("Setup already queued/running, not resubmitting:", _st["message"])
    else:
        try:
            chain.submit_setup(cfg)          # prints: setup  n2c48m24  <job id>
        except chain.ChainError as e:
            print(f"{type(e).__name__}: {e}")
            if getattr(e, "stderr", ""):
                print(e.stderr)
    _st = chain.setup_state(cfg)
    print("setup status:", _st["message"])
    _sj = paths.HPC / "setup" / "setup.json"
    if _sj.is_file():
        print("\nsetup.json:")
        print(json.dumps(json.loads(_sj.read_text()), indent=2))'''

# ------------------------------------------------------------------- Step 3
MD3 = r"""## Step 3: Chain

**Cell 3: submit the job chain from `prep`.**

Each Stage is one `sbatch` job with the account, partition, memory and CPU flags, `--export=NONE`, `--requeue`, `--gres=gpu:1` on GPU Stages, and `--dependency=afterok:<previous job id>`. If setup is still pending or running, `prep` waits on it with `afterok`; if setup has completed (with `setup.json`), `prep` has no dependency. If setup failed or was never submitted, nothing is submitted and the setup log path is shown. The first `sbatch` error stops the submission and prints its error output; no later Stage is submitted.

**Expected output:** one line per Stage, `stage  partition  job id`: 11 lines (`prep` … `assemble`) with self-training, 8 without. Together with setup that is 12 or 9 Stages. On a refusal you see `SetupNotReady: ...`, `SubmitError: ...` (plus the sbatch stderr) or `ConfigError: ...`."""

CODE3 = r'''# Cell 3: submit the chain (prep -> ... -> assemble)
if "cfg" not in globals() or "CONFIG_OK" not in globals():
    raise RuntimeError("Run Cell 0 and Cell 1 first.")

if not CONFIG_OK:
    print("Config has errors (Cell 1); chain NOT submitted:")
    for _e in CONFIG_ERRORS:
        print("  ", _e)
else:
    try:
        SUBMITTED = chain.submit(cfg, start="prep")   # prints stage / partition / job id
        print(f"\nsubmitted {len(SUBMITTED)} Stages for run {cfg.fingerprint()}")
    except chain.SetupNotReady as e:
        print("SetupNotReady:", e)
    except chain.SubmitError as e:
        print("SubmitError:", e)
        if e.stderr:
            print(e.stderr)
        if e.submitted:
            print("already submitted:", ", ".join(f"{s} [{j}]" for s, _, j in e.submitted))
    except chain.ConfigError as e:
        print("ConfigError:", e)'''

# ------------------------------------------------------------------- Step 4
MD4 = r"""## Step 4: Monitor

**Cell 4: show the state of every job. Re-run it any time.**

It shows, per job, the `squeue` state (or the `sacct` state and exit code once the job has left the queue), the last 40 lines of each Stage log (or `log not created yet`), and the done-markers (finished Checkpoints) of this run. For a job that `FAILED`, hit `TIMEOUT`, was `CANCELLED`, ran `OUT_OF_MEMORY` or exited non-zero, it prints `FAILED <stage>: log <path>; resubmit from this Stage with: python run_unlock.py submit --from <stage> --run <fp>`. Jobs that depended on it stay `PENDING (DependencyNeverSatisfied)`; cancel them with the printed `scancel` command before resubmitting (Cell 6 or the printed command).

Rough wall-clock times (estimates, not measured on Burst):

| Stage | Estimate |
| --- | --- |
| setup | 20–40 min first time, a few min later |
| prep | ~5–15 min |
| gpu_scan | ~10–30 min on L4 |
| pose_search | ~10–40 min |
| joint, pairs, verifier, validate | minutes each |
| selftrain_prep | minutes |
| selftrain_gpu | ~1.5–3 h on L4 |
| selftrain_pairs | ~10–30 min |
| assemble | minutes |

**Expected output:** `Run <fingerprint>  (research/data/hpc/<fingerprint>)`, a table `stage  job_id  partition  state`, `Done-markers: prep.done, ...`, then the log tails."""

CODE4 = r'''# Cell 4: monitor (re-run any time)
if "cfg" not in globals():
    raise RuntimeError("Run Cell 0 and Cell 1 first.")

print(chain.monitor(cfg))'''

# ------------------------------------------------------------------- Step 5
MD5 = r"""## Step 5: Report and download

**Cell 5: show the Run_Report and the candidate CSVs.**

Run it after `assemble` has finished (Cell 4 shows `assemble` `COMPLETED`).

**Expected output:** the rendered `hpc_unlock_report.md` (ranking, rejected configurations, self-training status, the "0.65 is the target, not guaranteed" note), the recommended two finals, and either a list of `submission_v13_*.csv` files with sizes or `NO_CANDIDATE: no candidate CSV was written`. Then the command to download the CSVs and the report."""

CODE5 = r'''# Cell 5: Run_Report, candidate CSVs, download command
import json
from pathlib import Path

if "WORK" not in globals():
    raise RuntimeError("Run Cell 0 first.")

_root = Path(WORK)
_md = _root / "hpc_unlock_report.md"
_js = _root / "hpc_unlock_report.json"

if _md.is_file():
    _text = _md.read_text(encoding="utf-8")
    try:
        from IPython.display import Markdown, display
        display(Markdown(_text))
    except ImportError:
        print(_text)
else:
    print(f"{_md.name} not found yet: wait for assemble to finish (Cell 4).")

_csvs = sorted(_root.glob("submission_v13_*.csv"))
if _csvs:
    print("\nCandidate CSVs:")
    for _p in _csvs:
        print(f"  {_p}  ({_p.stat().st_size / 1e6:.2f} MB)")
elif _md.is_file():
    print("\nNO_CANDIDATE: no candidate CSV was written.")

if _js.is_file():
    _rep = json.loads(_js.read_text(encoding="utf-8"))
    _rec = _rep.get("recommendation", {})
    print("\nstatus:", _rec.get("status", _rep.get("status")))
    for _i, _f in enumerate(_rec.get("finals", []), 1):
        print(f"final {_i}: {_f}")
    if _rec.get("note"):
        print("note:", _rec["note"])

print("\nDownload (from your own machine; `burst` = your ssh alias for Burst via greene):")
print('  scp "burst:/scratch/$USER/cellmatch/submission_v13_*.csv" .')
print('  scp "burst:/scratch/$USER/cellmatch/hpc_unlock_report.*" .')
print(f"  (replace $USER with your NetID if it differs locally; full path on Burst: {_root})")
print("  or use OOD Files: browse to the folder above and download the files.")'''

# ---------------------------------------------------------------- optional
MD6 = r"""## Optional: resubmit from a Stage

**Cell 6: resubmit the chain from one Stage after a failure.**

First cancel jobs left `PENDING (DependencyNeverSatisfied)` with the `scancel <ids>` command Cell 4 prints (or `scancel` the job IDs by hand). Then set `START_STAGE` to the failed Stage (for example `"pose_search"`) and run the cell. Finished Stages before it are not resubmitted; Stages whose done-marker matches are reused (`REUSE <stage>` in their log). Setup must have completed. To redo setup itself, re-run Cell 2 instead. Leave `START_STAGE = None` to do nothing.

**Expected output:** with `START_STAGE = None`: `START_STAGE is None; nothing submitted` and the list of valid Stages. Otherwise the same `stage  partition  job id` lines as Cell 3, from `START_STAGE` to `assemble`, or a `SetupNotReady` / `SubmitError` / `ConfigError` / `INVALID_START` message."""

CODE6 = r'''# Cell 6 (optional): resubmit from a Stage
START_STAGE = None   # e.g. "pose_search"; None = do nothing

if "cfg" not in globals() or "CONFIG_OK" not in globals():
    raise RuntimeError("Run Cell 0 and Cell 1 first.")

if START_STAGE is None:
    print("START_STAGE is None; nothing submitted. Valid Stages:",
          ", ".join(s for s in chain.STAGES if s != "setup"))
elif not CONFIG_OK:
    print("Config has errors (Cell 1); nothing submitted.")
else:
    print("equivalent shell command:", chain.resubmit_command(cfg, START_STAGE))
    try:
        chain.submit(cfg, start=START_STAGE)
    except ValueError as e:
        print("INVALID_START", e)
    except chain.SubmitError as e:
        print("SubmitError:", e)
        if e.stderr:
            print(e.stderr)
    except chain.ChainError as e:
        print(f"{type(e).__name__}: {e}")'''

FINAL = r"""## Notes

**Optional container kernel (OOD Jupyter).** This notebook only needs a plain Python 3 kernel. To run your own analysis with the pinned packages after setup has finished, register the container kernel from a Burst shell:

```bash
mkdir -p ~/.local/share/jupyter/kernels/cellmatch-unlock
cp /scratch/$USER/cellmatch/hpc/unlock/kernel/kernel.json ~/.local/share/jupyter/kernels/cellmatch-unlock/
```

It runs `ipykernel` inside `singularity exec --overlay ...:ro <sif>`. While it runs it holds the overlay read-only, so shut it down before resubmitting setup (Cell 2), which needs the overlay read-write.

**venv mode.** With `ENV_MODE = "venv"` no container files are needed. Setup sources `hpc/env.sh`, reuses `env/` if every core pin matches, else creates `env_unlock/` and leaves the old env untouched.

**What to submit on Kaggle.** Follow the two finals in the Run_Report (Cell 5): the top-ranked accepted candidate first, with `submission_v10_cpgate.csv` kept as the other final. Replace v10 with another candidate only if a candidate's public score clearly beats v10's 0.48893. With `NO_CANDIDATE`, keep `submission_v10_cpgate.csv` and `submission_v7_grow15.csv`."""


# ------------------------------------------------------------------- build
def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _md(i: int, text: str) -> dict:
    return {"cell_type": "markdown", "id": f"md-{i:02d}", "metadata": {}, "source": _lines(text)}


def _code(i: int, text: str) -> dict:
    return {"cell_type": "code", "id": f"code-{i:02d}", "metadata": {},
            "execution_count": None, "outputs": [], "source": _lines(text)}


CELLS = [("md", INTRO), ("md", MD0), ("code", CODE0), ("md", MD1), ("code", CODE1),
         ("md", MD2), ("code", CODE2), ("md", MD3), ("code", CODE3), ("md", MD4),
         ("code", CODE4), ("md", MD5), ("code", CODE5), ("md", MD6), ("code", CODE6),
         ("md", FINAL)]


def build() -> dict:
    cells = [(_md if kind == "md" else _code)(i, text) for i, (kind, text) in enumerate(CELLS)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
