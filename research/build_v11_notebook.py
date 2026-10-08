"""Write CellMatch_v11_Colab.ipynb (run once locally: .venv/bin/python research/build_v11_notebook.py)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "CellMatch_v11_Colab.ipynb"

INTRO = """# CellMatch v11: pseudo-label Cellpose-SAM for ex-vivo, plus the cellprob region gate

**Before you start**
1. Upload `cellmatch_v11_bundle.zip` (150 MB, in the project folder) to the root of **My Drive**.
2. `Runtime > Change runtime type > GPU` (A100 or L4 is best, T4 works but is ~3x slower).
3. Run the cells **in order**. Each cell says what to look for.

Checkpoints, flows and results go to `My Drive/cellmatch_v11/`, so if Colab disconnects,
just run cells 1-2 again and rerun the cell that was interrupted: finished folds are reused.

Nothing here writes a submission unless the leave-one-mouse-out score beats the current
pipeline (0.5125 held-out, `submission_v7_grow15.csv`, public 0.47488)."""

SETUP = """# Cell 1: GPU check, packages, Google Drive
import os, sys, zipfile, glob, json, shutil
from pathlib import Path
import torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > GPU, then run this cell again"
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install tifffile "cellpose==4.2.1.1"
from google.colab import drive
drive.mount("/content/drive")
PERSIST = Path("/content/drive/MyDrive/cellmatch_v11")
PERSIST.mkdir(parents=True, exist_ok=True)
print("persistent folder:", PERSIST)"""

BUNDLE = """# Cell 2: unpack the bundle and link persistent folders
WORK = Path("/content/work")
candidates = [Path("/content/drive/MyDrive/cellmatch_v11_bundle.zip"), Path("/content/cellmatch_v11_bundle.zip")]
bundle = next((p for p in candidates if p.exists()), None)
if bundle is None:
    from google.colab import files
    print("cellmatch_v11_bundle.zip not found on Drive; upload it now (slow, Drive is faster)")
    bundle = Path("/content") / next(iter(files.upload()))
if not (WORK / "pipeline.py").exists():
    with zipfile.ZipFile(bundle) as z:
        z.extractall("/content")
# Model checkpoints, flows and teacher masks live on Drive so a disconnect does not lose them.
for name in ("models", "flows", "pseudo_teacher"):
    target, link = PERSIST / name, WORK / name
    target.mkdir(exist_ok=True)
    if link.is_symlink() or not link.exists():
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)
os.chdir(WORK)
sys.path[:0] = [str(WORK), "/content"]
print("training subjects:", sorted(p.name for p in (WORK / "Project_2_Dataset/training").glob("subject_*")))
print("test regions:", len(list((WORK / "Project_2_Dataset/hidden_test").glob("*/*"))))
print("on Drive already:", {n: len(list((PERSIST / n).iterdir())) for n in ("models", "flows", "pseudo_teacher")})"""

QUICK_MD = """## Cell 3: quick check on one held-out mouse (~1 h on A100/L4, ~2.5 h on T4)
Trains on two mice, tests ex-vivo PQ on `subject_db6b8b` (current masks: **0.394**).
If the printed `best` PQ is not clearly above 0.394 (say < 0.41), stop here: the idea
does not work and the full run will not produce a submission. Tell me the printed numbers."""

QUICK = """!python -u /content/pseudo_cv_colab.py --fold subject_db6b8b
r = json.loads((WORK / "pseudo_cv_subject_db6b8b.json").read_text())
print("\\nbest:", r["best"], "\\nbaseline ex-vivo PQ on this mouse: 0.394")
shutil.copy(WORK / "pseudo_cv_subject_db6b8b.json", PERSIST)"""

FULL_MD = """## Cell 4: all three held-out mice (~2 h more on A100/L4)
Reuses the db6b8b fold from cell 3. Baseline pooled ex-vivo PQ is **0.390**."""

FULL = """!python -u /content/pseudo_cv_colab.py
r = json.loads((WORK / "pseudo_cv.json").read_text())
print("\\nbest:", r["best"], "\\nbaseline pooled ex-vivo PQ: 0.390")
shutil.copy(WORK / "pseudo_cv.json", PERSIST)"""

CAND_MD = """## Cell 5: full competition-score validation, then test CSVs (~1-1.5 h)
Re-runs registration + pair classifier on the new held-out masks and computes
0.25·PQ_iv + 0.25·PQ_ex + 0.5·F1. It prints `FULL_CV` per mask setting. Then either
`NO_CANDIDATE` (stop, keep `submission_v7_grow15.csv`) or two `CANDIDATE_READY` files."""

CAND = """!python -u /content/pseudo_candidate_colab.py
for name in ("pseudo_full_cv.json", "submission_pseudo_candidate.csv", "submission_pseudo_candidate_cpgate.csv"):
    if (WORK / name).exists():
        shutil.copy(WORK / name, PERSIST)
        print("saved to Drive:", PERSIST / name)
print(json.dumps(json.loads((WORK / "pseudo_full_cv.json").read_text())["winner"], indent=2))"""

DL = """# Cell 6: download the candidate CSVs (only exist if cell 5 printed CANDIDATE_READY)
from google.colab import files
for name in ("submission_pseudo_candidate_cpgate.csv", "submission_pseudo_candidate.csv"):
    if (WORK / name).exists():
        files.download(str(WORK / name))"""


def cell(kind, text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": [cell("markdown", INTRO), cell("code", SETUP), cell("code", BUNDLE),
              cell("markdown", QUICK_MD), cell("code", QUICK), cell("markdown", FULL_MD), cell("code", FULL),
              cell("markdown", CAND_MD), cell("code", CAND), cell("code", DL)],
}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
