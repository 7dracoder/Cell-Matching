"""Conditional full-score gate for the moderate sparse-label Colab experiment.

Run after mid_loss_cv.py. A CSV is produced only if held-out segmentation and
full matching beat the existing validation stack. Failure creates no fake
"new" submission.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

WORK = Path("/content/work")
CV_FILE = WORK / "mid_loss_cv.json"
FLOWS = WORK / "flows/exvivo_mid25_training.npz"
EVAL = WORK / "mid25_eval_compatible.json"
SUBJECTS = ("subject_5d294c", "subject_b2ba5e", "subject_db6b8b")


def main():
    if not CV_FILE.exists():
        raise FileNotFoundError(f"First-stage cross-validation has not completed: {CV_FILE}")
    cv = json.loads(CV_FILE.read_text())
    best = cv["best"]
    print("MID25_STAGE1", best, flush=True)
    if best["pooled_pq"] <= 0.3974:
        print("NO_CANDIDATE: held-out ex-vivo PQ did not improve", flush=True)
        return
    required = [
        WORK / "research/data/base_heldout_labels.npz",
        Path("/content/submission_v7_grow15.csv"),
        Path("/content/validate_submission.py"),
        Path("/content/partial_candidate.py"),
        Path("/content/v8_colab.py"),
        Path("/content/partial_loss_cv.py"),
        WORK / "Project_2_Dataset/hidden_test",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing full-validation inputs: " + ", ".join(missing))
    merged = {}
    for subject in SUBJECTS:
        path = WORK / "flows" / f"exvivo_mid0p25_{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as fold:
            for key in fold.files:
                merged[key] = fold[key]
    np.savez_compressed(FLOWS, **merged)
    EVAL.write_text(json.dumps({"all": [
        {"cellprob": row["cellprob"], "flow": row["flow"],
         "ex_pq": row["pooled_pq"]}
        for row in cv["all"]]}, indent=2))
    env = os.environ.copy()
    env.update({
        "CANDIDATE_FLOW_FILE": str(FLOWS),
        "CANDIDATE_EVAL_FILE": str(EVAL),
        "CANDIDATE_OUT": "/content/submission_mid25_candidate.csv",
        "CANDIDATE_TAG": "mid25",
        "CANDIDATE_BACKGROUND_WEIGHT": "0.25",
    })
    subprocess.run([sys.executable, "-u", "/content/partial_candidate.py"],
                   cwd="/content/work", env=env, check=True)


if __name__ == "__main__":
    main()
