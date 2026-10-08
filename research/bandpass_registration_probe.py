"""Test band-pass image evidence for cross-modal registration.

Inspired by the independent multi-scale channels in the supplied constellation
pipeline. This evaluates true and predicted poses on held-out training mice;
it does not alter a submission unless the evidence separates them reliably.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from common import ROOT
from reg_lab import err


HERE = Path(__file__).resolve().parent


def dog(image, low, high):
    image = image.astype(np.float32)
    a = cv2.GaussianBlur(image, (0, 0), low)
    b = cv2.GaussianBlur(image, (0, 0), high)
    return a - b


def pose_score(iv, ex, M, low, high):
    if M is None:
        return np.nan
    h, w = iv.shape
    inv = cv2.invertAffineTransform(M.astype(np.float32))
    ex_aligned = cv2.warpAffine(ex, inv, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(ex, dtype=np.uint8), inv, (w, h),
                           flags=cv2.INTER_NEAREST) > 0
    # Avoid the large zero-vs-zero regions outside the tissue.
    valid &= ex_aligned > np.percentile(ex_aligned[valid], 8) if valid.any() else False
    if valid.sum() < 500:
        return np.nan
    a = dog(iv, low, high)[valid]
    b = dog(ex_aligned, low, high)[valid]
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def main():
    with (HERE / "data/lab.pkl").open("rb") as stream:
        records = pickle.load(stream)
    with (HERE / "data/reg_window_vote5.pkl").open("rb") as stream:
        chosen = pickle.load(stream)
    cases = []
    for sid, record in records.items():
        Mtrue = record["gt_M"]
        pred = chosen[sid][0]
        if Mtrue is None or pred is None or err(record, pred) < 10:
            continue
        subject, region = sid.split("__")
        iv = cv2.imread(str(Path(ROOT) / "training" / subject / region / "invivo.tif"), cv2.IMREAD_UNCHANGED)
        ex = cv2.imread(str(Path(ROOT) / "training" / subject / region / "exvivo.tif"), cv2.IMREAD_UNCHANGED)
        if iv is None or ex is None:
            continue
        scores = []
        for low, high in ((1, 4), (2, 8), (4, 16), (8, 32)):
            true = pose_score(iv, ex, Mtrue, low, high)
            wrong = pose_score(iv, ex, pred, low, high)
            scores.append((true, wrong))
        cases.append((sid, scores))
    print("WRONG_POSE_CASES", len(cases))
    for j, scale in enumerate(((1, 4), (2, 8), (4, 16), (8, 32))):
        differences = np.array([row[1][j][0] - row[1][j][1] for row in cases])
        differences = differences[np.isfinite(differences)]
        print(scale, "n", len(differences), "true_better", int((differences > 0).sum()),
              "mean_margin", float(np.mean(differences)) if len(differences) else np.nan)


if __name__ == "__main__":
    main()
