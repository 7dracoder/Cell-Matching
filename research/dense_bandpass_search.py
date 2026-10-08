"""Test direct image-derived registration hypotheses on held-out mice.

No ground-truth information enters the search. Ground truth is used only to
evaluate candidates and decide whether the approach merits a submission.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from bandpass_registration_probe import dog, pose_score
from common import ROOT
from reg_lab import err
from registration import refine
from vote import vote_modes, P0

HERE = Path(__file__).resolve().parent


def search(iv: np.ndarray, ex: np.ndarray, scale_down: float = 0.25,
           top: int = 20, modes=None, offset=None):
    # Images are filtered at native resolution before resampling, so the band
    # pass retains the same physical scale as the pose-verification signal.
    iv_band = cv2.resize(dog(iv, 2, 8), None, fx=scale_down, fy=scale_down,
                         interpolation=cv2.INTER_AREA)
    ex_band = cv2.resize(dog(ex, 2, 8), None, fx=scale_down, fy=scale_down,
                         interpolation=cv2.INTER_AREA)
    h, w = iv_band.shape
    center = np.array([w / 2, h / 2], dtype=np.float32)
    half = min(h, w) // 2 - 12
    # One compact central patch avoids artificial black corners from rotation.
    box = (slice(int(center[1]) - half, int(center[1]) + half),
           slice(int(center[0]) - half, int(center[0]) + half))
    proposals = []
    for angle in np.arange(-19.0, 14.1, 2.0):
        for mag in np.arange(0.93, 1.051, 0.02):
            rotation = cv2.getRotationMatrix2D(tuple(center), -float(angle), float(mag))
            rotated = cv2.warpAffine(iv_band, rotation, (w, h), flags=cv2.INTER_LINEAR)
            patch = rotated[box]
            response = cv2.matchTemplate(ex_band, patch, cv2.TM_CCOEFF_NORMED)
            if modes is not None:
                yy, xx = np.indices(response.shape)
                ex_points = np.stack((xx + half, yy + half), axis=-1) / scale_down
                theta = np.deg2rad(angle)
                lin = mag * np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]], dtype=np.float32)
                allowed = np.zeros(response.shape, dtype=bool)
                for mode in modes:
                    if abs(angle - mode['angle']) > 7:
                        continue
                    predicted = mode['landing'] + lin @ (center / scale_down - (P0 - offset))
                    allowed |= np.sum((ex_points - predicted) ** 2, axis=-1) <= 150 ** 2
                response = np.where(allowed, response, -2.0)
            if response.max() <= -1:
                continue
            for index in np.argpartition(response.ravel(), -top)[-top:]:
                y, x = np.unravel_index(int(index), response.shape)
                if response[y, x] <= -1:
                    continue
                ex_center = np.array([x + half, y + half], dtype=np.float32) / scale_down
                theta = np.deg2rad(angle)
                linear = mag * np.array([[np.cos(theta), -np.sin(theta)],
                                         [np.sin(theta), np.cos(theta)]], dtype=np.float32)
                iv_center = center / scale_down
                matrix = np.c_[linear, ex_center - linear @ iv_center]
                proposals.append((float(response[y, x]), matrix))
    proposals.sort(key=lambda z: z[0], reverse=True)
    unique = []
    for score, matrix in proposals:
        center_ex = matrix[:, :2] @ (center / scale_down) + matrix[:, 2]
        if any(np.linalg.norm(center_ex - item[2]) < 20 and
               abs(np.arctan2(matrix[1, 0], matrix[0, 0]) - item[3]) < .05
               for item in unique):
            continue
        unique.append((score, matrix, center_ex,
                       np.arctan2(matrix[1, 0], matrix[0, 0])))
        if len(unique) >= top:
            break
    return [(score, matrix) for score, matrix, _, _ in unique]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    with (HERE / 'data/lab.pkl').open('rb') as f:
        records = pickle.load(f)
    with (HERE / 'data/reg_window_vote5.pkl').open('rb') as f:
        previous = pickle.load(f)
    with (HERE / 'data/vote_cands.pkl').open('rb') as f:
        candidates = pickle.load(f)
    modes = {subject: vote_modes({sid: candidates[sid] for sid, record in records.items()
                                 if record['subject'] == subject})
             for subject in {record['subject'] for record in records.values()}}
    cases = sorted(records, key=lambda sid: err(records[sid], previous[sid][0]),
                   reverse=True)
    for sid in cases[:args.limit]:
        subject, region = sid.split('__')
        folder = Path(ROOT) / 'training' / subject / region
        iv = cv2.imread(str(folder / 'invivo.tif'), cv2.IMREAD_UNCHANGED)
        ex = cv2.imread(str(folder / 'exvivo.tif'), cv2.IMREAD_UNCHANGED)
        found = search(iv, ex, modes=modes[subject], offset=records[sid]['offset'])
        results = [(round(score, 4), round(err(records[sid], matrix), 1),
                    round(pose_score(iv, ex, matrix, 2, 8), 4))
                   for score, matrix in found]
        truth = records[sid]['gt_M']
        true_image = pose_score(iv, ex, truth, 2, 8) if truth is not None else np.nan
        refined = []
        for image_score, matrix in found:
            fit, votes = refine(records[sid]['iv_c'], records[sid]['ex_c'], matrix)
            refined.append((round(image_score, 4), round(votes, 1),
                            round(err(records[sid], fit), 1),
                            round(pose_score(iv, ex, fit, 2, 8), 4)))
        print(sid, 'old', round(err(records[sid], previous[sid][0]), 1),
              'true_image', round(true_image, 4), 'top', results[:5],
              'best_error', min(x[1] for x in results),
              'refined_best', min(x[2] for x in refined),
              'refined_top', sorted(refined, key=lambda z: z[1], reverse=True)[:3],
              flush=True)


if __name__ == '__main__':
    main()
