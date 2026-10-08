"""Evaluate ex-vivo-to-ex-vivo canvas alignment against held-out GT poses.

Uses image content only to estimate canvas shifts. GT affine landmarks are
consulted only after estimation for this diagnostic, never as input.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from bandpass_registration_probe import dog
from common import ROOT
from vote import P0

HERE = Path(__file__).resolve().parent


def prep(path: Path, factor: int, low: float, high: float):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, None, fx=1 / factor, fy=1 / factor,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    p1, p99 = np.percentile(image, [1, 99.8])
    image = np.clip((image - p1) / max(p99 - p1, 1), 0, 1)
    return dog(image, low, high).astype(np.float32)


def phase(a, b, factor):
    if a.shape != b.shape:
        return np.array([np.nan, np.nan]), 0.0
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, confidence = cv2.phaseCorrelate(a, b, window)
    return np.array(shift) * factor, confidence


def main():
    with (HERE / 'data/lab.pkl').open('rb') as f:
        records = pickle.load(f)
    for subject in sorted({r['subject'] for r in records.values()}):
        entries = [(sid, r) for sid, r in records.items()
                   if r['subject'] == subject and r['gt_M'] is not None]
        ref_sid, ref = entries[0]
        ref_m = ref['gt_M']
        ref_land = ref_m[:, :2] @ (P0 - ref['offset']) + ref_m[:, 2]
        print('\n', subject, 'ref', ref_sid, 'land', ref_land.round(1), flush=True)
        images = {}
        for factor, low, high in [(4, 1, 4), (4, 3, 12), (8, 1, 4), (8, 3, 12)]:
            ref_image = prep(Path(ROOT) / 'training' / ref_sid.replace('__', '/') /
                             'exvivo.tif', factor, low, high)
            errors = []
            for sid, r in entries:
                key = (sid, factor, low, high)
                if key not in images:
                    images[key] = prep(Path(ROOT) / 'training' / sid.replace('__', '/') /
                                       'exvivo.tif', factor, low, high)
                estimated, confidence = phase(ref_image, images[key], factor)
                m = r['gt_M']
                land = m[:, :2] @ (P0 - r['offset']) + m[:, 2]
                desired = land - ref_land
                residual = min(np.linalg.norm(estimated - desired),
                               np.linalg.norm(-estimated - desired))
                errors.append((sid, float(residual), float(confidence),
                               estimated.round().tolist(), desired.round().tolist()))
            vals = [x[1] for x in errors]
            print('PARAM', (factor, low, high), 'median', round(float(np.median(vals)), 1),
                  '<10', sum(v < 10 for v in vals), '<25', sum(v < 25 for v in vals),
                  'n', len(vals), flush=True)
            print('worst', sorted(errors, key=lambda x: x[1], reverse=True)[:4], flush=True)


if __name__ == '__main__':
    main()
