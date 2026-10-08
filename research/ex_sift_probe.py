"""Probe ex-vivo frame alignment for regions sharing the exact in-vivo image."""
import hashlib
import pickle
from pathlib import Path

import cv2
import numpy as np

from common import ROOT
from vote import P0

HERE = Path(__file__).resolve().parent


def load(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    lo, hi = np.percentile(im, (5, 99.8))
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1) * 255,
                   0, 255).astype(np.uint8)


def land(row):
    m = row['gt_M']
    return m[:, :2] @ (P0 - row['offset']) + m[:, 2]


def main():
    with (HERE / 'data/lab.pkl').open('rb') as f:
        records = pickle.load(f)
    groups = {}
    for sid, row in records.items():
        if row['gt_M'] is None:
            continue
        path = Path(ROOT) / 'training' / sid.replace('__', '/') / 'invivo.tif'
        key = row['subject'], hashlib.md5(path.read_bytes()).hexdigest()
        groups.setdefault(key, []).append(sid)
    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher()
    for group in groups.values():
        if len(group) < 2:
            continue
        reference = group[0]
        ref_path = Path(ROOT) / 'training' / reference.replace('__', '/') / 'exvivo.tif'
        ref_im = load(ref_path)
        ka, da = sift.detectAndCompute(ref_im, None)
        for sid in group[1:]:
            target_path = Path(ROOT) / 'training' / sid.replace('__', '/') / 'exvivo.tif'
            kb, db = sift.detectAndCompute(load(target_path), None)
            raw = matcher.knnMatch(da, db, k=2)
            good = [a for a, b in raw if a.distance < .72 * b.distance]
            if len(good) < 3:
                print(reference[-6:], sid[-6:], 'matches', len(good), flush=True)
                continue
            a = np.float32([ka[m.queryIdx].pt for m in good])
            b = np.float32([kb[m.trainIdx].pt for m in good])
            fit, inlier = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                        ransacReprojThreshold=5)
            expected = land(records[sid]) - land(records[reference])
            predicted = fit[:, :2] @ land(records[reference]) + fit[:, 2] - land(records[reference]) if fit is not None else None
            print(reference[-6:], sid[-6:], 'matches', len(good),
                  'inliers', int(inlier.sum()) if inlier is not None else 0,
                  'pred_shift', None if predicted is None else predicted.round().tolist(),
                  'gt_shift', expected.round().tolist(),
                  'resid', None if predicted is None else round(float(np.linalg.norm(predicted - expected)), 1),
                  flush=True)


if __name__ == '__main__':
    main()
