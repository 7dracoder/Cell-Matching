"""Evaluate classical ex-vivo soma proposals as additions to saved Cellpose masks.

The validation split is by mouse, never by region. This is an experiment, not a
replacement for the score-0.47488 submission unless it wins held-out PQ.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

from cellmatch import ROOT, classical_segment, normalize, pq_score, read_image, region_centers, rle_to_labels


CV_MASKS = Path(__file__).parent / "data/v7/heldout_labels.npz"
CLASSICAL_THRESHOLD = 0.23


def candidate_features(image: np.ndarray, candidates: np.ndarray, existing: np.ndarray,
                       cellprob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized morphology and appearance measurements for each proposal."""
    n = int(candidates.max())
    if not n:
        return np.empty((0, 19), np.float32), np.empty(0, np.float32)
    flat = candidates.ravel()
    y, x = np.nonzero(candidates)
    ids = candidates[y, x]
    area = np.bincount(ids, minlength=n + 1)[1:].astype(np.float32)
    features = normalize(image, "exvivo")
    intensity, contrast = features
    p = cellprob.astype(np.float32)

    def mean(values: np.ndarray) -> np.ndarray:
        return (np.bincount(flat, weights=values.ravel(), minlength=n + 1)[1:] /
                np.maximum(area, 1))

    xi = x.astype(np.float32)
    yi = y.astype(np.float32)
    sx = np.bincount(ids, weights=xi, minlength=n + 1)[1:]
    sy = np.bincount(ids, weights=yi, minlength=n + 1)[1:]
    cx, cy = sx / area, sy / area
    vx = np.bincount(ids, weights=xi * xi, minlength=n + 1)[1:] / area - cx * cx
    vy = np.bincount(ids, weights=yi * yi, minlength=n + 1)[1:] / area - cy * cy
    cov = np.bincount(ids, weights=xi * yi, minlength=n + 1)[1:] / area - cx * cy
    anisotropy = np.sqrt((vx - vy) ** 2 + 4 * cov ** 2) / np.maximum(vx + vy, 0.01)

    minx = np.full(n + 1, image.shape[1], np.int32)
    miny = np.full(n + 1, image.shape[0], np.int32)
    maxx = np.zeros(n + 1, np.int32)
    maxy = np.zeros(n + 1, np.int32)
    np.minimum.at(minx, ids, x)
    np.minimum.at(miny, ids, y)
    np.maximum.at(maxx, ids, x)
    np.maximum.at(maxy, ids, y)
    bw = (maxx[1:] - minx[1:] + 1).astype(np.float32)
    bh = (maxy[1:] - miny[1:] + 1).astype(np.float32)
    bbox_fill = area / (bw * bh)

    edge = candidates > 0
    edge &= ((candidates != np.roll(candidates, 1, 0)) |
             (candidates != np.roll(candidates, -1, 0)) |
             (candidates != np.roll(candidates, 1, 1)) |
             (candidates != np.roll(candidates, -1, 1)))
    perimeter = np.bincount(candidates[edge], minlength=n + 1)[1:].astype(np.float32)
    circularity = 4 * np.pi * area / np.maximum(perimeter * perimeter, 1)
    overlap = mean((existing > 0).astype(np.float32))

    existing_centers = region_centers(existing)
    if len(existing_centers):
        nearest = cKDTree(existing_centers).query(np.stack([cx, cy], axis=1))[0].astype(np.float32)
    else:
        nearest = np.full(n, 1000, np.float32)
    pmax = ndi.maximum(p, labels=candidates, index=np.arange(1, n + 1)).astype(np.float32)
    imax = ndi.maximum(intensity, labels=candidates, index=np.arange(1, n + 1)).astype(np.float32)
    cmax = ndi.maximum(contrast, labels=candidates, index=np.arange(1, n + 1)).astype(np.float32)
    local_bg = ndi.gaussian_filter(intensity, 7)
    matrix = np.stack([
        np.log1p(area), cx / image.shape[1], cy / image.shape[0],
        np.log1p(bw), np.log1p(bh), bbox_fill, circularity, anisotropy,
        mean(intensity), mean(contrast), mean(p), imax, cmax, pmax,
        mean(local_bg), mean(intensity - local_bg),
        overlap, np.log1p(nearest), area / np.maximum(perimeter, 1),
    ], axis=1).astype(np.float32)
    return matrix, overlap


def successful_instances(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate truth links and IoUs under the exact 0.75 criterion."""
    npred, ngt = int(pred.max()), int(gt.max())
    best_gt = np.zeros(npred, np.int32)
    best_iou = np.zeros(npred, np.float32)
    if not npred or not ngt:
        return best_gt, best_iou
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pa, ga = joint.sum(1), joint.sum(0)
    rows, cols = np.nonzero(joint[1:, 1:])
    rows += 1
    cols += 1
    iou = joint[rows, cols] / (pa[rows] + ga[cols] - joint[rows, cols])
    for row, col, value in zip(rows, cols, iou):
        if value > best_iou[row - 1]:
            best_iou[row - 1] = value
            best_gt[row - 1] = col
    return best_gt, best_iou


def merge(existing: np.ndarray, proposals: np.ndarray, selected: np.ndarray) -> np.ndarray:
    labels = existing.copy()
    if not selected.any():
        return labels
    lut = np.zeros(int(proposals.max()) + 1, np.int32)
    lut[1:] = selected.astype(np.int32) * (np.arange(1, len(selected) + 1) + int(existing.max()))
    add = lut[proposals]
    labels[(labels == 0) & (add > 0)] = add[(labels == 0) & (add > 0)]
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def main() -> None:
    rows = list(csv.DictReader((ROOT / "training/train_ground_truth.csv").open()))
    data = np.load(CV_MASKS)
    records = {}
    for row in rows:
        sid = row["sample_id"]
        image = read_image(ROOT / "training" / sid.replace("__", "/") / "exvivo.tif")
        gt, _ = rle_to_labels(json.loads(row["exvivo_instances"]), image.shape)
        old = data[f"{sid}|exvivo"].astype(np.int32)
        cellprob = data[f"{sid}|exvivo|prob"]
        proposals = classical_segment(image, "exvivo", CLASSICAL_THRESHOLD)
        x, overlap = candidate_features(image, proposals, old, cellprob)
        old_links, old_iou = successful_instances(old, gt)
        new_links, new_iou = successful_instances(proposals, gt)
        old_hit = set(old_links[old_iou > .75].tolist())
        y = np.array([(new_iou[i] > .75 and new_links[i] not in old_hit)
                      for i in range(len(new_iou))], np.uint8)
        valid = overlap < .025
        records[sid] = (x, y, valid, proposals, old, gt)
        print(sid, "candidates", len(x), "valid", int(valid.sum()), "new TP", int((y & valid).sum()), flush=True)

    subjects = sorted({s.split("__")[0] for s in records})
    predictions = {}
    for held in subjects:
        train = [v for s, v in records.items() if not s.startswith(held)]
        tx = np.concatenate([r[0][r[2]] for r in train])
        ty = np.concatenate([r[1][r[2]] for r in train])
        print("train", held, len(tx), "positive", int(ty.sum()), flush=True)
        clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31,
                                             learning_rate=.055, l2_regularization=2.0,
                                             random_state=1)
        clf.fit(tx, ty)
        for sid, (x, y, valid, proposals, old, gt) in records.items():
            if sid.startswith(held):
                predictions[sid] = clf.predict_proba(x)[:, 1] * valid

    for threshold in (.05, .1, .2, .3, .4, .5, .7, .9):
        print("THRESHOLD", threshold)
        all_old, all_new = [], []
        for subject in subjects:
            old_score, new_score, added = [], [], 0
            for sid, (_, _, _, proposals, old, gt) in records.items():
                if not sid.startswith(subject):
                    continue
                selected = predictions[sid] >= threshold
                merged = merge(old, proposals, selected)
                old_score.append(pq_score(old, gt)[0])
                new_score.append(pq_score(merged, gt)[0])
                added += int(selected.sum())
            all_old += old_score
            all_new += new_score
            print(subject, "old", round(float(np.mean(old_score)), 4), "new", round(float(np.mean(new_score)), 4), "added", added)
        print("ALL", round(float(np.mean(all_old)), 4), round(float(np.mean(all_new)), 4), flush=True)


if __name__ == "__main__":
    main()
