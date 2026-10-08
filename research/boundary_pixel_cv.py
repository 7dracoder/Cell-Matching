"""Mouse-held-out pixel-level boundary refinement for ex-vivo Cellpose masks.

Only a thin ring around each predicted cell may change. Training targets are
assigned solely where a predicted cell overlaps an annotated cell; unknown
cells are ignored, rather than treated as background. Run with the project
virtualenv: .venv/bin/python research/boundary_pixel_cv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402

SOURCE = ROOT / "research/data/v7/heldout_labels.npz"
OUTPUT = ROOT / "research/data/boundary_pixel_cv.json"
THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75)
MAX_TRAIN_PER_IMAGE = 22000


def owner_and_band(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inside = labels > 0
    din = ndi.distance_transform_edt(inside).astype(np.float32)
    dout, indices = ndi.distance_transform_edt(~inside, return_indices=True)
    owner = labels.copy()
    outside = (~inside) & (dout <= 2.5)
    owner[outside] = labels[indices[0, outside], indices[1, outside]]
    band = ((inside & (din <= 2.5)) | outside) & (owner > 0)
    signed = np.where(inside, din, -dout).astype(np.float32)
    return owner, band, signed


def image_features(sample_id: str, labels: np.ndarray, probability: np.ndarray):
    image = P.read_image(P.region_path(sample_id) / "exvivo.tif").astype(np.float32)
    positive = image[image > 0]
    lo, hi = np.percentile(positive, (1, 99.5)) if positive.size else (0.0, 1.0)
    image = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.5)
    smooth = ndi.gaussian_filter(image, 1.0)
    background = ndi.gaussian_filter(image, 5.0)
    grad = np.hypot(ndi.sobel(smooth, 0), ndi.sobel(smooth, 1))
    owner, band, signed = owner_and_band(labels)
    count = np.bincount(labels.ravel())
    total = np.bincount(labels.ravel(), weights=smooth.ravel(), minlength=len(count))
    mean = total / np.maximum(count, 1)
    ys, xs = np.nonzero(band)
    ids = owner[ys, xs]
    features = np.stack((
        probability[ys, xs].astype(np.float32),
        image[ys, xs], smooth[ys, xs],
        (smooth - background)[ys, xs], grad[ys, xs],
        signed[ys, xs], np.log1p(count[ids]).astype(np.float32),
        (smooth[ys, xs] - mean[ids]).astype(np.float32),
    ), axis=1)
    return (ys, xs, ids), features


def linked_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    npred, ngt = int(pred.max()), int(gt.max())
    result = np.zeros(npred + 1, np.int32)
    if not npred or not ngt:
        return result
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    overlap = joint[1:, 1:]
    chosen = overlap.argmax(axis=1) + 1
    rows = np.arange(1, npred + 1)
    intersection = joint[rows, chosen]
    union = joint[rows].sum(axis=1) + joint[:, chosen].sum(axis=0) - intersection
    result[rows] = np.where(intersection / np.maximum(union, 1) >= 0.45, chosen, 0)
    return result


def predict_labels(base: np.ndarray, coordinates, features, probability, threshold):
    ys, xs, ids = coordinates
    answer = base.copy()
    inside_distance = ndi.distance_transform_edt(base > 0)
    answer[inside_distance <= 2.5] = 0
    selected = probability >= threshold
    answer[ys[selected], xs[selected]] = ids[selected]
    return answer


def main():
    truth = P.load_truth()
    cache = np.load(SOURCE)
    subjects = sorted({v["subject"] for v in truth.values()})
    records = {}
    for sid in truth:
        labels = cache[f"{sid}|exvivo"].astype(np.int32)
        prob = cache[f"{sid}|exvivo|prob"]
        coords, features = image_features(sid, labels, prob)
        gt = truth[sid]["exvivo"][0]
        mapping = linked_gt(labels, gt)
        ys, xs, ids = coords
        known = mapping[ids] > 0
        target = gt[ys[known], xs[known]] == mapping[ids[known]]
        train_x = features[known]
        train_y = target.astype(np.uint8)
        if len(train_x) > MAX_TRAIN_PER_IMAGE:
            rng = np.random.default_rng(419)
            keep = rng.choice(len(train_x), MAX_TRAIN_PER_IMAGE, replace=False)
            train_x, train_y = train_x[keep], train_y[keep]
        records[sid] = (labels, gt, coords, features, train_x, train_y)
        print("FEATURES", sid, len(features), len(train_x),
              int(train_y.sum()), flush=True)

    pq = {"baseline": [], **{str(t): [] for t in THRESHOLDS}}
    by_subject = {}
    for subject in subjects:
        xs = [records[s][4] for s in records if truth[s]["subject"] != subject]
        ys = [records[s][5] for s in records if truth[s]["subject"] != subject]
        x, y = np.concatenate(xs), np.concatenate(ys)
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=20,
                                               min_samples_leaf=120,
                                               learning_rate=0.07,
                                               l2_regularization=0.5,
                                               random_state=419)
        model.fit(x, y)
        values = {k: [] for k in pq}
        for sid in truth:
            if truth[sid]["subject"] != subject:
                continue
            base, gt, coords, feat, _, _ = records[sid]
            proba = model.predict_proba(feat)[:, 1]
            values["baseline"].append(pq_score(base, gt)[0])
            for threshold in THRESHOLDS:
                pred = predict_labels(base, coords, feat, proba, threshold)
                values[str(threshold)].append(pq_score(pred, gt)[0])
        for key in pq:
            pq[key].extend(values[key])
        by_subject[subject] = {k: float(np.mean(v)) for k, v in values.items()}
        print("FOLD", subject, by_subject[subject], flush=True)
    pooled = {k: float(np.mean(v)) for k, v in pq.items()}
    result = {"pooled": pooled, "by_subject": by_subject,
              "best_threshold": max(THRESHOLDS, key=lambda t: pooled[str(t)])}
    OUTPUT.write_text(json.dumps(result, indent=2))
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
