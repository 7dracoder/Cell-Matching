"""Mouse-held-out test of per-cell ex-vivo boundary growth.

This is diagnostic only. No hidden-test CSV is written unless the held-out
result justifies a separate submission-building step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cellmatch import ROOT, pq_score, read_image, rle_to_labels  # noqa: E402
from research.common import load_truth  # noqa: E402
from research.size_lab import grow  # noqa: E402


CACHE = Path(__file__).parent / "data/heldout_labels.npz"
FRACS = (0.0, 0.10, 0.20, 0.35)


def mean_by_label(labels: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    sums = np.bincount(labels.ravel(), weights=values.ravel(), minlength=n + 1)[1:]
    return sums / np.maximum(counts, 1)


def prepare(sid: str, truth: dict, cache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = read_image(ROOT / "training" / sid.replace("__", "/") / "exvivo.tif")
    gt, _ = rle_to_labels(truth[sid]["exvivo_instances"], image.shape)
    labels = cache[f"{sid}|exvivo"].astype(np.int32)
    prob = cache[f"{sid}|exvivo|prob"].astype(np.float32)
    variants = [labels] + [grow(labels, prob=prob, frac=f) for f in FRACS[1:]]
    n, ng = int(labels.max()), int(gt.max())
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.float32)
    norm = image.astype(np.float32)
    p1, p99 = np.percentile(norm, (1, 99))
    norm = np.clip((norm - p1) / max(p99 - p1, 1), 0, 2)
    full = grow(labels)
    ring = np.where(labels == 0, full, 0)
    ring_area = np.bincount(ring.ravel(), minlength=n + 1)[1:].astype(np.float32)
    y, x = np.indices(labels.shape, dtype=np.float32)
    cx = mean_by_label(labels, x / image.shape[1], n)
    cy = mean_by_label(labels, y / image.shape[0], n)
    inside_i = mean_by_label(labels, norm, n)
    inside_p = mean_by_label(labels, prob, n)
    ring_i = mean_by_label(ring, norm, n)
    ring_p = mean_by_label(ring, prob, n)
    local_bg = ndi.gaussian_filter(norm, 7)
    contrast = mean_by_label(labels, norm - local_bg, n)
    feature = np.stack([
        np.log1p(area), np.log1p(ring_area), ring_area / np.maximum(area, 1),
        inside_i, inside_p, ring_i, ring_p, inside_i - ring_i,
        inside_p - ring_p, contrast, cx, cy,
        np.full(n, np.log1p(np.median(area))),
        np.full(n, np.log1p(image.shape[0])),
        np.full(n, np.log1p(image.shape[1])),
    ], axis=1).astype(np.float32)

    joint = np.bincount((labels.astype(np.int64) * (ng + 1) + gt).ravel(),
                        minlength=(n + 1) * (ng + 1)).reshape(n + 1, ng + 1)
    ga = joint.sum(0)
    base_iou = joint[1:, 1:] / np.maximum(area[:, None] + ga[None, 1:] - joint[1:, 1:], 1)
    linked = base_iou.argmax(1) + 1 if ng else np.zeros(n, np.int32)
    reward = np.zeros((n, len(FRACS)), np.float32)
    for action, candidate in enumerate(variants):
        j = np.bincount((candidate.astype(np.int64) * (ng + 1) + gt).ravel(),
                        minlength=(n + 1) * (ng + 1)).reshape(n + 1, ng + 1)
        ca = j.sum(1)[1:]
        hit = j[np.arange(1, n + 1), linked]
        union = ca + ga[linked] - hit
        iou = hit / np.maximum(union, 1)
        reward[:, action] = np.where(iou > 0.75, iou, 0)
    return feature, reward, area


def choose(models, features: np.ndarray, margin: float) -> np.ndarray:
    gain = np.stack([model.predict(features) for model in models], axis=1)
    winner = gain.argmax(1) + 1
    best = gain.max(1)
    return np.where(best > margin, winner, 0).astype(np.int16)


def evaluate(sid: str, choice: np.ndarray, truth: dict, cache) -> tuple[float, float]:
    image = read_image(ROOT / "training" / sid.replace("__", "/") / "exvivo.tif")
    gt, _ = rle_to_labels(truth[sid]["exvivo_instances"], image.shape)
    labels = cache[f"{sid}|exvivo"].astype(np.int32)
    prob = cache[f"{sid}|exvivo|prob"].astype(np.float32)
    variants = [labels] + [grow(labels, prob=prob, frac=f) for f in FRACS[1:]]
    selected = np.r_[0, choice]
    out = np.zeros_like(labels)
    for action, candidate in enumerate(variants):
        keep = (candidate > 0) & (selected[candidate] == action)
        out[keep] = candidate[keep]
    return pq_score(labels, gt)[0], pq_score(out, gt)[0]


def main() -> None:
    truth = load_truth()
    cache = np.load(CACHE)
    records = {}
    for sid in truth:
        records[sid] = prepare(sid, truth, cache)
        print("PREPARED", sid, records[sid][0].shape[0], flush=True)
    subjects = sorted({sid.split("__")[0] for sid in truth})
    predictions = {}
    for held in subjects:
        train_records = [v for sid, v in records.items() if not sid.startswith(held)]
        X = np.concatenate([v[0] for v in train_records])
        Y = np.concatenate([v[1] for v in train_records])
        models = []
        for action in range(1, len(FRACS)):
            model = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                                                  learning_rate=0.05, l2_regularization=5.0,
                                                  min_samples_leaf=30, random_state=41)
            model.fit(X, Y[:, action] - Y[:, 0])
            models.append(model)
        for sid, (features, _, _) in records.items():
            if sid.startswith(held):
                predictions[sid] = np.stack([m.predict(features) for m in models], axis=1)
    for margin in (0.0, 0.01, 0.025, 0.05, 0.1):
        before, after = [], []
        per = {}
        for sid, (features, _, _) in records.items():
            gain = predictions[sid]
            choice = np.where(gain.max(1) > margin, gain.argmax(1) + 1, 0).astype(np.int16)
            a, b = evaluate(sid, choice, truth, cache)
            before.append(a)
            after.append(b)
            per.setdefault(sid.split("__")[0], [[], []])
            per[sid.split("__")[0]][0].append(a)
            per[sid.split("__")[0]][1].append(b)
        print("MARGIN", margin, "PQ", round(float(np.mean(before)), 4),
              "->", round(float(np.mean(after)), 4),
              {s: (round(float(np.mean(v[0])), 4), round(float(np.mean(v[1])), 4))
               for s, v in per.items()}, flush=True)


if __name__ == "__main__":
    main()
