"""Test rotation/translation-invariant soma-neighborhood descriptors.

This is a diagnostic on saved mouse-held-out predictions. It checks whether
local point patterns can recover verified pairs without a known global pose.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


HERE = Path(__file__).resolve().parent


def describe(points, k=16, bins=12, radius=90.0):
    if not len(points):
        return np.empty((0, bins), dtype=np.float32)
    distance = cdist(points, points)
    np.fill_diagonal(distance, np.inf)
    # The global scale differs by about 10%; a log-distance histogram tolerates
    # missing cells better than ordered nearest-neighbour distances.
    grid = np.geomspace(4.0, radius, bins + 1)
    neighbors = np.sort(distance, axis=1)[:, :k]
    histogram = np.stack([np.histogram(row, bins=grid)[0] for row in neighbors])
    histogram = np.sqrt(histogram.astype(np.float32))
    histogram /= np.maximum(np.linalg.norm(histogram, axis=1, keepdims=True), 1e-6)
    return histogram


def main():
    with (HERE / "data/lab.pkl").open("rb") as stream:
        records = pickle.load(stream)
    for radius in (45.0, 65.0, 90.0, 130.0):
        for k in (6, 10, 16, 24):
            ranks = []
            for record in records.values():
                iv, ex = record["iv_c"], record["ex_c"]
                if len(iv) < 5 or len(ex) < 5:
                    continue
                a, b = describe(iv, k=k, radius=radius), describe(ex, k=k, radius=radius)
                sim = a @ b.T
                ex_by_gt = {int(v): j for j, v in enumerate(record["ex_link"]) if v >= 0}
                for i, gt_i in enumerate(record["iv_link"]):
                    if gt_i < 0:
                        continue
                    true_ex = [ex_by_gt[e] for a_gt, e in record["gt_pairs"]
                               if a_gt == gt_i and e in ex_by_gt]
                    if not true_ex:
                        continue
                    rank = 1 + int((sim[i] > sim[i, true_ex[0]]).sum())
                    ranks.append(rank)
            if not ranks:
                continue
            print(radius, k, "n", len(ranks), "top1", np.mean(np.array(ranks) <= 1),
                  "top5", np.mean(np.array(ranks) <= 5),
                  "top10", np.mean(np.array(ranks) <= 10),
                  "median", float(np.median(ranks)))


if __name__ == "__main__":
    main()
