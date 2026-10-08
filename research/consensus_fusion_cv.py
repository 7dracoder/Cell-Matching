"""Fuse complementary ex-vivo cells only when independent models agree.

This tests saved held-out predictions without using hidden labels. Source masks
are never compared to ground truth until the candidate rule is fully applied.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cellmatch import pq_score, read_image, region_centers, rle_to_labels  # noqa: E402
from research.common import load_truth  # noqa: E402
from research.size_lab import grow  # noqa: E402

DATA = ROOT / "research/data"
PATHS = [DATA / name for name in ("heldout_labels_ens.npz", "heldout_labels_ensb.npz",
                                   "heldout_labels_v2a.npz", "heldout_labels_v2b.npz",
                                   "heldout_labels_v2c.npz")]


def boxes(labels):
    from scipy.ndimage import find_objects
    return find_objects(labels)


def candidates(base, alternatives, radius=4):
    centers = [region_centers(x) for x in alternatives]
    trees = [cKDTree(c) if len(c) else None for c in centers]
    boxes_by_source = [boxes(x) for x in alternatives]
    out = []
    for source, labels in enumerate(alternatives):
        for index, center in enumerate(centers[source], 1):
            box = boxes_by_source[source][index - 1]
            if box is None:
                continue
            cell = labels[box] == index
            area = int(cell.sum())
            if area < 17 or area > 350:
                continue
            if np.any(base[box][cell] > 0):
                continue
            votes = 1
            for other, tree in enumerate(trees):
                if other == source or tree is None:
                    continue
                nearby = tree.query_ball_point(center, radius)
                for partner in nearby:
                    other_box = boxes_by_source[other][partner]
                    if other_box is None:
                        continue
                    # Support is spatial and independent of exact mask area.
                    votes += 1
                    break
            out.append((votes, source, index, box, cell, area))
    return out


def fuse(base, alternatives, minimum_votes):
    result = base.copy()
    next_id = int(base.max())
    pool = candidates(base, alternatives)
    for votes, source, index, box, mask, area in sorted(pool, key=lambda x: (-x[0], x[1])):
        if votes < minimum_votes:
            break
        canvas = result[box]
        if np.count_nonzero(canvas[mask]) / area > 0.05:
            continue
        free = mask & (canvas == 0)
        if free.sum() < 17:
            continue
        next_id += 1
        canvas[free] = next_id
    _, inverse = np.unique(result, return_inverse=True)
    return inverse.reshape(result.shape).astype(np.int32), next_id - int(base.max())


def main():
    truth = load_truth()
    base_file = np.load(DATA / "v7/heldout_labels.npz")
    caches = [np.load(p) for p in PATHS]
    score = {k: [] for k in ("base", 2, 3, 4, 5)}
    added = {k: 0 for k in (2, 3, 4, 5)}
    mice = {k: {} for k in score}
    for sid, row in truth.items():
        subject = sid.split("__")[0]
        image = read_image(ROOT / "Project_2_Dataset/training" / sid.replace("__", "/") / "exvivo.tif")
        gt, _ = rle_to_labels(row["exvivo_instances"], image.shape)
        base = base_file[f"{sid}|exvivo"].astype(np.int32)
        base = grow(base, prob=base_file[f"{sid}|exvivo|prob"].astype(np.float32), frac=0.15)
        alternatives = [c[f"{sid}|exvivo"].astype(np.int32) for c in caches]
        pq = float(pq_score(base, gt)[0])
        score["base"].append(pq)
        mice["base"].setdefault(subject, []).append(pq)
        for k in added:
            out, n = fuse(base, alternatives, k)
            pq = float(pq_score(out, gt)[0])
            score[k].append(pq)
            added[k] += n
            mice[k].setdefault(subject, []).append(pq)
        print("REGION", sid, {k: round(v[-1], 3) for k, v in score.items()}, flush=True)
    for k, values in score.items():
        print("FINAL", k, "PQ", round(float(np.mean(values)), 4), "added", added.get(k, 0),
              "by_mouse", {s: round(float(np.mean(v)), 4) for s, v in mice[k].items()}, flush=True)


if __name__ == "__main__":
    main()
