"""Add only nonoverlapping complementary instances from saved ex-vivo models.

Assesses leave-one-mouse-out predictions, never writes a hidden-test CSV.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cellmatch import pq_score, read_image, rle_to_labels  # noqa: E402
from research.common import load_truth  # noqa: E402
from research.size_lab import grow  # noqa: E402

DATA = ROOT / "research/data"
SOURCES = {
    "ens": DATA / "heldout_labels_ens.npz",
    "ensb": DATA / "heldout_labels_ensb.npz",
    "v2a": DATA / "heldout_labels_v2a.npz",
    "v2b": DATA / "heldout_labels_v2b.npz",
    "v2c": DATA / "heldout_labels_v2c.npz",
}


def add(base, candidate, overlap_limit):
    sizes = np.bincount(candidate.ravel())
    crossing = np.bincount(candidate[base > 0], minlength=len(sizes))
    keep = crossing / np.maximum(sizes, 1) <= overlap_limit
    keep[0] = False
    result = base.copy()
    addable = (result == 0) & keep[candidate]
    result[addable] = candidate[addable] + int(base.max())
    _, inverse = np.unique(result, return_inverse=True)
    return inverse.reshape(result.shape).astype(np.int32), int(keep.sum())


def main():
    truth = load_truth()
    base_file = np.load(DATA / "v7/heldout_labels.npz")
    others = {name: np.load(path) for name, path in SOURCES.items()}
    limits = (0.0, 0.05, 0.1, 0.25, 0.5)
    metrics = {(name, limit): {"pq": [], "added": 0, "mice": {}}
               for name in SOURCES for limit in limits}
    baseline = []
    for sid, row in truth.items():
        subject = sid.split("__")[0]
        image = read_image(ROOT / "Project_2_Dataset/training" / sid.replace("__", "/") / "exvivo.tif")
        gt, _ = rle_to_labels(row["exvivo_instances"], image.shape)
        base = base_file[f"{sid}|exvivo"].astype(np.int32)
        base = grow(base, prob=base_file[f"{sid}|exvivo|prob"].astype(np.float32), frac=0.15)
        baseline.append(pq_score(base, gt)[0])
        for name, cache in others.items():
            candidate = cache[f"{sid}|exvivo"].astype(np.int32)
            for limit in limits:
                fused, n = add(base, candidate, limit)
                score = float(pq_score(fused, gt)[0])
                entry = metrics[(name, limit)]
                entry["pq"].append(score)
                entry["added"] += n
                entry["mice"].setdefault(subject, []).append(score)
    print("BASE", round(float(np.mean(baseline)), 4), flush=True)
    for (name, limit), entry in metrics.items():
        print(name, limit, "PQ", round(float(np.mean(entry["pq"])), 4),
              "ADDED", entry["added"],
              "BY_MOUSE", {s: round(float(np.mean(v)), 4) for s, v in entry["mice"].items()}, flush=True)


if __name__ == "__main__":
    main()
