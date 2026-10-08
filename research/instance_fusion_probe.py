"""Check whether saved ex-vivo models offer complementary IoU>.75 cells.

Read-only diagnostic; no hidden labels or submission files are used.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cellmatch import pq_score  # noqa: E402
from research.common import load_truth  # noqa: E402
from research.size_lab import grow  # noqa: E402

DATA = ROOT / "research/data"
SOURCES = {
    "v7": DATA / "v7/heldout_labels.npz",
    "ens": DATA / "heldout_labels_ens.npz",
    "ensb": DATA / "heldout_labels_ensb.npz",
    "v2a": DATA / "heldout_labels_v2a.npz",
    "v2b": DATA / "heldout_labels_v2b.npz",
    "v2c": DATA / "heldout_labels_v2c.npz",
}


def hits(pred, gt):
    n, m = int(pred.max()), int(gt.max())
    if not n or not m:
        return set()
    joint = np.bincount((pred.astype(np.int64) * (m + 1) + gt).ravel(),
                        minlength=(n + 1) * (m + 1)).reshape(n + 1, m + 1)
    pa, ga = joint.sum(1), joint.sum(0)
    i, j = np.nonzero(joint[1:, 1:])
    inter = joint[i + 1, j + 1]
    good = inter / (pa[i + 1] + ga[j + 1] - inter) > 0.75
    return set((j[good] + 1).tolist())


def main():
    truth = load_truth()
    caches = {name: np.load(path) for name, path in SOURCES.items()}
    counts = {name: {"pq": [], "hits": 0, "new": 0} for name in SOURCES}
    counts["v7+grow15"] = {"pq": [], "hits": 0, "new": 0}
    for sid, row in truth.items():
        gt = row["exvivo_instances"]
        # common.load_truth() preserves RLE dictionaries, unlike pipeline.load_truth().
        from cellmatch import rle_to_labels, read_image
        image = read_image(ROOT / "Project_2_Dataset/training" / sid.replace("__", "/") / "exvivo.tif")
        gt, _ = rle_to_labels(gt, image.shape)
        labels = {name: cache[f"{sid}|exvivo"].astype(np.int32)
                  for name, cache in caches.items()}
        base = labels["v7"]
        labels["v7+grow15"] = grow(base, prob=caches["v7"][f"{sid}|exvivo|prob"].astype(np.float32), frac=0.15)
        base_hit = hits(labels["v7+grow15"], gt)
        for name, pred in labels.items():
            matched = hits(pred, gt)
            counts[name]["pq"].append(float(pq_score(pred, gt)[0]))
            counts[name]["hits"] += len(matched)
            counts[name]["new"] += len(matched - base_hit)
    for name, count in counts.items():
        print(name, "PQ", round(float(np.mean(count["pq"])), 4),
              "TP", count["hits"], "complementary TP", count["new"])


if __name__ == "__main__":
    main()
