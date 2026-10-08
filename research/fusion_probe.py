"""Held-out Cellpose-SAM + DINO proposal fusion probe; no hidden labels.

Run on the existing Colab runtime after v8. This is diagnostic and never writes
a submission. It measures whether DINO adds correct ex-vivo instances beyond SAM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import normalize, pq_score  # noqa: E402

ROOT = Path("/content/work")
BASE = np.load(ROOT / "flows/exvivo_base+_v2_aug0_training.npz")
DINO = np.load(ROOT / "research/data/heldout_labels.npz")
TRUTH = P.load_truth()
CFG = {"cellprob": 0.0, "flow": 0.1}


def add_dino(base, dino, cp, image, max_overlap, min_cp, min_brightness):
    n = int(dino.max())
    if n == 0:
        return base.copy(), 0
    counts = np.bincount(dino.ravel(), minlength=n + 1)
    overlap = np.bincount(dino[base > 0].ravel(), minlength=n + 1)
    mean_cp = np.bincount(dino.ravel(), cp.ravel(), minlength=n + 1) / np.maximum(counts, 1)
    norm = normalize(image, "exvivo")[0]
    mean_b = np.bincount(dino.ravel(), norm.ravel(), minlength=n + 1) / np.maximum(counts, 1)
    keep = ((overlap / np.maximum(counts, 1) <= max_overlap) &
            (mean_cp >= min_cp) & (mean_b >= min_brightness) &
            (counts >= 17) & (counts <= 350))
    keep[0] = False
    ids = np.flatnonzero(keep)
    mapped = np.zeros(n + 1, np.int32)
    mapped[ids] = np.arange(int(base.max()) + 1, int(base.max()) + 1 + len(ids))
    result = base.copy()
    vacant = result == 0
    result[vacant] = mapped[dino[vacant]]
    return result, len(ids)


def linked(pred, gt):
    a, b = int(pred.max()), int(gt.max())
    if not a or not b:
        return set()
    z = np.bincount((pred.astype(np.int64) * (b + 1) + gt).ravel(),
                    minlength=(a + 1) * (b + 1)).reshape(a + 1, b + 1)
    pc, gc = z.sum(1), z.sum(0)
    i, j = np.nonzero(z[1:, 1:])
    iou = z[i + 1, j + 1] / (pc[i + 1] + gc[j + 1] - z[i + 1, j + 1])
    return set((j[iou > .75] + 1).tolist())


def run():
    rows = []
    for sid, item in TRUTH.items():
        flows = (BASE[f"{sid}|dp"], BASE[f"{sid}|cp"], int(BASE[f"{sid}|n"]))
        base = P.cellpose_labels(flows, "exvivo", CFG)
        dino = DINO[f"{sid}|exvivo"].astype(np.int32)
        cp = DINO[f"{sid}|exvivo|prob"].astype(np.float32)
        image = P.read_image(P.region_path(sid) / "exvivo.tif")
        gt, ids = item["exvivo"]
        iv = DINO[f"{sid}|invivo"].astype(np.int32)
        ivgt, ivids = item["invivo"]
        ihit = {ivids[i - 1] for i in linked(iv, ivgt)}
        rows.append((sid, base, dino, cp, image, gt, ids, ihit, item["pairs"]))
    baseline = np.mean([pq_score(b, gt)[0] for _, b, _, _, _, gt, _, _, _ in rows])
    print("BASE_PQ", round(float(baseline), 4), flush=True)
    for overlap in (0.0, 0.1, 0.25):
        for min_cp in (-1, 0, 0.5, 1, 2, 3):
            for min_b in (0.0, 0.4):
                pqs, adds, reach = [], 0, 0
                for _, base, dino, cp, image, gt, ids, ihit, pairs in rows:
                    pred, added = add_dino(base, dino, cp, image, overlap, min_cp, min_b)
                    pqs.append(pq_score(pred, gt)[0])
                    adds += added
                    ehit = {ids[i - 1] for i in linked(pred, gt)}
                    reach += sum(a in ihit and b in ehit for a, b in pairs)
                print("FUSE", overlap, min_cp, min_b, "PQ", round(float(np.mean(pqs)), 4),
                      "ADDED", adds, "REACH", reach, flush=True)


if __name__ == "__main__":
    run()
