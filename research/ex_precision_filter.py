"""Leak-free cell-level ex-vivo quality filter on v7 Cellpose masks.

Trains on two mice, predicts the third. Reports per-region PQ and matchable
verified pairs; does not write a submission until a held-out gain is shown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cellmatch import normalize, pq_score, read_image, rle_to_labels  # noqa: E402
from common import ROOT, load_truth  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = np.load(HERE / "data/v7/heldout_labels.npz")
TRUTH = load_truth()


def gt_hits(pred, gt):
    a, b = int(pred.max()), int(gt.max())
    joint = np.bincount((pred.astype(np.int64) * (b + 1) + gt).ravel(),
                        minlength=(a + 1) * (b + 1)).reshape(a + 1, b + 1)
    pc, gc = joint.sum(1), joint.sum(0)
    ii, jj = np.nonzero(joint[1:, 1:])
    iou = joint[ii + 1, jj + 1] / (pc[ii + 1] + gc[jj + 1] - joint[ii + 1, jj + 1])
    hit_pred = np.zeros(a + 1, bool)
    hit_gt = np.full(a + 1, -1, int)
    hit_pred[ii[iou > .75] + 1] = True
    hit_gt[ii[iou > .75] + 1] = jj[iou > .75]
    return hit_pred, hit_gt


def features(labels, image, cp):
    n = int(labels.max())
    if n == 0:
        return np.zeros((0, 16), np.float32)
    image, contrast = normalize(image, "exvivo")
    f = np.zeros((n, 16), np.float32)
    objects = ndi.find_objects(labels)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        y0, x0 = sl[0].start, sl[1].start
        y1, x1 = sl[0].stop, sl[1].stop
        pad = 3
        box = np.s_[max(0, y0-pad):min(labels.shape[0], y1+pad),
                    max(0, x0-pad):min(labels.shape[1], x1+pad)]
        m = labels[box] == i
        ys, xs = np.nonzero(m)
        area = len(xs)
        if not area:
            continue
        ring = ndi.binary_dilation(m, iterations=1) & (labels[box] == 0)
        ring_cp = cp[box][ring]
        ring_im = image[box][ring]
        vals_cp, vals_im, vals_con = cp[box][m], image[box][m], contrast[box][m]
        perim = float((m & ~ndi.binary_erosion(m)).sum())
        cov = np.cov(np.stack((xs, ys))) if area > 2 else np.eye(2)
        eig = np.linalg.eigvalsh(cov)
        ecc = np.sqrt(max(0, 1 - eig[0] / max(eig[1], 1e-3)))
        f[i-1] = [np.log1p(area), vals_im.mean(), vals_con.mean(), vals_cp.mean(),
                  vals_cp.max(), vals_cp.std(), area / ((y1-y0)*(x1-x0)),
                  4*np.pi*area/max(perim*perim, 1), ecc,
                  ring_cp.mean() if len(ring_cp) else 0,
                  ring_im.mean() if len(ring_im) else 0,
                  float((vals_cp > 2).mean()),
                  (x0+x1)/(2*labels.shape[1]), (y0+y1)/(2*labels.shape[0]),
                  np.log1p(n), np.log1p(labels.size)]
    return f


def relabel_filter(labels, probs, threshold):
    keep = np.r_[False, probs >= threshold]
    mapped = np.cumsum(keep).astype(np.int32)
    mapped[~keep] = 0
    return mapped[labels]


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


def main():
    records = {}
    for sid, t in TRUTH.items():
        subject, region = sid.split("__")
        path = Path(ROOT) / "training" / subject / region
        image = read_image(path / "exvivo.tif")
        pred = DATA[f"{sid}|exvivo"].astype(np.int32)
        cp = DATA[f"{sid}|exvivo|prob"].astype(np.float32)
        gt, ids = rle_to_labels(t["exvivo_instances"], image.shape)
        ivgt, ivids = rle_to_labels(t["invivo_instances"],
                                   read_image(path / "invivo.tif").shape)
        iv = DATA[f"{sid}|invivo"].astype(np.int32)
        ivhit = {ivids[i-1] for i in linked(iv, ivgt)}
        target, _ = gt_hits(pred, gt)
        records[sid] = (subject, pred, gt, ids, ivhit, t["match_pairs"],
                        features(pred, image, cp), target[1:])
        print("READY", sid, len(target)-1, int(target.sum()), flush=True)

    probs = {}
    for subject in sorted({v[0] for v in records.values()}):
        X = np.concatenate([r[6] for r in records.values() if r[0] != subject])
        y = np.concatenate([r[7] for r in records.values() if r[0] != subject])
        clf = HistGradientBoostingClassifier(max_iter=160, learning_rate=.05,
            max_leaf_nodes=15, l2_regularization=2, min_samples_leaf=20,
            random_state=0).fit(X, y)
        for sid, r in records.items():
            if r[0] == subject:
                probs[sid] = clf.predict_proba(r[6])[:, 1]

    for threshold in (0, .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8):
        scores, count, reach, subj = [], 0, 0, {}
        for sid, (subject, pred, gt, ids, ivhit, pairs, _, _) in records.items():
            out = relabel_filter(pred, probs[sid], threshold)
            pq = pq_score(out, gt)[0]
            scores.append(pq)
            subj.setdefault(subject, []).append(pq)
            count += int(out.max())
            ehit = {ids[i-1] for i in linked(out, gt)}
            reach += sum(a in ivhit and b in ehit for a, b in pairs)
        print("FILTER", threshold, "PQ", round(float(np.mean(scores)), 4),
              "CELLS", count, "REACH", reach,
              "BY_MOUSE", {k: round(float(np.mean(v)), 3) for k, v in subj.items()}, flush=True)


if __name__ == "__main__":
    main()
