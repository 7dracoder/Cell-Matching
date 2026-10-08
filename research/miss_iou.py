"""Best-IoU distribution of GT ex-vivo cells against held-out predictions (all vs paired cells)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from cellmatch import read_image, rle_to_labels  # noqa: E402
from common import ROOT, load_truth  # noqa: E402
from size_lab import grow  # noqa: E402


def best_iou(pred, gt):
    npred, ngt = int(pred.max()), int(gt.max())
    out = np.zeros(ngt)
    if not npred or not ngt:
        return out
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc, gc = joint.sum(1), joint.sum(0)
    inter = joint[1:, 1:]
    iou = inter / (pc[1:, None] + gc[None, 1:] - inter)
    return iou.max(0)


data = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/v7/heldout_labels.npz")
truth = load_truth()
bins = [0, 0.01, 0.3, 0.5, 0.6, 0.7, 0.75, 1.01]
res = {}
for sid, t in truth.items():
    subj = sid.split("__")[0][-6:]
    shape = read_image(os.path.join(ROOT, "training", *sid.split("__"), "exvivo.tif")).shape
    gt, ids = rle_to_labels(t["exvivo_instances"], shape)
    lab = grow(data[f"{sid}|exvivo"].astype(np.int32), prob=data[f"{sid}|exvivo|prob"].astype(np.float32), frac=0.15)
    b = best_iou(lab, gt)
    paired = np.array([i in {e for _, e in t["match_pairs"]} for i in ids])
    for name, sel in (("all", np.ones(len(ids), bool)), ("paired", paired)):
        res.setdefault((subj, name), []).append(b[sel])
for k in sorted(res):
    v = np.concatenate(res[k])
    h = np.histogram(v, bins)[0] / len(v)
    print(k, len(v), " ".join(f"[{bins[i]:.2f},{bins[i + 1]:.2f}):{h[i]:.2f}" for i in range(len(h))))
