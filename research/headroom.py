"""Where the held-out score is lost: PQ parts per modality, GT pairs in gated vs ungated regions."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from cellmatch import pq_score, read_image, rle_to_labels  # noqa: E402
from common import ROOT, load_truth  # noqa: E402
from size_lab import grow  # noqa: E402

data = np.load("data/v7/heldout_labels.npz")
truth = load_truth()
for m in ("invivo", "exvivo"):
    tp = fp = fn = 0
    iou_sum = 0.0
    for sid, t in truth.items():
        shape = read_image(os.path.join(ROOT, "training", *sid.split("__"), f"{m}.tif")).shape
        gt, _ = rle_to_labels(t[f"{m}_instances"], shape)
        lab = data[f"{sid}|{m}"].astype(np.int32)
        if m == "exvivo":
            lab = grow(lab, prob=data[f"{sid}|{m}|prob"].astype(np.float32), frac=0.15)
        pq, a, b, c = pq_score(lab, gt)
        tp, fp, fn = tp + a, fp + b, fn + c
        iou_sum += pq * (a + 0.5 * (b + c))
    print(f"{m}: TP {tp} FP {fp} FN {fn} | precision {tp / (tp + fp):.3f} recall {tp / (tp + fn):.3f} "
          f"| SQ {iou_sum / max(tp, 1):.3f}")

R = pickle.load(open("data/v7/lab.pkl", "rb"))
from margin_lab import margin  # noqa: E402
C = pickle.load(open("data/v7/vote_cands.pkl", "rb"))
W = pickle.load(open("data/v7/reg_window_vote5.pkl", "rb"))
from reg_lab import err  # noqa: E402
rows = []
for s, r in R.items():
    mg = margin(C[s], W[s][0], W[s][1], r)
    ok = err(r, W[s][0]) < 5 if W[s][0] is not None and r["gt_M"] is not None else False
    rows.append((s[-22:], r["n_gt_pairs"], len(r["gt_pairs"]), round(mg, 1), ok))
gated = [x for x in rows if x[3] >= 3]
print("GT pairs total", sum(x[1] for x in rows), "| in gated regions", sum(x[1] for x in gated),
      "| ungated but correct", sum(x[1] for x in rows if x[3] < 3 and x[4]),
      "| ungated and wrong", sum(x[1] for x in rows if x[3] < 3 and not x[4]))
for x in sorted(rows, key=lambda x: x[3]):
    print(x)
