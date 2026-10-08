"""Ex-vivo / in-vivo error anatomy on held-out predictions."""
import os, sys, pickle
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import rle_to_labels, read_image, pq_score  # noqa: E402
from common import load_truth, ROOT  # noqa: E402

HERE = os.path.dirname(__file__)
truth = load_truth()
mod = sys.argv[1] if len(sys.argv) > 1 else "exvivo"
source = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "heldout_labels.npz")
data = np.load(source)
key = f"{mod}_instances"
stats = {}
for sid, t in truth.items():
    subj = sid.split("__")[0]
    img = read_image(os.path.join(ROOT, "training", *sid.split("__"), f"{mod}.tif"))
    gt, _ = rle_to_labels(t[key], img.shape)
    pred = data[f"{sid}|{mod}"].astype(np.int32)
    npred, ngt = pred.max(), gt.max()
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(), minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc, gc = joint.sum(1), joint.sum(0)
    iou = joint[1:, 1:] / np.maximum(pc[1:, None] + gc[None, 1:] - joint[1:, 1:], 1)
    best_gt = iou.max(0) if npred else np.zeros(ngt)
    arg = iou.argmax(0) if npred else np.zeros(ngt, int)
    s = stats.setdefault(subj, {"pq": [], "gt": 0, "tp": 0, "near": 0, "miss": 0, "low": 0, "pred_bigger": 0, "pred_smaller": 0,
                                "fp": 0, "fp_any_gt_overlap": 0, "gt_area": [], "pred_area": []})
    s["pq"].append(pq_score(pred, gt)[0])
    s["gt"] += ngt
    s["tp"] += int((best_gt > 0.75).sum())
    near = (best_gt > 0.5) & (best_gt <= 0.75)
    s["near"] += int(near.sum())
    s["low"] += int(((best_gt > 0.1) & (best_gt <= 0.5)).sum())
    s["miss"] += int((best_gt <= 0.1).sum())
    for g in np.flatnonzero(near):
        p = arg[g]
        if pc[p + 1] > gc[g + 1]:
            s["pred_bigger"] += 1
        else:
            s["pred_smaller"] += 1
    best_pred = iou.max(1) if ngt else np.zeros(npred)
    fp = best_pred <= 0.75
    s["fp"] += int(fp.sum())
    s["fp_any_gt_overlap"] += int((fp & (joint[1:, 1:].sum(1) > 0)).sum())
    s["gt_area"] += list(gc[1:])
    s["pred_area"] += list(pc[1:])
for subj, s in stats.items():
    print(subj, f"PQ {np.mean(s['pq']):.3f} | GT {s['gt']} TP {s['tp']} near(0.5-0.75) {s['near']} "
          f"(pred bigger {s['pred_bigger']}, smaller {s['pred_smaller']}) low(0.1-0.5) {s['low']} missed {s['miss']} | "
          f"FP {s['fp']} (overlapping GT {s['fp_any_gt_overlap']}) | area med GT {np.median(s['gt_area']):.0f} pred {np.median(s['pred_area']):.0f}")
