"""Held-out full score when only the ex-vivo cells we put into pairs get a tighter boundary.

Uses the leave-one-mouse-out poses/gate of v10 (cp_pose_train.pkl, margin>=3 or cellprob z>=5),
the LOO pair classifier, and per-cell variant IoUs from shrink_probe.py.
"""
import pickle
import sys

import numpy as np

import cp_pose_lab as L
import pair_clf
from margin_lab import C, R, margin

THR = 0.025
CELLS = pickle.load(open("data/shrink_cells.pkl", "rb"))
BY = {(c["sid"], c["k"]): c for c in CELLS}
NGT = {}
from common import load_truth  # noqa: E402
for sid, t in load_truth().items():
    NGT[sid] = len(t["exvivo_instances"])


def pipeline():
    S = pickle.load(open("data/cp_pose_train.pkl", "rb"))
    regs, keep = {}, {}
    for s in R:
        ch = L.pose_choose(S[s], "score")
        regs[s] = (ch[0], ch[1])
        keep[s] = margin(C[s], ch[0], ch[1], R[s]) >= 3 or ch[2] >= 5
    rows = pair_clf.dataset(regs)
    probs = pair_clf.loo_predict(rows)
    chosen = {}
    for s, pairs, X, y in rows:
        chosen[s] = [tuple(p) for p, pr in zip(pairs, probs[s]) if pr >= THR] if keep[s] else []
    return chosen


def score(chosen, variant, scope="paired"):
    tp = pred = 0
    pq_by = {}
    for s, r in R.items():
        paired_ex = {j for _, j in chosen[s]}
        cells = [c for c in CELLS if c["sid"] == s]
        iou_sum, ntp = 0.0, 0
        hit = {}
        for c in cells:
            v = variant if (scope == "all" or (c["k"] - 1) in paired_ex) else "base"
            iou = c["iou"][v]
            if iou > 0.75:
                ntp += 1
                iou_sum += iou
                hit[c["k"] - 1] = c["gt"] - 1
        npred = len(cells)
        den = ntp + 0.5 * (npred - ntp) + 0.5 * (NGT[s] - ntp)
        pq_by.setdefault(r["subject"], []).append(iou_sum / den if den else 1.0)
        for i, j in chosen[s]:
            pred += 1
            li = r["iv_link"][i]
            if li >= 0 and j in hit and (li, hit[j]) in r["gt_pairs"]:
                tp += 1
    f1 = 2 * tp / (pred + pair_clf.TOTAL)
    pq = np.mean([v for vs in pq_by.values() for v in vs])
    return pq, f1, tp, pred, {k[-6:]: round(float(np.mean(v)), 3) for k, v in pq_by.items()}


if __name__ == "__main__":
    chosen = pipeline()
    pickle.dump(chosen, open("data/v10_train_pairs.pkl", "wb"))
    iv_pq = 0.7404
    for scope in ("paired", "all"):
        for v in ["base", "ep25", "ep50", "ep75", "cp1", "it20", "it30"]:
            pq, f1, tp, pred, by = score(chosen, v, scope)
            print(f"{scope:6s} {v:5s} exPQ {pq:.4f} F1 {f1:.4f} (tp {tp} pred {pred}) full {0.25 * (iv_pq + pq) + 0.5 * f1:.4f} {by}")
