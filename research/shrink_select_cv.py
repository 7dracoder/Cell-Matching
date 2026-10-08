"""LOO-mouse selector: for each ex cell we put in a pair, choose its boundary variant.

Per (cell, variant) a HistGradientBoosting classifier predicts IoU > 0.75 from mask/cellprob
features plus the in-vivo partner's area (a cross-modal size cue); the cell takes the variant
with the highest probability. Evaluated on held-out pairs/masks with the full score.
"""
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import pair_clf
from margin_lab import R
from paired_shrink_cv import CELLS, NGT

CAND = ["base", "ep10", "ep25", "ep35", "ep50", "ep60", "cp1", "it20"]
BY = {(c["sid"], c["k"] - 1): c for c in CELLS}


def feats(c, v, iv_area, scale2):
    a0 = c["areas"]["base"]
    av = c["areas"][v]
    return [CAND.index(v), av / a0, np.log(a0), np.log(av), c["cpmean"], c["cp_inner"], c["cp_max"],
            c["contrast"], c["peak"], np.log(iv_area * scale2), np.log(av / (iv_area * scale2))]


def build(chosen):
    X, y, key = [], [], []
    for s, pairs in chosen.items():
        r = R[s]
        for i, j in pairs:
            M = None
        for i, j in pairs:
            c = BY[(s, j)]
            iv_area = r["iv_f"]["area"][i]
            for v in CAND:
                X.append(feats(c, v, iv_area, 1.0))
                y.append(c["iou"][v] > 0.75)
                key.append((s, j, v))
    return np.array(X), np.array(y), key


def evaluate(chosen, pick):
    tp = pred = 0
    pq = {}
    for s, r in R.items():
        hit, iou_sum, ntp = {}, 0.0, 0
        cells = [c for c in CELLS if c["sid"] == s]
        for c in cells:
            v = pick.get((s, c["k"] - 1), "base")
            if c["iou"][v] > 0.75:
                ntp += 1
                iou_sum += c["iou"][v]
                hit[c["k"] - 1] = c["gt"] - 1
        den = ntp + 0.5 * (len(cells) - ntp) + 0.5 * (NGT[s] - ntp)
        pq.setdefault(r["subject"], []).append(iou_sum / den)
        for i, j in chosen[s]:
            pred += 1
            li = r["iv_link"][i]
            tp += li >= 0 and j in hit and (li, hit[j]) in r["gt_pairs"]
    return np.mean([v for vs in pq.values() for v in vs]), 2 * tp / (pred + pair_clf.TOTAL), tp


if __name__ == "__main__":
    chosen = pickle.load(open("data/v10_train_pairs.pkl", "rb"))
    X, y, key = build(chosen)
    subj = np.array([R[k[0]]["subject"] for k in key])
    prob = np.zeros(len(y))
    for g in np.unique(subj):
        tr = subj != g
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        prob[~tr] = clf.predict_proba(X[~tr])[:, 1]
    best = {}
    for (s, j, v), p in zip(key, prob):
        if (s, j) not in best or p > best[(s, j)][0]:
            best[(s, j)] = (p, v)
    pick = {k: v for k, (p, v) in best.items()}
    for name, pk in [("base", {}), ("ep25 fixed", {k: "ep25" for k in pick}), ("ep35 fixed", {k: "ep35" for k in pick}),
                     ("selector", pick)]:
        pq, f1, tp = evaluate(chosen, pk)
        print(f"{name:12s} exPQ {pq:.4f} F1 {f1:.4f} tp {tp} full {0.25 * (0.7404 + pq) + 0.5 * f1:.4f}")
    from collections import Counter
    print(Counter(pick.values()))
