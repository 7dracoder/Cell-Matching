"""Leave-one-mouse-out pose classifier on pose_cands.pkl features."""
import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from margin_lab import R
from reg_lab import err

D = pickle.load(open(os.path.join(os.path.dirname(__file__), "data", "pose_cands.pkl"), "rb"))
FEATURES = ["score", "score_gap", "rank", "n3", "n5", "bright_sum", "bright_both", "ncc", "ncc_blur",
            "d_angle", "d_land", "scale", "overlap"]


def loo_pick(seed=0):
    subjects = sorted({R[s]["subject"] for s in R})
    picks = {}
    for g in subjects:
        X, y = [], []
        for s, (feats, errs, mats) in D.items():
            if R[s]["subject"] == g or len(feats) == 0:
                continue
            X.append(feats)
            y.append(errs < 5)
        X, y = np.vstack(X), np.concatenate(y)
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=seed).fit(X, y)
        for s, (feats, errs, mats) in D.items():
            if R[s]["subject"] != g:
                continue
            if len(feats) == 0:
                picks[s] = (None, -1.0, np.inf)
                continue
            p = clf.predict_proba(feats)[:, 1]
            k = int(np.argmax(p))
            picks[s] = (mats[k], float(p[k]), float(errs[k]), p, errs)
    return picks


if __name__ == "__main__":
    picks = loo_pick()
    top1 = sum(bool(len(e) and e[0] < 5) for _, e, _ in D.values())
    oracle = sum(bool((e < 5).any()) for _, e, _ in D.values())
    clf_ok = sum(p[2] < 5 for p in picks.values())
    print(f"top1-by-score {top1}/47 | oracle-in-window {oracle}/47 | clf-pick {clf_ok}/47")

    # confidence gate on max prob
    for cut in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        g = [s for s, p in picks.items() if p[1] >= cut]
        print(f"p>={cut}: gated {len(g)} correct {sum(picks[s][2] < 5 for s in g)} "
              f"GT-pairs {sum(R[s]['n_gt_pairs'] for s in g if picks[s][2] < 5)}")

    # vs margin baseline
    W = pickle.load(open("data/reg_window_vote5.pkl", "rb"))
    from margin_lab import margin, C
    mg = {s: margin(C[s], W[s][0], W[s][1], R[s]) for s in R}
    print("margin>=3:", sum(mg[s] >= 3 for s in R), "correct",
          sum(err(R[s], W[s][0]) < 5 for s in R if mg[s] >= 3))

    # Feature importance via permutation on pooled
    X = np.vstack([D[s][0] for s in D if len(D[s][0])])
    y = np.concatenate([D[s][1] < 5 for s in D if len(D[s][0])])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=0).fit(X, y)
    base = clf.score(X, y)
    print("train acc", round(base, 3))
    for i, name in enumerate(FEATURES):
        Xp = X.copy()
        Xp[:, i] = np.random.permutation(Xp[:, i])
        print(f"  drop {name:12s} {base - clf.score(Xp, y):+.3f}")
