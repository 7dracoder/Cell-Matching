"""Gated LOO pair F1 for a registration pickle {sid: (M, score, margin, ok)}; argv: pickles."""
import pickle
import sys

import numpy as np

import pair_clf

for path in sys.argv[1:]:
    reg = pickle.load(open(path, "rb"))
    rows = pair_clf.dataset({s: (reg[s][0], reg[s][1]) for s in pair_clf.R})
    probs = pair_clf.loo_predict(rows)
    best = []
    for mcut in (-99, 0, 3, 5):
        table = []
        for th in np.arange(0.025, 0.31, 0.025):
            tp = sum(int(y[probs[s] >= th].sum()) for s, _, _, y in rows if reg[s][2] >= mcut)
            pred = sum(int((probs[s] >= th).sum()) for s, _, _, y in rows if reg[s][2] >= mcut)
            table.append((2 * tp / (pred + pair_clf.TOTAL), th, tp, pred))
        f1, th, tp, pred = max(table)
        best.append(f"margin>={mcut}: F1 {f1:.4f} (thr {th:.3f}, tp {tp}, pred {pred})")
    print(path, "|", " | ".join(best))
