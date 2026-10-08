"""Gated LOO pair F1 vs classifier threshold for the lab in data/."""
import pickle

import numpy as np

import pair_clf
from margin_lab import margin, C, R

if __name__ == "__main__":
    W = pickle.load(open("data/reg_window_vote5.pkl", "rb"))
    rows = pair_clf.dataset({s: (W[s][0], W[s][1]) for s in R})
    probs = pair_clf.loo_predict(rows)
    mg = {s: margin(C[s], W[s][0], W[s][1], R[s]) for s in R}
    for mcut in (2, 3, 5):
        out = []
        for th in np.arange(0.025, 0.31, 0.025):
            tp = sum(int(y[probs[s] >= th].sum()) for s, _, _, y in rows if mg[s] >= mcut)
            pred = sum(int((probs[s] >= th).sum()) for s, _, _, y in rows if mg[s] >= mcut)
            out.append(f"{th:.3f}:{2 * tp / (pred + pair_clf.TOTAL):.4f}")
        print(f"margin>={mcut}", " ".join(out))
