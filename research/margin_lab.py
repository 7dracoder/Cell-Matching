"""Does the margin of the chosen pose over the best clearly-different pose predict correctness?"""
import os, pickle
import numpy as np
from reg_lab import err
from window_lab import pose

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
C = pickle.load(open(os.path.join(HERE, "data", "vote_cands.pkl"), "rb"))
W = pickle.load(open(os.path.join(HERE, "data", "reg_window_vote.pkl"), "rb"))


def margin(cands, M, score, r):
    if M is None:
        return -99.0
    a, l = pose(r, M)
    null = [c[0] for c in cands if abs(c[1] - a) > 3 or np.linalg.norm(c[2] - l) > 60]
    return score - (max(null) if null else 0.0)


if __name__ == "__main__":
    rows = []
    for s, r in R.items():
        M, score = W[s]
        rows.append((margin(C[s], M, score, r), score, err(r, M) < 5, len(r["gt_pairs"]), s))
    rows.sort()
    for m, sc, ok, n, s in rows:
        print(f"{m:6.1f} score {sc:5.1f} {'OK ' if ok else 'BAD'} gtpairs {n:3d} {s[-22:]}")
