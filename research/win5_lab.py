import pickle
import numpy as np
from multiprocessing import Pool
import pair_clf
from hough import window_register
from vote import vote_modes, P0
from reg_lab import err
from margin_lab import margin, C, R

ANGLE_WIN = 5.0


def run(args):
    s, modes = args
    r = R[s]
    return s, window_register(r["iv_c"], r["ex_c"], r["offset"], modes, P0, angle_win=ANGLE_WIN)


if __name__ == "__main__":
    jobs = []
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        modes = vote_modes({s: C[s] for s in sids})
        jobs += [(s, modes) for s in sids]
    with Pool(8) as pool:
        W = dict(pool.map(run, jobs))
    print("correct", sum(err(R[s], W[s][0]) < 5 for s in R), flush=True)
    pickle.dump(W, open("data/reg_window_vote5.pkl", "wb"))
    rows = pair_clf.dataset({s: (W[s][0], W[s][1]) for s in R})
    probs = pair_clf.loo_predict(rows)
    mg = {s: margin(C[s], W[s][0], W[s][1], R[s]) for s in R}
    for mcut in (-99, 3):
        table = []
        for th in np.arange(0, 0.3, 0.025):
            tp = sum(int(y[probs[s] >= th].sum()) for s, _, _, y in rows if mg[s] >= mcut)
            pred = sum(int((probs[s] >= th).sum()) for s, _, _, y in rows if mg[s] >= mcut)
            table.append((2 * tp / (pred + pair_clf.TOTAL), th))
        f1, th = max(table)
        print(f"margin>={mcut} F1 {f1:.4f} thr {th:.3f} | correct above cut "
              f"{sum(err(R[s], W[s][0]) < 5 for s in R if mg[s] >= mcut)} of {sum(mg[s] >= mcut for s in R)}")
