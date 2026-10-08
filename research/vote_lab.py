import os, pickle, time
import numpy as np
from multiprocessing import Pool
from vote import region_candidates, vote_modes, P0
from hough import window_register
from reg_lab import err
from window_lab import pose

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))


def cands(sid):
    r = R[sid]
    return sid, region_candidates(r["iv_c"], r["ex_c"], r["offset"])


def win(args):
    sid, modes = args
    r = R[sid]
    return sid, window_register(r["iv_c"], r["ex_c"], r["offset"], modes, P0)


if __name__ == "__main__":
    started = time.time()
    with Pool(8) as pool:
        C = dict(pool.map(cands, list(R)))
    print(f"candidates {time.time() - started:.0f}s")
    jobs = []
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        modes = vote_modes({s: C[s] for s in sids})
        gt = [pose(R[s], R[s]["gt_M"]) for s in sids if R[s]["gt_M"] is not None]
        print(subj, "modes", [(round(m["angle"], 1), m["landing"].round(), m["n"], round(m["support"], 1)) for m in modes])
        print("    GT", sorted((round(a, 1), tuple(l.round())) for a, l in gt))
        jobs += [(s, modes) for s in sids]
    with Pool(8) as pool:
        res = dict(pool.map(win, jobs))
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        print(subj, "correct", sum(err(R[s], res[s][0]) < 5 for s in sids), "/", len(sids))
    print("total", sum(err(R[s], res[s][0]) < 5 for s in R))
    pickle.dump(res, open(os.path.join(HERE, "data", "reg_window_vote.pkl"), "wb"))
    pickle.dump(C, open(os.path.join(HERE, "data", "vote_cands.pkl"), "wb"))
