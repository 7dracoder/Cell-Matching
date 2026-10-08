import os, sys, pickle
import numpy as np
from multiprocessing import Pool
import consensus as C
from reg_lab import err

C.SCALES = np.arange(0.89, 1.001, 0.025)
HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
RADIUS = float(sys.argv[1]) if len(sys.argv) > 1 else 160
WIN = float(sys.argv[2]) if len(sys.argv) > 2 else 4.5


def gt_pose(s):
    M = R[s]["gt_M"]
    return np.degrees(np.arctan2(M[1, 0], M[0, 0])), M[:, :2] @ (C.P0 - R[s]["offset"]) + M[:, 2]


def reg(args):
    sid, modes = args
    r = R[sid]
    best = None
    for m in modes:
        out = C.constrained_register(r["iv_c"], r["ex_c"], r["ex_shape"], r["offset"], m, angle_win=WIN, radius=RADIUS)
        if best is None or out[1] > best[1]:
            best = out
    return sid, best


if __name__ == "__main__":
    jobs = []
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        poses = [gt_pose(s) for s in sids if R[s]["gt_M"] is not None]
        groups = [[p for p in poses if p[0] < -5], [p for p in poses if p[0] >= -5]]
        modes = [{"angle": float(np.median([a for a, _ in g])), "landing": np.median([l for _, l in g], 0)} for g in groups if g]
        jobs += [(s, modes) for s in sids]
    with Pool(8) as pool:
        res = dict(pool.map(reg, jobs))
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        print(subj, "correct", sum(err(R[s], res[s][0]) < 5 for s in sids), "/", len(sids))
    print("total", sum(err(R[s], res[s][0]) < 5 for s in R))
    pickle.dump(res, open(os.path.join(HERE, "data", "reg_oracle_consensus.pkl"), "wb"))
