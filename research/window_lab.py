"""Constrained Hough registration around (oracle or estimated) per-mouse consensus poses."""
import os, sys, pickle, time
import numpy as np
from multiprocessing import Pool
from hough import window_register
from reg_lab import err

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
H = pickle.load(open(os.path.join(HERE, "data", "reg_hough.pkl"), "rb"))
P0 = np.array([300.0, 300.0])
MODE = sys.argv[1] if len(sys.argv) > 1 else "oracle"
RADIUS = float(sys.argv[2]) if len(sys.argv) > 2 else 120
WIN = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5


def pose(r, M):
    return np.degrees(np.arctan2(M[1, 0], M[0, 0])), M[:, :2] @ (P0 - r["offset"]) + M[:, 2]


def cluster_modes(poses, weights, ang_tol=4.0, land_tol=120.0, max_modes=2, min_share=0.25):
    """Weighted densest clusters of (angle, landing) poses."""
    poses = list(poses)
    weights = np.asarray(weights, float)
    modes, used = [], np.zeros(len(poses), bool)
    for _ in range(max_modes):
        best = None
        for i, (a, l) in enumerate(poses):
            if used[i]:
                continue
            member = np.array([not used[j] and abs(a - b) < ang_tol and np.linalg.norm(l - m) < land_tol
                               for j, (b, m) in enumerate(poses)])
            w = weights[member].sum()
            if best is None or w > best[0]:
                best = (w, member)
        if best is None or (modes and best[0] < min_share * weights.sum()) or best[1].sum() < 2:
            break
        idx = np.flatnonzero(best[1])
        modes.append({"angle": float(np.median([poses[j][0] for j in idx])),
                      "landing": np.median([poses[j][1] for j in idx], axis=0), "n": len(idx)})
        used |= best[1]
    return modes


def subject_modes(sids):
    if MODE == "oracle":
        poses = [pose(R[s], R[s]["gt_M"]) for s in sids if R[s]["gt_M"] is not None]
        groups = [[p for p in poses if p[0] < -5], [p for p in poses if p[0] >= -5]]
        return [{"angle": float(np.median([a for a, _ in g])), "landing": np.median([l for _, l in g], 0)} for g in groups if g]
    poses = [pose(R[s], H[s][0]) for s in sids if H[s][0] is not None]
    scores = [H[s][1] for s in sids if H[s][0] is not None]
    weights = np.maximum(np.array(scores) - np.median(scores), 0) + 1e-3
    return cluster_modes(poses, weights)


def run(args):
    sid, modes = args
    r = R[sid]
    return sid, window_register(r["iv_c"], r["ex_c"], r["offset"], modes, P0, angle_win=WIN, radius=RADIUS)


if __name__ == "__main__":
    started = time.time()
    jobs = []
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        modes = subject_modes(sids)
        print(subj, [(round(m["angle"], 1), m["landing"].round(), m.get("n")) for m in modes])
        jobs += [(s, modes) for s in sids]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        print(subj, "correct", sum(err(R[s], res[s][0]) < 5 for s in sids), "/", len(sids))
    print("total", sum(err(R[s], res[s][0]) < 5 for s in R), f"{time.time() - started:.0f}s")
    pickle.dump(res, open(os.path.join(HERE, "data", f"reg_window_{MODE}.pkl"), "wb"))
