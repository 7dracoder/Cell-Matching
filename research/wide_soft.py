"""Mode-free wide pose search + soft mutual-nearest objective.

Candidates: similarity Hough over angle -35..35 (1 deg) x scale 0.85..1.13, top peaks refined
with registration.refine, deduplicated; plus the existing window/vote candidates.
Each candidate is re-scored with soft(sigma) = sum over mutual-nearest pairs of exp(-d^2/2s^2).
"""
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "..")
from registration import refine, transform  # noqa: E402
from hough import hough_candidates  # noqa: E402

ANGLES = np.arange(-35, 35.1, 1.0)
SCALES = np.arange(0.85, 1.131, 0.02)


def soft(iv_c, ex_c, M, sig=2.5):
    P = transform(iv_c, M)
    d, j = cKDTree(ex_c).query(P)
    _, i = cKDTree(P).query(ex_c)
    mutual = i[j] == np.arange(len(P))
    return float(np.sum(np.exp(-d[mutual] ** 2 / (2 * sig ** 2))))


STRETCH = [(1.0, 0.0)] + [(k, phi) for k in (0.92, 1.08) for phi in (0, 45, 90, 135)]


def stretch(k, phi):
    u = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    return np.eye(2) + (k - 1) * np.outer(u, u)


def wide_candidates(iv_c, ex_c, top=150, stretches=None):
    raw = []
    for k, phi in (stretches or STRETCH):
        Sm = stretch(k, phi)
        for votes, a, s, M in hough_candidates(iv_c @ Sm.T, ex_c, angles=ANGLES, scales=SCALES, per_pose=3,
                                               keep=top if k == 1.0 else top // 3):
            raw.append((votes, np.c_[M[:, :2] @ Sm, M[:, 2]]))
    raw.sort(key=lambda c: c[0], reverse=True)
    out = []
    for votes, M in raw:
        Rm, sc = refine(iv_c, ex_c, M)
        if any(np.abs(Rm - o[0]).max() < 1e-3 or (np.abs(Rm[:, :2] - o[0][:, :2]).max() < 0.01
               and np.linalg.norm(Rm[:, 2] - o[0][:, 2]) < 4) for o in out):
            continue
        out.append((Rm, sc))
    return out


def run(args):
    s, iv_c, ex_c, extra = args
    cands = wide_candidates(iv_c, ex_c) + [(M, sc) for M, sc in extra]
    return s, [(M, sc, soft(iv_c, ex_c, M)) for M, sc in cands]


if __name__ == "__main__":
    from margin_lab import R
    from reg_lab import err
    S = pickle.load(open("data/cp_pose_train.pkl", "rb"))
    jobs = [(s, R[s]["iv_c"], R[s]["ex_c"], [(c[0], c[1]) for c in S[s]]) for s in R]
    with Pool(8) as pool:
        W = dict(pool.map(run, jobs))
    pickle.dump(W, open("data/wide_soft_train.pkl", "wb"))
    present = by_soft = by_score = 0
    for s, cs in W.items():
        r = R[s]
        if r["gt_M"] is None:
            continue
        ok = [err(r, c[0]) < 5 for c in cs]
        present += any(ok)
        by_soft += ok[int(np.argmax([c[2] for c in cs]))]
        by_score += ok[int(np.argmax([c[1] for c in cs]))]
    print(f"correct candidate present {present}/46 | chosen by soft {by_soft} | by refine score {by_score}")
