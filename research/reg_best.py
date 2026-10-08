"""Per-region best of several registration hypotheses (stretch / none), scored by margin in the true frame.

Writes data/reg_best.pkl = {sid: (M, score, margin, variant)} for training lab.pkl subjects.
Also usable as REG_TRAIN for test_apply after converting to (M, score).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine  # noqa: E402
from hough import window_register  # noqa: E402
from margin_lab import margin, R  # noqa: E402
from reg_lab import err  # noqa: E402
from reg_points_lab import stretch, unstretch  # noqa: E402
from vote import P0, region_candidates, vote_modes  # noqa: E402

VARIANTS = os.environ.get("VARIANTS", "pred stretch0.92_75").split()


def cands(args):
    variant, sid = args
    r = R[sid]
    S = stretch(variant)
    return sid, variant, region_candidates(r["iv_c"], r["ex_c"] @ S.T, r["offset"])


def window(args):
    sid, variant, modes = args
    r = R[sid]
    S = stretch(variant)
    M, score = window_register(r["iv_c"], r["ex_c"] @ S.T, r["offset"], modes, P0, angle_win=5.0)
    return sid, variant, M, score


if __name__ == "__main__":
    # reuse cached if available
    cached = {}
    for v in VARIANTS:
        path = f"data/regpts_{v}.pkl"
        if os.path.exists(path):
            D = pickle.load(open(path, "rb"))
            cached[v] = D
            print("loaded", v, flush=True)

    need = [v for v in VARIANTS if v not in cached]
    if need:
        with Pool(8) as pool:
            raw = pool.map(cands, [(v, s) for v in need for s in R])
        by = {}
        for sid, variant, C in raw:
            by.setdefault(variant, {})[sid] = C
        for variant in need:
            jobs = []
            for subj in sorted({r["subject"] for r in R.values()}):
                sids = [s for s in R if R[s]["subject"] == subj]
                modes = vote_modes({s: by[variant][s] for s in sids})
                jobs += [(s, variant, modes) for s in sids]
            with Pool(8) as pool:
                wins = pool.map(window, jobs)
            out = {}
            for sid, variant, M, score in wins:
                C = by[variant][sid]
                out[sid] = ((M, score), margin(C, M, score, R[sid]),
                            M is not None and err(R[sid], unstretch(M, variant)) < 5, C)
            pickle.dump(out, open(f"data/regpts_{variant}.pkl", "wb"))
            cached[variant] = out
            print("computed", variant, flush=True)

    best = {}
    for s, r in R.items():
        choice = None
        for v in VARIANTS:
            M_s, score = cached[v][s][0]
            C = cached[v][s][3]
            M = unstretch(M_s, v)
            # remargin in true frame against unstretched candidates
            true_cands = []
            for sc, a, land, Mc in C:
                Rm, sc2 = refine(r["iv_c"], r["ex_c"], unstretch(Mc, v))
                a2 = float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])))
                land2 = Rm[:, :2] @ (P0 - r["offset"]) + Rm[:, 2]
                true_cands.append((sc2, a2, land2, Rm))
            if M is not None:
                Rm, sc2 = refine(r["iv_c"], r["ex_c"], M)
            else:
                Rm, sc2 = None, 0.0
            mg = margin(true_cands, Rm, sc2, r)
            ok = Rm is not None and err(r, Rm) < 5
            if choice is None or mg > choice[2] or (mg == choice[2] and sc2 > choice[1]):
                choice = (Rm, sc2, mg, v, ok, true_cands)
        best[s] = choice

    for cut in (3, 1, 0):
        g = [s for s in R if best[s][2] >= cut]
        print(f"best-of margin>={cut}: gated {len(g)} correct {sum(best[s][4] for s in g)} "
              f"GT-pairs {sum(R[s]['n_gt_pairs'] for s in g if best[s][4])}")
    print("overall correct", sum(b[4] for b in best.values()),
          "by variant", {v: sum(best[s][3] == v and best[s][4] for s in R) for v in VARIANTS})
    pickle.dump({s: (best[s][0], best[s][1], best[s][2], best[s][3]) for s in R},
                open("data/reg_best.pkl", "wb"))
    pickle.dump({s: (best[s][0], best[s][1]) for s in R}, open("data/reg_window_best.pkl", "wb"))
