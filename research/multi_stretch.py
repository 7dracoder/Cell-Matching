"""Combine several compression hypotheses: refine every hypothesis' poses in the true ex frame, pick the best.

usage: VARIANTS="stretch0.92_75 ..." multi_stretch.py   (needs data/regpts_{variant}.pkl from reg_points_lab)
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine  # noqa: E402
from margin_lab import margin  # noqa: E402
from reg_lab import err  # noqa: E402
from reg_points_lab import unstretch  # noqa: E402
from vote import P0  # noqa: E402

R = pickle.load(open("data/lab.pkl", "rb"))


def combine(r, per_variant):
    """per_variant: {variant: (window M, window cands)} in stretched frames -> (M, score, true-frame cands)."""
    best, cands = (None, 0.0), []
    for variant, (M, vc) in per_variant.items():
        if M is not None:
            Rm, score = refine(r["iv_c"], r["ex_c"], unstretch(M, variant))
            if score > best[1]:
                best = (Rm, score)
        for _, _, _, Mc in vc:
            Rc, sc = refine(r["iv_c"], r["ex_c"], unstretch(Mc, variant))
            a = float(np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0])))
            cands.append((sc, a, Rc[:, :2] @ (P0 - r["offset"]) + Rc[:, 2], Rc))
    return best[0], best[1], cands


if __name__ == "__main__":
    variants = os.environ["VARIANTS"].split()
    D = {v: pickle.load(open(f"data/regpts_{v}.pkl", "rb")) for v in variants}
    out = {}
    for s, r in R.items():
        M, score, cands = combine(r, {v: (D[v][s][0][0], D[v][s][3]) for v in variants})
        out[s] = (M, score, margin(cands, M, score, r), M is not None and err(r, M) < 5)
    for cut in (3, 5, 8):
        gated = [s for s in R if out[s][2] >= cut]
        print(f"margin>={cut}: gated {len(gated)} correct {sum(out[s][3] for s in gated)} "
              f"GT pairs {sum(R[s]['n_gt_pairs'] for s in gated if out[s][3])}")
    print("correct overall", sum(o[3] for o in out.values()),
          "GT pairs", sum(R[s]["n_gt_pairs"] for s in R if out[s][3]))
    pickle.dump(out, open("data/reg_multi_stretch.pkl", "wb"))
