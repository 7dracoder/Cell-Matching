"""How well does one affine explain each region's GT pairs? (residuals under gt_M)."""
import pickle

import numpy as np

from registration import transform

R = pickle.load(open("data/lab.pkl", "rb"))
GX = pickle.load(open("data/gt_ex_c.pkl", "rb"))
P = pickle.load(open("data/pose_cands.pkl", "rb"))
rows = []
for s, r in R.items():
    if r["gt_M"] is None:
        continue
    pairs = sorted(r["gt_pairs"])
    a = r["gt_iv_c"][[i for i, _ in pairs]]
    b = GX[s][[j for _, j in pairs]]
    res = np.linalg.norm(transform(a, r["gt_M"]) - b, axis=1)
    has = bool((P[s][1] < 5).any())
    rows.append((has, s[-22:], len(pairs), np.median(res), np.mean(res < 3), np.mean(res < 6), res.max()))
for x in sorted(rows):
    print(f"cand_ok {x[0]!s:5} {x[1]} pairs {x[2]:3d} median {x[3]:5.1f} <3px {x[4]:.2f} <6px {x[5]:.2f} max {x[6]:6.1f}")
