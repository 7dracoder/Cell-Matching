"""Is each region's true pose inside the voted window (angle +-5, landing radius 120) of its mouse?"""
import pickle

import numpy as np

from vote import vote_modes
from window_lab import pose

R = pickle.load(open("data/lab.pkl", "rb"))
C = pickle.load(open("data/vote_cands.pkl", "rb"))
P = pickle.load(open("data/regpts_gt.pkl", "rb"))
Q = pickle.load(open("data/regpts_pred.pkl", "rb"))
for subj in sorted({r["subject"] for r in R.values()}):
    sids = [s for s in R if R[s]["subject"] == subj]
    modes = vote_modes({s: C[s] for s in sids})
    print(subj, [(round(m["angle"], 1), m["landing"].round()) for m in modes])
    for s in sorted(sids, key=lambda s: -R[s]["n_gt_pairs"]):
        r = R[s]
        if r["gt_M"] is None:
            print("   ", s[-13:], "no gt_M")
            continue
        a, l = pose(r, r["gt_M"])
        d = min((abs(a - m["angle"]), float(np.linalg.norm(l - m["landing"]))) for m in modes)
        inside = d[0] <= 5 and d[1] <= 120
        print(f"    {s[-13:]} gtpairs {r['n_gt_pairs']:3d} true angle {a:6.1f} land {l.round()} "
              f"d_ang {d[0]:4.1f} d_land {d[1]:5.0f} {'IN ' if inside else 'OUT'} "
              f"pred_ok {Q[s][2]} gt_ok {P[s][2]} margin {Q[s][1]:.1f}")
