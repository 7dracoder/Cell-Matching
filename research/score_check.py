import os, sys, pickle
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine, paired_nearest, transform  # noqa: E402
from reg_lab import err

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
res = pickle.load(open(os.path.join(HERE, "data", "reg_oracle_consensus.pkl"), "rb"))


def count_close(r, M, d=5):
    if M is None:
        return 0
    return len(paired_nearest(r["iv_c"], r["ex_c"], M, d)[0])


def expected_random(r, d=5):
    """Expected chance pairs: ex density inside canvas x iv count x disc area."""
    dens = len(r["ex_c"]) / (r["ex_shape"][0] * r["ex_shape"][1])
    return len(r["iv_c"]) * dens * np.pi * d * d


for sid, r in R.items():
    if r["gt_M"] is None:
        continue
    gm, gscore = refine(r["iv_c"], r["ex_c"], r["gt_M"])
    ok = err(r, res[sid][0]) < 5
    print(f"{sid[-22:]} {'OK ' if ok else 'BAD'} chosen score {res[sid][1]:6.1f} close5 {count_close(r, res[sid][0]):3d} | "
          f"GT-refined score {gscore:6.1f} close5 {count_close(r, gm):3d} gtErr {err(r, gm):5.1f} | chance {expected_random(r):4.1f} "
          f"nIV {len(r['iv_c'])} nEX {len(r['ex_c'])} eligible {sum(a in set(r['iv_link']) and b in set(r['ex_link']) for a, b in r['gt_pairs'])}")
