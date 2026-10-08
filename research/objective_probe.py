"""Which pose objective ranks the true pose above every clearly-wrong candidate?"""
import pickle, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "..")
from registration import refine, transform
import cp_pose_lab as L
from margin_lab import R
from reg_lab import err
from window_lab import pose

S = pickle.load(open("data/cp_pose_train.pkl", "rb"))
Z = np.load("data/heldout_labels.npz")


def objectives(r, M, bm):
    P = transform(r["iv_c"], M)
    d, j = cKDTree(r["ex_c"]).query(P)
    db, i = cKDTree(P).query(r["ex_c"])
    mutual = i[j] == np.arange(len(P))
    out = {}
    for sig in (1.0, 1.5, 2.5):
        out[f"soft{sig}"] = float(np.sum(np.exp(-d[mutual] ** 2 / (2 * sig ** 2))))
    for t in (2, 3, 5):
        out[f"n{t}"] = int(np.sum(mutual & (d < t)))
    out["z"] = L.cp_z(bm, r["iv_c"], M)
    return out


if __name__ == "__main__":
    wins = {}
    nreg = 0
    for s, r in R.items():
        if r["gt_M"] is None:
            continue
        bm = L.cp_map(Z[s + "|exvivo|prob"].astype(np.float32))
        Mg, _ = refine(r["iv_c"], r["ex_c"], r["gt_M"])
        if err(r, Mg) > 5:
            Mg = r["gt_M"]
        og = objectives(r, Mg, bm)
        a, l = pose(r, Mg)
        wrong = [c for c in S[s] if err(r, c[0]) > 10]
        ow = [objectives(r, c[0], bm) for c in wrong]
        nreg += 1
        for k in og:
            best_wrong = max([o[k] for o in ow], default=-1e9)
            wins.setdefault(k, []).append(og[k] > best_wrong)
    for k, v in wins.items():
        print(f"{k:8s} true pose beats all wrong candidates in {sum(v)}/{nreg}")
