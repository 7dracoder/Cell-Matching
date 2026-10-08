import os, pickle
import numpy as np, tifffile
from sections import prep, align, compose
from common import ROOT

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
P0 = np.array([300.0, 300.0])


def to_mosaic(r, M):
    return np.c_[M[:, :2], M[:, 2] - M[:, :2] @ r["offset"]]


out = {}
for subj in sorted({r["subject"] for r in R.values()}):
    sids = [s for s in R if R[s]["subject"] == subj]
    ims = {s: prep(tifffile.imread(os.path.join(ROOT, "training", *s.split("__"), "exvivo.tif"))) for s in sids}
    ref = sids[0]
    print(subj, "ref", ref[-6:])
    raw, aligned = [], []
    for s in sids:
        S, cc = align(ims[s], ims[ref]) if s != ref else (np.c_[np.eye(2), np.zeros(2)], 1.0)
        out[s] = (S, cc, ref)
        if R[s]["gt_M"] is None:
            continue
        Mm = to_mosaic(R[s], R[s]["gt_M"])
        land = Mm[:, :2] @ P0 + Mm[:, 2]
        Pref = compose(S, Mm)
        land_ref = Pref[:, :2] @ P0 + Pref[:, 2]
        ang, ang_ref = (np.degrees(np.arctan2(M[1, 0], M[0, 0])) for M in (Mm, Pref))
        raw.append((ang, *land)); aligned.append((ang_ref, *land_ref))
        print(f"   {s[-6:]} ecc {cc:.3f} raw ang {ang:6.1f} land {land.round()} -> ref ang {ang_ref:6.1f} land {land_ref.round()}")
    raw, aligned = np.array(raw), np.array(aligned)
    mad = lambda a: np.median(np.abs(a - np.median(a, 0)), 0).round(1)
    print("   MAD raw (ang, x, y)", mad(raw), " aligned", mad(aligned))
pickle.dump(out, open(os.path.join(HERE, "data", "sections.pkl"), "wb"))
