import os, pickle
import numpy as np
from consensus import landing_volume, ANGLES, SCALES, BIN, MARGIN, P0

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
for sid in ["subject_5d294c__region_952df0", "subject_db6b8b__region_65121e", "subject_db6b8b__region_ce8b99",
            "subject_b2ba5e__region_66750c"]:
    r = R[sid]
    v = landing_volume(r["iv_c"], r["ex_c"], r["ex_shape"], r["offset"], spread=8)
    M = r["gt_M"]
    ang = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    sc = np.sqrt(abs(np.linalg.det(M[:, :2])))
    land = M[:, :2] @ (P0 - r["offset"]) + M[:, 2]
    a, s = np.argmin(abs(ANGLES - ang)), np.argmin(abs(SCALES - sc))
    by, bx = int(round((land[1] + MARGIN) / BIN)), int(round((land[0] + MARGIN) / BIN))
    local = v[max(a - 1, 0):a + 2, max(s - 1, 0):s + 2, by - 3:by + 4, bx - 3:bx + 4].max()
    am = np.unravel_index(np.argmax(v), v.shape)
    print(sid[-14:], f"GT ang {ang:.1f} sc {sc:.3f} land {land.round()} | vol@GT {local:.2f} max {v.max():.2f} at ang {ANGLES[am[0]]} sc {SCALES[am[1]]:.2f} land {(np.array(am[:1:-1]) * BIN - MARGIN)}")
