import os, pickle
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import maximum_filter
import consensus as C

C.SCALES = np.arange(0.89, 1.001, 0.03)
HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))


def vol(sid):
    r = R[sid]
    return sid, C.landing_volume(r["iv_c"], r["ex_c"], r["ex_shape"], r["offset"], spread=8).astype(np.float16)


def peak(total):
    a, s, y, x = np.unravel_index(np.argmax(total), total.shape)
    return C.ANGLES[a], np.array([x * C.BIN - C.MARGIN, y * C.BIN - C.MARGIN])


VARIANTS = {
    "clip0": lambda z: z,
    "z>2": lambda z: np.where(z > 2, z - 2, 0),
    "z>2 sq": lambda z: np.where(z > 2, z - 2, 0) ** 2,
    "z>2.5 norm": lambda z: np.where(z > 2.5, z - 2.5, 0) / max(z.max() - 2.5, 1e-3),
}

if __name__ == "__main__":
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        with Pool(8) as pool:
            vols = dict(pool.map(vol, sids))
        gt = [(np.degrees(np.arctan2(R[s]["gt_M"][1, 0], R[s]["gt_M"][0, 0])),
               R[s]["gt_M"][:, :2] @ (C.P0 - R[s]["offset"]) + R[s]["gt_M"][:, 2]) for s in sids if R[s]["gt_M"] is not None]
        print(subj, "GT angle median", np.median([a for a, _ in gt]).round(1), "landing median", np.median([l for _, l in gt], 0).round())
        for spread in (64, 128):
            r = spread // C.BIN
            for name, f in VARIANTS.items():
                total = sum(maximum_filter(f(vols[s].astype(np.float32)), size=(3, 1, 2 * r + 1, 2 * r + 1)) for s in sids)
                ang, land = peak(total)
                hits = sum(abs(a - ang) < 5 and np.linalg.norm(l - land) < 160 for a, l in gt)
                print(f"   spread {spread} {name:12s} -> ang {ang:5.1f} land {land} | GT poses within window {hits}/{len(gt)}")
