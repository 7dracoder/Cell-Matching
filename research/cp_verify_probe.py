"""Probe: does dense ex-vivo cellprob evidence separate correct vs wrong poses?

For a pose M (in-vivo -> ex-vivo affine), project every predicted in-vivo centroid into the
ex-vivo frame and read the local max of the ex-vivo cellprob map. Compare against a null made by
jittering the pose translation by 15-40 px. Output a z-score per region for the chosen window pose
and report how it separates correct (err < 5 px) from wrong regions, especially below the margin gate.
"""
import pickle

import cv2
import numpy as np

from margin_lab import C, R, margin
from reg_lab import err

Z = np.load("data/heldout_labels.npz")
W = pickle.load(open("data/reg_window_vote5.pkl", "rb"))
rng = np.random.default_rng(0)
JIT = [(dx, dy) for dx in range(-40, 41, 8) for dy in range(-40, 41, 8) if 15 <= np.hypot(dx, dy) <= 45]


def evidence(cpmax, pts):
    h, w = cpmax.shape
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if ok.sum() < 10:
        return np.nan, ok.sum()
    return float(np.mean(cpmax[y[ok], x[ok]] > 0)), ok.sum()


def zscore(cpmax, iv_c, M):
    pts = iv_c @ M[:, :2].T + M[:, 2]
    e0, n = evidence(cpmax, pts)
    null = [evidence(cpmax, pts + np.array(j))[0] for j in JIT]
    null = np.array([v for v in null if np.isfinite(v)])
    return (e0 - null.mean()) / (null.std() + 1e-3), e0, n


if __name__ == "__main__":
    rows = []
    for s, r in R.items():
        M = W[s][0]
        if M is None or r["gt_M"] is None:
            continue
        cp = Z[s + "|exvivo|prob"].astype(np.float32)
        cpmax = cv2.dilate(cp, np.ones((5, 5), np.uint8))
        z, e0, n = zscore(cpmax, r["iv_c"], M)
        e = err(r, M)
        mg = margin(C[s], M, W[s][1], r)
        rows.append((s, z, e0, n, e, mg))
        print(f"{s[-15:]} z {z:6.2f} hit {e0:.2f} n {n:4d} err {e:7.1f} margin {mg:6.1f}", flush=True)
    pickle.dump(rows, open("/tmp/cp_verify_rows.pkl", "wb"))
    ok = np.array([r[4] < 5 for r in rows])
    z = np.array([r[1] for r in rows])
    mg = np.array([r[5] for r in rows])
    print("all: correct z mean %.2f  wrong z mean %.2f" % (z[ok].mean(), z[~ok].mean()))
    lo = mg < 3
    print("below gate: correct", sorted(np.round(z[lo & ok], 2)), "\n wrong", sorted(np.round(z[lo & ~ok], 2)))
