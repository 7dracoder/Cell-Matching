"""Candidate poses per region inside the voted windows, with features for a learned pose scorer.

Training: writes data/pose_cands.pkl  {sid: (features [K, F], errors [K], matrices)}.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import normalize, read_image  # noqa: E402
from common import ROOT  # noqa: E402
from hough import window_candidates  # noqa: E402
from registration import refine, transform  # noqa: E402
from vote import P0, vote_modes  # noqa: E402

FEATURES = ["score", "score_gap", "rank", "n3", "n5", "bright_sum", "bright_both", "ncc", "ncc_blur",
            "d_angle", "d_land", "scale", "overlap"]


def rank(v):
    return v.argsort().argsort() / max(len(v) - 1, 1)


def candidates(iv_c, ex_c, offset, modes, angle_win=5.0, radius=120, top=30, keep=20):
    out = []
    for mode in modes:
        for votes, M in window_candidates(iv_c, ex_c, offset, mode, P0, angle_win, radius)[:top]:
            R, score = refine(iv_c, ex_c, M)
            land = R[:, :2] @ (P0 - offset) + R[:, 2]
            ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            if np.linalg.norm(land - mode["landing"]) > radius * 1.3 or abs(ang - mode["angle"]) > angle_win + 2:
                continue
            if any(abs(ang - a) < 1 and np.linalg.norm(land - l) < 10 for _, a, l, _, _ in out):
                continue
            out.append((score, ang, land, R, mode))
    out.sort(key=lambda c: c[0], reverse=True)
    return out[:keep]


def ncc(a, b, mask):
    a, b = a[mask], b[mask]
    if a.size < 100:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def features(cands, iv_c, ex_c, iv_f, ex_f, iv_img, ex_img):
    ivr, exr = rank(iv_f["contrast"]), rank(ex_f["contrast"])
    iv_n, ex_n = normalize(iv_img, "invivo")[1], normalize(ex_img, "exvivo")[1]
    ex_blur = cv2.GaussianBlur(ex_n, (0, 0), 2)
    tree = cKDTree(ex_c)
    best = cands[0][0] if cands else 0.0
    rows = []
    for k, (score, ang, land, M, mode) in enumerate(cands):
        P = transform(iv_c, M)
        d, j = tree.query(P)
        back = cKDTree(P).query(ex_c)[1]
        mutual = back[j] == np.arange(len(P))
        m5 = mutual & (d < 5)
        warped = cv2.warpAffine(iv_n, M.astype(np.float32), ex_n.shape[::-1], flags=cv2.INTER_LINEAR)
        cover = cv2.warpAffine(np.ones_like(iv_n), M.astype(np.float32), ex_n.shape[::-1]) > 0.99
        rows.append([score, best - score, k, int((mutual & (d < 3)).sum()), int(m5.sum()),
                     float((ivr[m5] * exr[j[m5]]).sum()), int(((ivr[m5] > 0.5) & (exr[j[m5]] > 0.5)).sum()),
                     ncc(warped, ex_n, cover), ncc(cv2.GaussianBlur(warped, (0, 0), 2), ex_blur, cover),
                     abs(ang - mode["angle"]), float(np.linalg.norm(land - mode["landing"])),
                     float(np.sqrt(abs(np.linalg.det(M[:, :2])))), float(cover.mean())])
    return np.array(rows, float).reshape(-1, len(FEATURES))


def training_job(args):
    sid, modes = args
    from reg_lab import err
    r = R[sid]
    path = os.path.join(ROOT, "training", *sid.split("__"))
    cands = candidates(r["iv_c"], r["ex_c"], r["offset"], modes)
    X = features(cands, r["iv_c"], r["ex_c"], r["iv_f"], r["ex_f"],
                 read_image(path + "/invivo.tif"), read_image(path + "/exvivo.tif"))
    errs = np.array([err(r, c[3]) for c in cands])
    return sid, (X, errs, [c[3] for c in cands])


R = pickle.load(open(os.path.join(os.path.dirname(__file__), "data", "lab.pkl"), "rb"))
C = pickle.load(open(os.path.join(os.path.dirname(__file__), "data", "vote_cands.pkl"), "rb"))

if __name__ == "__main__":
    jobs = []
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        modes = vote_modes({s: C[s] for s in sids})
        jobs += [(s, modes) for s in sids]
    with Pool(8) as pool:
        out = dict(pool.map(training_job, jobs))
    pickle.dump(out, open(os.path.join(os.path.dirname(__file__), "data", "pose_cands.pkl"), "wb"))
    has = sum(bool((e < 5).any()) for _, e, _ in out.values())
    top1 = sum(bool(len(e) and e[0] < 5) for _, e, _ in out.values())
    print(f"regions with a correct candidate {has}/47 | top-1 by score correct {top1}/47")
    print("GT pairs in regions with a correct candidate",
          sum(R[s]["n_gt_pairs"] for s, (_, e, _) in out.items() if (e < 5).any()))
