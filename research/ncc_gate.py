"""NCC of the chosen window pose; can it rescue low-margin regions safely?"""
import os
import pickle
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import normalize, read_image  # noqa: E402
from common import ROOT  # noqa: E402
from margin_lab import R, C, margin  # noqa: E402
from pose_cands import ncc  # noqa: E402
from reg_lab import err  # noqa: E402
from window_lab import pose  # noqa: E402

W = pickle.load(open("data/reg_window_vote5.pkl", "rb"))


def ncc_of(sid):
    r = R[sid]
    M, score = W[sid]
    if M is None:
        return sid, 0.0, 0.0, -99.0, False
    path = os.path.join(ROOT, "training", *sid.split("__"))
    iv_n = normalize(read_image(path + "/invivo.tif"), "invivo")[1]
    ex_n = normalize(read_image(path + "/exvivo.tif"), "exvivo")[1]
    warped = cv2.warpAffine(iv_n, M.astype(np.float32), ex_n.shape[::-1], flags=cv2.INTER_LINEAR)
    cover = cv2.warpAffine(np.ones_like(iv_n), M.astype(np.float32), ex_n.shape[::-1]) > 0.99
    return sid, ncc(warped, ex_n, cover), ncc(cv2.GaussianBlur(warped, (0, 0), 2),
                                                 cv2.GaussianBlur(ex_n, (0, 0), 2), cover), \
        margin(C[sid], M, score, r), err(r, M) < 5


if __name__ == "__main__":
    with Pool(8) as pool:
        rows = pool.map(ncc_of, list(R))
    print(f"{'sid':22s} {'ncc':6s} {'nccb':6s} {'mg':6s} ok")
    for sid, n, nb, mg, ok in sorted(rows, key=lambda t: t[3]):
        print(f"{sid[-22:]} {n:6.3f} {nb:6.3f} {mg:6.1f} {'OK' if ok else 'BAD'}")
    # gates: margin>=3 OR (margin>=lo AND nccb>=thr)
    for lo in (-1, 0, 1, 2):
        for thr in (0.15, 0.2, 0.25, 0.3, 0.35):
            g = [(ok, R[s]["n_gt_pairs"]) for s, n, nb, mg, ok in rows if mg >= 3 or (mg >= lo and nb >= thr)]
            if not g:
                continue
            print(f"mg>=3 or (mg>={lo} & nccb>={thr}): gated {len(g)} correct {sum(o for o,_ in g)} "
                  f"GT {sum(p for o,p in g if o)} wrongGT {sum(p for o,p in g if not o)}")
