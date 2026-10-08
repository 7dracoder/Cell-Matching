"""Registration chain (candidates -> vote -> window -> margin) on alternative point sets.

usage: VARIANTS="pred gt" reg_points_lab.py
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import region_centers, rle_to_labels, read_image  # noqa: E402
from common import ROOT, load_truth  # noqa: E402
from hough import window_register  # noqa: E402
from margin_lab import margin  # noqa: E402
from reg_lab import err  # noqa: E402
from vote import P0, region_candidates, vote_modes  # noqa: E402

R = pickle.load(open(os.path.join(os.path.dirname(__file__), "data", "lab.pkl"), "rb"))
GT_EX = os.path.join(os.path.dirname(__file__), "data", "gt_ex_c.pkl")


def gt_ex_centres():
    truth = load_truth()
    out = {}
    for sid, t in truth.items():
        shape = read_image(os.path.join(ROOT, "training", *sid.split("__"), "exvivo.tif")).shape
        out[sid] = region_centers(rle_to_labels(t["exvivo_instances"], shape)[0])
    pickle.dump(out, open(GT_EX, "wb"))


def top_frac(c, f, frac):
    if frac >= 1:
        return c
    keep = f["contrast"] >= np.quantile(f["contrast"], 1 - frac)
    return c[keep]


def stretch(variant):
    """Inverse of the sectioning compression: variant 'stretch{k}_{phi}' (k ratio, phi axis in ex degrees)."""
    if not variant.startswith("stretch"):
        return np.eye(2)
    k, phi = (float(x) for x in variant[7:].split("_"))
    u = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    return np.eye(2) + (1 / k - 1) * np.outer(u, u)


def unstretch(M, variant):
    return None if M is None else np.linalg.inv(stretch(variant)) @ M


def points(variant, sid):
    r = R[sid]
    if variant.startswith("stretch"):
        return r["iv_c"], r["ex_c"] @ stretch(variant).T
    if variant == "pred":
        return r["iv_c"], r["ex_c"]
    if variant == "gt":
        return r["gt_iv_c"], pickle.load(open(GT_EX, "rb"))[sid]
    if variant.startswith("bright"):
        iv_f, ex_f = (float(x) for x in variant[6:].split("_"))
        return top_frac(r["iv_c"], r["iv_f"], iv_f), top_frac(r["ex_c"], r["ex_f"], ex_f)
    raise ValueError(variant)


def cands(args):
    variant, sid = args
    iv, ex = points(variant, sid)
    return sid, (iv, ex, region_candidates(iv, ex, R[sid]["offset"]))


def window(args):
    sid, iv, ex, modes = args
    return sid, window_register(iv, ex, R[sid]["offset"], modes, P0, angle_win=5.0)


if __name__ == "__main__":
    if not os.path.exists(GT_EX):
        gt_ex_centres()
    for variant in os.environ["VARIANTS"].split():
        with Pool(8) as pool:
            C = dict(pool.map(cands, [(variant, s) for s in R]))
        jobs = []
        for subj in sorted({r["subject"] for r in R.values()}):
            sids = [s for s in R if R[s]["subject"] == subj]
            modes = vote_modes({s: C[s][2] for s in sids})
            jobs += [(s, C[s][0], C[s][1], modes) for s in sids]
        with Pool(8) as pool:
            W = dict(pool.map(window, jobs))
        ok = {s: W[s][0] is not None and err(R[s], unstretch(W[s][0], variant)) < 5 for s in R}
        mg = {s: margin(C[s][2], W[s][0], W[s][1], R[s]) for s in R}
        gated = [s for s in R if mg[s] >= 3]
        pairs_ok = sum(R[s]["n_gt_pairs"] for s in R if ok[s])
        pairs_gated_ok = sum(R[s]["n_gt_pairs"] for s in gated if ok[s])
        print(f"{variant:14s} correct {sum(ok.values())}/47 (GT pairs {pairs_ok}) | gated {len(gated)} "
              f"correct-in-gate {sum(ok[s] for s in gated)} (GT pairs {pairs_gated_ok})", flush=True)
        pickle.dump({s: (W[s], mg[s], ok[s], C[s][2]) for s in R}, open(f"data/regpts_{variant}.pkl", "wb"))
