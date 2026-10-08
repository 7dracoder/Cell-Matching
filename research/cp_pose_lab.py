"""Pose selection + gating with dense ex-vivo cellprob evidence (held-out, leave-one-mouse-out).

For each region: candidate poses = refined windowed-Hough candidates around the per-mouse vote
modes + the region's own vote candidates. Each candidate gets a cellprob z-score: fraction of
projected in-vivo centroids that land on ex-vivo cellprob > 0 (5x5 dilated), versus a null of
the same pose shifted by 15-45 px. Selection and gating use z instead of / in addition to the
centroid-count refine score and margin.

Writes data/cp_pose.pkl: sid -> dict(M, score, z, margin, cands=[(M, score, z)]).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import cv2
import numpy as np

import pair_clf
from hough import window_candidates
from margin_lab import C, R, margin
from reg_lab import err
from vote import P0, vote_modes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine  # noqa: E402

HERE = os.path.dirname(__file__)
JIT = np.array([(dx, dy) for dx in range(-40, 41, 8) for dy in range(-40, 41, 8) if 15 <= np.hypot(dx, dy) <= 45], float)


def cp_map(cp):
    return cv2.dilate((cp > 0).astype(np.uint8), np.ones((5, 5), np.uint8))


def hit(binmap, pts):
    h, w = binmap.shape
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if ok.sum() < 10:
        return np.nan
    return float(binmap[y[ok], x[ok]].mean())


def cp_z(binmap, iv_c, M):
    pts = iv_c @ M[:, :2].T + M[:, 2]
    e0 = hit(binmap, pts)
    if not np.isfinite(e0):
        return -9.0
    null = np.array([hit(binmap, pts + j) for j in JIT])
    null = null[np.isfinite(null)]
    if len(null) < 5:
        return -9.0
    return float((e0 - null.mean()) / (null.std() + 1e-3))


def all_window(iv_c, ex_c, offset, modes, angle_win=5.0, radius=120, top=30):
    out = []
    for mode in modes:
        for votes, M in window_candidates(iv_c, ex_c, offset, mode, P0, angle_win, radius)[:top]:
            Rm, score = refine(iv_c, ex_c, M)
            land = Rm[:, :2] @ (P0 - offset) + Rm[:, 2]
            ang = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
            if np.linalg.norm(land - mode["landing"]) > radius * 1.3 or abs(ang - mode["angle"]) > angle_win + 2:
                continue
            out.append((Rm, score, "win"))
    return out


def process(rec, binmap, own_cands, modes):
    """rec needs iv_c, ex_c, offset. Returns candidate list [(M, score, z, src)] and chosen indices."""
    cands = all_window(rec["iv_c"], rec["ex_c"], rec["offset"], modes)
    cands += [(c[3], c[0], "vote") for c in own_cands]
    scored = [(M, s, cp_z(binmap, rec["iv_c"], M), src) for M, s, src in cands]
    return scored


def run(args):
    s, modes, cp_path, key = args
    r = R[s]
    cp = np.load(cp_path)[key].astype(np.float32)
    return s, process(r, cp_map(cp), C[s], modes)


def pose_choose(scored, rule):
    win = [c for c in scored if c[3] == "win"]
    if not win:
        return None
    if rule == "score":
        return max(win, key=lambda c: c[1])
    if rule == "z":
        return max(win, key=lambda c: c[2])
    if rule == "mix":
        sc = np.array([c[1] for c in win])
        zz = np.array([c[2] for c in win])
        v = (sc - sc.mean()) / (sc.std() + 1e-6) + (zz - zz.mean()) / (zz.std() + 1e-6)
        return win[int(np.argmax(v))]
    raise ValueError(rule)


def z_margin(scored, chosen, r):
    """z of chosen minus best z among clearly different candidates (any source)."""
    from window_lab import pose
    a, l = pose(r, chosen[0])
    null = []
    for M, s, z, src in scored:
        b, m = pose(r, M)
        if abs(b - a) > 3 or np.linalg.norm(m - l) > 60:
            null.append(z)
    return chosen[2] - (max(null) if null else 0.0)


def evaluate(S, rule, gate):
    regs, info = {}, {}
    for s, scored in S.items():
        ch = pose_choose(scored, rule)
        if ch is None:
            regs[s] = (None, 0.0)
            info[s] = (-99, -99, -99, np.inf)
            continue
        regs[s] = (ch[0], ch[1])
        info[s] = (margin(C[s], ch[0], ch[1], R[s]), ch[2], z_margin(S[s], ch, R[s]), err(R[s], ch[0]))
    rows = pair_clf.dataset(regs)
    probs = pair_clf.loo_predict(rows)
    res = []
    for name, fn in gate.items():
        keep_reg = {s: fn(*info[s][:3]) for s in R}
        best = (0, 0)
        for th in np.arange(0, 0.3, 0.025):
            tp = sum(int(y[probs[s] >= th].sum()) for s, _, _, y in rows if keep_reg[s])
            pred = sum(int((probs[s] >= th).sum()) for s, _, _, y in rows if keep_reg[s])
            best = max(best, (2 * tp / (pred + pair_clf.TOTAL), th))
        nk = sum(keep_reg.values())
        ok = sum(info[s][3] < 5 for s in R if keep_reg[s])
        res.append((name, best[0], best[1], ok, nk))
    correct = sum(info[s][3] < 5 for s in R)
    return correct, res, info


GATES = {
    "margin>=3": lambda mg, z, zm: mg >= 3,
    "all": lambda mg, z, zm: True,
    "z>=4": lambda mg, z, zm: z >= 4,
    "z>=5": lambda mg, z, zm: z >= 5,
    "z>=6": lambda mg, z, zm: z >= 6,
    "mg3|z>=5": lambda mg, z, zm: mg >= 3 or z >= 5,
    "mg3|z>=6": lambda mg, z, zm: mg >= 3 or z >= 6,
    "mg3|zm>=1": lambda mg, z, zm: mg >= 3 or zm >= 1,
    "mg3|zm>=2": lambda mg, z, zm: mg >= 3 or zm >= 2,
    "zm>=1": lambda mg, z, zm: zm >= 1,
    "zm>=2": lambda mg, z, zm: zm >= 2,
    "mg3|(z>=4&zm>=1)": lambda mg, z, zm: mg >= 3 or (z >= 4 and zm >= 1),
}


if __name__ == "__main__":
    cp_path = os.path.join(HERE, "data", "heldout_labels.npz")
    jobs = []
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        modes = vote_modes({s: C[s] for s in sids})
        jobs += [(s, modes, cp_path, s + "|exvivo|prob") for s in sids]
    with Pool(8) as pool:
        S = dict(pool.map(run, jobs))
    pickle.dump(S, open(os.path.join(HERE, "data", "cp_pose_train.pkl"), "wb"))
    for rule in ("score", "z", "mix"):
        correct, res, info = evaluate(S, rule, GATES)
        print(f"rule {rule}: correct {correct}/47")
        for name, f1, th, ok, nk in res:
            print(f"   gate {name:18s} F1 {f1:.4f} thr {th:.3f} kept {nk} correct {ok}")
        if rule == "z":
            for s in sorted(R):
                mg, z, zm, e = info[s]
                print(f"      {s[-15:]} mg {mg:6.1f} z {z:6.2f} zm {zm:6.2f} err {e:7.1f}")
