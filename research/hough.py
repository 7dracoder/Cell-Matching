"""Exhaustive similarity-grid Hough registration: counts cell pairs per translation bin."""
import os, sys
import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine  # noqa: E402

ANGLES = np.arange(-35, 35.1, 1.0)
SCALES = np.arange(0.88, 1.011, 0.015)


def rotation(angle):
    t = np.deg2rad(angle)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def hough_candidates(iv_c, ex_c, angles=ANGLES, scales=SCALES, bin_size=6.0, per_pose=2, keep=40):
    lo = -2500.0
    nb = int(5000 / bin_size) + 1
    cands = []
    for angle in angles:
        rot = rotation(angle)
        for scale in scales:
            lin = scale * rot
            diff = (ex_c[:, None, :] - (iv_c @ lin.T)[None]).reshape(-1, 2)
            ij = np.floor((diff - lo) / bin_size).astype(np.int64)
            ok = (ij >= 0).all(1) & (ij < nb).all(1)
            # 2x2-bin box sum makes the count robust to bin-edge splits
            flat = np.bincount(ij[ok, 1] * nb + ij[ok, 0], minlength=nb * nb).reshape(nb, nb)
            occupied = np.unique(ij[ok, 1] * nb + ij[ok, 0])
            ys, xs = occupied // nb, occupied % nb
            box = (flat[ys, xs] + flat[np.minimum(ys + 1, nb - 1), xs] + flat[ys, np.minimum(xs + 1, nb - 1)]
                   + flat[np.minimum(ys + 1, nb - 1), np.minimum(xs + 1, nb - 1)])
            for k in np.argsort(box)[-per_pose:]:
                y, x = ys[k], xs[k]
                sel = ok.copy()
                sel[ok] = (np.abs(ij[ok, 0] - x - 0.5) <= 1.5) & (np.abs(ij[ok, 1] - y - 0.5) <= 1.5)
                shift = np.median(diff[sel], axis=0)
                cands.append((int(box[k]), angle, scale, np.c_[lin, shift]))
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands[:keep]


def window_candidates(iv_c, ex_c, offset, mode, probe, angle_win=3.5, radius=120, bin_size=5.0, keep=40):
    """Hough peaks whose mosaic-probe landing lies within radius of the mode's landing."""
    cands = []
    for angle in np.arange(mode["angle"] - angle_win, mode["angle"] + angle_win + 0.01, 0.5):
        rot = rotation(angle)
        for scale in np.arange(0.885, 1.006, 0.01):
            lin = scale * rot
            base = lin @ (probe - offset)  # landing = shift + base
            diff = (ex_c[:, None, :] - (iv_c @ lin.T)[None]).reshape(-1, 2)
            land = diff + base
            near = np.sum((land - mode["landing"]) ** 2, axis=1) <= radius ** 2
            if near.sum() < 3:
                continue
            d = diff[near]
            ij = np.floor(d / bin_size).astype(np.int64)
            ij -= ij.min(0)
            nb = ij.max(0) + 2
            flat = np.bincount(ij[:, 1] * nb[0] + ij[:, 0], minlength=nb[0] * nb[1]).reshape(nb[1], nb[0])
            box = flat[:-1, :-1] + flat[1:, :-1] + flat[:-1, 1:] + flat[1:, 1:]
            for k in np.argsort(box.ravel())[-3:]:
                y, x = divmod(int(k), box.shape[1])
                sel = (ij[:, 0] >= x) & (ij[:, 0] <= x + 1) & (ij[:, 1] >= y) & (ij[:, 1] <= y + 1)
                if sel.sum() >= 3:
                    cands.append((int(box[y, x]), np.c_[lin, np.median(d[sel], axis=0)]))
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands[:keep]


def window_register(iv_c, ex_c, offset, modes, probe, angle_win=3.5, radius=120, top=30):
    best = (None, 0.0)
    for mode in modes:
        for votes, M in window_candidates(iv_c, ex_c, offset, mode, probe, angle_win, radius)[:top]:
            R, score = refine(iv_c, ex_c, M)
            land = R[:, :2] @ (probe - offset) + R[:, 2]
            ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            if np.linalg.norm(land - mode["landing"]) > radius * 1.3 or abs(ang - mode["angle"]) > angle_win + 2:
                continue
            if score > best[1]:
                best = (R, score)
    return best


def hough_register(iv_c, ex_c, top=25, **kw):
    if len(iv_c) < 3 or len(ex_c) < 3:
        return None, 0.0, []
    cands = hough_candidates(iv_c, ex_c, **kw)
    refined = []
    for votes, angle, scale, M in cands[:top]:
        R, score = refine(iv_c, ex_c, M)
        refined.append((score, votes, R))
    refined.sort(key=lambda c: c[0], reverse=True)
    return refined[0][2], refined[0][0], refined
