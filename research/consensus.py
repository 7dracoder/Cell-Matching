"""Per-mouse consensus pose from pooled NCC score volumes in in-vivo mosaic coordinates,
then a constrained per-region search around it. Needs no labels."""
import numpy as np, cv2
from scipy.ndimage import maximum_filter

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine, blob_map  # noqa: E402

ANGLES = np.arange(-33, 33.1, 1.5)
SCALES = np.arange(0.86, 1.05, 0.03)
P0 = np.array([300.0, 300.0])  # mosaic probe point
BIN, MARGIN = 8, 640


def rotation(angle):
    t = np.deg2rad(angle)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def ncc_map(iv_c, ex_c, ex_shape, linear, down):
    proj = iv_c @ linear.T / down
    origin = proj.min(0) - 4
    tp = proj - origin
    tmpl = blob_map(tp, (int(tp[:, 1].max()) + 5, int(tp[:, 0].max()) + 5), 1.2)
    ph, pw = tmpl.shape[0] // 2, tmpl.shape[1] // 2
    shape = (int(np.ceil(ex_shape[0] / down)), int(np.ceil(ex_shape[1] / down)))
    img = cv2.copyMakeBorder(blob_map(ex_c / down, shape, 1.2), ph, ph, pw, pw, cv2.BORDER_CONSTANT)
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCORR_NORMED)
    # res[y, x] <-> shift = ((x, y) - origin - (pw, ph)) * down
    return res, -(origin + [pw, ph]) * down


def landing_volume(iv_c, ex_c, ex_shape, offset, down=4.0, spread=96):
    """Normalised NCC over (angle, scale, landing bin of the mosaic probe point)."""
    H = (ex_shape[0] + 2 * MARGIN) // BIN + 1
    W = (ex_shape[1] + 2 * MARGIN) // BIN + 1
    vol = np.zeros((len(ANGLES), len(SCALES), H, W), np.float32)
    k = int(BIN / down)
    for a, angle in enumerate(ANGLES):
        for s, scale in enumerate(SCALES):
            lin = scale * rotation(angle)
            res, shift0 = ncc_map(iv_c, ex_c, ex_shape, lin, down)
            h, w = res.shape[0] // k * k, res.shape[1] // k * k
            pooled = res[:h, :w].reshape(h // k, k, w // k, k).max(axis=(1, 3))
            land0 = shift0 + lin @ (P0 - offset) + MARGIN  # landing of res[0, 0], in margin coords
            bx, by = int(round(land0[0] / BIN)), int(round(land0[1] / BIN))
            y0, x0 = max(by, 0), max(bx, 0)
            y1, x1 = min(by + pooled.shape[0], H), min(bx + pooled.shape[1], W)
            if y1 > y0 and x1 > x0:
                vol[a, s, y0:y1, x0:x1] = pooled[y0 - by:y1 - by, x0 - bx:x1 - bx]
    flat = vol[vol > 0]
    vol = np.clip((vol - np.median(flat)) / (flat.std() + 1e-6), 0, None)
    r = spread // BIN
    return maximum_filter(vol, size=(3, 3, 2 * r + 1, 2 * r + 1))


def consensus_modes(volumes, n_modes=2, min_sep_deg=9):
    total = np.sum(volumes, axis=0)
    modes = []
    work = total.copy()
    for _ in range(n_modes):
        a, s, y, x = np.unravel_index(np.argmax(work), work.shape)
        modes.append({"angle": float(ANGLES[a]), "scale": float(SCALES[s]),
                      "landing": np.array([x * BIN - MARGIN, y * BIN - MARGIN], float), "value": float(work[a, s, y, x])})
        lo = np.abs(ANGLES - ANGLES[a]) < min_sep_deg
        work[lo] = 0
    return modes


def constrained_register(iv_c, ex_c, ex_shape, offset, mode, angle_win=4.5, radius=160, down=2.0, top=6):
    """Best pose whose rotation is near the mode's and whose probe landing is within radius."""
    cands = []
    for angle in np.arange(mode["angle"] - angle_win, mode["angle"] + angle_win + 0.1, 1.0):
        for scale in SCALES:
            lin = scale * rotation(angle)
            res, shift0 = ncc_map(iv_c, ex_c, ex_shape, lin, down)
            # landing(x, y) = shift0 + (x, y) * down + lin @ (P0 - offset)
            base = shift0 + lin @ (P0 - offset)
            yy, xx = np.indices(res.shape)
            land_x, land_y = base[0] + xx * down, base[1] + yy * down
            mask = (land_x - mode["landing"][0]) ** 2 + (land_y - mode["landing"][1]) ** 2 <= radius ** 2
            if not mask.any():
                continue
            masked = np.where(mask, res, -1)
            y, x = np.unravel_index(np.argmax(masked), masked.shape)
            cands.append((float(masked[y, x]), np.c_[lin, shift0 + np.array([x, y]) * down]))
    cands.sort(key=lambda c: c[0], reverse=True)
    results = []
    for v, M in cands[:top]:
        R, score = refine(iv_c, ex_c, M)
        land = R[:, :2] @ (P0 - offset) + R[:, 2]
        if np.linalg.norm(land - mode["landing"]) <= radius * 1.25:
            results.append((R, score, v))
        else:
            results.append((M, refine_score(iv_c, ex_c, M), v))
    return max(results, key=lambda r: r[1]) if results else (None, 0.0, 0.0)


def refine_score(iv_c, ex_c, M):
    from registration import paired_nearest
    _, _, d = paired_nearest(iv_c, ex_c, M, 8)
    return float(np.sum(1 + np.maximum(0, 1 - d / 8)))
