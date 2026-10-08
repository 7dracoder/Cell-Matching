"""Hypothesis: annotators traced each ex-vivo cell at a fixed fraction of its own local contrast.

For every GT cell: peak = high percentile inside the mask, bg = median of a ring 3-7 px outside
(excluding other GT cells). Boundary level f = (I_edge - bg) / (peak - bg), with I_edge the mean of
the inner and outer boundary pixels. If f is stable across mice, an oracle-seeded relative
threshold mask {I > bg + f (peak - bg)} (component at the cell) should reproduce GT at IoU > 0.75.

usage: contour_probe.py [modality=exvivo] [sigma=0.7]
"""
import sys
from collections import defaultdict

import cv2
import numpy as np
from scipy import ndimage as ndi

from common import label_map, load_truth, read, regions

MOD = sys.argv[1] if len(sys.argv) > 1 else "exvivo"
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
FRACS = np.round(np.arange(0.1, 0.81, 0.05), 2)
CROSS = ndi.generate_binary_structure(2, 1)


def cell_window(lab, k, pad=10):
    ys, xs = np.nonzero(lab == k)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, lab.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, lab.shape[1])
    return slice(y0, y1), slice(x0, x1)


def stats(img, lab, k):
    sl = cell_window(lab, k)
    L, I = lab[sl], img[sl]
    m = L == k
    others = (L > 0) & ~m
    d = ndi.distance_transform_edt(~m)
    ring = (d >= 3) & (d <= 7) & ~ndi.binary_dilation(others, iterations=2)
    if ring.sum() < 10 or m.sum() < 8:
        return None
    bg = np.median(I[ring])
    peak = np.percentile(I[m], 90)
    if peak - bg <= 1e-6:
        return None
    inner = m & ~ndi.binary_erosion(m, CROSS)
    outer = ndi.binary_dilation(m, CROSS) & ~m & ~others
    f = (0.5 * (I[inner].mean() + I[outer].mean()) - bg) / (peak - bg)
    # oracle: seed at the brightest interior pixel, threshold at each fraction
    seed = np.unravel_index(np.argmax(np.where(m, I, -np.inf)), I.shape)
    ious = []
    for fr in FRACS:
        fg = (I > bg + fr * (peak - bg)) & ~others
        cc, _ = ndi.label(fg, CROSS)
        c = cc[seed]
        if c == 0:
            ious.append(0.0)
            continue
        p = cc == c
        ious.append((p & m).sum() / (p | m).sum())
    return f, np.array(ious), (peak - bg), m.sum()


if __name__ == "__main__":
    truth = load_truth()
    F, IO = defaultdict(list), defaultdict(list)
    for subj, reg, path in regions("training"):
        sid = f"{subj}__{reg}"
        img = read(f"{path}/{MOD}.tif")
        img = cv2.GaussianBlur(img, (0, 0), SIGMA) if SIGMA > 0 else img
        lab, _ = label_map(truth[sid][f"{MOD}_instances"], img.shape)
        for k in range(1, lab.max() + 1):
            if not (lab == k).any():
                continue
            s = stats(img, lab, k)
            if s is None:
                continue
            F[subj].append(s[0])
            IO[subj].append(s[1])
    allio = []
    for subj in F:
        f, io = np.array(F[subj]), np.array(IO[subj])
        allio.append(io)
        best = FRACS[np.argmax((io > 0.75).mean(0))]
        print(f"{subj}: n {len(f)} f median {np.median(f):.2f} IQR [{np.percentile(f, 25):.2f}, {np.percentile(f, 75):.2f}]"
              f" | oracle IoU>.75 at best fixed frac {best}: {(io > 0.75).mean(0).max():.2f}"
              f" | per-cell best frac: {(io.max(1) > 0.75).mean():.2f}")
        print("   IoU>.75 by frac:", dict(zip(FRACS.tolist(), np.round((io > 0.75).mean(0), 2).tolist())))
    io = np.vstack(allio)
    print("pooled best frac", FRACS[np.argmax((io > 0.75).mean(0))], "rate", (io > 0.75).mean(0).max().round(3))
