"""Per-cell boundary variants for predicted ex-vivo masks (held-out), by cell brightness.

Finding that motivates this: predicted masks of *verified-pair* ex cells that miss IoU 0.75 are
too big in all three mice (pred/GT area ~1.35), unlike the whole population.
For every predicted ex cell we compute the IoU with its GT cell under several shrink variants
and store features, so a brightness-conditioned rule can be chosen leave-one-mouse-out.
Writes data/shrink_cells.pkl: list of dicts (sid, subj, k, feats, iou per variant, gt index,
gt verified flag).
"""
import pickle
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import ndimage as ndi

from common import label_map, load_truth, read, regions

CROSS = ndi.generate_binary_structure(2, 1)
Z = np.load("data/heldout_labels.npz")
TRUTH = load_truth()
VARIANTS = ["base", "e1", "ep10", "ep25", "ep35", "ep50", "ep60", "ep75", "it20", "it30", "it40", "it50", "cp0", "cp1", "cp2"]


def variants(m, I, cp, others):
    out = {"base": m}
    inner = m & ~ndi.binary_erosion(m, CROSS)
    e1 = m & ~inner
    out["e1"] = e1 if e1.sum() >= 8 else m
    vals = cp[inner]
    for q in (10, 25, 35, 50, 60, 75):
        cut = np.quantile(vals, q / 100) if len(vals) else -np.inf
        v = m & ~(inner & (cp <= cut))
        out[f"ep{q}"] = v if v.sum() >= 8 else m
    d = ndi.distance_transform_edt(~m)
    ring = (d >= 3) & (d <= 7) & ~ndi.binary_dilation(others, iterations=2)
    bg = np.median(I[ring]) if ring.sum() > 5 else np.percentile(I, 20)
    peak = np.percentile(I[m], 95)
    seed = np.unravel_index(np.argmax(np.where(m, I, -np.inf)), I.shape)
    for f in (20, 30, 40, 50):
        fg = m & (I >= bg + f / 100 * (peak - bg))
        cc, _ = ndi.label(fg, CROSS)
        v = cc == cc[seed] if cc[seed] else m
        out[f"it{f}"] = v if v.sum() >= 8 else m
    for t in (0, 1, 2):
        v = m & (cp > t)
        cc, _ = ndi.label(v, CROSS)
        if cc.max():
            v = cc == (np.argmax(np.bincount(cc[cc > 0])) )
        out[f"cp{t}"] = v if v.sum() >= 8 else m
    return out, bg, peak


def region(sr):
    subj, reg, path = sr
    sid = f"{subj}__{reg}"
    P = Z[f"{sid}|exvivo"].astype(np.int32)
    cp = Z[f"{sid}|exvivo|prob"].astype(np.float32)
    img = cv2.GaussianBlur(read(f"{path}/exvivo.tif"), (0, 0), 0.7)
    t = TRUTH[sid]
    G, gid = label_map(t["exvivo_instances"], P.shape)
    ids = list(t["exvivo_instances"])
    verified = {ids.index(b) + 1 for _, b in t["match_pairs"]}
    lo, hi = np.percentile(img, [1, 99.8])
    objs = ndi.find_objects(P)
    rows = []
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        pad = 10
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, P.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, P.shape[1])
        Lw, Gw, Iw, Cw = P[y0:y1, x0:x1], G[y0:y1, x0:x1], img[y0:y1, x0:x1], cp[y0:y1, x0:x1]
        m = Lw == k
        V, bg, peak = variants(m, Iw, Cw, (Lw > 0) & ~m)
        g = np.bincount(Gw[m], minlength=1)
        g[0] = 0
        gk = int(np.argmax(g)) if g.max() else 0
        gm = Gw == gk if gk else np.zeros_like(m)
        ious, areas = {}, {}
        for n, v in V.items():
            ious[n] = (v & gm).sum() / (v | gm).sum() if gk else 0.0
            areas[n] = int(v.sum())
        rows.append(dict(sid=sid, subj=subj, k=k, gt=gk, verified=gk in verified, iou=ious, areas=areas, cp_inner=float(Cw[m & ~ndi.binary_erosion(m, CROSS)].mean()), cp_max=float(Cw[m].max()),
                         area=int(m.sum()), peak=(peak - lo) / (hi - lo), contrast=(peak - bg) / (hi - lo),
                         cpmean=float(Cw[m].mean()), gt_area=int(gm.sum())))
    return rows


if __name__ == "__main__":
    with Pool(8) as pool:
        out = [r for rows in pool.map(region, regions("training")) for r in rows]
    pickle.dump(out, open("data/shrink_cells.pkl", "wb"))
    for subj in sorted({r["subj"] for r in out}):
        for sel, name in ((lambda r: True, "all"), (lambda r: r["verified"], "verified")):
            rs = [r for r in out if r["subj"] == subj and sel(r)]
            print(subj[-6:], f"{name:8s} n {len(rs):4d}", "  ".join(
                f"{v} {np.mean([r['iou'][v] > .75 for r in rs]):.3f}" for v in VARIANTS))
