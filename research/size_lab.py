"""Per-mouse held-out PQ when ex/in-vivo masks are grown or shrunk by a boundary layer."""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from cellmatch import pq_score, read_image, rle_to_labels  # noqa: E402
from common import ROOT, load_truth  # noqa: E402

CROSS = ndi.generate_binary_structure(2, 1)


def grow(lab, steps=1, prob=None, thr=None, frac=1.0):
    """Add background pixels touching a cell (4-neighbour); frac<1 keeps only the most cell-like ones."""
    lab = lab.copy()
    for _ in range(steps):
        fg = lab > 0
        ring = ndi.binary_dilation(fg, CROSS) & ~fg
        if prob is not None and thr is not None:
            ring &= prob > thr
        if frac < 1.0 and prob is not None:
            vals = prob[ring]
            if len(vals):
                ring &= prob >= np.quantile(vals, 1 - frac)
        near = ndi.grey_dilation(lab, footprint=CROSS)
        lab[ring] = near[ring]
    return lab


def shrink(lab):
    edge = ndi.grey_dilation(lab, footprint=CROSS) != ndi.grey_erosion(lab, footprint=CROSS)
    out = lab.copy()
    out[edge & (lab > 0)] = 0
    return out


VARIANTS = {
    "shrink1": lambda l, p: shrink(l),
    "none": lambda l, p: l,
    "grow_p25": lambda l, p: grow(l, prob=p, frac=0.25),
    "grow_p50": lambda l, p: grow(l, prob=p, frac=0.5),
    "grow1": lambda l, p: grow(l),
    "grow1_thr-1": lambda l, p: grow(l, prob=p, thr=-1.0),
    "grow1_thr-2": lambda l, p: grow(l, prob=p, thr=-2.0),
    "grow2": lambda l, p: grow(l, 2),
}
IMG_FRACS = (0.15, 0.25, 0.35, 0.5)
if os.environ.get("FINE"):
    VARIANTS = {"none": VARIANTS["none"]}
    VARIANTS.update({f"grow_p{int(f * 100)}": (lambda f: lambda l, p: grow(l, prob=p, frac=f))(f)
                     for f in (0.1, 0.15, 0.2, 0.3, 0.35, 0.4)})
    VARIANTS["grow_p25x2"] = lambda l, p: grow(grow(l, prob=p, frac=0.25), prob=p, frac=0.25)
    IMG_FRACS = ()

if __name__ == "__main__":
    modality = sys.argv[2] if len(sys.argv) > 2 else "exvivo"
    data = np.load(sys.argv[1])
    truth = load_truth()
    res = {}
    for sid, t in truth.items():
        subj = sid.split("__")[0][-6:]
        image = read_image(os.path.join(ROOT, "training", *sid.split("__"), f"{modality}.tif"))
        shape = image.shape
        gt, _ = rle_to_labels(t[f"{modality}_instances"], shape)
        smooth = ndi.gaussian_filter(image.astype(np.float32), 1.0)
        lab = data[f"{sid}|{modality}"].astype(np.int32)
        prob = data[f"{sid}|{modality}|prob"].astype(np.float32) if f"{sid}|{modality}|prob" in data.files else None
        for name, fn in VARIANTS.items():
            if prob is None and "_p" in name or prob is None and "thr" in name:
                continue
            res.setdefault(name, {}).setdefault(subj, []).append(pq_score(fn(lab, prob), gt)[0])
        for f in IMG_FRACS:
            res.setdefault(f"img_{f}", {}).setdefault(subj, []).append(pq_score(grow(lab, prob=smooth, frac=f), gt)[0])
    for name, per in res.items():
        allv = [x for v in per.values() for x in v]
        print(f"{name:12s} all {np.mean(allv):.4f}", {s: round(float(np.mean(v)), 3) for s, v in per.items()})
