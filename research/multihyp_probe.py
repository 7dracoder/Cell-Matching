"""Held-out PQ when each predicted cell is submitted with extra size hypotheses (overlapping masks).

Competition PQ: TP = GT cells with some prediction at IoU > 0.75 (each GT once), FP = all other
predictions, FN = unhit GT. Masks need not be disjoint for this definition. Variants per cell:
  g1  4-neighbour grow by 1 px (only into background)    s1  erode by 1 px
  g15 cellprob-ranked ring grow (top 15%) as in v7_grow15
Prints per-mouse and pooled PQ for several variant sets, plus how many GT become hit.
"""
import itertools
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi

from common import label_map, load_truth, regions
from size_lab import grow

CROSS = ndi.generate_binary_structure(2, 1)
Z = np.load("data/heldout_labels.npz")


def shrink(lab):
    out = lab.copy()
    edge = (ndi.grey_erosion(lab, footprint=CROSS) != lab) | (ndi.grey_dilation(lab, footprint=CROSS) != lab)
    out[edge] = 0
    # keep at least the core of tiny cells
    lost = np.setdiff1d(np.unique(lab), np.unique(out))
    for k in lost:
        if k:
            out[lab == k] = k
    return out


def iou_table(pred, gt):
    npd, ng = int(pred.max()), int(gt.max())
    J = np.bincount((pred.astype(np.int64) * (ng + 1) + gt).ravel(), minlength=(npd + 1) * (ng + 1)).reshape(npd + 1, ng + 1)
    pc, gc = J.sum(1), J.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        I = J[1:, 1:] / (pc[1:, None] + gc[None, 1:] - J[1:, 1:])
    return np.nan_to_num(I)


def pq_multi(tables, ngt):
    """tables: list of (npred_k, ngt) IoU arrays, one per variant set member."""
    npred = sum(t.shape[0] for t in tables)
    best = np.max(np.vstack([t.max(0, initial=0)[None] for t in tables]), axis=0) if ngt else np.zeros(0)
    hit = best > 0.75
    tp = int(hit.sum())
    fp, fn = npred - tp, ngt - tp
    den = tp + 0.5 * (fp + fn)
    return (best[hit].sum() / den if den else 1.0), tp, fp, fn


if __name__ == "__main__":
    truth = load_truth()
    SETS = [("base",), ("g15",), ("base", "g1"), ("base", "s1"), ("g15", "s1"), ("base", "g1", "s1"),
            ("g15", "g1"), ("g15", "g1", "s1"), ("s1", "base", "g1", "g2")]
    res = defaultdict(lambda: defaultdict(list))
    for mod in ("exvivo", "invivo"):
        for subj, reg, path in regions("training"):
            sid = f"{subj}__{reg}"
            base = Z[f"{sid}|{mod}"].astype(np.int32)
            gt, _ = label_map(truth[sid][f"{mod}_instances"], base.shape)
            prob = Z[f"{sid}|{mod}|prob"].astype(np.float32)
            V = {"base": base, "g1": grow(base), "s1": shrink(base), "g15": grow(base, prob=prob, frac=0.15)}
            V["g2"] = grow(V["g1"])
            T = {k: iou_table(v, gt) for k, v in V.items()}
            for S in SETS:
                res[mod][S].append((subj,) + pq_multi([T[k] for k in S], int(gt.max())))
        print(f"== {mod}")
        for S in SETS:
            rows = res[mod][S]
            per = {s: np.mean([r[1] for r in rows if r[0] == s]) for s in sorted({r[0] for r in rows})}
            pooled = np.mean([r[1] for r in rows])
            tp = sum(r[2] for r in rows)
            print(f"  {'+'.join(S):18s} PQ {pooled:.4f}  TP {tp:5d}  " + "  ".join(f"{k[-6:]} {v:.3f}" for k, v in per.items()))
