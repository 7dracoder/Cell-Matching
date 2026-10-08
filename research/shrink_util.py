"""Tighter boundary for selected (paired) ex-vivo cells."""
import numpy as np
from scipy import ndimage as ndi

CROSS = ndi.generate_binary_structure(2, 1)


def shrink_pairs(base, grown, cp, paired, q=25):
    """paired: iterable of 1-based labels. Their pixels come from `base` minus low-cellprob inner ring."""
    out = grown.copy()
    objs = ndi.find_objects(base)
    for k in paired:
        sl = objs[k - 1]
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
        m = base[sl] == k
        inner = m & ~ndi.binary_erosion(m, CROSS)
        vals = cp[sl][inner]
        v = m.copy()
        if len(vals):
            v &= ~(inner & (cp[sl] <= np.quantile(vals, q / 100)))
        if v.sum() < 8:
            v = m
        o = out[sl]
        o[o == k] = 0
        o[v] = k
    return out
