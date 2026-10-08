"""Exact held-out score of the v11 mask rule on real label maps.

Unpaired ex cells: cellprob-ranked 15% ring grow (as v7_grow15). Paired ex cells: base mask minus
the lowest-cellprob Q% of its inner boundary pixels (shrink_pairs). Sweeps the pair threshold and gate.
"""
import pickle
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "..")
import cp_pose_lab as L  # noqa: E402
import pair_clf  # noqa: E402
from cellmatch import pq_score  # noqa: E402
from common import label_map, load_truth  # noqa: E402
from lab_build import gt_link  # noqa: E402
from margin_lab import C, R, margin  # noqa: E402
from size_lab import grow  # noqa: E402
from shrink_util import shrink_pairs  # noqa: E402

CROSS = ndi.generate_binary_structure(2, 1)


if __name__ == "__main__":
    Q = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    Z = np.load("data/heldout_labels.npz")
    T = load_truth()
    S = pickle.load(open("data/cp_pose_train.pkl", "rb"))
    regs, info = {}, {}
    for s in R:
        ch = L.pose_choose(S[s], "score")
        regs[s] = (ch[0], ch[1])
        info[s] = (margin(C[s], ch[0], ch[1], R[s]), ch[2])
    rows = pair_clf.dataset(regs)
    probs = pair_clf.loo_predict(rows)
    cache = {}
    for s in R:
        base = Z[f"{s}|exvivo"].astype(np.int32)
        cp = Z[f"{s}|exvivo|prob"].astype(np.float32)
        gt, _ = label_map(T[s]["exvivo_instances"], base.shape)
        cache[s] = (base, grow(base, prob=cp, frac=0.15), cp, gt)
    for gate_name, gate in [("mg3|z5", lambda m, z: m >= 3 or z >= 5), ("mg3|z4.25", lambda m, z: m >= 3 or z >= 4.25)]:
        for th in (0.025, 0.05, 0.1):
            for shrink in (False, True):
                tp = pred = 0
                pqs = []
                for s, pairs, X, y in rows:
                    base, grown, cp, gt = cache[s]
                    keep = probs[s] >= th if gate(*info[s]) else np.zeros(len(pairs), bool)
                    sel = pairs[keep] if len(pairs) else pairs
                    lab = shrink_pairs(base, grown, cp, [j + 1 for _, j in sel], Q) if shrink else grown
                    pqs.append(pq_score(lab, gt)[0])
                    link = gt_link(lab, gt)
                    r = R[s]
                    for i, j in sel:
                        pred += 1
                        tp += r["iv_link"][i] >= 0 and link[j] >= 0 and (r["iv_link"][i], link[j]) in r["gt_pairs"]
                f1 = 2 * tp / (pred + pair_clf.TOTAL)
                pq = float(np.mean(pqs))
                print(f"{gate_name:9s} thr {th:.3f} shrink {str(shrink):5s} exPQ {pq:.4f} F1 {f1:.4f} tp {tp} pred {pred} "
                      f"full {0.25 * (0.7404 + pq) + 0.5 * f1:.4f}", flush=True)
