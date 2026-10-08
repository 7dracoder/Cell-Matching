"""Registration correctness and matching F1 on held-out predictions."""
import os, sys, pickle
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import match_points, transform  # noqa: E402

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
TOTAL = sum(r["n_gt_pairs"] for r in R.values())


def reg_error(r, M):
    if M is None or r["gt_M"] is None:
        return np.inf
    return float(np.median(np.linalg.norm(transform(r["gt_iv_c"], M) - transform(r["gt_iv_c"], r["gt_M"]), axis=1)))


def baseline_pairs(r, M, distance=8, min_b=0.45, ratio=2.5):
    if M is None:
        return []
    keep = np.flatnonzero(r["ex_f"]["mean"] >= min_b)
    if len(keep) < 3:
        keep = np.arange(len(r["ex_c"]))
    pairs = match_points(r["iv_c"], r["ex_c"][keep], M, distance, "greedy")
    if not pairs:
        return []
    second = cKDTree(r["ex_c"][keep]).query(transform(r["iv_c"], M), k=2)[0][:, 1]
    return [(i, int(keep[j])) for i, j, d in pairs if second[i] >= ratio * max(d, 0.5)]


def score(pairs_by_region):
    tp = pred = 0
    for sid, pairs in pairs_by_region.items():
        r = R[sid]
        tp += sum((r["iv_link"][i], r["ex_link"][j]) in r["gt_pairs"] for i, j in pairs)
        pred += len(pairs)
    return tp, pred, 2 * tp / (pred + TOTAL)


if __name__ == "__main__":
    eligible = sum(sum(a in set(r["iv_link"]) and b in set(r["ex_link"]) for a, b in r["gt_pairs"]) for r in R.values())
    print("total GT pairs", TOTAL, "eligible", eligible)
    for subj in sorted({r["subject"] for r in R.values()}):
        items = [r for r in R.values() if r["subject"] == subj]
        for m in ("iv", "ex"):
            tp = sum((r[f"{m}_link"] >= 0).sum() for r in items)
            npred = sum(len(r[f"{m}_link"]) for r in items)
            ngt = sum(len(r["gt_iv_c"]) if m == "iv" else 0 for r in items)
            print(f"  {subj} {m}: pred {npred} tp {tp}")
    errs = {sid: reg_error(r, r["ncc"][0]) for sid, r in R.items()}
    ok = {sid for sid, e in errs.items() if e < 5}
    print("ncc registration correct (<5px):", len(ok), "/", len(R))
    for sid, e in errs.items():
        print(f"  {sid[-22:]} err {e:8.1f} score {R[sid]['ncc'][1]:.1f} pairsGT {len(R[sid]['gt_pairs'])}")
    base = {sid: baseline_pairs(r, r["ncc"][0]) for sid, r in R.items()}
    print("baseline (no gate):", score(base))
    print("baseline, correct regions only:", score({s: p for s, p in base.items() if s in ok}))
    oracle = {sid: baseline_pairs(r, r["gt_M"]) for sid, r in R.items()}
    print("oracle registration, baseline matching:", score(oracle))
