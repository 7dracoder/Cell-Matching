"""Pair F1 under GT registration and reachable GT pairs for the lab.pkl given by argv[1]."""
import pickle
import sys

import pair_clf

pair_clf.R = R = pickle.load(open(sys.argv[1], "rb"))
pair_clf.TOTAL = sum(r["n_gt_pairs"] for r in R.values())

def linked(link):
    return set(link.values() if isinstance(link, dict) else list(link))


reach = sum(sum(a in linked(r["iv_link"]) and b in linked(r["ex_link"]) for a, b in r["gt_pairs"])
            for r in R.values())
print(sys.argv[1], "gt pairs", pair_clf.TOTAL, "reachable", reach,
      "iv cells", sum(len(r["iv_c"]) for r in R.values()), "ex cells", sum(len(r["ex_c"]) for r in R.values()))
pair_clf.report("GT registration", {s: (r["gt_M"], 100.0) for s, r in R.items()})
