"""v11: v10 pairs, with each *paired* ex-vivo cell given a tighter boundary.

Unpaired ex cells keep the v7_grow15 masks. Paired ex cells use the ungrown v7 mask minus the
lowest-cellprob Q% of its inner boundary pixels (paired_shrink_exact.shrink_pairs).
usage: make_v11.py Q OUT.csv
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "..")
from cellmatch import labels_to_rles, read_image, rle_to_labels  # noqa: E402
from common import ROOT  # noqa: E402
from shrink_util import shrink_pairs  # noqa: E402

csv.field_size_limit(sys.maxsize)
Q, OUT = int(sys.argv[1]), sys.argv[2]
v10 = list(csv.DictReader(open("../submission_v10_cpgate.csv")))
base = {r["sample_id"]: r for r in csv.DictReader(open("data/submission.csv"))}
cp = np.load("data/test_cp_base.npz")
changed = 0
for row in v10:
    sid = row["sample_id"]
    pairs = json.loads(row["match_pairs"])
    if not pairs:
        continue
    shape = read_image(os.path.join(ROOT, "hidden_test", *sid.split("__"), "exvivo.tif")).shape
    grown, ids = rle_to_labels(json.loads(row["exvivo_instances"]), shape)
    b, bids = rle_to_labels(json.loads(base[sid]["exvivo_instances"]), shape)
    assert ids == bids
    index = {k: n for n, k in enumerate(ids, 1)}
    lab = shrink_pairs(b, grown, cp[f"{sid}|cp"].astype(np.float32), [index[e] for _, e in pairs], Q)
    rles = labels_to_rles(lab, "X")
    assert len(rles) == len(ids)
    row["exvivo_instances"] = json.dumps({ids[int(k[2:]) - 1]: v for k, v in rles.items()})
    changed += len(pairs)
    print(sid[-22:], "paired", len(pairs), "area", int((grown > 0).sum()), "->", int((lab > 0).sum()))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(v10[0].keys()))
    w.writeheader()
    w.writerows(v10)
print("saved", OUT, "paired cells reshaped", changed)
