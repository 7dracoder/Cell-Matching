"""How many pairs of submission A reappear in B (both masks IoU > 0.75 with A's masks)."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import read_image, rle_to_labels  # noqa: E402
from common import ROOT  # noqa: E402
from lab_build import gt_link  # noqa: E402

csv.field_size_limit(sys.maxsize)
A = {r["sample_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
B = {r["sample_id"]: r for r in csv.DictReader(open(sys.argv[2]))}
ta = tb = same = 0
for sid in A:
    pa, pb = json.loads(A[sid]["match_pairs"]), json.loads(B[sid]["match_pairs"])
    if not pa and not pb:
        continue
    path = os.path.join(ROOT, "hidden_test", *sid.split("__"))
    link = {}
    for m in ("invivo", "exvivo"):
        shape = read_image(f"{path}/{m}.tif").shape
        la, ia = rle_to_labels(json.loads(A[sid][f"{m}_instances"]), shape)
        lb, ib = rle_to_labels(json.loads(B[sid][f"{m}_instances"]), shape)
        l = gt_link(lb, la)
        link[m] = {ib[k]: ia[v] for k, v in enumerate(l) if v >= 0}
    sa = {tuple(p) for p in pa}
    mapped = {(link["invivo"].get(a), link["exvivo"].get(b)) for a, b in pb}
    n = len(sa & mapped)
    ta, tb, same = ta + len(pa), tb + len(pb), same + n
    print(sid[-22:], "A", len(pa), "B", len(pb), "shared", n)
print("total A", ta, "B", tb, "shared", same)
