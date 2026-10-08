"""Grow ex-vivo masks of a submission into the most cell-like boundary ring pixels; ids and pairs unchanged.

usage: grow_submission.py SRC.csv CP.npz FRAC [STEPS] OUT.csv
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import labels_to_rles, read_image, rle_to_labels  # noqa: E402
from common import ROOT  # noqa: E402
from size_lab import grow  # noqa: E402

csv.field_size_limit(sys.maxsize)
src, cp_path, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
steps, out = (int(sys.argv[4]), sys.argv[5]) if len(sys.argv) > 5 else (1, sys.argv[4])
cp = np.load(cp_path)
rows = list(csv.DictReader(open(src)))
for row in rows:
    sid = row["sample_id"]
    shape = read_image(os.path.join(ROOT, "hidden_test", *sid.split("__"), "exvivo.tif")).shape
    lab, ids = rle_to_labels(json.loads(row["exvivo_instances"]), shape)
    prob = cp[f"{sid}|cp"].astype(np.float32)
    assert prob.shape == shape, (prob.shape, shape)
    grown = lab
    for _ in range(steps):
        grown = grow(grown, prob=prob, frac=frac)
    rles = labels_to_rles(grown, "X")
    row["exvivo_instances"] = json.dumps({ids[int(k[2:]) - 1]: v for k, v in rles.items()})
    print(sid[-22:], "area", int((lab > 0).sum()), "->", int((grown > 0).sum()))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", out)
