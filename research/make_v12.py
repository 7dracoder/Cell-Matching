"""v12 probe: opposite of v11. Paired ex-vivo cells get one extra cellprob-ranked ring grow (top FRAC);
unpaired cells and everything else stay exactly as v10 (0.48893).
usage: make_v12.py FRAC OUT.csv
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "..")
from cellmatch import labels_to_rles, read_image, rle_to_labels  # noqa: E402
from common import ROOT  # noqa: E402
from size_lab import grow  # noqa: E402

csv.field_size_limit(sys.maxsize)
FRAC, OUT = float(sys.argv[1]), sys.argv[2]
rows = list(csv.DictReader(open("../submission_v10_cpgate.csv")))
cp = np.load("data/test_cp_base.npz")
for row in rows:
    sid = row["sample_id"]
    pairs = json.loads(row["match_pairs"])
    if not pairs:
        continue
    shape = read_image(os.path.join(ROOT, "hidden_test", *sid.split("__"), "exvivo.tif")).shape
    lab, ids = rle_to_labels(json.loads(row["exvivo_instances"]), shape)
    index = {k: n for n, k in enumerate(ids, 1)}
    paired = np.zeros(len(ids) + 1, bool)
    paired[[index[e] for _, e in pairs]] = True
    g = grow(lab, prob=cp[f"{sid}|cp"].astype(np.float32), frac=FRAC)
    add = (lab == 0) & (g > 0) & paired[g]
    out = lab.copy()
    out[add] = g[add]
    rles = labels_to_rles(out, "X")
    assert len(rles) == len(ids)
    row["exvivo_instances"] = json.dumps({ids[int(k[2:]) - 1]: v for k, v in rles.items()})
    print(sid[-22:], "paired", len(pairs), "added px", int(add.sum()))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", OUT)
