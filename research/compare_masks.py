"""Per-region agreement (PQ of B against A as reference) and cell counts for two submission CSVs."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import pq_score, read_image, rle_to_labels  # noqa: E402
from common import ROOT  # noqa: E402

csv.field_size_limit(sys.maxsize)
A = {r["sample_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
B = {r["sample_id"]: r for r in csv.DictReader(open(sys.argv[2]))}
tot = {m: [] for m in ("invivo", "exvivo")}
for sid in A:
    path = os.path.join(ROOT, "hidden_test", *sid.split("__"))
    out = []
    for m in ("invivo", "exvivo"):
        shape = read_image(f"{path}/{m}.tif").shape
        a, _ = rle_to_labels(json.loads(A[sid][f"{m}_instances"]), shape)
        b, _ = rle_to_labels(json.loads(B[sid][f"{m}_instances"]), shape)
        pq, tp, fp, fn = pq_score(b, a)
        tot[m].append(pq)
        out.append(f"{m[:2]} n {int(a.max()):4d}->{int(b.max()):4d} agreePQ {pq:.3f} shape {shape}")
    print(sid[-22:], " | ".join(out))
print({m: round(float(np.mean(v)), 3) for m, v in tot.items()})
