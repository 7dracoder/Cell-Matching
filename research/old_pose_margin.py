"""Margin of the old submission's implied pose for the regions where old and new disagree."""
import os, sys, csv, json
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine  # noqa: E402
from common import crop_offsets  # noqa: E402
from test_apply import build, cands  # noqa: E402
from margin_lab import margin  # noqa: E402
from window_lab import pose  # noqa: E402

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(__file__)
rows = {r["sample_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "submission.csv")))}
offsets, _ = crop_offsets("hidden_test")
for reg in ("subject_78b6a7__region_7c48e3", "subject_d7a97c__region_1b3772", "subject_d7a97c__region_dd13e1"):
    sid, rec = build(rows[reg])
    rec["offset"] = offsets[tuple(sid.split("__"))]
    _, C = cands((sid, rec))
    ii = {k: n for n, k in enumerate(rec["iv_ids"])}
    ee = {k: n for n, k in enumerate(rec["ex_ids"])}
    pairs = json.loads(rows[reg]["match_pairs"])
    M0, _ = cv2.estimateAffine2D(rec["iv_c"][[ii[a] for a, _ in pairs]], rec["ex_c"][[ee[b] for _, b in pairs]],
                                 method=cv2.RANSAC, ransacReprojThreshold=6)
    M, score = refine(rec["iv_c"], rec["ex_c"], M0)
    a, l = pose(rec, M)
    print(f"{reg[-22:]} old pose angle {a:.1f} landing {l.round()} score {score:.1f} margin {margin(C, M, score, rec):.1f}")
