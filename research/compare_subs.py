"""Per-region comparison of old vs re-matched submission: pair overlap and implied poses."""
import os, sys, csv, json
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import rle_to_labels, region_centers, read_image  # noqa: E402
from common import crop_offsets, ROOT  # noqa: E402

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(__file__)
old = {r["sample_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "submission.csv")))}
new = {r["sample_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "submission_rematch.csv")))}
offsets, _ = crop_offsets("hidden_test")
P0 = np.array([300.0, 300.0])


def implied_pose(sid, pairs):
    if len(pairs) < 4:
        return None
    path = os.path.join(ROOT, "hidden_test", *sid.split("__"))
    iv = json.loads(old[sid]["invivo_instances"])
    ex = json.loads(old[sid]["exvivo_instances"])
    ivl, ivk = rle_to_labels(iv, read_image(path + "/invivo.tif").shape)
    exl, exk = rle_to_labels(ex, read_image(path + "/exvivo.tif").shape)
    ivc, exc = region_centers(ivl), region_centers(exl)
    ii = {k: n for n, k in enumerate(ivk)}
    ee = {k: n for n, k in enumerate(exk)}
    A = ivc[[ii[a] for a, _ in pairs]]
    B = exc[[ee[b] for _, b in pairs]]
    M, _ = cv2.estimateAffine2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=6)
    ang = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    land = M[:, :2] @ (P0 - offsets[tuple(sid.split("__"))]) + M[:, 2]
    return round(float(ang), 1), land.round()


for sid in old:
    po, pn = json.loads(old[sid]["match_pairs"]), json.loads(new[sid]["match_pairs"])
    common = len({tuple(p) for p in po} & {tuple(p) for p in pn})
    print(f"{sid[-22:]} old {len(po):3d} new {len(pn):3d} common {common:3d} | old pose {implied_pose(sid, po)} | new pose {implied_pose(sid, pn)}")
