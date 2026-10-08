"""Probe unusual geometry (flip, large rotation, scale) for one test mouse."""
import os, sys, csv, json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import rle_to_labels, region_centers, read_image  # noqa: E402
from common import ROOT  # noqa: E402
from hough import hough_candidates  # noqa: E402
from registration import refine  # noqa: E402

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(__file__)
SUBJ = sys.argv[1] if len(sys.argv) > 1 else "subject_78b6a7"
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "submission.csv"))) if r["sample_id"].startswith(SUBJ)]


def centers(row):
    path = os.path.join(ROOT, "hidden_test", *row["sample_id"].split("__"))
    out = []
    for m, key in (("invivo", "invivo_instances"), ("exvivo", "exvivo_instances")):
        shape = read_image(f"{path}/{m}.tif").shape
        out.append(region_centers(rle_to_labels(json.loads(row[key]), shape)[0]))
    return out


def probe(args):
    row, flip = args
    iv, ex = centers(row)
    if flip:
        iv = iv * np.array([-1, 1])
    best = []
    for scale_set in (np.arange(0.6, 0.86, 0.04), np.arange(0.86, 1.2, 0.03), np.arange(1.2, 1.65, 0.05)):
        c = hough_candidates(iv, ex, angles=np.arange(-180, 180, 1.5), scales=scale_set, per_pose=1, keep=10)
        for votes, angle, scale, M in c:
            R, score = refine(iv, ex, M)
            best.append((score, votes, angle, round(float(scale), 2)))
    best.sort(reverse=True)
    return row["sample_id"][-6:], flip, best[:3]


if __name__ == "__main__":
    with Pool(8) as pool:
        for sid, flip, best in pool.map(probe, [(r, f) for r in rows[:8] for f in (False, True)]):
            print(sid, "flip" if flip else "    ", [(round(s, 1), v, a, sc) for s, v, a, sc in best])
