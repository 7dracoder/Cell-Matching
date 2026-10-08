"""Conservative cross-modal image-evidence add-on for the current best CSV.

The existing v7_grow15 masks and matches are left untouched. We only add
high-probability pairs to regions rejected by the original registration gate
when a distinct 2–8 px band-pass channel strongly favors one affine pose.
The independent source idea is adapted from the supplied constellation code;
all thresholds were selected on held-out training mice, not hidden labels.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_apply as T  # noqa: E402
from bandpass_registration_probe import pose_score  # noqa: E402
from registration import transform  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "submission_v7_grow15.csv"
OUT = HERE.parent / "submission_bandpass_candidate.csv"
IMAGE_GAP = 0.03
PAIR_THRESHOLD = 0.10


def choice(rec, original, candidates, iv, ex):
    M, score, _ = original
    old_margin = T.margin(candidates, M, score, rec)
    if old_margin >= 3:
        return None, old_margin, 0.0
    poses = [("previous", M, score)] + [(f"vote_{j}", item[3], float(item[0]))
                                          for j, item in enumerate(candidates)]
    ranked = []
    for source, matrix, votes in poses:
        if matrix is None:
            continue
        value = pose_score(iv, ex, matrix, 2, 8)
        if np.isfinite(value):
            ranked.append((value, source, matrix, votes))
    if not ranked:
        return None, old_margin, 0.0
    ranked.sort(key=lambda row: row[0], reverse=True)
    best = ranked[0]
    h, w = rec["iv_shape"]
    anchors = np.array([[0, 0], [w / 2, h / 2], [w, h]], dtype=np.float32)
    points = transform(anchors, best[2])
    alternate = next((row for row in ranked[1:]
                      if np.max(np.linalg.norm(transform(anchors, row[2]) - points,
                                               axis=1)) > 50), None)
    gap = best[0] - alternate[0] if alternate else 0.0
    return (best if gap >= IMAGE_GAP else None), old_margin, gap


def main():
    csv.field_size_limit(sys.maxsize)
    with SRC.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with Pool(8) as pool:
        recs = dict(pool.map(T.build, rows))
    offsets, _ = T.crop_offsets("hidden_test")
    for sid, rec in recs.items():
        rec["offset"] = offsets[tuple(sid.split("__"))]
    with Pool(8) as pool:
        candidates = dict(pool.map(T.cands, list(recs.items())))
    groups = {}
    for sid, rec in recs.items():
        groups.setdefault((rec["subject"], rec["ex_shape"]), []).append(sid)
    jobs = []
    for sids in groups.values():
        seen, voters = set(), {}
        for sid in sids:
            if recs[sid]["key"] not in seen:
                seen.add(recs[sid]["key"])
                voters[sid] = candidates[sid]
        modes = T.vote_modes(voters)
        jobs += [(sid, recs[sid], modes) for sid in sids]
    with Pool(8) as pool:
        original = dict(pool.map(T.window, jobs))
    os.environ["REG_TRAIN"] = str(HERE / "data/reg_window_vote5.pkl")
    T.REG_TRAIN = os.environ["REG_TRAIN"]
    classifier = T.train_classifier()
    total_added, regions = 0, 0
    for row in rows:
        sid = row["sample_id"]
        subject, region = sid.split("__")
        path = Path(T.ROOT) / "hidden_test" / subject / region
        iv = cv2.imread(str(path / "invivo.tif"), cv2.IMREAD_UNCHANGED)
        ex = cv2.imread(str(path / "exvivo.tif"), cv2.IMREAD_UNCHANGED)
        picked, old_margin, image_gap = choice(recs[sid], original[sid],
                                               candidates[sid], iv, ex)
        if picked is None:
            print("SKIP", sid, "vote_margin", round(old_margin, 2),
                  "image_gap", round(image_gap, 4), flush=True)
            continue
        _, source, matrix, score = picked
        pairs, features = T.pair_clf.candidates(recs[sid], matrix, score)
        probabilities = (classifier.predict_proba(features)[:, 1]
                         if len(features) else np.zeros(0, dtype=float))
        keep = probabilities >= PAIR_THRESHOLD
        previous = json.loads(row["match_pairs"])
        iv_used = {pair[0] for pair in previous}
        ex_used = {pair[1] for pair in previous}
        new = []
        for (i, j), accepted in zip(pairs, keep):
            if not accepted:
                continue
            iv_id, ex_id = recs[sid]["iv_ids"][i], recs[sid]["ex_ids"][j]
            if iv_id not in iv_used and ex_id not in ex_used:
                new.append([iv_id, ex_id])
                iv_used.add(iv_id)
                ex_used.add(ex_id)
        row["match_pairs"] = json.dumps(previous + new)
        regions += 1
        total_added += len(new)
        print("ADD", sid, source, "image_gap", round(image_gap, 4),
              "pairs", len(new), "max_pair_probability",
              round(float(probabilities.max()), 3) if len(probabilities) else 0.0,
              flush=True)
    if total_added == 0:
        print("NO_CANDIDATE: no high-confidence new pairs", flush=True)
        return
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("CANDIDATE", OUT, "regions", regions, "added_pairs", total_added,
          "public score unverified", flush=True)


if __name__ == "__main__":
    main()
