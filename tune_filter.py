"""Cross-validated brightness and distance sweep for geometric matches."""

import json
from pathlib import Path

import numpy as np

from registration import match_points, register_with_priors


def main():
    records = json.loads(Path("features_cv.json").read_text())
    pooled = {}
    for record in records:
        ci = np.asarray(record["invivo_centers"], np.float32)
        ce = np.asarray(record["exvivo_centers"], np.float32)
        brightness = np.asarray(record["exvivo_brightness"])
        matrix, score, mode = max((register_with_priors(ci, ce, mode=k) for k in (0, 1)),
                                  key=lambda result: result[1])
        truth = {tuple(pair) for pair in record["truth_pairs"]}
        subject = record["sample_id"].split("__")[0]
        for minimum in (0.0, 0.45, 0.55, 0.65, 0.75):
            keep = np.flatnonzero(brightness >= minimum)
            if len(keep) < 3:
                continue
            for distance in (2, 3, 4, 5, 6):
                pairs = match_points(ci, ce[keep], matrix, distance)
                tp = sum((record["invivo_gt"][i], record["exvivo_gt"][keep[j]]) in truth
                         for i, j, _ in pairs)
                key = (subject, minimum, distance)
                count = pooled.setdefault(key, np.zeros(3, np.int64))
                count += [tp, len(pairs), len(truth)]
        print("registered", record["sample_id"], mode, round(score), flush=True)
    for subject in sorted({key[0] for key in pooled} | {"ALL"}):
        print("SUBJECT", subject)
        ranking = []
        for minimum in (0.0, 0.45, 0.55, 0.65, 0.75):
            for distance in (2, 3, 4, 5, 6):
                count = (sum((pooled[(s, minimum, distance)] for s in
                              sorted({key[0] for key in pooled})), start=np.zeros(3, np.int64))
                         if subject == "ALL" else pooled[(subject, minimum, distance)])
                f1 = 2 * count[0] / (count[1] + count[2])
                ranking.append((f1, minimum, distance, count.tolist()))
        for entry in sorted(ranking, reverse=True)[:10]:
            print(entry)


if __name__ == "__main__":
    main()
