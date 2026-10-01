"""Cross-validated test of subject-level orientation consensus."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from registration import match_points, register_with_priors


def main():
    rows = json.loads(Path("features_cv.json").read_text())
    scores = defaultdict(list)
    for row in rows:
        subject = row["sample_id"].split("__")[0]
        ci = np.asarray(row["invivo_centers"], np.float32)
        ce = np.asarray(row["exvivo_centers"], np.float32)
        keep = np.flatnonzero(np.asarray(row["exvivo_brightness"]) >= 0.65)
        true = {tuple(p) for p in row["truth_pairs"]}
        modes = []
        for mode in (0, 1):
            matrix, score, _ = register_with_priors(ci, ce, mode=mode)
            metrics = []
            for distance in (4, 5):
                pairs = match_points(ci, ce[keep], matrix, distance)
                tp = sum((row["invivo_gt"][i], row["exvivo_gt"][keep[j]]) in true
                         for i, j, _ in pairs)
                metrics.append(np.array([tp, len(pairs), len(true)]))
            modes.append((score, metrics))
        scores[subject].append(modes)
    for subject, records in scores.items():
        print(subject, "best mode counts", [sum(r[0][0] > r[1][0] for r in records),
                                            sum(r[1][0] > r[0][0] for r in records)])
        for method in ("best", "force0", "force1"):
            for idx, distance in enumerate((4, 5)):
                total = np.zeros(3, int)
                for modes in records:
                    chosen = max(range(2), key=lambda k: modes[k][0]) if method == "best" else (
                        0 if method == "force0" else 1)
                    total += modes[chosen][1][idx]
                print(method, distance, total.tolist(), round(2 * total[0] / (total[1] + total[2]), 4))


if __name__ == "__main__":
    main()
