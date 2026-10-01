"""Analyze cross-validated registration hypotheses and independent masks."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cellmatch import ROOT, read_image, region_centers, rle_to_labels
from registration import match_points, register_with_priors, transform


def true_matrix(sample_id: str, row) -> np.ndarray | None:
    path = ROOT / "training" / sample_id.replace("__", "/")
    iv, iv_ids = rle_to_labels(json.loads(row.invivo_instances), read_image(path / "invivo.tif").shape)
    ex, ex_ids = rle_to_labels(json.loads(row.exvivo_instances), read_image(path / "exvivo.tif").shape)
    iv_centers, ex_centers = region_centers(iv), region_centers(ex)
    iv_lookup, ex_lookup = {x: i for i, x in enumerate(iv_ids)}, {x: i for i, x in enumerate(ex_ids)}
    pairs = [(iv_lookup[a], ex_lookup[b]) for a, b in json.loads(row.match_pairs)
             if a in iv_lookup and b in ex_lookup]
    if len(pairs) < 3:
        return None
    matrix, _ = cv2.estimateAffine2D(
        iv_centers[[i for i, j in pairs]].astype(np.float32),
        ex_centers[[j for i, j in pairs]].astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=5)
    return matrix


def main() -> None:
    records = json.loads(Path("features_cv.json").read_text())
    frame = pd.read_csv(ROOT / "training/train_ground_truth.csv").set_index("sample_id")
    output = []
    for record in records:
        ci, ce = np.asarray(record["invivo_centers"], np.float32), np.asarray(record["exvivo_centers"], np.float32)
        brightness = np.asarray(record["exvivo_brightness"])
        keep = np.flatnonzero(brightness >= 0.65)
        ce_filtered = ce[keep]
        true = true_matrix(record["sample_id"], frame.loc[record["sample_id"]])
        sample = np.vstack((ci.mean(axis=0), np.quantile(ci, .2, axis=0), np.quantile(ci, .8, axis=0)))
        truth_pairs = {tuple(pair) for pair in record["truth_pairs"]}
        candidates = []
        for mode in (0, 1):
            all_results = register_with_priors(ci, ce_filtered, mode=mode, return_candidates=True)
            for matrix, score, mode_index in all_results:
                pairs = match_points(ci, ce_filtered, matrix, 4)
                predicted = [(record["invivo_gt"][i], record["exvivo_gt"][keep[j]])
                             for i, j, _ in pairs]
                tp = sum(pair in truth_pairs for pair in predicted)
                error = float(np.mean(np.linalg.norm(transform(sample, matrix) - transform(sample, true), axis=1))) \
                    if true is not None else float("nan")
                candidates.append({"score": score, "mode": mode_index, "matrix": matrix.tolist(),
                                   "tp": tp, "pred": len(pairs), "error": error})
        item = {"sample_id": record["sample_id"], "exvivo_shape": record["exvivo_shape"],
                "invivo_shape": record["invivo_shape"], "eligible": sum(
                    a in record["invivo_gt"] and b in record["exvivo_gt"] for a, b in truth_pairs),
                "truth_count": len(truth_pairs), "true_matrix": true.tolist() if true is not None else None,
                "bright_center": np.median(ce_filtered, axis=0).tolist(),
                "iv_center": np.median(ci, axis=0).tolist(), "candidates": candidates}
        output.append(item)
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
        best_good = min((k for k, c in enumerate(ranked) if c["error"] < 20), default=-1)
        error_text = round(ranked[0]["error"]) if np.isfinite(ranked[0]["error"]) else None
        print(record["sample_id"], "best", round(ranked[0]["score"]), error_text,
              ranked[0]["tp"], "good_rank", best_good, flush=True)
    Path("candidates_cv.json").write_text(json.dumps(output))
    print("saved candidates_cv.json", len(output), flush=True)


if __name__ == "__main__":
    main()
