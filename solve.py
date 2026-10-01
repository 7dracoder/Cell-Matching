"""Validate on held-out mice or generate the 29-row Kaggle submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2

from cellmatch import (ROOT, labels_to_rles, normalize, pq_score, read_image, region_centers,
                       rle_to_labels, write_submission)
from learned import load_model, predict_probabilities, probabilities_to_labels
from registration import match_points, register_with_priors


def predicted_to_ground_truth(pred: np.ndarray, truth: np.ndarray) -> dict[int, int]:
    npred, ngt = int(pred.max()), int(truth.max())
    if not npred or not ngt:
        return {}
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + truth).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc = joint.sum(axis=1)
    gc = joint.sum(axis=0)
    rows, cols = np.nonzero(joint[1:, 1:])
    rows += 1
    cols += 1
    iou = joint[rows, cols] / (pc[rows] + gc[cols] - joint[rows, cols])
    return {int(p): int(g) for p, g, score in zip(rows, cols, iou) if score > 0.75}


def predict_one(path: Path, iv_model, ex_model, device: str, tta: bool,
                iv_threshold: float, ex_threshold: float, ex_min_intensity: float) -> dict:
    images = {}
    labels = {}
    centers = {}
    for modality, models, threshold in (("invivo", iv_model, iv_threshold),
                                        ("exvivo", ex_model, ex_threshold)):
        image = read_image(path / f"{modality}.tif")
        probs = np.mean([predict_probabilities(model, image, modality, device=device, tta=tta)
                         for model in models], axis=0)
        instances = probabilities_to_labels(probs, modality, threshold, 0.5)
        images[modality] = image
        labels[modality] = instances
        centers[modality] = region_centers(instances)
    ex_labels = labels["exvivo"]
    ex_brightness = normalize(images["exvivo"], "exvivo")[0]
    ex_sizes = np.bincount(ex_labels.ravel())
    ex_means = np.bincount(ex_labels.ravel(), weights=ex_brightness.ravel()) / np.maximum(ex_sizes, 1)
    ex_indices = np.flatnonzero(ex_means[1:] >= ex_min_intensity)
    if len(ex_indices) < 3:
        ex_indices = np.arange(len(centers["exvivo"]))
    ex_registration_centers = centers["exvivo"][ex_indices]
    # Use every detected cell for geometric registration.  Verified pairs are
    # brightness-biased in ex-vivo images, so restrict the final pair list only.
    positive = register_with_priors(centers["invivo"], centers["exvivo"], mode=0)
    negative = register_with_priors(centers["invivo"], centers["exvivo"], mode=1)
    return {"images": images, "labels": labels, "centers": centers,
            "registration": (positive, negative), "ex_registration_centers": ex_registration_centers,
            "ex_registration_indices": ex_indices}


def matched_indices(item: dict, matrix: np.ndarray | None, distance: float):
    if matrix is None:
        return []
    pairs = match_points(item["centers"]["invivo"], item["ex_registration_centers"], matrix, distance)
    return [(i, int(item["ex_registration_indices"][j]), d) for i, j, d in pairs]


def subject_mode(items: list[dict]) -> int | None:
    # Square ex-vivo canvases share a stable orientation across training regions.
    # Rectangular canvases contain mixed orientations, so select per region.
    square_fraction = np.mean([
        abs(np.log(item["images"]["exvivo"].shape[0] / item["images"]["exvivo"].shape[1])) < 0.1
        for item in items
    ])
    if square_fraction < 0.65:
        return None
    winners = np.asarray([item["registration"][0][1] > item["registration"][1][1]
                          for item in items])
    if np.mean(winners) >= 0.65:
        return 0
    if np.mean(~winners) >= 0.65:
        return 1
    return None


def select_registration(item: dict, mode: int | None) -> tuple[np.ndarray | None, float, int]:
    candidates = item["registration"]
    return candidates[mode] if mode is not None else max(candidates, key=lambda x: x[1])


def validate(frame: pd.DataFrame, records: dict, threshold_list=(4, 5, 6, 7, 8)):
    pq = {"invivo": [], "exvivo": []}
    counts = {distance: [0, 0, 0] for distance in threshold_list}
    eligible = 0
    oracle = [0, 0]
    for subject, group in frame.groupby(frame.sample_id.str.split("__").str[0], sort=False):
        mode = subject_mode([records[row.sample_id] for row in group.itertuples(index=False)])
        print(f"{subject} subject-level mode: {mode}", flush=True)
        for row in group.itertuples(index=False):
            item = records[row.sample_id]
            matrix, score, chosen_mode = select_registration(item, mode)
            mappings = {}
            gt_ids = {}
            for modality in ("invivo", "exvivo"):
                ground_truth, ids = rle_to_labels(json.loads(getattr(row, f"{modality}_instances")),
                                                  item["images"][modality].shape)
                pred = item["labels"][modality]
                pq[modality].append(pq_score(pred, ground_truth)[0])
                mappings[modality] = predicted_to_ground_truth(pred, ground_truth)
                gt_ids[modality] = ids
            truth_pairs = {tuple(pair) for pair in json.loads(row.match_pairs)}
            iv_reverse = {gt_ids["invivo"][gt - 1]: pred for pred, gt in mappings["invivo"].items()}
            ex_reverse = {gt_ids["exvivo"][gt - 1]: pred for pred, gt in mappings["exvivo"].items()}
            eligible += sum(a in iv_reverse and b in ex_reverse for a, b in truth_pairs)
            # Diagnostic upper bound: fit from labels, then apply to predicted cells.
            iv_truth, _ = rle_to_labels(json.loads(row.invivo_instances),
                                         item["images"]["invivo"].shape)
            ex_truth, _ = rle_to_labels(json.loads(row.exvivo_instances),
                                         item["images"]["exvivo"].shape)
            iv_truth_centers = region_centers(iv_truth)
            ex_truth_centers = region_centers(ex_truth)
            iv_lookup = {cell_id: i for i, cell_id in enumerate(gt_ids["invivo"])}
            ex_lookup = {cell_id: i for i, cell_id in enumerate(gt_ids["exvivo"])}
            known = [(iv_lookup[a], ex_lookup[b]) for a, b in truth_pairs
                     if a in iv_lookup and b in ex_lookup]
            if len(known) >= 3:
                fitted, _ = cv2.estimateAffine2D(
                    iv_truth_centers[[i for i, j in known]].astype(np.float32),
                    ex_truth_centers[[j for i, j in known]].astype(np.float32),
                    method=cv2.RANSAC, ransacReprojThreshold=5)
                if fitted is not None:
                    oracle_pairs = matched_indices(item, fitted, 4)
                    oracle[1] += len(oracle_pairs)
                    oracle[0] += sum(
                        (gt_ids["invivo"][mappings["invivo"].get(i + 1, 0) - 1],
                         gt_ids["exvivo"][mappings["exvivo"].get(j + 1, 0) - 1]) in truth_pairs
                        for i, j, _ in oracle_pairs
                        if mappings["invivo"].get(i + 1) and mappings["exvivo"].get(j + 1)
                    )
            for distance in threshold_list:
                pairs = matched_indices(item, matrix, distance)
                correct = 0
                for iv_index, ex_index, _ in pairs:
                    iv_gt = mappings["invivo"].get(iv_index + 1)
                    ex_gt = mappings["exvivo"].get(ex_index + 1)
                    if iv_gt is not None and ex_gt is not None and (
                        gt_ids["invivo"][iv_gt - 1], gt_ids["exvivo"][ex_gt - 1]
                    ) in truth_pairs:
                        correct += 1
                counts[distance][0] += correct
                counts[distance][1] += len(pairs) - correct
                counts[distance][2] += len(truth_pairs) - correct
            print(row.sample_id, "PQ", round(pq["invivo"][-1], 3), round(pq["exvivo"][-1], 3),
                  "cells", len(item["centers"]["invivo"]), len(item["centers"]["exvivo"]),
                  "registration", chosen_mode, round(score, 1), flush=True)
    iv_pq, ex_pq = np.mean(pq["invivo"]), np.mean(pq["exvivo"])
    print("Mean PQ", iv_pq, ex_pq, flush=True)
    print("Eligible matched masks", eligible, "oracle registration", oracle, flush=True)
    for distance, (tp, fp, fn) in counts.items():
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
        final = 0.25 * (iv_pq + ex_pq) + 0.5 * f1
        print(f"distance {distance}: TP={tp} FP={fp} FN={fn} F1={f1:.4f} final={final:.4f}", flush=True)


def submit(frame: pd.DataFrame, records: dict, output: Path, match_distance: float):
    rows = []
    for subject, group in frame.groupby(frame.sample_id.str.split("__").str[0], sort=False):
        mode = subject_mode([records[row.sample_id] for row in group.itertuples(index=False)])
        print(f"{subject} subject-level mode: {mode}", flush=True)
        for row in group.itertuples(index=False):
            item = records[row.sample_id]
            matrix, score, chosen_mode = select_registration(item, mode)
            iv = labels_to_rles(item["labels"]["invivo"], "IVP")
            ex = labels_to_rles(item["labels"]["exvivo"], "EXP")
            pairs = []
            if matrix is not None:
                pairs = [[f"IVP_{i+1:06d}", f"EXP_{j+1:06d}"] for i, j, _ in
                         matched_indices(item, matrix, match_distance)]
            print(row.sample_id, "cells", len(iv), len(ex), "pairs", len(pairs),
                  "registration", chosen_mode, round(score, 1), flush=True)
            rows.append({"sample_id": row.sample_id, "invivo_instances": iv,
                         "exvivo_instances": ex, "match_pairs": pairs})
    write_submission(rows, output)
    print(f"Saved {output} ({output.stat().st_size:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "submit"], required=True)
    parser.add_argument("--subject", default="")
    parser.add_argument("--iv-model", type=Path, nargs="+", required=True)
    parser.add_argument("--ex-model", type=Path, nargs="+", required=True)
    parser.add_argument("--iv-threshold", type=float, default=0.5)
    parser.add_argument("--ex-threshold", type=float, default=0.55)
    parser.add_argument("--ex-min-intensity", type=float, default=0.65)
    parser.add_argument("--match-distance", type=float, default=6)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    iv_model = [load_model(path, device) for path in args.iv_model]
    ex_model = [load_model(path, device) for path in args.ex_model]
    if args.mode == "validate":
        if not args.subject:
            raise ValueError("--subject is required for validation")
        frame = pd.read_csv(ROOT / "training/train_ground_truth.csv")
        frame = frame[frame.sample_id.str.startswith(args.subject)]
        directory = ROOT / "training"
    else:
        frame = pd.read_csv(ROOT / "sample_submission.csv")
        directory = ROOT / "hidden_test"
    records = {}
    for row in frame.itertuples(index=False):
        path = directory / row.sample_id.replace("__", "/")
        records[row.sample_id] = predict_one(path, iv_model, ex_model, device, args.tta,
                                             args.iv_threshold, args.ex_threshold,
                                             args.ex_min_intensity)
        print("inferred", row.sample_id, flush=True)
    if args.mode == "validate":
        validate(frame, records)
    else:
        submit(frame, records, args.output, args.match_distance)


if __name__ == "__main__":
    main()
