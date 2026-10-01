"""Cell instance segmentation, registration, and Kaggle submission utilities.

Images and run lengths retain their original (row, column) coordinates throughout.
The routines here are intentionally independent of Kaggle's hidden labels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


ROOT = Path(__file__).resolve().parent / "Project_2_Dataset"


def read_image(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image: {path}, shape={image.shape}")
    return image


def decode_rle(rle: str, shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    runs = np.fromstring(rle, sep=" ", dtype=np.int64)
    if len(runs) % 2:
        raise ValueError("Odd number of RLE entries")
    for start, length in runs.reshape(-1, 2):
        if start < 0 or length <= 0 or start + length > flat.size:
            raise ValueError("Invalid RLE run")
        flat[start : start + length] = True
    return flat.reshape(shape)


def rle_to_labels(instances: dict[str, str], shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    labels = np.zeros(shape[0] * shape[1], np.int32)
    ids = list(instances)
    for index, cell_id in enumerate(ids, 1):
        runs = np.fromstring(instances[cell_id], sep=" ", dtype=np.int64)
        if len(runs) % 2:
            raise ValueError(f"Invalid RLE for {cell_id}")
        for start, length in runs.reshape(-1, 2):
            labels[start : start + length] = index
    return labels.reshape(shape), ids


def encode_rle(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    edges = np.flatnonzero(np.diff(np.r_[False, flat, False].astype(np.int8)))
    return " ".join(f"{start} {end-start}" for start, end in edges.reshape(-1, 2))


def labels_to_rles(labels: np.ndarray, prefix: str) -> dict[str, str]:
    flat = np.asarray(labels, dtype=np.int32).ravel(order="C")
    edges = np.r_[0, np.flatnonzero(np.diff(flat)) + 1, len(flat)]
    runs: dict[int, list[str]] = {}
    for start, end in zip(edges[:-1], edges[1:]):
        cell = int(flat[start])
        if cell:
            runs.setdefault(cell, []).append(f"{start} {end-start}")
    return {f"{prefix}_{cell:06d}": " ".join(chunks) for cell, chunks in sorted(runs.items())}


def normalize(image: np.ndarray, modality: str) -> np.ndarray:
    """Multi-scale contrast normalized to a fairly stable [0,1] range."""
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99.8])
    image = np.clip((image - low) / max(high - low, 1), 0, 1)
    # Local contrast makes bright somata comparable across uneven illumination.
    background = cv2.GaussianBlur(image, (0, 0), 12 if modality == "invivo" else 9)
    contrast = image - background
    scale = np.percentile(contrast[contrast > 0], 99) if np.any(contrast > 0) else 1
    contrast = np.clip(contrast / max(scale, 0.02), 0, 1)
    return np.stack([image, contrast], axis=0).astype(np.float32)


def classical_segment(image: np.ndarray, modality: str, threshold: float = 0.19) -> np.ndarray:
    """Fallback segmentation based on cell-sized local contrast and watershed."""
    features = normalize(image, modality)
    contrast = features[1]
    sigma = 1.1 if modality == "invivo" else 0.9
    smooth = ndi.gaussian_filter(contrast, sigma)
    radius = 4 if modality == "invivo" else 3
    coords = peak_local_max(smooth, min_distance=radius, threshold_abs=threshold)
    markers = np.zeros(image.shape, np.int32)
    for i, (row, col) in enumerate(coords, 1):
        markers[row, col] = i
    foreground = smooth > (threshold * (0.48 if modality == "invivo" else 0.53))
    labels = watershed(-smooth, markers, mask=foreground)
    min_area, max_area = (18, 340) if modality == "invivo" else (15, 360)
    counts = np.bincount(labels.ravel())
    keep = (counts >= min_area) & (counts <= max_area)
    keep[0] = False
    labels[~keep[labels]] = 0
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def pq_score(pred: np.ndarray, truth: np.ndarray, cutoff: float = 0.75) -> tuple[float, int, int, int]:
    """Competition-style panoptic quality for a pair of disjoint label maps."""
    npred, ngt = int(pred.max()), int(truth.max())
    if not npred and not ngt:
        return 1.0, 0, 0, 0
    if not npred or not ngt:
        return 0.0, 0, npred, ngt
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + truth).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc = joint.sum(axis=1)
    gc = joint.sum(axis=0)
    rows, cols = np.nonzero(joint[1:, 1:])
    rows += 1
    cols += 1
    iou = joint[rows, cols] / (pc[rows] + gc[cols] - joint[rows, cols])
    hits = iou > cutoff
    tp = int(hits.sum())
    fp, fn = npred - tp, ngt - tp
    denominator = tp + 0.5 * (fp + fn)
    return float(iou[hits].sum() / denominator), tp, fp, fn


def region_centers(labels: np.ndarray) -> np.ndarray:
    n = int(labels.max())
    if n == 0:
        return np.empty((0, 2), np.float32)
    yy, xx = np.indices(labels.shape)
    count = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    x = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=n + 1)[1:]
    y = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=n + 1)[1:]
    return np.stack([x / np.maximum(count, 1), y / np.maximum(count, 1)], axis=1).astype(np.float32)


def write_submission(rows: list[dict], path: Path, sample_csv: Path = ROOT / "sample_submission.csv") -> None:
    columns = ["sample_id", "invivo_instances", "exvivo_instances", "match_pairs"]
    with sample_csv.open(newline="") as stream:
        expected = [row["sample_id"] for row in csv.DictReader(stream)]
    if [row["sample_id"] for row in rows] != expected:
        raise ValueError("Submission sample IDs/order do not match the sample file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            iv, ex, pairs = row["invivo_instances"], row["exvivo_instances"], row["match_pairs"]
            assert len({pair[0] for pair in pairs}) == len(pairs)
            assert len({pair[1] for pair in pairs}) == len(pairs)
            assert all(a in iv and b in ex for a, b in pairs)
            writer.writerow({"sample_id": row["sample_id"], "invivo_instances": json.dumps(iv),
                             "exvivo_instances": json.dumps(ex), "match_pairs": json.dumps(pairs)})
