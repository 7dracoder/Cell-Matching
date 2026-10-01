"""Robust registration of sparse cell centers across imaging modalities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from cellmatch import ROOT, read_image, region_centers, rle_to_labels


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def initial_candidates(invivo: np.ndarray, exvivo: np.ndarray, angle_limit: int = 45,
                       bin_size: float = 12, count: int = 36) -> list[np.ndarray]:
    """Hough voting for angle, scale, and translation from all point pairs."""
    if len(invivo) < 3 or len(exvivo) < 3:
        return []
    angles = np.arange(-angle_limit, angle_limit + 0.1, 3)
    scales = np.arange(0.79, 1.141, 0.04)
    pool = []
    # In-vivo coordinates are in a much smaller canvas; translations are bounded.
    limit = 2500
    bins = int(2 * limit / bin_size) + 1
    for angle in angles:
        theta = np.deg2rad(angle)
        base = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for scale in scales:
            linear = scale * base
            rotated = invivo @ linear.T
            differences = (exvivo[:, None, :] - rotated[None, :, :]).reshape(-1, 2)
            xy = np.floor((differences + limit) / bin_size).astype(np.int32)
            valid = (xy >= 0).all(axis=1) & (xy < bins).all(axis=1)
            votes = np.bincount(xy[valid, 1] * bins + xy[valid, 0], minlength=bins * bins)
            peak_ids = np.argpartition(votes, -3)[-3:]
            for peak in peak_ids:
                ix, iy = peak % bins, peak // bins
                around = (np.abs(xy[valid, 0] - ix) <= 1) & (np.abs(xy[valid, 1] - iy) <= 1)
                if around.sum() < 3:
                    continue
                shift = np.median(differences[valid][around], axis=0)
                matrix = np.c_[linear, shift]
                _, _, distances = paired_nearest(invivo, exvivo, matrix, 16)
                score = np.maximum(0, 1 - distances / 16).sum()
                pool.append((float(score), matrix))
    pool.sort(key=lambda pair: pair[0], reverse=True)
    return [matrix for _, matrix in pool[:count]]


def fixed_linear_candidates(invivo: np.ndarray, exvivo: np.ndarray, linear: np.ndarray,
                            bin_size: float = 8, count: int = 20) -> list[np.ndarray]:
    """Find translations when a subject-level linear transform is known approximately."""
    differences = (exvivo[:, None, :] - (invivo @ linear.T)[None, :, :]).reshape(-1, 2)
    limit = 2500
    bins = int(2 * limit / bin_size) + 1
    xy = np.floor((differences + limit) / bin_size).astype(np.int32)
    valid = (xy >= 0).all(axis=1) & (xy < bins).all(axis=1)
    histogram = np.bincount(xy[valid, 1] * bins + xy[valid, 0], minlength=bins * bins).reshape(bins, bins)
    histogram = gaussian_filter(histogram.astype(np.float32), 0.8)
    pool = []
    for _ in range(count):
        peak = np.unravel_index(np.argmax(histogram), histogram.shape)
        if histogram[peak] == 0:
            break
        nearby = (np.abs(xy[valid, 0] - peak[1]) <= 2) & (np.abs(xy[valid, 1] - peak[0]) <= 2)
        shift = np.median(differences[valid][nearby], axis=0)
        pool.append(np.c_[linear, shift])
        y, x = peak
        histogram[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = 0
    return pool


def paired_nearest(invivo: np.ndarray, exvivo: np.ndarray, matrix: np.ndarray,
                   distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = transform(invivo, matrix)
    tree = cKDTree(exvivo)
    nearest_distance, nearest_index = tree.query(projected, distance_upper_bound=distance)
    valid = np.isfinite(nearest_distance)
    sources = np.flatnonzero(valid)
    if not len(sources):
        return np.array([], int), np.array([], int), np.array([], float)
    destinations = nearest_index[valid]
    distances = nearest_distance[valid]
    # Retain a single source for each ex-vivo cell.
    order = np.argsort(distances)
    used = set()
    keep = []
    for k in order:
        if int(destinations[k]) not in used:
            used.add(int(destinations[k]))
            keep.append(k)
    return sources[keep], destinations[keep], distances[keep]


def refine(invivo: np.ndarray, exvivo: np.ndarray, initial: np.ndarray,
           target_weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    matrix = initial.copy()
    for distance in (30, 20, 12, 7):
        sources, destinations, _ = paired_nearest(invivo, exvivo, matrix, distance)
        if len(sources) < 3:
            break
        fit, inliers = cv2.estimateAffine2D(invivo[sources], exvivo[destinations],
                                             method=cv2.RANSAC, ransacReprojThreshold=max(4, distance * 0.5),
                                             maxIters=2000, refineIters=20)
        if fit is None or not 0.65 < abs(np.linalg.det(fit[:, :2])) < 1.3:
            break
        matrix = fit
    _, destination, distances = paired_nearest(invivo, exvivo, matrix, 8)
    # True registrations support many close, unique cell pairs.
    pair_weights = np.ones(len(distances)) if target_weights is None else target_weights[destination]
    score = np.sum(pair_weights * (1 + np.maximum(0, 1 - distances / 8)))
    return matrix, float(score)


def register_points(invivo: np.ndarray, exvivo: np.ndarray) -> tuple[np.ndarray | None, float]:
    candidates = initial_candidates(invivo, exvivo)
    if not candidates:
        return None, 0
    results = [refine(invivo, exvivo, candidate) for candidate in candidates]
    return max(results, key=lambda result: result[1])


# Median in-vivo to ex-vivo linear transforms estimated from the 47 labeled
# regions, using verified pairs only.  Translation is estimated per region.
TRAINING_LINEAR_PRIORS = (
    np.array([[0.967, -0.161], [0.121, 0.909]], dtype=np.float32),
    np.array([[0.982, 0.227], [-0.256, 0.927]], dtype=np.float32),
)


def estimate_linear_priors(exclude_subjects: set[str] | None = None) -> tuple[np.ndarray, ...]:
    """Fit orientation-clustered linear priors from verified GT pairs.

    When exclude_subjects is set, those mice are held out (leak-free CV).
    """
    frame = pd.read_csv(ROOT / "training/train_ground_truth.csv")
    matrices = []
    for row in frame.itertuples(index=False):
        subject = row.sample_id.split("__")[0]
        if exclude_subjects and subject in exclude_subjects:
            continue
        path = ROOT / "training" / row.sample_id.replace("__", "/")
        iv, iv_ids = rle_to_labels(json.loads(row.invivo_instances),
                                   read_image(path / "invivo.tif").shape)
        ex, ex_ids = rle_to_labels(json.loads(row.exvivo_instances),
                                   read_image(path / "exvivo.tif").shape)
        iv_centers, ex_centers = region_centers(iv), region_centers(ex)
        iv_lookup = {cell_id: i for i, cell_id in enumerate(iv_ids)}
        ex_lookup = {cell_id: i for i, cell_id in enumerate(ex_ids)}
        pairs = [(iv_lookup[a], ex_lookup[b]) for a, b in json.loads(row.match_pairs)
                 if a in iv_lookup and b in ex_lookup]
        if len(pairs) < 3:
            continue
        fit, _ = cv2.estimateAffine2D(
            iv_centers[[i for i, _ in pairs]].astype(np.float32),
            ex_centers[[j for _, j in pairs]].astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=5)
        if fit is None:
            continue
        linear = fit[:, :2]
        if not 0.65 < abs(np.linalg.det(linear)) < 1.3:
            continue
        matrices.append(linear.astype(np.float32))
    # Two orientation clusters, matching the order of TRAINING_LINEAR_PRIORS.
    positive = [m for m in matrices if np.arctan2(m[1, 0], m[0, 0]) >= 0]
    negative = [m for m in matrices if np.arctan2(m[1, 0], m[0, 0]) < 0]
    if not positive or not negative:
        return TRAINING_LINEAR_PRIORS
    return (
        np.median(np.stack(positive), axis=0).astype(np.float32),
        np.median(np.stack(negative), axis=0).astype(np.float32),
    )


def register_with_priors(invivo: np.ndarray, exvivo: np.ndarray,
                         mode: int | None = None,
                         target_weights: np.ndarray | None = None,
                         return_candidates: bool = False,
                         priors: tuple[np.ndarray, ...] | None = None,
                         ) -> tuple[np.ndarray | None, float, int] | list[tuple[np.ndarray, float, int]]:
    """Search label-derived geometric priors and solve translation from data."""
    if len(invivo) < 3 or len(exvivo) < 3:
        return None, 0, -1
    prior_set = priors if priors is not None else TRAINING_LINEAR_PRIORS
    results = []
    modes = range(len(prior_set)) if mode is None else (mode,)
    for mode_index in modes:
        base = prior_set[mode_index]
        for angle in (-5, 0, 5):
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            for scale in (0.95, 1.0, 1.05):
                linear = (scale * rotation @ base).astype(np.float32)
                for candidate in fixed_linear_candidates(invivo, exvivo, linear, count=4):
                    matrix, score = refine(invivo, exvivo, candidate, target_weights)
                    results.append((matrix, score, mode_index))
    if return_candidates:
        return sorted(results, key=lambda item: item[1], reverse=True)
    return max(results, key=lambda item: item[1]) if results else (None, 0, -1)


def blob_map(points: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    image = np.zeros(shape, np.float32)
    ij = np.round(points[:, ::-1]).astype(int)
    inside = (ij >= 0).all(axis=1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.add.at(image, (ij[inside, 0], ij[inside, 1]), 1.0)
    return cv2.GaussianBlur(image, (0, 0), sigma)


def register_ncc(invivo: np.ndarray, exvivo: np.ndarray, priors: tuple[np.ndarray, ...],
                 down: float = 2.0, sigma: float = 1.2, top: int = 3) -> tuple[np.ndarray | None, float, int]:
    """Normalized cross-correlation of cell-density maps over every translation.

    Normalization discounts dense clusters of false ex-vivo detections, which is what
    fools the raw pair-count score used by register_with_priors.
    """
    if len(invivo) < 3 or len(exvivo) < 3:
        return None, 0, -1
    target = exvivo / down
    candidates = []
    for mode, base in enumerate(priors):
        for angle in np.arange(-9, 9.1, 1.5):
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            for scale in np.arange(0.93, 1.101, 0.02):
                linear = scale * rotation @ base
                projected = invivo @ linear.T / down
                origin = projected.min(axis=0) - 4
                template_points = projected - origin
                template = blob_map(template_points, (int(template_points[:, 1].max()) + 5,
                                                      int(template_points[:, 0].max()) + 5), sigma)
                pad = np.array(template.shape) // 2
                image = blob_map(target + pad[::-1], (int(target[:, 1].max()) + 5 + 2 * pad[0],
                                                      int(target[:, 0].max()) + 5 + 2 * pad[1]), sigma)
                image = cv2.copyMakeBorder(image, 0, max(0, template.shape[0] - image.shape[0]),
                                           0, max(0, template.shape[1] - image.shape[1]), cv2.BORDER_CONSTANT)
                _, value, _, location = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED))
                shift = (np.array(location, float) - origin - pad[::-1]) * down
                candidates.append((value, np.c_[linear, shift], mode))
    candidates.sort(key=lambda item: item[0], reverse=True)
    refined = [(*refine(invivo, exvivo, matrix), mode) for _, matrix, mode in candidates[:top]]
    return max(refined, key=lambda item: item[1])


def match_points(invivo: np.ndarray, exvivo: np.ndarray, matrix: np.ndarray,
                 max_distance: float = 5.0, method: str = "greedy") -> list[tuple[int, int, float]]:
    if method == "greedy" or len(invivo) == 0 or len(exvivo) == 0:
        sources, destinations, distances = paired_nearest(invivo, exvivo, matrix, max_distance)
        return [(int(i), int(j), float(d)) for i, j, d in zip(sources, destinations, distances)]
    # Hungarian: the most one-to-one pairs within max_distance, then the least total distance.
    close = cKDTree(transform(invivo, matrix)).sparse_distance_matrix(
        cKDTree(exvivo), max_distance, output_type="ndarray")
    if not len(close):
        return []
    rows, row_index = np.unique(close["i"], return_inverse=True)
    cols, col_index = np.unique(close["j"], return_inverse=True)
    cost = np.full((len(rows), len(cols)), 1e6)
    cost[row_index, col_index] = close["v"]
    chosen_rows, chosen_cols = linear_sum_assignment(cost)
    return [(int(rows[i]), int(cols[j]), float(cost[i, j]))
            for i, j in zip(chosen_rows, chosen_cols) if cost[i, j] < 1e6]
