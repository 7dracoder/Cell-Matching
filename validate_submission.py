"""Strict structural checks for the generated Kaggle submission."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# In Colab the validation script is uploaded to /content while the dataset and
# current project modules live in /content/work. Keep local execution unchanged.
if Path("/content/work/Project_2_Dataset").is_dir():
    sys.path.insert(0, "/content/work")

from cellmatch import ROOT, read_image


def check_instances(instances: dict[str, str], shape: tuple[int, int]) -> int:
    occupied = np.zeros(shape[0] * shape[1], np.bool_)
    total = 0
    for cell_id, rle in instances.items():
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("Invalid cell ID")
        runs = np.fromstring(rle, sep=" ", dtype=np.int64)
        if len(runs) == 0 or len(runs) % 2:
            raise ValueError(f"Invalid run count: {cell_id}")
        for start, length in runs.reshape(-1, 2):
            if start < 0 or length <= 0 or start + length > occupied.size:
                raise ValueError(f"RLE out of bounds: {cell_id}")
            if occupied[start:start + length].any():
                raise ValueError(f"Overlapping masks: {cell_id}")
            occupied[start:start + length] = True
            total += int(length)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    # A region's RLE field can exceed the csv module's default 128 KB limit.
    csv.field_size_limit(sys.maxsize)
    with (ROOT / "sample_submission.csv").open(newline="") as stream:
        expected = [row["sample_id"] for row in csv.DictReader(stream)]
    with args.path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["sample_id", "invivo_instances", "exvivo_instances", "match_pairs"]
        rows = list(reader)
    if [row["sample_id"] for row in rows] != expected:
        raise ValueError("Rows do not match sample_submission.csv")
    total = {"invivo_cells": 0, "exvivo_cells": 0, "matches": 0}
    for row in rows:
        iv, ex, pairs = (json.loads(row[key]) for key in
                         ("invivo_instances", "exvivo_instances", "match_pairs"))
        if not isinstance(iv, dict) or not isinstance(ex, dict) or not isinstance(pairs, list):
            raise ValueError(f"Wrong JSON type in {row['sample_id']}")
        path = ROOT / "hidden_test" / row["sample_id"].replace("__", "/")
        for modality, instances in (("invivo", iv), ("exvivo", ex)):
            shape = read_image(path / f"{modality}.tif").shape
            check_instances(instances, shape)
            total[f"{modality}_cells"] += len(instances)
        if len({a for a, _ in pairs}) != len(pairs) or len({b for _, b in pairs}) != len(pairs):
            raise ValueError(f"Repeated match IDs in {row['sample_id']}")
        if any(a not in iv or b not in ex for a, b in pairs):
            raise ValueError(f"Match refers to absent mask in {row['sample_id']}")
        total["matches"] += len(pairs)
    print(f"PASS {len(rows)} rows, {total}, {args.path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
