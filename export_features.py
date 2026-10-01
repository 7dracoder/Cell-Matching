"""Export held-out cell-center features for registration tuning without label leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellmatch import ROOT, normalize, pq_score, read_image, region_centers, rle_to_labels
from learned import load_model, predict_probabilities, probabilities_to_labels
from solve import predicted_to_ground_truth


CHECKPOINTS = {
    "subject_5d294c": ("models/invivo_validation_5d.pt", "models/exvivo_validation_5d.pt"),
    "subject_b2ba5e": ("models/invivo_validation.pt", "models/exvivo_validation.pt"),
    "subject_db6b8b": ("models/invivo_validation_db.pt", "models/exvivo_validation_db.pt"),
}


def main() -> None:
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = pd.read_csv(ROOT / "training/train_ground_truth.csv")
    result = []
    for subject, (iv_checkpoint, ex_checkpoint) in CHECKPOINTS.items():
        models = (load_model(Path(iv_checkpoint), device), load_model(Path(ex_checkpoint), device))
        for row in frame[frame.sample_id.str.startswith(subject)].itertuples(index=False):
            path = ROOT / "training" / row.sample_id.replace("__", "/")
            item: dict = {"sample_id": row.sample_id,
                          "truth_pairs": json.loads(row.match_pairs)}
            for modality, model in zip(("invivo", "exvivo"), models):
                image = read_image(path / f"{modality}.tif")
                probs = predict_probabilities(model, image, modality, device=device, tta=True)
                threshold = 0.5 if modality == "invivo" and subject == "subject_db6b8b" else (
                    0.45 if modality == "invivo" else 0.6)
                labels = probabilities_to_labels(probs, modality, threshold, 0.5)
                truth, ids = rle_to_labels(json.loads(getattr(row, f"{modality}_instances")), image.shape)
                matches = predicted_to_ground_truth(labels, truth)
                item[f"{modality}_centers"] = region_centers(labels).round(2).tolist()
                item[f"{modality}_gt"] = [ids[matches[i + 1] - 1] if i + 1 in matches else None
                                             for i in range(int(labels.max()))]
                item[f"{modality}_pq"] = pq_score(labels, truth)[0]
                item[f"{modality}_shape"] = list(image.shape)
                if modality == "exvivo":
                    brightness = normalize(image, modality)[0]
                    counts = np.bincount(labels.ravel())
                    means = np.bincount(labels.ravel(), weights=brightness.ravel()) / np.maximum(counts, 1)
                    item["exvivo_brightness"] = means[1:].round(3).tolist()
            result.append(item)
            print("exported", row.sample_id, flush=True)
    Path("features_cv.json").write_text(json.dumps(result))
    print("saved features_cv.json", len(result), flush=True)


if __name__ == "__main__":
    main()
