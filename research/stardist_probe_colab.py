"""Zero-shot StarDist pilot for fluorescent ex-vivo somas, mouse-stratified.

Requires ``pip install stardist``. The pretrained fluorescence-nuclei model
has never seen this competition's masks. Probe a few regions per mouse before
committing compute to a full leave-one-mouse-out/fine-tuning experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D

WORK = Path("/content/work") if Path("/content/work/pipeline.py").exists() else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-mouse", type=int, default=3)
    args = parser.parse_args()
    truth = P.load_truth()
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    rows = []
    for subject in P.SUBJECTS:
        ids = sorted(sid for sid, row in truth.items() if row["subject"] == subject)
        ids = ids[:args.per_mouse]
        for sid in ids:
            image = P.read_image(P.region_path(sid) / "exvivo.tif")
            gt = truth[sid]["exvivo"][0]
            norm = normalize(image, 1, 99.8).astype(np.float32)
            for scale in (1.0, 1.5, 2.0):
                labels, details = model.predict_instances(norm, scale=scale,
                                                           prob_thresh=0.45,
                                                           n_tiles=(2, 2),
                                                           show_tile_progress=False)
                if labels.shape != image.shape:
                    labels = cv2.resize(labels.astype(np.int32), (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                score = float(pq_score(labels.astype(np.int32), gt)[0])
                row = {"sample_id": sid, "subject": subject, "scale": scale,
                       "pq": score, "predicted": int(labels.max()),
                       "gt": int(gt.max())}
                rows.append(row)
                print("PROBE", row, flush=True)
                (WORK / "stardist_probe.json").write_text(json.dumps(rows, indent=2))
    for scale in (1.0, 1.5, 2.0):
        selected = [x for x in rows if x["scale"] == scale]
        print("FINAL", scale, float(np.mean([x["pq"] for x in selected])), flush=True)


if __name__ == "__main__":
    main()
