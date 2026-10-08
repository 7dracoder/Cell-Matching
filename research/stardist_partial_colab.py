"""Pilot partially supervised StarDist on one held-out mouse.

Unlike ordinary mask training, StarDist explicitly supports negative label
values for unknown pixels. Here only annotated somas and a narrow background
ring are supervised; the rest of each crop is ignored. A pretrained
fluorescence model supplies the initial morphology prior. This pilot tests a
single untouched mouse before deciding whether a full CV run is justified.

Run in the prepared Colab runtime after ``pip install stardist``::
    python -u /content/stardist_partial_colab.py --fold subject_5d294c
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from csbdeep.utils import normalize
from csbdeep.utils.tf import keras_import
from scipy import ndimage as ndi
from stardist.models import StarDist2D

WORK = Path("/content/work") if Path("/content/work/pipeline.py").exists() else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
import pipeline as P  # noqa: E402
from cellmatch import pq_score, region_centers  # noqa: E402

MODELS = WORK / "models"
SCALE = 3.0
SIZE = 96


def training_data(truth, subject, per_region=40, seed=83):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for sid, row in truth.items():
        if row["subject"] == subject:
            continue
        image = P.read_image(P.region_path(sid) / "exvivo.tif")
        gt = row["exvivo"][0]
        image = normalize(image, 1, 99.8).astype(np.float32)
        centers = region_centers(gt)
        if not len(centers):
            continue
        chosen = rng.choice(len(centers), min(per_region, len(centers)), replace=False)
        for x, y in centers[chosen]:
            y0 = int(np.clip(round(y) - SIZE // 2, 0, image.shape[0] - SIZE))
            x0 = int(np.clip(round(x) - SIZE // 2, 0, image.shape[1] - SIZE))
            patch = image[y0:y0 + SIZE, x0:x0 + SIZE]
            instance = gt[y0:y0 + SIZE, x0:x0 + SIZE]
            # Pixels with no annotation are unverified, not known background.
            # A 3-pixel ring around annotated instances provides reliable
            # local background and prevents the model from bloating the masks.
            known = ndi.binary_dilation(instance > 0, iterations=3)
            target = np.full(instance.shape, -1, dtype=np.int32)
            target[known] = 0
            target[instance > 0] = instance[instance > 0]
            patch = cv2.resize(patch, (int(SIZE * SCALE),) * 2,
                               interpolation=cv2.INTER_CUBIC)
            target = cv2.resize(target, (int(SIZE * SCALE),) * 2,
                                interpolation=cv2.INTER_NEAREST)
            X.append(patch.astype(np.float32))
            Y.append(target.astype(np.int32))
        print("TILES", sid, len(chosen), flush=True)
    return X, Y


def fit(truth, subject, epochs, steps):
    MODELS.mkdir(exist_ok=True, parents=True)
    name = f"stardist_partial_fold_{subject}"
    path = MODELS / name
    ready = path / "trained.flag"
    if ready.exists():
        print("REUSE", path, flush=True)
        return StarDist2D(None, name=name, basedir=str(MODELS))
    source = StarDist2D.from_pretrained("2D_versatile_fluo")
    if not path.exists():
        shutil.copytree(source.logdir, path)
    del source
    model = StarDist2D(None, name=name, basedir=str(MODELS))
    model.config.train_n_val_patches = 32
    X, Y = training_data(truth, subject)
    perm = np.random.default_rng(71).permutation(len(X))
    val = set(perm[:max(16, len(X) // 20)].tolist())
    Xtr, Ytr = [x for i, x in enumerate(X) if i not in val], [y for i, y in enumerate(Y) if i not in val]
    Xval, Yval = [x for i, x in enumerate(X) if i in val], [y for i, y in enumerate(Y) if i in val]
    print("TRAIN", subject, len(Xtr), "VAL", len(Xval), "EPOCHS", epochs, flush=True)
    model.prepare_for_training(keras_import().optimizers.Adam(1e-5))
    model.train(Xtr, Ytr, validation_data=(Xval, Yval),
                epochs=epochs, steps_per_epoch=steps, workers=1)
    ready.touch()
    return model


def decode(model, image, probability):
    scaled = normalize(image, 1, 99.8).astype(np.float32)
    labels, _ = model.predict_instances(scaled, scale=SCALE,
                                        prob_thresh=probability, n_tiles=(3, 3),
                                        show_tile_progress=False)
    if labels.shape != image.shape:
        labels = cv2.resize(labels.astype(np.int32),
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    count = np.bincount(labels.ravel())
    keep = (count >= 17) & (count <= 350)
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(image.shape).astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="subject_5d294c")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only this many held-out regions (0 means all)")
    parser.add_argument("--probabilities", default="0.3,0.5,0.7")
    args = parser.parse_args()
    probabilities = tuple(float(value) for value in args.probabilities.split(","))
    truth = P.load_truth()
    if args.fold not in P.SUBJECTS:
        parser.error(f"unknown training mouse {args.fold}")
    model = fit(truth, args.fold, args.epochs, args.steps)
    rows = []
    heldout = [(sid, row) for sid, row in truth.items() if row["subject"] == args.fold]
    if args.limit:
        heldout = heldout[:args.limit]
    for sid, row in heldout:
        image = P.read_image(P.region_path(sid) / "exvivo.tif")
        for probability in probabilities:
            labels = decode(model, image, probability)
            score, tp, fp, fn = pq_score(labels, row["exvivo"][0])
            result = {"sample_id": sid, "probability": probability,
                      "pq": float(score), "tp": int(tp), "fp": int(fp),
                      "fn": int(fn)}
            rows.append(result)
            print("EVAL", result, flush=True)
            (WORK / "stardist_partial_pilot.json").write_text(json.dumps(rows, indent=2))
    for probability in probabilities:
        scores = [row["pq"] for row in rows if row["probability"] == probability]
        print("FINAL", probability, float(np.mean(scores)), flush=True)


if __name__ == "__main__":
    main()
