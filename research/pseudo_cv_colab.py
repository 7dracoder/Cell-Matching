"""Leave-one-mouse-out Cellpose-SAM with conservative pseudo-label augmentation.

Ground-truth instances are sparse: every other visible soma in a training crop
is currently treated as background. A *pretrained* teacher proposes only
high-confidence, non-overlapping cells; annotated masks always win. The
teacher has never seen competition labels, so the mouse-held-out check is
leak-free. This is a diagnostic: it does not make a submission before CV wins.

Run in the prepared Colab runtime:
    python -u /content/pseudo_cv_colab.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import models, train
from scipy import ndimage as ndi

WORK = Path("/content/work") if Path("/content/work/pipeline.py").exists() else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
import pipeline as P  # noqa: E402
from cellmatch import pq_score, region_centers  # noqa: E402

OUTPUT = WORK / "pseudo_cv.json"
CACHE = WORK / "pseudo_teacher"
FLOWS = WORK / "flows"
MODELS = WORK / "models"
P.CELLPOSE_TILE = 96
GRID = [(cp, flow) for cp in (-0.5, 0.0, 0.25, 0.5, 0.75)
        for flow in (0.08, 0.15, 0.25, 0.4)]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def model(path=None):
    return models.CellposeModel(gpu=torch.cuda.is_available(),
                                pretrained_model=str(path) if path else "cpsam_v2",
                                use_bfloat16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def teacher_labels(sid, image, teacher):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{sid}.npz"
    if path.exists():
        with np.load(path) as packed:
            return packed["labels"].astype(np.int32)
    labels, _, _ = teacher.eval(image.astype(np.float32), diameter=12,
                                cellprob_threshold=0.5, flow_threshold=0.15,
                                min_size=12, batch_size=16)
    labels = np.asarray(labels, dtype=np.int32)
    sizes = np.bincount(labels.ravel())
    keep = (sizes >= 17) & (sizes <= 350)
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    _, inverse = np.unique(labels, return_inverse=True)
    labels = inverse.reshape(labels.shape).astype(np.int32)
    np.savez_compressed(path, labels=labels.astype(np.uint16))
    log("TEACHER", sid, int(labels.max()))
    return labels


def fuse(gt, pseudo):
    """Keep only teacher objects clear of every known GT instance."""
    danger = ndi.binary_dilation(gt > 0, iterations=2)
    touch = np.bincount(pseudo[danger], minlength=int(pseudo.max()) + 1)
    keep = touch == 0
    keep[0] = False
    chosen = np.where(keep[pseudo], pseudo, 0)
    _, chosen = np.unique(chosen, return_inverse=True)
    chosen = chosen.reshape(gt.shape).astype(np.int32)
    chosen[gt > 0] = gt[gt > 0] + int(chosen.max())
    return chosen, int(keep.sum())


def make_tiles(subjects, truth, teacher, per_region=100, seed=27):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    half = P.CELLPOSE_TILE // 2
    for sid, row in truth.items():
        if row["subject"] not in subjects:
            continue
        image = P.read_image(P.region_path(sid) / "exvivo.tif")
        gt = row["exvivo"][0]
        pseudo = teacher_labels(sid, image, teacher)
        fused, added = fuse(gt, pseudo)
        centers = region_centers(gt)
        image = P.percentile_normalize(image)
        chosen = rng.choice(len(centers), min(per_region, len(centers)), replace=False)
        for x, y in centers[chosen]:
            y0 = int(np.clip(round(y) - half, 0, image.shape[0] - P.CELLPOSE_TILE))
            x0 = int(np.clip(round(x) - half, 0, image.shape[1] - P.CELLPOSE_TILE))
            crop = fused[y0:y0 + P.CELLPOSE_TILE, x0:x0 + P.CELLPOSE_TILE]
            _, crop = np.unique(crop, return_inverse=True)
            images.append(image[y0:y0 + P.CELLPOSE_TILE, x0:x0 + P.CELLPOSE_TILE])
            labels.append(crop.reshape(P.CELLPOSE_TILE, P.CELLPOSE_TILE).astype(np.int32))
        log("TILES", sid, len(chosen), "GT", int(gt.max()), "PSEUDO", added)
    return images, labels


def fit(subject, truth, teacher, epochs, per_epoch):
    path = MODELS / f"exvivo_cpsam_pseudo_fold_{subject}"
    MODELS.mkdir(parents=True, exist_ok=True)
    if path.exists():
        log("REUSE_MODEL", path)
        return path
    other = [s for s in P.SUBJECTS if s != subject]
    images, labels = make_tiles(other, truth, teacher)
    log("TRAIN", subject, len(images), "tiles")
    network = model().net
    train.train_seg(network, train_data=images, train_labels=labels,
                    normalize=False, rescale=True, batch_size=4,
                    n_epochs=epochs, nimg_per_epoch=per_epoch,
                    learning_rate=1e-5, weight_decay=0.1,
                    min_train_masks=1, save_path=str(WORK), model_name=path.name)
    del network, images, labels
    gc.collect()
    torch.cuda.empty_cache()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def infer(subject, truth, path):
    FLOWS.mkdir(parents=True, exist_ok=True)
    output = FLOWS / f"exvivo_pseudo_{subject}.npz"
    if not output.exists():
        network = model(path)
        packed = {}
        for sid, row in truth.items():
            if row["subject"] != subject:
                continue
            image = P.read_image(P.region_path(sid) / "exvivo.tif")
            dp, cp, n = P.cellpose_flows(network, image)
            packed[f"{sid}|dp"] = dp
            packed[f"{sid}|cp"] = cp
            packed[f"{sid}|n"] = np.int32(n)
            log("INFER", sid)
        np.savez_compressed(output, **packed)
        del network
        gc.collect()
        torch.cuda.empty_cache()
    with np.load(output) as packed:
        return {sid: (packed[f"{sid}|dp"], packed[f"{sid}|cp"], int(packed[f"{sid}|n"]))
                for sid, row in truth.items() if row["subject"] == subject}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--per-epoch", type=int, default=128)
    parser.add_argument("--fold", choices=P.SUBJECTS,
                        help="Run a single untouched mouse first; omit for all three folds")
    args = parser.parse_args()
    truth = P.load_truth()
    teacher = model()
    all_flows = {}
    folds = [args.fold] if args.fold else list(P.SUBJECTS)
    for subject in folds:
        path = fit(subject, truth, teacher, args.epochs, args.per_epoch)
        all_flows.update(infer(subject, truth, path))
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    scores = []
    for cp, flow in GRID:
        cfg = {"cellprob": cp, "flow": flow}
        per_mouse = {}
        for subject in folds:
            values = [pq_score(P.cellpose_labels(all_flows[sid], "exvivo", cfg), row["exvivo"][0])[0]
                      for sid, row in truth.items() if row["subject"] == subject]
            per_mouse[subject] = float(np.mean(values))
        pooled = float(np.mean([pq_score(P.cellpose_labels(all_flows[sid], "exvivo", cfg),
                                         row["exvivo"][0])[0]
                                for sid, row in truth.items() if row["subject"] in folds]))
        scores.append({"cellprob": cp, "flow": flow, "pooled_pq": pooled,
                       "by_subject": per_mouse})
        log("GRID", cp, flow, pooled, per_mouse)
    best = max(scores, key=lambda row: row["pooled_pq"])
    output = OUTPUT if not args.fold else WORK / f"pseudo_cv_{args.fold}.json"
    output.write_text(json.dumps({"folds": folds, "best": best, "all": scores}, indent=2))
    log("FINAL", best, "BASELINE", 0.39014881255484807)


if __name__ == "__main__":
    main()
