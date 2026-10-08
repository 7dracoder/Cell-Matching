"""A100 leave-one-mouse-out ex-vivo Cellpose test at moderate background weight.

Run in Colab after the project dataset and pipeline.py are in /content/work.
Each fold's outputs are checkpointed immediately in /content/drive/MyDrive/
cellmatch_models/mid_loss_cv if Drive is mounted. This is a diagnostic only;
it intentionally does not create a test submission without full-score CV.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import models, train

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402
from partial_loss_probe import make_partial_loss  # noqa: E402

WORK = Path("/content/work")
DRIVE = Path("/content/drive/MyDrive/cellmatch_models/mid_loss_cv")
P.CELLPOSE_TILE = 96
GRID = [(cp, flow) for cp in (-0.5, 0.0, 0.25, 0.5, 0.75)
        for flow in (0.08, 0.15, 0.25, 0.4)]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def checkpoint(path: Path) -> None:
    if DRIVE.parent.is_dir():
        DRIVE.mkdir(exist_ok=True)
        shutil.copy2(path, DRIVE / path.name)
        log("DRIVE_SAVED", path.name)


def model(path=None):
    return models.CellposeModel(gpu=True,
                                pretrained_model=str(path) if path else "cpsam_v2",
                                use_bfloat16=torch.cuda.is_bf16_supported())


def train_fold(subject: str, weight: float) -> Path:
    tag = f"mid{str(weight).replace('.', 'p')}_fold_{subject}"
    path = WORK / "models" / f"exvivo_cpsam_{tag}"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        log("REUSE", subject)
        return path
    saved = DRIVE / path.name
    if saved.exists():
        shutil.copy2(saved, path)
        log("RESTORED", subject)
        return path
    others = [s for s in P.SUBJECTS if s != subject]
    images, labels = P.cellpose_tiles("exvivo", others, per_image=100)
    log("TRAIN", subject, "weight", weight, "tiles", len(images))
    train._loss_fn_seg = make_partial_loss(weight)
    net = model().net
    train.train_seg(net, train_data=images, train_labels=labels,
                    normalize=False, rescale=True, batch_size=4, n_epochs=60,
                    nimg_per_epoch=128, learning_rate=1e-5, weight_decay=0.1,
                    min_train_masks=1, save_path=str(WORK), model_name=path.name)
    del net, images, labels
    gc.collect()
    torch.cuda.empty_cache()
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint(path)
    return path


def infer(net, image):
    diameter = float(net.net.diam_labels.item())
    _, output, _ = net.eval(image.astype(np.float32), diameter=diameter,
                            compute_masks=False, batch_size=16)
    return (output[1].astype(np.float16), output[2].astype(np.float16),
            int(200 * diameter / 30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-weight", type=float, default=0.25)
    args = parser.parse_args()
    truth = P.load_truth()
    scores = {}
    for subject in P.SUBJECTS:
        output = WORK / "flows" / f"exvivo_mid{str(args.background_weight).replace('.', 'p')}_{subject}.npz"
        output.parent.mkdir(exist_ok=True)
        if not output.exists() and (DRIVE / output.name).exists():
            shutil.copy2(DRIVE / output.name, output)
        if output.exists():
            packed = np.load(output)
            flows = {sid: (packed[f"{sid}|dp"], packed[f"{sid}|cp"],
                           int(packed[f"{sid}|n"]))
                     for sid in truth if sid.startswith(subject)}
            log("REUSE_FLOWS", subject)
        else:
            path = train_fold(subject, args.background_weight)
            net = model(path)
            flows = {}
            for sid in truth:
                if sid.startswith(subject):
                    image = P.read_image(P.region_path(sid) / "exvivo.tif")
                    flows[sid] = infer(net, image)
            del net
            gc.collect()
            torch.cuda.empty_cache()
            np.savez_compressed(output, **{f"{sid}|{key}": value
                                             for sid, values in flows.items()
                                             for key, value in zip(("dp", "cp", "n"), values)})
            checkpoint(output)
            log("INFERRED", subject, len(flows))
        grid = []
        for cp, flow in GRID:
            values = [pq_score(P.cellpose_labels(flows[sid], "exvivo",
                      {"cellprob": cp, "flow": flow}), truth[sid]["exvivo"][0])[0]
                      for sid in flows]
            grid.append({"cellprob": cp, "flow": flow,
                         "region_pq": {sid: float(v) for sid, v in zip(flows, values)},
                         "fold_pq": float(np.mean(values))})
        scores[subject] = grid
        best = max(grid, key=lambda x: x["fold_pq"])
        log("FOLD", subject, "BEST", best["fold_pq"], best["cellprob"], best["flow"])
        result = WORK / f"mid_loss_{subject}.json"
        result.write_text(json.dumps(grid, indent=2))
        checkpoint(result)
        del flows
        gc.collect()
    pooled = []
    for i, (cp, flow) in enumerate(GRID):
        region_values = [v for subject in P.SUBJECTS
                         for v in scores[subject][i]["region_pq"].values()]
        pooled.append({"cellprob": cp, "flow": flow,
                       "pooled_pq": float(np.mean(region_values)),
                       "by_subject": {s: scores[s][i]["fold_pq"] for s in P.SUBJECTS}})
    best = max(pooled, key=lambda x: x["pooled_pq"])
    result = WORK / "mid_loss_cv.json"
    result.write_text(json.dumps({"background_weight": args.background_weight,
                                  "best": best, "all": pooled}, indent=2))
    checkpoint(result)
    log("FINAL", best, "old_pq", 0.39014881255484807)


if __name__ == "__main__":
    main()
