"""Leave-one-mouse-out ex-vivo Cellpose-SAM with 64px cell-centred crops.

Diagnostic only: trains fold checkpoints and prints held-out PQ/reachable pairs.
Unlike v8, it never copies or emits a submission when validation is weak.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import models, train

sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402

WORK = Path("/content/work")
MODELS = WORK / "models"
CACHE = WORK / "flows"
P.CELLPOSE_TILE = 64
EPOCHS = 80
PER_EPOCH = 128
GRID = [(c, f) for c in (-0.5, 0.0, 0.25, 0.5, 0.75)
        for f in (0.08, 0.15, 0.25, 0.4)]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def model(path=None):
    return models.CellposeModel(gpu=True, pretrained_model=str(path) if path else "cpsam_v2",
                                use_bfloat16=torch.cuda.is_bf16_supported())


def train_fold(subject, truth):
    path = MODELS / f"exvivo_cpsam_tight_fold_{subject}"
    if path.exists():
        log("REUSE", path)
        return path
    others = [s for s in P.SUBJECTS if s != subject]
    images, labels = P.cellpose_tiles("exvivo", others, per_image=100)
    log("TRAIN", subject, len(images), "64px tiles", EPOCHS, "epochs")
    net = model().net
    train.train_seg(net, train_data=images, train_labels=labels, normalize=False,
                    rescale=True, batch_size=4, n_epochs=EPOCHS, nimg_per_epoch=PER_EPOCH,
                    learning_rate=1e-5, weight_decay=0.1, min_train_masks=1,
                    save_path=str(WORK), model_name=path.name)
    del net, images, labels
    gc.collect()
    torch.cuda.empty_cache()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def infer(net, image):
    d = float(net.net.diam_labels.item())
    _, out, _ = net.eval(image.astype(np.float32), diameter=d,
                         compute_masks=False, batch_size=16)
    return out[1].astype(np.float16), out[2].astype(np.float16), int(200 * d / 30)


def linked(pred, gt):
    a, b = int(pred.max()), int(gt.max())
    if not a or not b:
        return set()
    z = np.bincount((pred.astype(np.int64) * (b + 1) + gt).ravel(),
                    minlength=(a + 1) * (b + 1)).reshape(a + 1, b + 1)
    pc, gc = z.sum(1), z.sum(0)
    i, j = np.nonzero(z[1:, 1:])
    iou = z[i + 1, j + 1] / (pc[i + 1] + gc[j + 1] - z[i + 1, j + 1])
    return set((j[iou > .75] + 1).tolist())


def main():
    truth = P.load_truth()
    out = {}
    for subject in P.SUBJECTS:
        path = train_fold(subject, truth)
        net = model(path)
        for sid in truth:
            if sid.startswith(subject):
                image = P.read_image(P.region_path(sid) / "exvivo.tif")
                out[sid] = infer(net, image)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        log("INFER", subject, len(out), "regions")
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "exvivo_tight_training.npz"
    np.savez(path, **{f"{sid}|{key}": value for sid, values in out.items()
                      for key, value in zip(("dp", "cp", "n"), values)})
    log("FLOWS_SAVED", path)

    iv = np.load(CACHE / "invivo_base_aug1_training.npz")
    ivhit = {}
    for sid, item in truth.items():
        flows = (iv[f"{sid}|dp"], iv[f"{sid}|cp"], int(iv[f"{sid}|n"]))
        pred = P.cellpose_labels(flows, "invivo", {"cellprob": 0.0, "flow": 0.2})
        gt, ids = item["invivo"]
        ivhit[sid] = {ids[i - 1] for i in linked(pred, gt)}
    results = []
    for cp, flow in GRID:
        pq, reach, count = [], 0, 0
        for sid, item in truth.items():
            pred = P.cellpose_labels(out[sid], "exvivo", {"cellprob": cp, "flow": flow})
            gt, ids = item["exvivo"]
            pq.append(pq_score(pred, gt)[0])
            ehit = {ids[i - 1] for i in linked(pred, gt)}
            reach += sum(a in ivhit[sid] and b in ehit for a, b in item["pairs"])
            count += int(pred.max())
        score = float(np.mean(pq))
        results.append({"cellprob": cp, "flow": flow, "ex_pq": score,
                        "reachable_pairs": reach, "pred_cells": count})
        log("GRID", cp, flow, "EX_PQ", round(score, 4), "REACH", reach, "CELLS", count)
    best = max(results, key=lambda r: r["ex_pq"])
    (WORK / "tight_ex_eval.json").write_text(json.dumps({"best": best, "all": results}, indent=2))
    log("BEST", best, "baseline_ex_pq", 0.3974)


if __name__ == "__main__":
    main()
