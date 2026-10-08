"""Leave-one-mouse-out validation for sparse-label Cellpose fine-tuning.

Run in the existing Colab runtime after partial_loss_probe.py. Reuses its
b2ba5e checkpoint and writes held-out flow predictions, not a submission.
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

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402
from partial_loss_probe import partial_loss  # noqa: E402


WORK = Path("/content/work")
P.CELLPOSE_TILE = 96
GRID = [(cp, flow) for cp in (-0.5, 0.0, 0.25, 0.5, 0.75)
        for flow in (0.08, 0.15, 0.25, 0.4)]
BASELINE_BY_MOUSE = {"subject_5d294c": 0.4667,
                     "subject_b2ba5e": 0.2909,
                     "subject_db6b8b": 0.3938}


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def model(path=None):
    return models.CellposeModel(gpu=True,
                                pretrained_model=str(path) if path else "cpsam_v2",
                                use_bfloat16=torch.cuda.is_bf16_supported())


def train_fold(subject):
    path = WORK / "models" / f"exvivo_cpsam_partial96_fold_{subject}"
    if path.exists():
        log("REUSE_PARTIAL", subject)
        return path
    others = [s for s in P.SUBJECTS if s != subject]
    images, labels = P.cellpose_tiles("exvivo", others, per_image=100)
    log("TRAIN_PARTIAL", subject, len(images), "96px tiles 60 epochs")
    train._loss_fn_seg = partial_loss
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
    log("TRAINED_PARTIAL", subject)
    return path


def infer(net, image):
    diameter = float(net.net.diam_labels.item())
    _, output, _ = net.eval(image.astype(np.float32), diameter=diameter,
                            compute_masks=False, batch_size=16)
    return (output[1].astype(np.float16), output[2].astype(np.float16),
            int(200 * diameter / 30))


def main():
    truth = P.load_truth()
    flows = {}
    for subject in P.SUBJECTS:
        path = train_fold(subject)
        net = model(path)
        for sid in truth:
            if sid.startswith(subject):
                image = P.read_image(P.region_path(sid) / "exvivo.tif")
                flows[sid] = infer(net, image)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        log("INFER_PARTIAL", subject, sum(s.startswith(subject) for s in flows))

        scores = []
        for cp, flow in GRID:
            vals = [pq_score(P.cellpose_labels(flows[sid], "exvivo",
                    {"cellprob": cp, "flow": flow}), truth[sid]["exvivo"][0])[0]
                    for sid in truth if sid.startswith(subject)]
            scores.append((float(np.mean(vals)), cp, flow))
        best = max(scores)
        log("FOLD_BEST", subject, "PQ", round(best[0], 4),
            "CP", best[1], "FLOW", best[2],
            "OLD", BASELINE_BY_MOUSE[subject])

    output = WORK / "flows/exvivo_partial96_training.npz"
    output.parent.mkdir(exist_ok=True)
    np.savez_compressed(output, **{f"{sid}|{key}": value
                                  for sid, vals in flows.items()
                                  for key, value in zip(("dp", "cp", "n"), vals)})
    log("FLOWS_SAVED", output)

    results = []
    for cp, flow in GRID:
        by_subject = {}
        for subject in P.SUBJECTS:
            vals = [pq_score(P.cellpose_labels(flows[sid], "exvivo",
                    {"cellprob": cp, "flow": flow}), truth[sid]["exvivo"][0])[0]
                    for sid in truth if sid.startswith(subject)]
            by_subject[subject] = vals
        pooled = float(np.mean([v for vals in by_subject.values() for v in vals]))
        results.append({"cellprob": cp, "flow": flow, "ex_pq": pooled,
                        "by_subject": {k: round(float(np.mean(v)), 4)
                                       for k, v in by_subject.items()}})
        log("GRID", cp, flow, "PQ", round(pooled, 4), results[-1]["by_subject"])
    best = max(results, key=lambda r: r["ex_pq"])
    (WORK / "partial96_cv.json").write_text(json.dumps({"best": best,
                                                          "all": results}, indent=2))
    log("PARTIAL_CV_BEST", best, "baseline", 0.3901)


if __name__ == "__main__":
    main()
