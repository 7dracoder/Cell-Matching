"""Single-mouse probe: downweight unlabeled background in Cellpose training.

Ground truth may omit real cells, so treating every zero pixel as a definite
negative can suppress recall. This is a diagnostic on the held-out b2ba5e
mouse; it never makes a test submission.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from cellpose import train

sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402

SUBJECT = "subject_b2ba5e"
OUT = Path("/content/work/models/exvivo_cpsam_partial96_fold_" + SUBJECT)
P.CELLPOSE_TILE = 96


def make_partial_loss(background_weight: float):
    """Build a loss with reduced weight on potentially unannotated cells."""
    if not 0 < background_weight <= 1:
        raise ValueError("background_weight must be in (0, 1]")

    def loss(lbl, pred, device):
        pos = (lbl[:, -3] > 0.5).to(pred.dtype)
        near = F.max_pool2d(pos[:, None], kernel_size=19, stride=1, padding=9)[:, 0]
        weight = background_weight + (1.0 - background_weight) * near
        norm = weight.mean().clamp_min(0.05)
        flow = (pred[:, -3:-1] - 5.0 * lbl[:, -2:]).square()
        flow_loss = (flow * weight[:, None]).mean() / (2.0 * norm)
        binary = F.binary_cross_entropy_with_logits(pred[:, -1], pos, reduction="none")
        binary_loss = (binary * weight).mean() / norm
        return flow_loss + binary_loss

    return loss


partial_loss = make_partial_loss(0.06)


def main():
    from cellpose import models

    train._loss_fn_seg = partial_loss
    if not OUT.exists():
        others = [s for s in P.SUBJECTS if s != SUBJECT]
        images, labels = P.cellpose_tiles("exvivo", others, per_image=100)
        print(time.strftime("%H:%M:%S"), "TRAIN_PARTIAL", SUBJECT,
              len(images), "96px tiles 60 epochs", flush=True)
        net = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2",
                                    use_bfloat16=torch.cuda.is_bf16_supported()).net
        train.train_seg(net, train_data=images, train_labels=labels,
                        normalize=False, rescale=True, batch_size=4,
                        n_epochs=60, nimg_per_epoch=128, learning_rate=1e-5,
                        weight_decay=0.1, min_train_masks=1,
                        save_path="/content/work", model_name=OUT.name)
        del net, images, labels
        gc.collect()
        torch.cuda.empty_cache()
    if not OUT.exists():
        raise FileNotFoundError(OUT)

    net = models.CellposeModel(gpu=True, pretrained_model=str(OUT),
                                use_bfloat16=torch.cuda.is_bf16_supported())
    truth = P.load_truth()
    flows = {}
    for sid in truth:
        if sid.startswith(SUBJECT):
            image = P.read_image(P.region_path(sid) / "exvivo.tif")
            diameter = float(net.net.diam_labels.item())
            _, output, _ = net.eval(image.astype(np.float32), diameter=diameter,
                                    compute_masks=False, batch_size=16)
            flows[sid] = (output[1].astype(np.float16), output[2].astype(np.float16),
                          int(200 * diameter / 30))
    del net
    torch.cuda.empty_cache()
    print(time.strftime("%H:%M:%S"), "INFER_PARTIAL", len(flows), flush=True)

    for cp in (-0.5, 0.0, 0.25, 0.5, 0.75):
        for flow in (0.08, 0.15, 0.25, 0.4):
            scores, cells = [], 0
            for sid, values in flows.items():
                pred = P.cellpose_labels(values, "exvivo", {"cellprob": cp, "flow": flow})
                scores.append(pq_score(pred, truth[sid]["exvivo"][0])[0])
                cells += int(pred.max())
            print("PARTIAL_GRID", cp, flow, "PQ", round(float(np.mean(scores)), 4),
                  "CELLS", cells, flush=True)


if __name__ == "__main__":
    main()
