"""Colab A100 experiment: CellposeDINO ex-vivo segmentation and full rematching.

Only a leave-one-mouse-out CV improvement can create submission_v8.csv.
No hidden labels are used. A failed experiment produces no new submission.
Run from a Colab notebook with::

    !python -u /content/v8_colab.py

Requires the existing /content/work dataset, pipeline modules and CellposeDINO
dependency, plus v8_support.zip and submission_v7_grow15.csv uploaded to /content.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

WORK = Path("/content/work")
sys.path.insert(0, str(WORK))
import pipeline as P  # noqa: E402
from cellmatch import labels_to_rles, pq_score  # noqa: E402
from cellpose import models, train  # noqa: E402

SRC = Path("/content/submission_v7_grow15.csv")
OUT = WORK / "submission_v8.csv"
SUPPORT = Path("/content/v8_support.zip")
MODELS = WORK / "models"
RESEARCH = WORK / "research"
DATA = RESEARCH / "data"
BACKBONE = "cpdino-vitb"
EPOCHS = int(os.environ.get("V8_EPOCHS", "60"))
PER_EPOCH = int(os.environ.get("V8_PER_EPOCH", "64"))
BASELINE_CV = 0.5125  # .25*.74 + .25*.39 + .5*.46, from held-out v7
GRID = [(c, f) for c in (-0.5, 0.0, 0.5, 1.0) for f in (0.1, 0.25)]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def model(path: Path | None = None):
    return models.CellposeModel(gpu=True, pretrained_model=str(path) if path else BACKBONE,
                                use_bfloat16=torch.cuda.is_bf16_supported())


def train_fold(subjects, name: str):
    path = MODELS / name
    if path.exists():
        log("reuse", path)
        return path
    images, labels = P.cellpose_tiles("exvivo", subjects, per_image=40)
    log("train", name, "tiles", len(images), "epochs", EPOCHS)
    net = model().net
    train.train_seg(net, train_data=images, train_labels=labels, normalize=False,
                    rescale=True, batch_size=4, n_epochs=EPOCHS,
                    nimg_per_epoch=PER_EPOCH, learning_rate=1e-5,
                    weight_decay=0.1, min_train_masks=1,
                    save_path=str(WORK), model_name=name)
    del net, images, labels
    gc.collect()
    torch.cuda.empty_cache()
    if not path.exists():
        raise FileNotFoundError(path)
    log("trained", path)
    return path


def infer_flows(net, image):
    diameter = float(net.net.diam_labels.item())
    _, output, _ = net.eval(image.astype(np.float32), diameter=diameter,
                            compute_masks=False, batch_size=8)
    return (output[1].astype(np.float16), output[2].astype(np.float16),
            int(200 * diameter / 30))


def decode(flows, modality, cfg):
    return P.cellpose_labels(flows, modality, {"cellprob": cfg[0], "flow": cfg[1]})


def linked(pred, gt):
    """Ground-truth labels reached at the competition's strict IoU > .75."""
    npred, ngt = int(pred.max()), int(gt.max())
    if not npred or not ngt:
        return set()
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc, gc = joint.sum(1), joint.sum(0)
    r, c = np.nonzero(joint[1:, 1:])
    iou = joint[r + 1, c + 1] / (pc[r + 1] + gc[c + 1] - joint[r + 1, c + 1])
    return set((c[iou > 0.75] + 1).tolist())


def fold_flows(truth):
    result = {}
    for subject in P.SUBJECTS:
        others = [s for s in P.SUBJECTS if s != subject]
        path = train_fold(others, f"exvivo_cpdino_fold_{subject}")
        net = model(path)
        for sid in (s for s in truth if s.startswith(subject)):
            image = P.read_image(P.region_path(sid) / "exvivo.tif")
            result[sid] = infer_flows(net, image)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        log("held-out inference", subject, len(result), "regions")
    return result


def existing_iv(truth):
    cache = np.load(WORK / "flows/invivo_base_aug1_training.npz")
    labels, probs = {}, {}
    for sid in truth:
        flows = (cache[f"{sid}|dp"], cache[f"{sid}|cp"], int(cache[f"{sid}|n"]))
        labels[sid] = decode(flows, "invivo", (0, 0.2)).astype(np.int16)
        probs[sid] = flows[1]
    return labels, probs


def tune(flows, truth, iv):
    iv_hit = {}
    for sid, item in truth.items():
        gt, ids = item["invivo"]
        iv_hit[sid] = {ids[i - 1] for i in linked(iv[sid], gt)}
    rows = []
    for cfg in GRID:
        pqs, reachable, cells = [], 0, 0
        for sid, item in truth.items():
            pred = decode(flows[sid], "exvivo", cfg)
            gt, ids = item["exvivo"]
            pqs.append(pq_score(pred, gt)[0])
            ex_hit = {ids[i - 1] for i in linked(pred, gt)}
            reachable += sum(a in iv_hit[sid] and b in ex_hit for a, b in item["pairs"])
            cells += int(pred.max())
        pq = float(np.mean(pqs))
        # Recall-sensitive proxy: choose from real held-out labels, not hidden data.
        proxy = 0.40 * pq + 0.60 * reachable / 1139
        rows.append((proxy, pq, reachable, cells, cfg))
        log("GRID", cfg, "EX_PQ", round(pq, 4), "reachable", reachable,
            "pred_cells", cells, "proxy", round(proxy, 4))
    return max(rows)[-1]


def save_heldout(truth, iv, ivprob, exflows, cfg):
    data = {}
    for sid in truth:
        data[f"{sid}|invivo"] = iv[sid]
        data[f"{sid}|invivo|prob"] = ivprob[sid]
        data[f"{sid}|exvivo"] = decode(exflows[sid], "exvivo", cfg).astype(np.int16)
        data[f"{sid}|exvivo|prob"] = exflows[sid][1]
    np.savez_compressed(DATA / "heldout_labels.npz", **data)
    log("saved held-out labels", DATA / "heldout_labels.npz")


def command(args, *, cwd=WORK, env=None):
    log("run", " ".join(map(str, args)))
    result = subprocess.run(args, cwd=cwd, env=env, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    print(result.stdout, flush=True)
    return result.stdout


def rematch_cv():
    command([sys.executable, RESEARCH / "lab_build.py"])
    command([sys.executable, RESEARCH / "vote_lab.py"])
    output = command([sys.executable, RESEARCH / "win5_lab.py"], cwd=RESEARCH)
    matches = re.findall(r"margin>=3 F1 ([0-9.]+) thr ([0-9.]+)", output)
    if not matches:
        raise ValueError("Could not parse matching CV result")
    f1, threshold = map(float, matches[-1])
    return f1, threshold


def test_masks(cfg):
    final = train_fold(P.SUBJECTS, "exvivo_cpdino_all")
    net = model(final)
    rows = list(csv.DictReader(SRC.open(newline="")))
    for row in rows:
        sid = row["sample_id"]
        image = P.read_image(P.region_path(sid, "hidden_test") / "exvivo.tif")
        labels = decode(infer_flows(net, image), "exvivo", cfg)
        row["exvivo_instances"] = json.dumps(labels_to_rles(labels, "EXD"))
        row["match_pairs"] = "[]"
        log("test masks", sid, "cells", int(labels.max()))
    path = WORK / "masks_v8_dino.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    del net
    torch.cuda.empty_cache()
    return path


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Upload the current best submission first: {SRC}")
    if not (P.ROOT / "training/train_ground_truth.csv").exists():
        raise FileNotFoundError("The existing Colab dataset is missing")
    with zipfile.ZipFile(SUPPORT) as archive:
        archive.extractall(WORK)
    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    log("baseline available", SRC, "bytes", SRC.stat().st_size)

    truth = P.load_truth()
    iv, ivprob = existing_iv(truth)
    iv_pq = float(np.mean([pq_score(iv[s], truth[s]["invivo"][0])[0] for s in truth]))
    log("existing held-out IV_PQ", round(iv_pq, 4))
    exflows = fold_flows(truth)
    cfg = tune(exflows, truth, iv)
    ex_pq = float(np.mean([pq_score(decode(exflows[s], "exvivo", cfg),
                                    truth[s]["exvivo"][0])[0] for s in truth]))
    log("selected EX config", cfg, "EX_PQ", round(ex_pq, 4))
    save_heldout(truth, iv, ivprob, exflows, cfg)
    del exflows
    gc.collect()

    f1, threshold = rematch_cv()
    cv = 0.25 * iv_pq + 0.25 * ex_pq + 0.5 * f1
    log("DINO held-out full score", round(cv, 4), "matching_F1", round(f1, 4),
        "baseline_estimate", BASELINE_CV)
    (WORK / "v8_cv.json").write_text(json.dumps({"cv_score": cv, "iv_pq": iv_pq,
        "ex_pq": ex_pq, "matching_f1": f1, "cellprob": cfg[0],
        "flow": cfg[1], "pair_threshold": threshold}, indent=2))
    if cv <= BASELINE_CV + 0.005:
        log("No convincing held-out improvement; no new submission generated")
        return

    masks = test_masks(cfg)
    candidate = WORK / "submission_v8_dino_candidate.csv"
    env = os.environ.copy()
    env["MASKS_CSV"] = str(masks)
    env["REG_TRAIN"] = str(DATA / "reg_window_vote5.pkl")
    command([sys.executable, RESEARCH / "test_apply.py", candidate,
             f"{threshold:.4f}", "3"], cwd=RESEARCH, env=env)
    command([sys.executable, WORK / "validate_submission.py", candidate])
    shutil.copy2(candidate, OUT)
    log("NEW CANDIDATE READY", OUT, "CV", round(cv, 4),
        "Kaggle score still unverified")


if __name__ == "__main__":
    main()
