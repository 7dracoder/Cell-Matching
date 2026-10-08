"""Validate pseudo-labelled Cellpose against the complete competition score.

Run after ``pseudo_cv_colab.py`` in the existing Colab runtime. This deliberately
does not create a submission unless leave-one-mouse-out validation beats the
previous pipeline. Unmatched cells are never treated as negative pair labels.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from cellpose import train

WORK = Path("/content/work")
sys.path.insert(0, str(WORK))
sys.path.insert(0, "/content")
import pipeline as P  # noqa: E402
import pseudo_cv_colab as S  # noqa: E402
import v8_colab as V  # noqa: E402
from cellmatch import labels_to_rles, pq_score  # noqa: E402

BASELINE = Path("/content/submission_v7_grow15.csv")
OUT = WORK / "submission_pseudo_candidate.csv"
OUT_CP = WORK / "submission_pseudo_candidate_cpgate.csv"
TEST_CP = WORK / "test_cp_pseudo.npz"
BASELINE_CV = 0.5125
MIN_GAIN = 0.0075


def load_heldout(truth):
    flows = {}
    for subject in P.SUBJECTS:
        path = WORK / "flows" / f"exvivo_pseudo_{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as packed:
            for sid, row in truth.items():
                if row["subject"] == subject:
                    flows[sid] = (packed[f"{sid}|dp"], packed[f"{sid}|cp"],
                                  int(packed[f"{sid}|n"]))
    return flows


def selected_configs(flows, truth, iv):
    iv_hit = {}
    for sid, row in truth.items():
        gt, ids = row["invivo"]
        iv_hit[sid] = {ids[i - 1] for i in V.linked(iv[sid], gt)}
    ranked = []
    for cp, flow in S.GRID:
        cfg = {"cellprob": cp, "flow": flow}
        scores, reachable, cells = [], 0, 0
        for sid, row in truth.items():
            pred = P.cellpose_labels(flows[sid], "exvivo", cfg)
            gt, ids = row["exvivo"]
            scores.append(pq_score(pred, gt)[0])
            ex_hit = {ids[i - 1] for i in V.linked(pred, gt)}
            reachable += sum(a in iv_hit[sid] and b in ex_hit for a, b in row["pairs"])
            cells += int(pred.max())
        pq = float(np.mean(scores))
        proxy = 0.4 * pq + 0.6 * reachable / 1139
        item = {"cfg": cfg, "ex_pq": pq, "reachable_pairs": reachable,
                "pred_cells": cells, "proxy": proxy}
        ranked.append(item)
        print("MASK", item, flush=True)
    best_pq = max(ranked, key=lambda x: x["ex_pq"])
    best_proxy = max(ranked, key=lambda x: x["proxy"])
    return [best_pq] + ([best_proxy] if best_proxy != best_pq else [])


def fit_all(truth):
    path = S.MODELS / "exvivo_cpsam_pseudo_all"
    if path.exists():
        return path
    teacher = S.model()
    images, labels = S.make_tiles(P.SUBJECTS, truth, teacher)
    print("TRAIN_ALL", len(images), "tiles", flush=True)
    del teacher
    torch.cuda.empty_cache()
    net = S.model().net
    train.train_seg(net, train_data=images, train_labels=labels,
                    normalize=False, rescale=True, batch_size=4,
                    n_epochs=60, nimg_per_epoch=128, learning_rate=1e-5,
                    weight_decay=0.1, min_train_masks=1,
                    save_path=str(WORK), model_name=path.name)
    del net, images, labels
    gc.collect()
    torch.cuda.empty_cache()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def test_masks(path, cfg):
    net = S.model(path)
    with BASELINE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cps = {}
    for row in rows:
        sid = row["sample_id"]
        image = P.read_image(P.region_path(sid, "hidden_test") / "exvivo.tif")
        flows = P.cellpose_flows(net, image)
        labels = P.cellpose_labels(flows, "exvivo", cfg)
        cps[f"{sid}|cp"] = flows[1]
        row["exvivo_instances"] = json.dumps(labels_to_rles(labels, "EXP"))
        row["match_pairs"] = "[]"
        print("TEST_MASKS", sid, int(labels.max()), flush=True)
    np.savez_compressed(TEST_CP, **cps)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    masks = WORK / "masks_pseudo_candidate.csv"
    with masks.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return masks


def main():
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)
    if not (V.RESEARCH / 'lab_build.py').exists():
        with zipfile.ZipFile('/content/v8_support.zip') as archive:
            archive.extractall(WORK)
    V.DATA.mkdir(parents=True, exist_ok=True)
    truth = P.load_truth()
    flows = load_heldout(truth)
    old_iv = WORK / 'flows/invivo_base_aug1_training.npz'
    if old_iv.exists():
        iv, ivprob = V.existing_iv(truth)
    else:
        cache = Path('/content/iv_baseline_cache.npz')
        if not cache.exists():
            raise FileNotFoundError(cache)
        with np.load(cache) as packed:
            iv = {sid: packed[f'{sid}|invivo'].astype(np.int16) for sid in truth}
            ivprob = {sid: packed[f'{sid}|invivo|prob'] for sid in truth}
    iv_pq = float(np.mean([pq_score(iv[sid], row["invivo"][0])[0]
                           for sid, row in truth.items()]))
    results = []
    for candidate in selected_configs(flows, truth, iv):
        cfg = candidate["cfg"]
        V.save_heldout(truth, iv, ivprob, flows, (cfg["cellprob"], cfg["flow"]))
        f1, threshold = V.rematch_cv()
        full_cv = 0.25 * (iv_pq + candidate["ex_pq"]) + 0.5 * f1
        result = {**candidate, "iv_pq": iv_pq, "matching_f1": f1,
                  "pair_threshold": threshold, "full_cv": full_cv}
        results.append(result)
        print("FULL_CV", result, flush=True)
    winner = max(results, key=lambda x: x["full_cv"])
    (WORK / "pseudo_full_cv.json").write_text(json.dumps({"candidates": results,
                                                            "winner": winner}, indent=2))
    if winner["full_cv"] <= BASELINE_CV + MIN_GAIN:
        print("NO_CANDIDATE: full held-out score did not improve enough", flush=True)
        return
    cfg = winner["cfg"]
    V.save_heldout(truth, iv, ivprob, flows, (cfg["cellprob"], cfg["flow"]))
    V.rematch_cv()  # rebuild matching inputs for the winning mask variant
    masks = test_masks(fit_all(truth), cfg)
    env = os.environ.copy()
    env["MASKS_CSV"] = str(masks)
    env["REG_TRAIN"] = str(V.DATA / "reg_window_vote5.pkl")
    subprocess.run([sys.executable, V.RESEARCH / "test_apply.py", str(OUT),
                    f"{winner['pair_threshold']:.4f}", "3"],
                   cwd=V.RESEARCH, env=env, check=True)
    subprocess.run([sys.executable, WORK / "validate_submission.py", str(OUT)], check=True)
    if hashlib.sha256(OUT.read_bytes()).digest() == hashlib.sha256(BASELINE.read_bytes()).digest():
        raise RuntimeError("New candidate is byte-identical to the baseline")
    print("CANDIDATE_READY", OUT, "public score unverified", flush=True)
    # Same masks, plus regions whose pose is confirmed by the ex-vivo cellprob evidence (z >= 5).
    env["WRITE_CSV"] = str(masks)
    env["CP_NPZ"] = str(TEST_CP)
    subprocess.run([sys.executable, V.RESEARCH / "test_cp_apply.py", str(OUT_CP),
                    f"{winner['pair_threshold']:.4f}", "3", "5"],
                   cwd=V.RESEARCH, env=env, check=True)
    subprocess.run([sys.executable, WORK / "validate_submission.py", str(OUT_CP)], check=True)
    print("CANDIDATE_READY", OUT_CP, "public score unverified", flush=True)


if __name__ == "__main__":
    main()
