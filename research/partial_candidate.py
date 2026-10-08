"""Validate sparse-label Cellpose on all mice; build a CSV only on a CV gain.

Requires partial_loss_cv.py to have saved all leave-one-mouse-out flows in
/content/work/flows/exvivo_partial96_training.npz. Run on Colab with the
existing v7 held-out masks, v8 support scripts, and best submission uploaded.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from cellpose import train

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
import v8_colab as V  # noqa: E402
from partial_loss_cv import model, infer  # noqa: E402
from partial_loss_probe import make_partial_loss  # noqa: E402
from cellmatch import labels_to_rles, pq_score  # noqa: E402


WORK = Path("/content/work")
BASE_MASKS = Path(os.environ.get("CANDIDATE_BASE_MASKS",
                                 str(WORK / "research/data/base_heldout_labels.npz")))
FLOW_FILE = Path(os.environ.get("CANDIDATE_FLOW_FILE",
                                str(WORK / "flows/exvivo_partial96_training.npz")))
EVAL_FILE = Path(os.environ.get("CANDIDATE_EVAL_FILE",
                                str(WORK / "partial96_cv.json")))
BASELINE_CSV = Path("/content/submission_v7_grow15.csv")
OUT = Path(os.environ.get("CANDIDATE_OUT",
                          str(WORK / "submission_partial96_candidate.csv")))
RUN_TAG = os.environ.get("CANDIDATE_TAG", "partial96")
BACKGROUND_WEIGHT = float(os.environ.get("CANDIDATE_BACKGROUND_WEIGHT", "0.06"))
P.CELLPOSE_TILE = 96


def linked(pred, gt):
    a, b = int(pred.max()), int(gt.max())
    if not a or not b:
        return set()
    joint = np.bincount((pred.astype(np.int64) * (b + 1) + gt).ravel(),
                        minlength=(a + 1) * (b + 1)).reshape(a + 1, b + 1)
    pa, ga = joint.sum(1), joint.sum(0)
    i, j = np.nonzero(joint[1:, 1:])
    iou = joint[i + 1, j + 1] / (pa[i + 1] + ga[j + 1] - joint[i + 1, j + 1])
    return set((j[iou > 0.75] + 1).tolist())


def train_all():
    path = WORK / "models" / f"exvivo_cpsam_{RUN_TAG}_all"
    if path.exists():
        print("REUSE_ALL", path, flush=True)
        return path
    images, labels = P.cellpose_tiles("exvivo", P.SUBJECTS, per_image=100)
    print("TRAIN_ALL_PARTIAL", RUN_TAG, BACKGROUND_WEIGHT, len(images),
          "96px tiles 60 epochs", flush=True)
    train._loss_fn_seg = make_partial_loss(BACKGROUND_WEIGHT)
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
    return path


def test_masks(path, config):
    net = model(path)
    rows = list(csv.DictReader(BASELINE_CSV.open(newline="")))
    for row in rows:
        sid = row["sample_id"]
        image = P.read_image(P.region_path(sid, "hidden_test") / "exvivo.tif")
        labels = P.cellpose_labels(infer(net, image), "exvivo", config)
        row["exvivo_instances"] = json.dumps(labels_to_rles(labels, "EXP"))
        row["match_pairs"] = "[]"
        print("TEST_MASKS", sid, int(labels.max()), flush=True)
    del net
    torch.cuda.empty_cache()
    path = WORK / "masks_partial96.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    for path in (BASE_MASKS, FLOW_FILE, EVAL_FILE, BASELINE_CSV):
        if not path.exists():
            raise FileNotFoundError(path)
    truth = P.load_truth()
    base = np.load(BASE_MASKS)
    iv = {sid: base[f"{sid}|invivo"].astype(np.int16) for sid in truth}
    ivprob = {sid: base[f"{sid}|invivo|prob"] for sid in truth}
    ivpq = float(np.mean([pq_score(iv[s], truth[s]["invivo"][0])[0]
                          for s in truth]))
    ivhit = {sid: {truth[sid]["invivo"][1][i - 1]
                  for i in linked(iv[sid], truth[sid]["invivo"][0])}
             for sid in truth}

    packed = np.load(FLOW_FILE)
    flows = {sid: (packed[f"{sid}|dp"], packed[f"{sid}|cp"],
                   int(packed[f"{sid}|n"])) for sid in truth}
    evals = json.loads(EVAL_FILE.read_text())["all"]
    scored = []
    for item in evals:
        cfg = {"cellprob": item["cellprob"], "flow": item["flow"]}
        reachable = 0
        for sid, t in truth.items():
            pred = P.cellpose_labels(flows[sid], "exvivo", cfg)
            ehit = {t["exvivo"][1][i - 1]
                    for i in linked(pred, t["exvivo"][0])}
            reachable += sum(a in ivhit[sid] and b in ehit for a, b in t["pairs"])
        scored.append((item, reachable))
        print("REACH", cfg, "PQ", round(item["ex_pq"], 4),
              "PAIRS", reachable, flush=True)
    bestpq = max(scored, key=lambda x: x[0]["ex_pq"])[0]
    eligible = [(item, reach) for item, reach in scored
                if item["ex_pq"] >= bestpq["ex_pq"] - 0.015]
    bestreach = max(eligible, key=lambda x: (x[1], x[0]["ex_pq"]))[0]
    configs = [bestpq]
    if (bestreach["cellprob"], bestreach["flow"]) != (
            bestpq["cellprob"], bestpq["flow"]):
        configs.append(bestreach)
    del packed, base
    gc.collect()

    results = []
    for item in configs:
        cfg = (item["cellprob"], item["flow"])
        V.save_heldout(truth, iv, ivprob, flows, cfg)
        f1, threshold = V.rematch_cv()
        combined = 0.25 * ivpq + 0.25 * item["ex_pq"] + 0.5 * f1
        result = {"iv_pq": ivpq, "ex_pq": item["ex_pq"],
                  "matching_f1": f1, "combined_cv": combined,
                  "cellprob": cfg[0], "flow": cfg[1],
                  "pair_threshold": threshold}
        print("PARTIAL_FULL_CV", result, flush=True)
        results.append(result)
    best = max(results, key=lambda r: r["combined_cv"])
    (WORK / "partial96_full_cv.json").write_text(json.dumps({"best": best,
                                                               "all": results}, indent=2))
    # The old estimate is about 0.5125; require a real margin before spending
    # GPU time on an all-mice model and hidden-test inference.
    if best["ex_pq"] <= 0.3974 or best["combined_cv"] <= 0.5200:
        print("NO_CANDIDATE", best, flush=True)
        return

    if best != results[-1]:
        V.save_heldout(truth, iv, ivprob, flows,
                       (best["cellprob"], best["flow"]))
        V.rematch_cv()  # restore registration and pair-classifier train data
    del flows, iv, ivprob
    gc.collect()
    path = train_all()
    masks = test_masks(path, {"cellprob": best["cellprob"], "flow": best["flow"]})
    env = os.environ.copy()
    env["MASKS_CSV"] = str(masks)
    env["REG_TRAIN"] = str(WORK / "research/data/reg_window_vote5.pkl")
    subprocess.run([sys.executable, WORK / "research/test_apply.py", str(OUT),
                    f"{best['pair_threshold']:.4f}", "3"],
                   cwd=WORK / "research", env=env, check=True)
    subprocess.run([sys.executable, "/content/validate_submission.py", str(OUT)],
                   check=True)
    current = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if current == hashlib.sha256(BASELINE_CSV.read_bytes()).hexdigest():
        raise RuntimeError("Candidate is identical to baseline")
    print("NEW_CANDIDATE", OUT, "SHA256", current,
          "Kaggle score unverified", flush=True)


if __name__ == "__main__":
    main()
