"""Make a *new* tight-crop candidate only if leave-one-mouse-out CV improves.

Run after tight_ex_colab.py in the existing Colab runtime. Never writes a
baseline copy under a new name. A positive validation gate triggers an
all-training-mice model, hidden inference, rematching, and CSV validation.
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

sys.path.insert(0, "/content/work")
sys.path.insert(0, "/content")
import pipeline as P  # noqa: E402
import tight_ex_colab as T  # noqa: E402
import v8_colab as V  # noqa: E402
from cellmatch import labels_to_rles, pq_score  # noqa: E402

WORK = Path("/content/work")
EVAL = WORK / "tight_ex_eval.json"
FLOWS = WORK / "flows/exvivo_tight_training.npz"
BASE_TRAIN = WORK / "flows/exvivo_base+_v2_aug0_training.npz"
BASE_TEST = WORK / "flows/exvivo_base+_v2_aug0_hidden_test.npz"
BASELINE = Path("/content/submission_v7_grow15.csv")
OUT = WORK / "submission_v9_tight_candidate.csv"
EX_PQ_GATE = 0.35
FULL_CV_GATE = 0.525


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_flows(truth):
    cache = np.load(FLOWS)
    return {sid: (cache[f"{sid}|dp"], cache[f"{sid}|cp"], int(cache[f"{sid}|n"]))
            for sid in truth}


def load_base(path, ids):
    cache = np.load(path)
    return {sid: (cache[f"{sid}|dp"], cache[f"{sid}|cp"], int(cache[f"{sid}|n"]))
            for sid in ids}


def blend(base, tight, alpha):
    return ((base[0].astype(np.float32) * (1 - alpha) +
             tight[0].astype(np.float32) * alpha).astype(np.float16),
            (base[1].astype(np.float32) * (1 - alpha) +
             tight[1].astype(np.float32) * alpha).astype(np.float16),
            round((1 - alpha) * base[2] + alpha * tight[2]))


def choose_source(tight, truth, best):
    """Try flow-level ensembles, scored only on leave-one-mouse-out labels."""
    chosen = dict(best, source="tight", alpha=1.0)
    base = load_base(BASE_TRAIN, truth)
    grid = [(cp, flow) for cp in (0.0, 0.25, 0.5)
            for flow in (0.1, 0.15, 0.25, 0.4)]
    for alpha in (0.25, 0.5, 0.75):
        merged = {sid: blend(base[sid], tight[sid], alpha) for sid in truth}
        for cp, flow in grid:
            scores = [pq_score(P.cellpose_labels(merged[sid], "exvivo",
                        {"cellprob": cp, "flow": flow}), truth[sid]["exvivo"][0])[0]
                      for sid in truth]
            pq = float(np.mean(scores))
            print("BLEND", alpha, cp, flow, "EX_PQ", round(pq, 4), flush=True)
            if pq > chosen["ex_pq"]:
                chosen = {"source": "blend", "alpha": alpha,
                          "cellprob": cp, "flow": flow, "ex_pq": pq}
        del merged
    return chosen, base


def fit_all():
    path = WORK / "models/exvivo_cpsam_tight_all"
    if path.exists():
        print("REUSE", path, flush=True)
        return path
    P.CELLPOSE_TILE = 64
    images, labels = P.cellpose_tiles("exvivo", P.SUBJECTS, per_image=100)
    print("TRAIN_ALL", len(images), "tiles", T.EPOCHS, "epochs", flush=True)
    net = T.model().net
    train.train_seg(net, train_data=images, train_labels=labels, normalize=False,
                    rescale=True, batch_size=4, n_epochs=T.EPOCHS,
                    nimg_per_epoch=T.PER_EPOCH, learning_rate=1e-5,
                    weight_decay=0.1, min_train_masks=1,
                    save_path=str(WORK), model_name=path.name)
    del net, images, labels
    gc.collect()
    torch.cuda.empty_cache()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def infer_test(path, cfg, source, alpha):
    net = T.model(path)
    rows = list(csv.DictReader(BASELINE.open(newline="")))
    base = load_base(BASE_TEST, [row["sample_id"] for row in rows]) if source == "blend" else {}
    for row in rows:
        sid = row["sample_id"]
        img = P.read_image(P.region_path(sid, "hidden_test") / "exvivo.tif")
        flow = T.infer(net, img)
        if source == "blend":
            flow = blend(base[sid], flow, alpha)
        labels = P.cellpose_labels(flow, "exvivo", cfg)
        row["exvivo_instances"] = json.dumps(labels_to_rles(labels, "EXT"))
        row["match_pairs"] = "[]"
        print("TEST", sid, int(labels.max()), flush=True)
    del net
    torch.cuda.empty_cache()
    masks = WORK / "masks_v9_tight.csv"
    with masks.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return masks


def main():
    if not EVAL.exists() or not FLOWS.exists():
        raise FileNotFoundError("Complete tight_ex_colab.py first")
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)
    mask_eval = json.loads(EVAL.read_text())
    best = mask_eval["best"]
    print("BEST_MASK_CONFIG", best, flush=True)
    truth = P.load_truth()
    tight = load_flows(truth)
    best, base = choose_source(tight, truth, best)
    print("CHOSEN_MASK_SOURCE", best, flush=True)
    if best["ex_pq"] <= EX_PQ_GATE:
        print("NO_CANDIDATE: ex-vivo PQ gate failed; previous 0.47488 remains best", flush=True)
        return

    iv, ivprob = V.existing_iv(truth)
    iv_pq = float(np.mean([pq_score(iv[s], truth[s]["invivo"][0])[0] for s in truth]))
    alternatives = [best]
    high_reach = max((r for r in mask_eval["all"] if r["ex_pq"] >= EX_PQ_GATE),
                     key=lambda r: (r["reachable_pairs"], r["ex_pq"]))
    high_reach = dict(high_reach, source="tight", alpha=1.0)
    if (high_reach["source"], high_reach["cellprob"], high_reach["flow"]) != (
            best["source"], best["cellprob"], best["flow"]):
        alternatives.append(high_reach)
    results = []
    for candidate in alternatives:
        cfg = {"cellprob": candidate["cellprob"], "flow": candidate["flow"]}
        flows = ({sid: blend(base[sid], tight[sid], candidate["alpha"]) for sid in truth}
                 if candidate["source"] == "blend" else tight)
        V.save_heldout(truth, iv, ivprob, flows, (cfg["cellprob"], cfg["flow"]))
        del flows
        gc.collect()
        f1, threshold = V.rematch_cv()
        cv = 0.25 * (iv_pq + candidate["ex_pq"]) + 0.5 * f1
        result = {"iv_pq": iv_pq, "ex_pq": candidate["ex_pq"], "matching_f1": f1,
                  "combined_cv": cv, "pair_threshold": threshold, "mask_config": cfg,
                  "source": candidate["source"], "tight_weight": candidate["alpha"]}
        print("FULL_CV_CANDIDATE", result, flush=True)
        results.append(result)
    result = max(results, key=lambda r: r["combined_cv"])
    cv = result["combined_cv"]
    threshold = result["pair_threshold"]
    cfg = result["mask_config"]
    (WORK / "v9_tight_cv.json").write_text(json.dumps(result, indent=2))
    print("VALIDATION", result, flush=True)
    if cv <= FULL_CV_GATE:
        print("NO_CANDIDATE: combined CV gate failed; previous 0.47488 remains best", flush=True)
        return

    if result != results[-1]:
        flows = ({sid: blend(base[sid], tight[sid], result["tight_weight"]) for sid in truth}
                 if result["source"] == "blend" else tight)
        V.save_heldout(truth, iv, ivprob, flows, (cfg["cellprob"], cfg["flow"]))
        del flows
        V.rematch_cv()  # rebuild pair-classifier inputs for the winning mask variant

    model_path = fit_all()
    masks = infer_test(model_path, cfg, result["source"], result["tight_weight"])
    env = os.environ.copy()
    env["MASKS_CSV"] = str(masks)
    env["REG_TRAIN"] = str(V.DATA / "reg_window_vote5.pkl")
    subprocess.run([sys.executable, V.RESEARCH / "test_apply.py", OUT,
                    f"{threshold:.4f}", "3"], cwd=V.RESEARCH, env=env, check=True)
    subprocess.run([sys.executable, WORK / "validate_submission.py", OUT], check=True)
    if sha256(OUT) == sha256(BASELINE):
        raise RuntimeError("Candidate is identical to baseline; not a new submission")
    print("NEW_CANDIDATE", OUT, "sha256", sha256(OUT), flush=True)


if __name__ == "__main__":
    main()
