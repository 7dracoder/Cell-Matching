"""NYU HPC entry point: `python run_hpc.py gpu`, then `python run_hpc.py cpu`.

gpu: all training plus every prediction (held-out and test), cached under cache/.
cpu: leak-free search on the cached held-out predictions, then submission.csv from the cached
     test predictions.

The stages are separate jobs because the search is CPU-bound, and Torch cancels GPU jobs whose
GPU sits idle. Every step skips work already on disk, so a cancelled job can simply be resubmitted.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

import pipeline as P
from cellmatch import read_image
from learned import load_model, predict_probabilities

WIDTH = 32
STEPS = {"invivo": 3000, "exvivo": 4000}
BATCH = 16
SCALES = (1.0,)
FINAL_SEEDS = (0, 1)
TTA = True
# Cellpose-SAM fine-tuning budgets, name -> (epochs, cell-centred tiles per epoch).
CELLPOSE_VARIANTS = {"cellpose": (60, 64), "cellpose_long": (150, 96)}
CELLPOSE_BATCH = 4

CACHE = P.ROOT.parent / "cache"
BEST = P.ROOT.parent / "best_config.json"
OUTPUT = P.ROOT.parent / "submission.csv"


def ensure(name: str, compute) -> None:
    path = CACHE / f"{name}.pkl"
    if path.exists():
        print("reusing", path.name, flush=True)
        return
    started = time.time()
    value = compute()
    CACHE.mkdir(exist_ok=True)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    partial.rename(path)
    print(f"cached {path.name} in {(time.time() - started) / 60:.1f} min", flush=True)


def load(name: str):
    with (CACHE / f"{name}.pkl").open("rb") as stream:
        return pickle.load(stream)


def test_ids() -> list[str]:
    return list(pd.read_csv(P.ROOT / "sample_submission.csv").sample_id)


def test_image(sample_id: str, modality: str) -> np.ndarray:
    return read_image(P.region_path(sample_id, "hidden_test") / f"{modality}.tif")


def gpu_stage() -> None:
    assert torch.cuda.is_available(), "the gpu stage needs a GPU node"
    truth = P.load_truth()
    started = time.time()
    for scale in SCALES:
        for subject in P.SUBJECTS:
            others = [s for s in P.SUBJECTS if s != subject]
            for modality in P.MODALITIES:
                P.train(modality, others, P.model_path(modality, WIDTH, scale, f"fold_{subject}", 0),
                        STEPS[modality], WIDTH, scale, seed=0, batch=BATCH)
    for variant, (epochs, per_epoch) in CELLPOSE_VARIANTS.items():
        suffix = variant.removeprefix("cellpose")
        for subject in P.SUBJECTS:
            others = [s for s in P.SUBJECTS if s != subject]
            for modality in P.MODALITIES:
                P.train_cellpose(modality, others, P.cellpose_path(modality, f"fold_{subject}{suffix}"),
                                 epochs, per_epoch, CELLPOSE_BATCH)
    print(f"fold training done: {(time.time() - started) / 60:.1f} min", flush=True)

    for scale in SCALES:
        for modality in P.MODALITIES:
            ensure(f"heldout_{modality}_{P.unet_source(scale)}",
                   lambda: P.fold_probabilities(truth, modality, WIDTH, scale, tta=TTA))
    for variant in CELLPOSE_VARIANTS:
        for modality in P.MODALITIES:
            ensure(f"heldout_{modality}_{variant}", lambda: P.fold_cellpose_flows(truth, modality, variant))

    # Final models and test predictions for every source, so the CPU search can pick freely.
    for scale in SCALES:
        for modality in P.MODALITIES:
            paths = [P.train(modality, P.SUBJECTS, P.model_path(modality, WIDTH, scale, "all", seed),
                             STEPS[modality], WIDTH, scale, seed=seed, batch=BATCH) for seed in FINAL_SEEDS]

            def unet_test():
                models = [load_model(path, P.DEVICE) for path in paths]
                return {s: np.mean([predict_probabilities(model, test_image(s, modality), modality, P.DEVICE,
                                                          tta=TTA, scale=scale) for model in models],
                                   axis=0).astype(np.float16) for s in test_ids()}

            ensure(f"test_{modality}_{P.unet_source(scale)}", unet_test)
    for variant, (epochs, per_epoch) in CELLPOSE_VARIANTS.items():
        suffix = variant.removeprefix("cellpose")
        for modality in P.MODALITIES:
            final = P.train_cellpose(modality, P.SUBJECTS, P.cellpose_path(modality, f"all{suffix}"),
                                     epochs, per_epoch, CELLPOSE_BATCH)

            def cellpose_test():
                paths = [final] + [P.cellpose_path(modality, f"fold_{s}{suffix}") for s in P.SUBJECTS]
                models = [P.cellpose_model(path) for path in paths]
                return {s: P.cellpose_flows(models, test_image(s, modality)) for s in test_ids()}

            ensure(f"test_{modality}_{variant}", cellpose_test)
    print(f"gpu stage done: {(time.time() - started) / 60:.1f} min", flush=True)


def cpu_stage() -> None:
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    sources = [P.unet_source(scale) for scale in SCALES] + list(CELLPOSE_VARIANTS)
    if BEST.exists():
        best = json.loads(BEST.read_text())
        print("reusing", BEST.name, "- delete it to rerun the search", flush=True)
    else:
        started = time.time()
        truth = P.load_truth()
        caches = {(modality, source): load(f"heldout_{modality}_{source}")
                  for source in sources for modality in P.MODALITIES}
        best = P.tune(caches, truth)
        del caches
        BEST.write_text(json.dumps(best, indent=2))
        print(f"\nsearch: {(time.time() - started) / 60:.1f} min")
    print("CHOSEN (leave-one-mouse-out CV score %.4f):" % best["cv_score"])
    print(json.dumps(best, indent=2), flush=True)

    tests = {}

    def test(modality: str, source: str) -> dict:
        if (modality, source) not in tests:
            tests[(modality, source)] = load(f"test_{modality}_{source}")
        return tests[(modality, source)]

    predictions = {}
    for sample_id in test_ids():
        predictions[sample_id] = {}
        for modality in P.MODALITIES:
            config = best[modality]
            if config["source"] == "hybrid":
                predictions[sample_id][modality] = (test(modality, config["cellpose"])[sample_id],
                                                    test(modality, config["unet"])[sample_id])
            else:
                predictions[sample_id][modality] = test(modality, config["source"])[sample_id]
    P.assemble_submission(predictions, best, OUTPUT)
    subprocess.run([sys.executable, str(P.ROOT.parent / "validate_submission.py"), str(OUTPUT)], check=True)


if __name__ == "__main__":
    {"gpu": gpu_stage, "cpu": cpu_stage}[sys.argv[1]]()
