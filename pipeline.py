"""End-to-end pipeline used by the Colab notebook.

Stages: leave-one-mouse-out training, held-out prediction cache, leak-free search of
segmentation + matching settings on the competition score, final models, submission.

Segmentation sources compete in the search: our U-Net at each input scale ("unet_x1", "unet_x2"),
Cellpose-SAM fine-tuned on these masks at one or more training budgets ("cellpose",
"cellpose_long"), and Cellpose masks filtered by the U-Net's core probability and shape features
("hybrid").
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from cellmatch import (ROOT, labels_to_rles, normalize, pq_score, read_image, region_centers,
                       rle_to_labels, write_submission)
from learned import (RandomPatches, UNet, decode_instances, filter_instances, instance_features,
                     load_model, load_training, loss_function, predict_probabilities)
from registration import estimate_linear_priors, match_points, register_ncc, register_with_priors, transform

SUBJECTS = ("subject_5d294c", "subject_b2ba5e", "subject_db6b8b")
MODALITIES = ("invivo", "exvivo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ROOT.parent / "models"
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))  # DataLoader workers for U-Net training

BODIES = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
CORES = {"invivo": (0.4, 0.5, 0.6), "exvivo": (0.5,)}
FILTERS = {
    "invivo": [{}, {"min_core": 0.2}, {"min_core": 0.3},
               {"min_core": 0.2, "min_circularity": 0.4, "max_eccentricity": 0.97, "min_fill": 0.3}],
    "exvivo": [{}, {"min_core": 0.3}, {"min_core": 0.4},
               {"min_core": 0.3, "min_circularity": 0.45, "max_eccentricity": 0.95, "min_fill": 0.35},
               {"min_core": 0.3, "min_circularity": 0.55, "max_eccentricity": 0.93, "min_fill": 0.4},
               {"min_circularity": 0.5, "max_eccentricity": 0.94, "min_fill": 0.38}],
}
METHODS = ("greedy", "hungarian")
DISTANCES = (3, 4, 5, 6, 7, 8, 10, 12)
BRIGHTNESS_LEVELS = (0.0, 0.45, 0.55, 0.65, 0.75, 0.85)
RATIOS = (None, 1.5, 2.0, 2.5, 3.0)          # second-nearest / nearest distance required for a pair
CELLPOSE_GRID = tuple(itertools.product((-1.0, 0.0, 1.0), (0.1, 0.2, 0.3, 0.4)))  # (cellprob, flow)
FIELD_PRIOR = np.array([0.48, 0.54])  # mean true in-vivo field centre on the ex-vivo canvas, training mice
CELLPOSE_TILE = 128
REGISTRATIONS = ("none", "soma", "ncc")
CENTRE_GATES = (None, 0.06, 0.08, 0.10, 0.14)


def model_path(modality: str, width: int, scale: float, tag: str, seed: int) -> Path:
    return MODELS / f"{modality}_w{width}_x{scale:g}_{tag}_seed{seed}.pt"


def seed_worker(worker_id: int) -> None:
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def train(modality: str, subjects, output: Path, steps: int, width: int, scale: float,
          seed: int, batch: int = 16) -> Path:
    if output.exists():
        print("reusing", output.name, flush=True)
        return output
    torch.manual_seed(seed)
    np.random.seed(seed)
    records = load_training(modality, list(subjects), scale)
    loader = torch.utils.data.DataLoader(
        RandomPatches(records, length=steps * batch), batch_size=batch, num_workers=WORKERS,
        worker_init_fn=seed_worker, pin_memory=DEVICE == "cuda")
    model = UNet(width).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    started = time.monotonic()
    model.train()
    for step, (images, masks) in enumerate(loader, 1):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(images)
        loss = loss_function(logits.float(), masks, modality)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step % 500 == 0 or step == steps:
            print(f"  {output.name} step {step}/{steps} loss={loss.item():.4f} "
                  f"{time.monotonic() - started:.0f}s", flush=True)
        if step >= steps:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output)
    return output


def unet_source(scale: float) -> str:
    return f"unet_x{scale:g}"


def cellpose_path(modality: str, tag: str) -> Path:
    return MODELS / f"{modality}_cpsam_{tag}"


def cellpose_model(path: Path | None = None):
    from cellpose import models

    return models.CellposeModel(gpu=DEVICE == "cuda", pretrained_model=str(path) if path else "cpsam_v2",
                                use_bfloat16=DEVICE == "cuda" and torch.cuda.is_bf16_supported())


def percentile_normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    return ((image.astype(np.float32) - low) / max(high - low, 1e-6)).astype(np.float32)


def cellpose_tiles(modality: str, subjects, per_image: int = 40, seed: int = 0):
    """Crops centred on labelled cells, normalized with whole-image percentiles.

    Cellpose trains on random 256-px crops of the image upsampled to ~30-px cells; on a full
    ex-vivo canvas such a crop holds well under one labelled cell on average.
    """
    rng = np.random.default_rng(seed)
    frame = pd.read_csv(ROOT / "training/train_ground_truth.csv")
    images, labels = [], []
    size, half = CELLPOSE_TILE, CELLPOSE_TILE // 2
    for row in frame.itertuples(index=False):
        if row.sample_id.split("__")[0] not in subjects:
            continue
        image = percentile_normalize(read_image(region_path(row.sample_id) / f"{modality}.tif"))
        label, _ = rle_to_labels(json.loads(getattr(row, f"{modality}_instances")), image.shape)
        centres = region_centers(label)
        for x, y in centres[rng.choice(len(centres), min(per_image, len(centres)), replace=False)]:
            y0 = int(np.clip(round(y) - half, 0, image.shape[0] - size))
            x0 = int(np.clip(round(x) - half, 0, image.shape[1] - size))
            _, crop = np.unique(label[y0:y0 + size, x0:x0 + size], return_inverse=True)
            images.append(image[y0:y0 + size, x0:x0 + size])
            labels.append(crop.reshape(size, size).astype(np.int32))
    return images, labels


def train_cellpose(modality: str, subjects, output: Path, epochs: int, images_per_epoch: int,
                   batch: int = 4) -> Path:
    if output.exists():
        print("reusing", output.name, flush=True)
        return output
    from cellpose import train as cellpose_train

    started = time.monotonic()
    images, labels = cellpose_tiles(modality, list(subjects))
    model = cellpose_model()
    cellpose_train.train_seg(model.net, train_data=images, train_labels=labels, normalize=False,
                             rescale=True, batch_size=batch, n_epochs=epochs,
                             nimg_per_epoch=images_per_epoch, learning_rate=1e-5, weight_decay=0.1,
                             min_train_masks=1, save_path=str(MODELS.parent), model_name=output.name)
    print(f"  {output.name}: {len(images)} tiles, {time.monotonic() - started:.0f}s", flush=True)
    return output


def cellpose_flows(models, image: np.ndarray) -> tuple:
    """Flows and cell probability at the original resolution, averaged over one or more models."""
    models = models if isinstance(models, (list, tuple)) else [models]
    flows, probabilities, diameters = [], [], []
    for model in models:
        diameter = float(model.net.diam_labels.item())
        _, output, _ = model.eval(image.astype(np.float32), diameter=diameter, compute_masks=False,
                                  batch_size=16)
        flows.append(output[1])
        probabilities.append(output[2])
        diameters.append(diameter)
    return (np.mean(flows, axis=0).astype(np.float16), np.mean(probabilities, axis=0).astype(np.float16),
            int(200 * np.mean(diameters) / 30))


def cellpose_labels(flows: tuple, modality: str, config: dict) -> np.ndarray:
    from cellpose import dynamics

    dP, cellprob, niter = flows
    masks = dynamics.resize_and_compute_masks(
        dP.astype(np.float32), cellprob.astype(np.float32), niter=niter,
        cellprob_threshold=config["cellprob"], flow_threshold=config["flow"], min_size=10,
        device=torch.device(DEVICE))
    counts = np.bincount(masks.ravel())
    low, high = (20, 330) if modality == "invivo" else (17, 350)
    keep = (counts >= low) & (counts <= high)
    keep[0] = False
    masks = np.where(keep[masks], masks, 0)
    _, inverse = np.unique(masks, return_inverse=True)
    return inverse.reshape(masks.shape).astype(np.int32)


def fold_cellpose_flows(truth: dict, modality: str, source: str = "cellpose") -> dict:
    """Cellpose flows for every training region from the fine-tune that never saw its mouse."""
    cache = {}
    for subject in SUBJECTS:
        model = cellpose_model(cellpose_path(modality, f"fold_{subject}{source.removeprefix('cellpose')}"))
        for sample_id in (s for s, t in truth.items() if t["subject"] == subject):
            cache[sample_id] = cellpose_flows(model, read_image(region_path(sample_id) / f"{modality}.tif"))
    return cache


def region_path(sample_id: str, split: str = "training") -> Path:
    return ROOT / split / sample_id.replace("__", "/")


def load_truth() -> dict:
    frame = pd.read_csv(ROOT / "training/train_ground_truth.csv")
    truth = {}
    for row in frame.itertuples(index=False):
        item = {"subject": row.sample_id.split("__")[0],
                "pairs": {tuple(pair) for pair in json.loads(row.match_pairs)}}
        for modality in MODALITIES:
            shape = read_image(region_path(row.sample_id) / f"{modality}.tif").shape
            item[modality] = rle_to_labels(json.loads(getattr(row, f"{modality}_instances")), shape)
        truth[row.sample_id] = item
    return truth


def fold_probabilities(truth: dict, modality: str, width: int, scale: float, tta: bool = True) -> dict:
    """Probabilities for every training region from the model that never saw its mouse."""
    cache = {}
    for subject in SUBJECTS:
        model = load_model(model_path(modality, width, scale, f"fold_{subject}", 0), DEVICE)
        for sample_id in (s for s, t in truth.items() if t["subject"] == subject):
            image = read_image(region_path(sample_id) / f"{modality}.tif")
            cache[sample_id] = predict_probabilities(model, image, modality, DEVICE, tta=tta,
                                                     scale=scale).astype(np.float16)
    return cache


def decode(probs, modality: str, config: dict) -> np.ndarray:
    if config["source"].startswith("cellpose"):
        return cellpose_labels(probs, modality, config)
    if config["source"] == "hybrid":
        flows, unet_probs = probs
        labels = cellpose_labels(flows, modality, config)
        if not labels.max():
            return labels
        return filter_instances(labels, instance_features(labels, unet_probs.astype(np.float32)),
                                **FILTERS[modality][config["filter"]])
    probs = probs.astype(np.float32)
    labels = decode_instances(probs, modality, config["body"], config["core"])
    if not labels.max():
        return labels
    return filter_instances(labels, instance_features(labels, probs), **FILTERS[modality][config["filter"]])


def pq_table(probs: dict, truth: dict, modality: str, scale: float) -> list[dict]:
    rows = []
    filters = FILTERS[modality]
    for body, core in itertools.product(BODIES, CORES[modality]):
        scores = np.zeros((len(filters), len(probs)))
        for k, (sample_id, region_probs) in enumerate(probs.items()):
            region_probs = region_probs.astype(np.float32)
            raw = decode_instances(region_probs, modality, body, core)
            features = instance_features(raw, region_probs)
            for f, params in enumerate(filters):
                scores[f, k] = pq_score(filter_instances(raw, features, **params), truth[sample_id][modality][0])[0]
        rows += [{"source": unet_source(scale), "scale": scale, "body": body, "core": core, "filter": f,
                  "pq": float(scores[f].mean())} for f in range(len(filters))]
    return rows


def cellpose_pq_table(flows: dict, truth: dict, modality: str, source: str) -> list[dict]:
    rows = []
    for cellprob, flow in CELLPOSE_GRID:
        config = {"source": source, "cellprob": cellprob, "flow": flow}
        scores = [pq_score(cellpose_labels(region_flows, modality, config), truth[sample_id][modality][0])[0]
                  for sample_id, region_flows in flows.items()]
        rows.append({**config, "pq": float(np.mean(scores))})
    return rows


def hybrid_pq_table(data: dict, truth: dict, modality: str, unet: str, cellpose: str) -> list[dict]:
    """Cellpose masks, kept or dropped by the U-Net's per-instance core mean and shape filters."""
    rows = []
    filters = FILTERS[modality]
    for cellprob, flow in CELLPOSE_GRID:
        config = {"source": "hybrid", "unet": unet, "cellpose": cellpose, "cellprob": cellprob, "flow": flow}
        scores = np.zeros((len(filters), len(data)))
        for k, (sample_id, (flows, unet_probs)) in enumerate(data.items()):
            labels = cellpose_labels(flows, modality, config)
            features = instance_features(labels, unet_probs.astype(np.float32))
            for f, params in enumerate(filters):
                scores[f, k] = pq_score(filter_instances(labels, features, **params), truth[sample_id][modality][0])[0]
        rows += [{**config, "filter": f, "pq": float(scores[f].mean())} for f in range(len(filters))]
    return rows


def pq_breakdown(labels: dict, truth: dict, modality: str) -> dict:
    """Pooled TP/FP/FN, mean IoU of true positives, and near misses (best IoU in 0.5-0.75)."""
    tp = fp = fn = near = 0
    iou_sum = 0.0
    for sample_id, pred in labels.items():
        gt = truth[sample_id][modality][0]
        npred, ngt = int(pred.max()), int(gt.max())
        joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                            minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
        pc, gc = joint.sum(axis=1), joint.sum(axis=0)
        iou = joint[1:, 1:] / np.maximum(pc[1:, None] + gc[None, 1:] - joint[1:, 1:], 1)
        best = iou.max(axis=1) if ngt else np.zeros(npred)
        hits = best > 0.75
        tp += int(hits.sum())
        fp += npred - int(hits.sum())
        fn += ngt - int(hits.sum())
        near += int(((best > 0.5) & ~hits).sum())
        iou_sum += float(best[hits].sum())
    return {"tp": tp, "fp": fp, "fn": fn, "mean_tp_iou": iou_sum / max(tp, 1),
            "near_misses_iou_0.5_0.75": near}


def predicted_to_ground_truth(pred: np.ndarray, truth: np.ndarray) -> dict[int, int]:
    npred, ngt = int(pred.max()), int(truth.max())
    if not npred or not ngt:
        return {}
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + truth).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc, gc = joint.sum(axis=1), joint.sum(axis=0)
    rows, cols = np.nonzero(joint[1:, 1:])
    rows += 1
    cols += 1
    iou = joint[rows, cols] / (pc[rows] + gc[cols] - joint[rows, cols])
    return {int(p): int(g) for p, g, score in zip(rows, cols, iou) if score > 0.75}


def ground_truth_ids(pred: np.ndarray, truth_labels: np.ndarray, ids: list[str]) -> list[str | None]:
    mapping = predicted_to_ground_truth(pred, truth_labels)
    return [ids[mapping[i] - 1] if i in mapping else None for i in range(1, int(pred.max()) + 1)]


def region_record(iv_labels: np.ndarray, ex_labels: np.ndarray, ex_image: np.ndarray, priors,
                  registrations=REGISTRATIONS) -> dict:
    iv_centers, ex_centers = region_centers(iv_labels), region_centers(ex_labels)
    n = int(ex_labels.max())
    counts = np.bincount(ex_labels.ravel(), minlength=n + 1)
    brightness = (np.bincount(ex_labels.ravel(), weights=normalize(ex_image, "exvivo")[0].ravel(),
                              minlength=n + 1) / np.maximum(counts, 1))[1:]
    circularity = instance_features(ex_labels)[:, 1]
    # "soma" weighting keeps dense bright processes from making a wrong alignment look good.
    target_weights = {"none": None, "soma": np.clip(brightness, 0, 1) * np.clip(circularity, 0, 1)}
    registration = {}
    for name in registrations:
        if name == "ncc":
            registration[name] = (register_ncc(iv_centers, ex_centers, priors),)
        else:
            registration[name] = tuple(register_with_priors(iv_centers, ex_centers, mode=mode,
                                                            target_weights=target_weights[name], priors=priors)
                                       for mode in (0, 1))
    return {"iv_centers": iv_centers, "ex_centers": ex_centers, "brightness": brightness,
            "iv_shape": iv_labels.shape, "ex_shape": ex_labels.shape, "registration": registration}


def subject_modes(records: dict, registration: str) -> dict:
    """Square ex-vivo canvases share one orientation per mouse; others choose per region."""
    modes = {}
    subjects = sorted({sample_id.split("__")[0] for sample_id in records})
    if registration == "ncc":
        return dict.fromkeys(subjects)
    for subject in subjects:
        items = [r for s, r in records.items() if s.startswith(subject)]
        square = np.mean([abs(np.log(r["ex_shape"][0] / r["ex_shape"][1])) < 0.1 for r in items])
        winners = np.array([r["registration"][registration][0][1] > r["registration"][registration][1][1]
                            for r in items])
        modes[subject] = None
        if square >= 0.65 and winners.mean() >= 0.65:
            modes[subject] = 0
        elif square >= 0.65 and (~winners).mean() >= 0.65:
            modes[subject] = 1
    return modes


def chosen_matrix(record: dict, registration: str, mode: int | None):
    candidates = record["registration"][registration]
    return (candidates[mode] if mode is not None else max(candidates, key=lambda c: c[1]))[0]


def consistent_regions(records: dict, matrices: dict, gate: float | None) -> set:
    """Regions whose in-vivo field lands near its mouse's median spot on the ex-vivo canvas.

    Within a mouse that spot varies by only a few percent of the canvas, so a far-off
    alignment is almost always wrong, and every pair it produces would be a false positive.
    """
    if gate is None:
        return set(records)
    # Anchor on the densest cluster of field centres (ties go to the training-mean position):
    # when a mouse's alignments split into two groups, a median lands between them and keeps none.
    centres = {}
    for sample_id, matrix in matrices.items():
        if matrix is not None:
            record = records[sample_id]
            centre = matrix[:, :2] @ np.array([record["iv_shape"][1] / 2, record["iv_shape"][0] / 2]) + matrix[:, 2]
            centres[sample_id] = centre / np.array([record["ex_shape"][1], record["ex_shape"][0]])
    keep = set()
    for subject in {sample_id.split("__")[0] for sample_id in centres}:
        members = [s for s in centres if s.startswith(subject)]
        points = np.array([centres[s] for s in members])
        near = np.linalg.norm(points[:, None] - points[None], axis=2) <= gate
        densest = np.flatnonzero(near.sum(axis=1) == near.sum(axis=1).max())
        seed = densest[np.argmin(np.linalg.norm(points[densest] - FIELD_PRIOR, axis=1))]
        anchor = np.median(points[near[seed]], axis=0)
        keep |= {s for s, point in zip(members, points) if np.linalg.norm(point - anchor) <= gate}
    return keep


def predict_pairs(record: dict, matrix, method: str, distance: float, min_brightness: float,
                  ratio: float | None = None):
    if matrix is None:
        return []
    keep = np.flatnonzero(record["brightness"] >= min_brightness)
    if len(keep) < 3:
        keep = np.arange(len(record["ex_centers"]))
    pairs = match_points(record["iv_centers"], record["ex_centers"][keep], matrix, distance, method)
    if ratio is None or not pairs:
        return [(i, int(keep[j])) for i, j, _ in pairs]
    # Ratio test: an unambiguous pair has no second ex-vivo candidate nearly as close.
    second = cKDTree(record["ex_centers"][keep]).query(transform(record["iv_centers"], matrix), k=2)[0][:, 1]
    return [(i, int(keep[j])) for i, j, d in pairs if second[i] >= ratio * max(d, 0.5)]


def cv_records(iv_probs: dict, ex_probs: dict, truth: dict, iv_config: dict, ex_config: dict,
               priors_by_subject: dict) -> dict:
    records = {}
    for sample_id, item in truth.items():
        iv = decode(iv_probs[sample_id], "invivo", iv_config)
        ex = decode(ex_probs[sample_id], "exvivo", ex_config)
        ex_image = read_image(region_path(sample_id) / "exvivo.tif")
        record = region_record(iv, ex, ex_image, priors_by_subject[item["subject"]])
        record["iv_gt"] = ground_truth_ids(iv, *item["invivo"])
        record["ex_gt"] = ground_truth_ids(ex, *item["exvivo"])
        records[sample_id] = record
    return records


def matching_search(records: dict, truth: dict) -> list[dict]:
    total = sum(len(item["pairs"]) for item in truth.values())
    results = []
    for registration in REGISTRATIONS:
        modes = subject_modes(records, registration)
        matrices = {s: chosen_matrix(r, registration, modes[s.split("__")[0]]) for s, r in records.items()}
        kept = {gate: consistent_regions(records, matrices, gate) for gate in CENTRE_GATES}
        for method, distance, level, ratio in itertools.product(METHODS, DISTANCES, BRIGHTNESS_LEVELS, RATIOS):
            counts = {}
            for sample_id, record in records.items():
                pairs = predict_pairs(record, matrices[sample_id], method, distance, level, ratio)
                counts[sample_id] = (sum((record["iv_gt"][i], record["ex_gt"][j]) in truth[sample_id]["pairs"]
                                         for i, j in pairs), len(pairs))
            for gate, regions in kept.items():
                tp = sum(counts[s][0] for s in regions)
                predicted = sum(counts[s][1] for s in regions)
                results.append({"registration": registration, "centre_gate": gate, "method": method,
                                "distance": distance, "min_brightness": level, "ratio": ratio,
                                "tp": tp, "predicted": predicted,
                                "precision": tp / max(predicted, 1), "recall": tp / total,
                                "f1": 2 * tp / (predicted + total)})
    return results


def tune(caches: dict, truth: dict, top: int = 2) -> dict:
    """caches[(modality, source)] -> {sample_id: probs or flows}; adds hybrid sources in place.

    Returns the best leak-free config. Matching is searched for the best setting of each of the
    top `top` segmentation sources per modality.
    """
    segmentation = {m: [] for m in MODALITIES}
    for (modality, source), data in list(caches.items()):
        started = time.monotonic()
        if source.startswith("cellpose"):
            segmentation[modality] += cellpose_pq_table(data, truth, modality, source)
        else:
            segmentation[modality] += pq_table(data, truth, modality, float(source.removeprefix("unet_x")))
        print(f"PQ grid {modality} {source}: {time.monotonic() - started:.0f}s", flush=True)
    for modality in MODALITIES:
        unets = [r for r in segmentation[modality] if r["source"].startswith("unet")]
        cellposes = [r for r in segmentation[modality] if r["source"].startswith("cellpose")]
        if cellposes and unets:
            unet = max(unets, key=lambda row: row["pq"])["source"]
            cellpose = max(cellposes, key=lambda row: row["pq"])["source"]
            started = time.monotonic()
            caches[(modality, "hybrid")] = {s: (flows, caches[(modality, unet)][s])
                                            for s, flows in caches[(modality, cellpose)].items()}
            segmentation[modality] += hybrid_pq_table(caches[(modality, "hybrid")], truth, modality, unet, cellpose)
            print(f"PQ grid {modality} hybrid ({cellpose} + {unet}): {time.monotonic() - started:.0f}s", flush=True)
    for modality in MODALITIES:
        segmentation[modality].sort(key=lambda row: row["pq"], reverse=True)
        print(f"\nTop {modality} settings (held-out PQ):")
        for row in segmentation[modality][:5]:
            print("  ", row, FILTERS[modality][row["filter"]] if "filter" in row else "")
        for source in sorted({row["source"] for row in segmentation[modality]}):
            print(f"   best {source}: {max(r['pq'] for r in segmentation[modality] if r['source'] == source):.4f}")
    priors = {subject: estimate_linear_priors({subject}) for subject in SUBJECTS}
    total = sum(len(item["pairs"]) for item in truth.values())
    best = None
    candidates = {}
    for modality in MODALITIES:
        per_source = {}
        for row in segmentation[modality]:
            per_source.setdefault(row["source"], row)
        candidates[modality] = sorted(per_source.values(), key=lambda row: row["pq"], reverse=True)[:top]
    for iv_config, ex_config in itertools.product(candidates["invivo"], candidates["exvivo"]):
        records = cv_records(caches[("invivo", iv_config["source"])], caches[("exvivo", ex_config["source"])],
                             truth, iv_config, ex_config, priors)
        eligible = sum(a in set(r["iv_gt"]) and b in set(r["ex_gt"])
                       for s, r in records.items() for a, b in truth[s]["pairs"])
        segmentation_score = 0.25 * (iv_config["pq"] + ex_config["pq"])
        match = max(matching_search(records, truth), key=lambda row: row["f1"])
        final = segmentation_score + 0.5 * match["f1"]
        print(f"\nIV {iv_config} | EX {ex_config}\n  eligible pairs {eligible}/{total} "
              f"(F1 ceiling {2 * eligible / (eligible + total):.3f}) best match {match}\n"
              f"  CV score = {final:.4f}", flush=True)
        if best is None or final > best["cv_score"]:
            best = {"cv_score": final, "invivo": iv_config, "exvivo": ex_config, "match": match}
    for modality in MODALITIES:
        config = best[modality]
        labels = {s: decode(caches[(modality, config["source"])][s], modality, config) for s in truth}
        print(f"\n{modality} PQ breakdown for the chosen setting:", pq_breakdown(labels, truth, modality))
    return best


def assemble_submission(predictions: dict, best: dict, output: Path) -> Path:
    """predictions[sample_id][modality] is what decode() needs for best[modality] on that test region."""
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    priors = estimate_linear_priors()
    match = best["match"]
    records, labels = {}, {}
    for sample_id in sample.sample_id:
        region_labels = {modality: decode(predictions[sample_id][modality], modality, best[modality])
                         for modality in MODALITIES}
        ex_image = read_image(region_path(sample_id, "hidden_test") / "exvivo.tif")
        records[sample_id] = region_record(region_labels["invivo"], region_labels["exvivo"], ex_image,
                                           priors, registrations=(match["registration"],))
        labels[sample_id] = region_labels
        print("decoded", sample_id, {m: int(l.max()) for m, l in region_labels.items()}, flush=True)
    modes = subject_modes(records, match["registration"])
    matrices = {s: chosen_matrix(r, match["registration"], modes[s.split("__")[0]]) for s, r in records.items()}
    kept = consistent_regions(records, matrices, match["centre_gate"])
    print(f"centre gate {match['centre_gate']}: pairs kept for {len(kept)}/{len(records)} regions", flush=True)
    rows = []
    for sample_id, record in records.items():
        pairs = (predict_pairs(record, matrices[sample_id], match["method"], match["distance"],
                               match["min_brightness"], match["ratio"])
                 if sample_id in kept else [])
        rows.append({"sample_id": sample_id,
                     "invivo_instances": labels_to_rles(labels[sample_id]["invivo"], "IVP"),
                     "exvivo_instances": labels_to_rles(labels[sample_id]["exvivo"], "EXP"),
                     "match_pairs": [[f"IVP_{i + 1:06d}", f"EXP_{j + 1:06d}"] for i, j in pairs]})
    write_submission(rows, output)
    print(f"saved {output} with {sum(len(r['match_pairs']) for r in rows)} pairs", flush=True)
    return output


def make_submission(models: dict, best: dict, output: Path, tta: bool = True) -> Path:
    """models[modality] = {"cellpose": [Cellpose models] or None, "unet": [U-Nets] or None}.

    Several Cellpose models (final + fold fine-tunes) are averaged at the flow level.
    """
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    predictions = {}
    for sample_id in sample.sample_id:
        path = region_path(sample_id, "hidden_test")
        predictions[sample_id] = {}
        for modality in MODALITIES:
            image = read_image(path / f"{modality}.tif")
            config = best[modality]
            if config["source"].startswith("cellpose"):
                prediction = cellpose_flows(models[modality]["cellpose"], image)
            elif config["source"] == "hybrid":
                scale = float(config["unet"].removeprefix("unet_x"))
                prediction = (cellpose_flows(models[modality]["cellpose"], image),
                              np.mean([predict_probabilities(model, image, modality, DEVICE, tta=tta, scale=scale)
                                       for model in models[modality]["unet"]], axis=0))
            else:
                prediction = np.mean([predict_probabilities(model, image, modality, DEVICE, tta=tta,
                                                            scale=config["scale"])
                                      for model in models[modality]["unet"]], axis=0)
            predictions[sample_id][modality] = prediction
        print("inferred", sample_id, flush=True)
    return assemble_submission(predictions, best, output)
