"""Colab: held-out (fold) and test masks for a chosen Cellpose model set and threshold config.

Env per modality M in {IV, EX}: M_TAGS (e.g. "base,_v2"), M_CFG ("cellprob:flow"), M_AUG ("0"/"1").
Flows are cached in /content/work/flows/ so other thresholds only cost decoding.
MODE=grid: score EX configs in GRID on held-out PQ, ex cell count and reachable GT pairs.
MODE=held: only the held-out labels (no probabilities). MODE=final: write /content/work/heldout_labels_{OUT}.npz and /content/work/masks_{OUT}.csv (no pairs).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import labels_to_rles, pq_score, write_submission  # noqa: E402

OUT = os.environ.get("OUT", "new")
MODE = os.environ.get("MODE", "final")
CACHE = Path("/content/work/flows")
SPEC = {m: dict(tags=os.environ[f"{k}_TAGS"].split(","),
                cfg=dict(zip(("cellprob", "flow"), map(float, os.environ[f"{k}_CFG"].split(":")))),
                aug=os.environ.get(f"{k}_AUG", "0") == "1")
        for m, k in (("invivo", "IV"), ("exvivo", "EX"))}


def flows(models, image, aug):
    kw = dict(compute_masks=False, batch_size=16)
    if aug:
        kw.update(augment=True, tile_overlap=0.5)
    dps, cps, ds = [], [], []
    for model in models:
        d = float(model.net.diam_labels.item())
        _, o, _ = model.eval(image.astype(np.float32), diameter=d, **kw)
        dps.append(o[1]); cps.append(o[2]); ds.append(d)
    return np.mean(dps, 0).astype(np.float16), np.mean(cps, 0).astype(np.float16), int(200 * np.mean(ds) / 30)


def load(modality, names):
    return [P.cellpose_model(P.cellpose_path(modality, f"{n}{'' if t == 'base' else t}"))
            for t in SPEC[modality]["tags"] for n in names]


def cached_flows(modality, split, sids):
    """split 'training' uses each mouse's fold models; 'hidden_test' uses final + all fold models."""
    spec = SPEC[modality]
    path = CACHE / f"{modality}_{'+'.join(spec['tags'])}_aug{int(spec['aug'])}_{split}.npz"
    if path.exists():
        data = np.load(path)
        return {s: (data[f"{s}|dp"], data[f"{s}|cp"], int(data[f"{s}|n"])) for s in sids}
    out = {}
    groups = ({s: [s2 for s2 in sids if s2.startswith(s)] for s in P.SUBJECTS} if split == "training"
              else {"all": list(sids)})
    for group, members in groups.items():
        models = load(modality, [f"fold_{group}"] if split == "training"
                      else ["all"] + [f"fold_{s}" for s in P.SUBJECTS])
        for sid in members:
            out[sid] = flows(models, P.read_image(P.region_path(sid, split) / f"{modality}.tif"), spec["aug"])
        print("flows", modality, split, group, round(time.time() - t0), flush=True)
    CACHE.mkdir(exist_ok=True)
    np.savez(path, **{f"{s}|{k}": v for s, f in out.items() for k, v in zip(("dp", "cp", "n"), f)})
    return out


def linked(pred, gt):
    """GT label ids (1-based) hit by some predicted instance at IoU > 0.75."""
    npred, ngt = int(pred.max()), int(gt.max())
    if not npred or not ngt:
        return set()
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc, gc = joint.sum(1), joint.sum(0)
    r, c = np.nonzero(joint[1:, 1:])
    iou = joint[r + 1, c + 1] / (pc[r + 1] + gc[c + 1] - joint[r + 1, c + 1])
    return set((c[iou > 0.75] + 1).tolist())


t0 = time.time()
truth = P.load_truth()
train_ids = sorted(truth)

if MODE == "grid":
    iv = cached_flows("invivo", "training", train_ids)
    ex = cached_flows("exvivo", "training", train_ids)
    iv_hit, pairs = {}, {}
    for sid in train_ids:
        (gl, gids) = truth[sid]["invivo"]
        (el, eids) = truth[sid]["exvivo"]
        iv_hit[sid] = {gids[i - 1] for i in linked(P.cellpose_labels(iv[sid], "invivo", SPEC["invivo"]["cfg"]), gl)}
        pairs[sid] = truth[sid]["pairs"]
    for g in os.environ.get("GRID", "0:0.1").split(","):
        cfg = dict(zip(("cellprob", "flow"), map(float, g.split(":"))))
        pq, cells, reach = [], 0, 0
        for sid in train_ids:
            el, eids = truth[sid]["exvivo"]
            lab = P.cellpose_labels(ex[sid], "exvivo", cfg)
            pq.append(pq_score(lab, el)[0])
            cells += int(lab.max())
            ex_hit = {eids[i - 1] for i in linked(lab, el)}
            reach += sum(a in iv_hit[sid] and b in ex_hit for a, b in pairs[sid])
        print("GRID", "+".join(SPEC["exvivo"]["tags"]), g, "pq", round(float(np.mean(pq)), 4), "cells", cells,
              "reach", reach, flush=True)
    sys.exit()

held = {}
for modality in P.MODALITIES:
    fl = cached_flows(modality, "training", train_ids)
    for sid in train_ids:
        held[f"{sid}|{modality}"] = P.cellpose_labels(fl[sid], modality, SPEC[modality]["cfg"]).astype(np.int16)
        if MODE != "held":
            held[f"{sid}|{modality}|prob"] = fl[sid][1]
np.savez_compressed(f"/content/work/heldout_labels_{OUT}.npz", **held)
if MODE == "held":
    print("HELD_DONE", OUT, round(time.time() - t0), flush=True)
    sys.exit()

sample = pd.read_csv(P.ROOT / "sample_submission.csv")
labels = {sid: {} for sid in sample.sample_id}
for modality in P.MODALITIES:
    fl = cached_flows(modality, "hidden_test", list(sample.sample_id))
    for sid in sample.sample_id:
        labels[sid][modality] = P.cellpose_labels(fl[sid], modality, SPEC[modality]["cfg"])
rows = [{"sample_id": sid,
         "invivo_instances": labels_to_rles(labels[sid]["invivo"], "IVP"),
         "exvivo_instances": labels_to_rles(labels[sid]["exvivo"], "EXP"),
         "match_pairs": []} for sid in sample.sample_id]
write_submission(rows, Path(f"/content/work/masks_{OUT}.csv"))
print("FINAL_DONE", OUT, round(time.time() - t0), flush=True)
