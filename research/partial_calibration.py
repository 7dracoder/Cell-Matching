"""Mouse-held-out selection of sparse-label Cellpose decoding settings.

Only image/model predictions and the old submission masks determine a setting.
Ground truth is used to train on the other mice and to measure held-out PQ.
No hidden-test predictions or CSV are written by this diagnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402


WORK = Path("/content/work")
FLOW = np.load(WORK / "flows/exvivo_partial96_training.npz")
BASE = np.load(WORK / "research/data/base_heldout_labels.npz")
TRUTH = P.load_truth()
GRID = [(cp, flow) for cp in (-0.5, 0.0, 0.25, 0.5, 0.75)
        for flow in (0.08, 0.15, 0.25, 0.4)]


def agreement(pred, old):
    a, b = int(pred.max()), int(old.max())
    if not a or not b:
        return 0.0, 0, a, b
    joint = np.bincount((pred.astype(np.int64) * (b + 1) + old).ravel(),
                        minlength=(a + 1) * (b + 1)).reshape(a + 1, b + 1)
    pa, ba = joint.sum(1), joint.sum(0)
    i, j = np.nonzero(joint[1:, 1:])
    iou = joint[i + 1, j + 1] / (pa[i + 1] + ba[j + 1] - joint[i + 1, j + 1])
    hit = iou > 0.5
    return float(iou[hit].sum()), int(hit.sum()), a, b


def build():
    records = {}
    for sid, item in TRUTH.items():
        old = BASE[f"{sid}|exvivo"].astype(np.int32)
        gt = item["exvivo"][0]
        values = (FLOW[f"{sid}|dp"], FLOW[f"{sid}|cp"], int(FLOW[f"{sid}|n"]))
        rows = []
        for cp, flow in GRID:
            pred = P.cellpose_labels(values, "exvivo",
                                     {"cellprob": cp, "flow": flow})
            quality = float(pq_score(pred, gt)[0])
            overlap, matches, npred, nold = agreement(pred, old)
            features = [cp, flow, npred / max(nold, 1),
                        overlap / max(nold, 1), overlap / max(npred, 1),
                        matches / max(nold, 1), matches / max(npred, 1),
                        np.log1p(nold), np.log1p(gt.size)]
            rows.append({"cp": cp, "flow": flow, "pq": quality,
                         "overlap": overlap, "matches": matches,
                         "n_pred": npred, "n_old": nold,
                         "features": features})
        records[sid] = rows
        print("REGION", sid, "best", max(r["pq"] for r in rows), flush=True)
    return records


def report(name, chosen, records):
    per = {}
    for sid, index in chosen.items():
        per.setdefault(sid.split("__")[0], []).append(records[sid][index]["pq"])
    pooled = float(np.mean([q for vals in per.values() for q in vals]))
    print(name, "PQ", round(pooled, 4),
          {s: round(float(np.mean(v)), 4) for s, v in per.items()}, flush=True)
    return pooled


def main():
    records = build()
    subjects = sorted({sid.split("__")[0] for sid in records})
    np.savez_compressed(WORK / "partial96_calibration_table.npz",
                        **{sid: np.array([[r["pq"], r["overlap"], r["matches"],
                                            r["n_pred"], r["n_old"]] for r in rows],
                                          np.float32) for sid, rows in records.items()})
    global_best = json.loads((WORK / "partial96_cv.json").read_text())["best"]
    index = GRID.index((global_best["cellprob"], global_best["flow"]))
    report("GLOBAL", {sid: index for sid in records}, records)
    report("ORACLE_REGION", {sid: int(np.argmax([r["pq"] for r in rows]))
                             for sid, rows in records.items()}, records)

    # Alpha penalizes model-only cells relative to instances confirmed by the
    # old masks. Choose alpha on two mice, then test the held-out third mouse.
    alphas = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
    rule_predictions = {}
    for held in subjects:
        scored = []
        for alpha in alphas:
            selection = {sid: int(np.argmax([
                r["overlap"] / (r["n_old"] + alpha *
                 max(r["n_pred"] - r["matches"], 0) + 1e-6)
                for r in rows])) for sid, rows in records.items()}
            train_pq = np.mean([records[sid][selection[sid]]["pq"]
                                for sid in records if not sid.startswith(held)])
            scored.append((float(train_pq), alpha, selection))
        _, alpha, selection = max(scored, key=lambda x: x[0])
        print("RULE_FOLD", held, "alpha", alpha, flush=True)
        rule_predictions.update({sid: choice for sid, choice in selection.items()
                                 if sid.startswith(held)})
    report("RULE_LOMO", rule_predictions, records)

    model_predictions = {}
    for held in subjects:
        X = np.array([r["features"] for sid, rows in records.items()
                      if not sid.startswith(held) for r in rows], np.float32)
        y = np.array([r["pq"] for sid, rows in records.items()
                      if not sid.startswith(held) for r in rows], np.float32)
        model = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15,
                                               learning_rate=0.05,
                                               l2_regularization=5,
                                               min_samples_leaf=20,
                                               random_state=31).fit(X, y)
        for sid, rows in records.items():
            if sid.startswith(held):
                features = np.array([r["features"] for r in rows], np.float32)
                model_predictions[sid] = int(model.predict(features).argmax())
    report("MODEL_LOMO", model_predictions, records)
    print("CALIBRATION_DONE", flush=True)


if __name__ == "__main__":
    main()
