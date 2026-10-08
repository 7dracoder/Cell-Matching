"""Held-out PQ sweep: diameter scale, classical merge, multi-threshold union.

Env: TAGS=base|_v2|base+_v2  GRID=cellprob:flow,...  DIAMS=0.7,0.85,1,1.15,1.3
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/content/work")
import pipeline as P  # noqa: E402
from cellmatch import classical_segment, pq_score  # noqa: E402

TAGS = ["" if t == "base" else t for t in os.environ.get("TAGS", "base").split("+")]
GRID = [tuple(map(float, g.split(":"))) for g in
        os.environ.get("GRID", "0:0.1,-0.5:0.1,-0.5:0.15,0:0.15").split(",")]
DIAMS = [float(x) for x in os.environ.get("DIAMS", "1.0").split(",")]
MERGE = os.environ.get("MERGE", "0") == "1"  # add classical peaks missing from cellpose
MOD = os.environ.get("MOD", "exvivo")


def flows(models, image, diam_scale):
    dps, cps, ds = [], [], []
    for model in models:
        d = float(model.net.diam_labels.item()) * diam_scale
        _, o, _ = model.eval(image.astype(np.float32), diameter=d, compute_masks=False, batch_size=16)
        dps.append(o[1]); cps.append(o[2]); ds.append(d)
    return np.mean(dps, 0).astype(np.float16), np.mean(cps, 0).astype(np.float16), int(200 * np.mean(ds) / 30)


def merge_classical(lab, image, modality):
    """Add classical instances that don't overlap an existing cell (IoU-free: no pixel overlap)."""
    classic = classical_segment(image, modality)
    out = lab.copy()
    next_id = int(out.max()) + 1
    for cid in range(1, int(classic.max()) + 1):
        mask = classic == cid
        if not mask.any():
            continue
        if out[mask].any():
            continue  # overlaps existing
        out[mask] = next_id
        next_id += 1
    return out


t0 = time.time()
truth = P.load_truth()
results = {}  # diam -> cfg -> sid -> pq
for subject in P.SUBJECTS:
    models = [P.cellpose_model(P.cellpose_path(MOD, f"fold_{subject}{t}")) for t in TAGS]
    sids = [s for s, t in truth.items() if t["subject"] == subject]
    for sid in sids:
        img = P.read_image(P.region_path(sid) / f"{MOD}.tif")
        gt = truth[sid][MOD][0]
        for dscale in DIAMS:
            fl = flows(models, img, dscale)
            for c, f in GRID:
                lab = P.cellpose_labels(fl, MOD, {"cellprob": c, "flow": f})
                if MERGE:
                    lab = merge_classical(lab, img, MOD)
                results.setdefault(dscale, {}).setdefault(f"{c}|{f}", {})[sid] = pq_score(lab, gt)[0]
    print(subject, "done", round(time.time() - t0), flush=True)

for dscale, grid in results.items():
    rows = sorted(((float(np.mean(list(v.values()))), k) for k, v in grid.items()), reverse=True)
    best = rows[0]
    per = {s[-6:]: round(float(np.mean([v for sid, v in grid[best[1]].items() if sid.startswith(s)])), 3)
           for s in P.SUBJECTS}
    print("RESULT", "+".join(TAGS) or "base", "diam", dscale, "merge", int(MERGE),
          "best", round(best[0], 4), best[1], per, "| top3", [(round(a, 4), b) for a, b in rows[:3]], flush=True)
