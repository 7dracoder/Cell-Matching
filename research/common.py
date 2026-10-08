import csv, sys, json, glob, os
import numpy as np, tifffile, cv2

csv.field_size_limit(sys.maxsize)
ROOT = os.path.join(os.path.dirname(__file__), "..", "Project_2_Dataset")


def rle_indices(rle):
    vals = np.array(rle.split(), dtype=np.int64)
    if vals.size == 0:
        return vals
    return np.concatenate([np.arange(s, s + l) for s, l in zip(vals[::2], vals[1::2])])


def label_map(instances, shape):
    lab = np.zeros(shape[0] * shape[1], np.int32)
    ids = {}
    for n, (key, rle) in enumerate(instances.items(), start=1):
        lab[rle_indices(rle)] = n
        ids[n] = key
    return lab.reshape(shape), ids


def centroids(instances, shape):
    out = {}
    for key, rle in instances.items():
        idx = rle_indices(rle)
        out[key] = np.array([(idx % shape[1]).mean(), (idx // shape[1]).mean()])  # (x, y)
    return out


def load_truth():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "training", "train_ground_truth.csv"))))
    return {r["sample_id"]: {k: json.loads(r[k]) for k in ("invivo_instances", "exvivo_instances", "match_pairs")} for r in rows}


def regions(split):
    out = []
    for subj in sorted(glob.glob(os.path.join(ROOT, split, "subject_*"))):
        for reg in sorted(glob.glob(os.path.join(subj, "region_*"))):
            out.append((os.path.basename(subj), os.path.basename(reg), reg))
    return out


def read(path):
    return tifffile.imread(path).astype(np.float32)


def crop_offsets(split):
    """Offset (x, y) of each in-vivo crop inside the largest in-vivo crop of its mouse."""
    by_subj = {}
    for subj, reg, path in regions(split):
        by_subj.setdefault(subj, []).append((reg, read(os.path.join(path, "invivo.tif"))))
    offsets, refs = {}, {}
    pad = 400
    for subj, items in by_subj.items():
        ref_reg, ref = max(items, key=lambda t: t[1].size)
        refs[subj] = ref_reg
        refp = cv2.copyMakeBorder(ref, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        for reg, im in items:
            h, w = im.shape
            t = im[h // 4:3 * h // 4, w // 4:3 * w // 4]
            res = cv2.matchTemplate(refp, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            offsets[(subj, reg)] = np.array([loc[0] - pad - w // 4, loc[1] - pad - h // 4], float)
    return offsets, refs
