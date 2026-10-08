import sys, os, numpy as np
sys.path.insert(0, ".."); sys.path.insert(0, ".")
from common import load_truth, ROOT
from cellmatch import rle_to_labels, read_image
truth = load_truth()
labs = {k: np.load(f"data/{k}") for k in ["v7/heldout_labels.npz", "heldout_labels_ensb.npz", "heldout_labels_v2a.npz"]}
res = {}
for sid, t in truth.items():
    subj = sid.split("__")[0][-6:]
    path = os.path.join(ROOT, "training", *sid.split("__"))
    for m in ("invivo", "exvivo"):
        shape = read_image(f"{path}/{m}.tif").shape
        g, _ = rle_to_labels(t[f"{m}_instances"], shape)
        res.setdefault((subj, m, "GT"), []).extend(np.bincount(g.ravel())[1:][np.bincount(g.ravel())[1:] > 0])
        for k, d in labs.items():
            p = d[f"{sid}|{m}"]
            c = np.bincount(p.ravel().astype(np.int64))[1:]
            res.setdefault((subj, m, k.split("/")[0][:14]), []).extend(c[c > 0])
for k in sorted(res):
    print(k, len(res[k]), np.median(res[k]))
