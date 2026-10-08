"""How are GT pairs chosen? Compare GT pairs with geometric mutual-nearest pairs under the GT affine."""
import os, json
import numpy as np, cv2
from scipy.spatial import cKDTree
from common import load_truth, read, centroids, rle_indices, ROOT

truth = load_truth()
rows = []
tot = {"gt": 0, "mnn": 0, "mnn_gt": 0, "gt_resid_gt8": 0}
feat = {"paired": [], "unpaired_covered": []}
for sid, t in truth.items():
    subj, reg = sid.split("__")
    path = os.path.join(ROOT, "training", subj, reg)
    iv_img, ex_img = read(path + "/invivo.tif"), read(path + "/exvivo.tif")
    ivc = centroids(t["invivo_instances"], iv_img.shape)
    exc = centroids(t["exvivo_instances"], ex_img.shape)
    pairs = [(a, b) for a, b in t["match_pairs"] if a in ivc and b in exc]
    if len(pairs) < 4:
        print("few pairs", sid, len(pairs), len(t["match_pairs"]))
        continue
    A = np.array([ivc[a] for a, _ in pairs], np.float32)
    B = np.array([exc[b] for _, b in pairs], np.float32)
    M, inl = cv2.estimateAffine2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=6)
    resid = np.linalg.norm(A @ M[:, :2].T + M[:, 2] - B, axis=1)
    iv_keys, ex_keys = list(ivc), list(exc)
    P = np.array([ivc[k] for k in iv_keys]) @ M[:, :2].T + M[:, 2]
    E = np.array([exc[k] for k in ex_keys])
    d1, j1 = cKDTree(E).query(P)
    d2, i2 = cKDTree(P).query(E)
    gtset = set(pairs)
    mnn = [(iv_keys[i], ex_keys[j1[i]], d1[i]) for i in range(len(P)) if i2[j1[i]] == i and d1[i] < 8]
    mnn_gt = sum((a, b) in gtset for a, b, _ in mnn)
    # ex cells covered by the projected in-vivo field
    h, w = iv_img.shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) @ M[:, :2].T + M[:, 2]
    inside = np.array([cv2.pointPolygonTest(corners.astype(np.float32), tuple(map(float, e)), True) for e in E])
    covered = inside > 10
    paired_ex = {b for _, b in pairs}
    exn = ex_img / np.percentile(ex_img, 99.5)
    for k, e in zip(ex_keys, E):
        idx = rle_indices(t["exvivo_instances"][k])
        f = (exn.ravel()[idx].mean(), len(idx))
        if k in paired_ex:
            feat["paired"].append(f)
    for k, e, c in zip(ex_keys, E, covered):
        if c and k not in paired_ex:
            idx = rle_indices(t["exvivo_instances"][k])
            feat["unpaired_covered"].append((exn.ravel()[idx].mean(), len(idx)))
    rows.append((sid[-20:], len(ivc), len(exc), int(covered.sum()), len(pairs), len(mnn), mnn_gt,
                 float(np.median(resid)), int((resid > 8).sum()), round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 1),
                 round(float(np.sqrt(abs(np.linalg.det(M[:, :2])))), 3)))
    tot["gt"] += len(pairs); tot["mnn"] += len(mnn); tot["mnn_gt"] += mnn_gt; tot["gt_resid_gt8"] += int((resid > 8).sum())

print("sid nIV nEX exCovered gtPairs mnn<8 mnn&gt medResid resid>8 rot scale")
for r in rows:
    print(*r)
print(tot)
for k, v in feat.items():
    v = np.array(v)
    print(k, len(v), "bright q10/50/90", np.percentile(v[:, 0], [10, 50, 90]).round(3), "area q10/50/90", np.percentile(v[:, 1], [10, 50, 90]))
