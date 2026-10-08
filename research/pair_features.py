"""Separability of GT pairs among geometric mutual-nearest candidates (GT masks, GT affine)."""
import os
import numpy as np, cv2
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import load_truth, read, label_map, ROOT


def stats(labels, image):
    n = labels.max()
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n + 1)[1:]
    yy, xx = np.indices(labels.shape)
    cx = np.bincount(flat, xx.ravel(), n + 1)[1:] / np.maximum(cnt, 1)
    cy = np.bincount(flat, yy.ravel(), n + 1)[1:] / np.maximum(cnt, 1)
    lo, hi = np.percentile(image, [1, 99.5])
    im = np.clip((image - lo) / (hi - lo), 0, 1)
    mean = np.bincount(flat, im.ravel(), n + 1)[1:] / np.maximum(cnt, 1)
    bg = cv2.GaussianBlur(im, (0, 0), 15)
    contrast = mean - np.bincount(flat, bg.ravel(), n + 1)[1:] / np.maximum(cnt, 1)
    return np.c_[cx, cy], cnt, mean, contrast


truth = load_truth()
X, y, groups = [], [], []
for sid, t in truth.items():
    subj, reg = sid.split("__")
    path = os.path.join(ROOT, "training", subj, reg)
    iv_img, ex_img = read(path + "/invivo.tif"), read(path + "/exvivo.tif")
    ivl, ivid = label_map(t["invivo_instances"], iv_img.shape)
    exl, exid = label_map(t["exvivo_instances"], ex_img.shape)
    ivc, iva, ivm, ivk = stats(ivl, iv_img)
    exc, exa, exm, exk = stats(exl, ex_img)
    ivk_ = {v: k - 1 for k, v in ivid.items()}
    exk_ = {v: k - 1 for k, v in exid.items()}
    pairs = [(ivk_[a], exk_[b]) for a, b in t["match_pairs"] if a in ivk_ and b in exk_]
    if len(pairs) < 4:
        continue
    M, _ = cv2.estimateAffine2D(ivc[[a for a, _ in pairs]].astype(np.float32), exc[[b for _, b in pairs]].astype(np.float32),
                                method=cv2.RANSAC, ransacReprojThreshold=6)
    P = ivc @ M[:, :2].T + M[:, 2]
    d, j = cKDTree(exc).query(P, k=2)
    dback, iback = cKDTree(P).query(exc, k=2)
    gt = set(pairs)
    # rank-normalised brightness within the region
    exr = exm.argsort().argsort() / max(len(exm) - 1, 1)
    ivr = ivm.argsort().argsort() / max(len(ivm) - 1, 1)
    for i in range(len(P)):
        jj = j[i, 0]
        if d[i, 0] < 10 and iback[jj, 0] == i:
            X.append([d[i, 0], d[i, 1] / max(d[i, 0], .5), dback[jj, 1], exm[jj], exk[jj], exa[jj], exr[jj],
                      ivm[i], ivk[i], iva[i], ivr[i], exa[jj] / iva[i]])
            y.append((i, jj) in gt)
            groups.append(subj)
X, y, groups = np.array(X), np.array(y), np.array(groups)
names = "dist ratio2 exNN2 exMean exContrast exArea exRank ivMean ivContrast ivArea ivRank areaRatio".split()
print("candidates", len(y), "gt", y.sum())
for k, n in enumerate(names):
    print(f"  {n:10s} AUC {roc_auc_score(y, X[:, k]):.3f}")
for g in np.unique(groups):
    tr, te = groups != g, groups == g
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    best = max(((2 * ((p >= th) & y[te]).sum() / ((p >= th).sum() + y[te].sum()), th) for th in np.linspace(0.05, 0.9, 35)))
    print(g, "AUC", round(roc_auc_score(y[te], p), 3), "F1 all", round(2 * y[te].sum() / (len(p) + y[te].sum()), 3), "F1 best thr", np.round(best, 3))
