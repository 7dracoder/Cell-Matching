"""GT affine from in-vivo mosaic coordinates to each ex-vivo canvas; ex-vivo image sharing."""
import os, hashlib
import numpy as np, cv2
from common import load_truth, read, centroids, crop_offsets, regions, ROOT

truth = load_truth()
offsets, refs = crop_offsets("training")
by_subj = {}
for sid, t in truth.items():
    subj, reg = sid.split("__")
    path = os.path.join(ROOT, "training", subj, reg)
    iv_img, ex_img = read(path + "/invivo.tif"), read(path + "/exvivo.tif")
    ivc = centroids(t["invivo_instances"], iv_img.shape)
    exc = centroids(t["exvivo_instances"], ex_img.shape)
    pairs = [(a, b) for a, b in t["match_pairs"] if a in ivc and b in exc]
    if len(pairs) < 4:
        continue
    A = np.array([ivc[a] for a, _ in pairs], np.float32) + offsets[(subj, reg)].astype(np.float32)
    B = np.array([exc[b] for _, b in pairs], np.float32)
    M, _ = cv2.estimateAffine2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=6)
    h = hashlib.md5(ex_img.tobytes()).hexdigest()[:6]
    hi = hashlib.md5(iv_img.tobytes()).hexdigest()[:6]
    by_subj.setdefault(subj, []).append((reg, M, ex_img.shape, h, hi, offsets[(subj, reg)]))

for subj, items in by_subj.items():
    print(subj, "ref", refs[subj])
    probe = np.array([[300, 300]], np.float32)
    for reg, M, shape, h, hi, off in items:
        rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        sc = np.sqrt(abs(np.linalg.det(M[:, :2])))
        shear = np.degrees(np.arctan2(M[1, 1], M[0, 1]) - np.arctan2(M[1, 0], M[0, 0])) - 90
        land = (probe @ M[:, :2].T + M[:, 2])[0]
        print(f"  {reg[-6:]} ex {shape} exhash {h} ivhash {hi} off {off.astype(int)} rot {rot:6.1f} scale {sc:.3f} "
              f"shear {shear:5.1f} mosaic(300,300)->{land.round(0)}")
