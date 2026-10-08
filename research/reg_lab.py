"""Wide-angle registration variants on held-out predictions; report correctness vs GT affine."""
import os, sys, pickle, time
import numpy as np, cv2
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine, transform, blob_map  # noqa: E402

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
DATA = np.load(os.path.join(HERE, "data", "heldout_labels.npz"))
ANGLES = np.arange(-30, 30.1, 1.5)
SCALES = np.arange(0.89, 1.001, 0.025)
DOWN = 2.0


def target_map(sid, variant):
    r = R[sid]
    shape = (int(np.ceil(r["ex_shape"][0] / DOWN)), int(np.ceil(r["ex_shape"][1] / DOWN)))
    if variant == "points":
        return blob_map(r["ex_c"] / DOWN, shape, 1.2)
    if variant == "wpoints":
        image = np.zeros(shape, np.float32)
        ij = np.clip(np.round(r["ex_c"][:, ::-1] / DOWN).astype(int), 0, np.array(shape) - 1)
        np.add.at(image, (ij[:, 0], ij[:, 1]), np.clip(r["ex_f"]["contrast"], 0.05, 1))
        return cv2.GaussianBlur(image, (0, 0), 1.2)
    prob = DATA[f"{sid}|exvivo|prob"].astype(np.float32)
    p = 1 / (1 + np.exp(-prob))
    p = cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    if variant == "prob":
        return p
    # local peaks only: suppress diffuse bright areas
    return np.clip(p - cv2.GaussianBlur(p, (0, 0), 6), 0, None)


def dog(image, wide):
    return image - cv2.GaussianBlur(image, (0, 0), wide)


def search(iv_c, target, angles, scales, top=5, method=cv2.TM_CCORR_NORMED, wide=None):
    cands = []
    if wide:
        target = dog(target, wide)
    for angle in angles:
        t = np.deg2rad(angle)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        for s in scales:
            lin = s * rot
            proj = iv_c @ lin.T / DOWN
            origin = proj.min(0) - 4
            tp = proj - origin
            tmpl = blob_map(tp, (int(tp[:, 1].max()) + 5, int(tp[:, 0].max()) + 5), 1.2)
            if wide:
                tmpl = dog(tmpl, wide)
            ph, pw = tmpl.shape[0] // 2, tmpl.shape[1] // 2
            img = cv2.copyMakeBorder(target, ph, ph, pw, pw, cv2.BORDER_CONSTANT)
            res = cv2.matchTemplate(img, tmpl, method)
            _, v, _, loc = cv2.minMaxLoc(res)
            shift = (np.array(loc, float) - origin - [pw, ph]) * DOWN
            cands.append((v, np.c_[lin, shift]))
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands[:top]


def run(args):
    sid, variant = args
    r = R[sid]
    base, _, wide = variant.partition("-dog")
    method = cv2.TM_CCORR if wide else cv2.TM_CCORR_NORMED
    cands = search(r["iv_c"], target_map(sid, base), ANGLES, SCALES, method=method,
                   wide=float(wide) if wide else None)
    refined = [refine(r["iv_c"], r["ex_c"], M) + (v,) for v, M in cands]
    return sid, refined


def err(r, M):
    if M is None or r["gt_M"] is None:
        return np.inf
    return float(np.median(np.linalg.norm(transform(r["gt_iv_c"], M) - transform(r["gt_iv_c"], r["gt_M"]), axis=1)))


if __name__ == "__main__":
    out = {}
    for variant in sys.argv[1:]:
        started = time.time()
        with Pool(8) as pool:
            res = dict(pool.map(run, [(s, variant) for s in R]))
        out[variant] = res
        top1 = sum(err(R[s], max(c, key=lambda x: x[1])[0]) < 5 for s, c in res.items())
        anyk = sum(any(err(R[s], m) < 5 for m, _, _ in c) for s, c in res.items())
        ncctop = sum(err(R[s], c[0][0]) < 5 for s, c in res.items())
        print(f"{variant}: correct by refine-score {top1}/47, by ncc rank1 {ncctop}/47, in top5 {anyk}/47 "
              f"({time.time() - started:.0f}s)", flush=True)
    pickle.dump(out, open(os.path.join(HERE, "data", "reg_" + "_".join(sys.argv[1:]) + ".pkl"), "wb"))
