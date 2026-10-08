"""Cache per-region held-out predictions, GT links, GT affine, and current-pipeline registration."""
import os, sys, json, pickle, time
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import rle_to_labels, region_centers, normalize, read_image  # noqa: E402
from registration import register_ncc, estimate_linear_priors  # noqa: E402
from common import load_truth, crop_offsets, ROOT  # noqa: E402

HERE = os.path.dirname(__file__)


def gt_link(pred, gt):
    """pred instance index (0-based) -> gt instance index (0-based) when IoU > 0.75."""
    npred, ngt = int(pred.max()), int(gt.max())
    out = np.full(npred, -1)
    if not npred or not ngt:
        return out
    joint = np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                        minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)
    pc, gc = joint.sum(1), joint.sum(0)
    r, c = np.nonzero(joint[1:, 1:])
    iou = joint[r + 1, c + 1] / (pc[r + 1] + gc[c + 1] - joint[r + 1, c + 1])
    hit = iou > 0.75
    out[r[hit]] = c[hit]
    return out


def instance_stats(labels, image, modality, prob=None):
    n = int(labels.max())
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n + 1)[1:].astype(float)
    norm = normalize(image, modality)
    mean = np.bincount(flat, norm[0].ravel(), n + 1)[1:] / np.maximum(cnt, 1)
    contrast = np.bincount(flat, norm[1].ravel(), n + 1)[1:] / np.maximum(cnt, 1)
    out = {"area": cnt, "mean": mean, "contrast": contrast}
    if prob is not None:
        out["prob"] = np.bincount(flat, prob.astype(np.float32).ravel(), n + 1)[1:] / np.maximum(cnt, 1)
    return out


def main():
    truth = load_truth()
    data = np.load(os.path.join(HERE, "data", "heldout_labels.npz"))
    offsets, _ = crop_offsets("training")
    priors = {s: estimate_linear_priors({s}) for s in {k.split("__")[0] for k in truth}}
    records = {}
    for sid, t in truth.items():
        started = time.time()
        subj, reg = sid.split("__")
        path = os.path.join(ROOT, "training", subj, reg)
        iv_img, ex_img = read_image(path + "/invivo.tif"), read_image(path + "/exvivo.tif")
        ivg, ivg_ids = rle_to_labels(t["invivo_instances"], iv_img.shape)
        exg, exg_ids = rle_to_labels(t["exvivo_instances"], ex_img.shape)
        ivp, exp = data[f"{sid}|invivo"].astype(np.int32), data[f"{sid}|exvivo"].astype(np.int32)
        rec = {"subject": subj, "iv_shape": iv_img.shape, "ex_shape": ex_img.shape, "offset": offsets[(subj, reg)]}
        rec["iv_c"], rec["ex_c"] = region_centers(ivp), region_centers(exp)
        prob = {m: data[f"{sid}|{m}|prob"] if f"{sid}|{m}|prob" in data.files else None for m in ("invivo", "exvivo")}
        rec["iv_f"] = instance_stats(ivp, iv_img, "invivo", prob["invivo"])
        rec["ex_f"] = instance_stats(exp, ex_img, "exvivo", prob["exvivo"])
        rec["iv_link"], rec["ex_link"] = gt_link(ivp, ivg), gt_link(exp, exg)
        ivl = {k: i for i, k in enumerate(ivg_ids)}
        exl = {k: i for i, k in enumerate(exg_ids)}
        rec["gt_pairs"] = {(ivl[a], exl[b]) for a, b in t["match_pairs"] if a in ivl and b in exl}
        rec["n_gt_pairs"] = len(t["match_pairs"])
        gi, ge = region_centers(ivg), region_centers(exg)
        rec["gt_iv_c"] = gi
        pairs = sorted(rec["gt_pairs"])
        rec["gt_M"] = None
        if len(pairs) >= 3:
            M, _ = cv2.estimateAffine2D(gi[[a for a, _ in pairs]], ge[[b for _, b in pairs]],
                                        method=cv2.RANSAC, ransacReprojThreshold=6)
            rec["gt_M"] = M
        rec["ncc"] = register_ncc(rec["iv_c"], rec["ex_c"], priors[subj])
        records[sid] = rec
        print(sid, f"{time.time() - started:.1f}s", flush=True)
    with open(os.path.join(HERE, "data", "lab.pkl"), "wb") as f:
        pickle.dump(records, f)


if __name__ == "__main__":
    main()
