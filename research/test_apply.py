"""Re-match test regions: keep submitted masks, new registration + pair classifier."""
import os, sys, csv, json, pickle
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import rle_to_labels, region_centers, read_image  # noqa: E402
from common import crop_offsets, ROOT  # noqa: E402
from lab_build import instance_stats  # noqa: E402
from hough import window_register  # noqa: E402
from vote import region_candidates, vote_modes  # noqa: E402
from window_lab import pose, P0  # noqa: E402
import pair_clf  # noqa: E402
from margin_lab import margin  # noqa: E402

HERE = os.path.dirname(__file__)
csv.field_size_limit(sys.maxsize)
SRC = os.environ.get("MASKS_CSV", os.path.join(HERE, "data", "submission.csv"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "submission_rematch.csv")
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
MIN_MARGIN = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
# "k_phi": undo sectioning compression k along ex-frame axis phi before registration; "" = off.
STRETCH = os.environ.get("STRETCH", "")
REG_TRAIN = os.environ.get("REG_TRAIN", os.path.join(HERE, "data", "reg_window_vote5.pkl"))


def stretch_matrix():
    if not STRETCH:
        return np.eye(2)
    k, phi = (float(x) for x in STRETCH.split("_"))
    u = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    return np.eye(2) + (1 / k - 1) * np.outer(u, u)


def build(row):
    sid = row["sample_id"]
    path = os.path.join(ROOT, "hidden_test", *sid.split("__"))
    iv_img, ex_img = read_image(path + "/invivo.tif"), read_image(path + "/exvivo.tif")
    iv_ids, ex_ids = list(json.loads(row["invivo_instances"])), list(json.loads(row["exvivo_instances"]))
    ivl, _ = rle_to_labels(json.loads(row["invivo_instances"]), iv_img.shape)
    exl, _ = rle_to_labels(json.loads(row["exvivo_instances"]), ex_img.shape)
    rec = {"subject": sid.split("__")[0], "iv_shape": iv_img.shape, "ex_shape": ex_img.shape,
           "iv_c": region_centers(ivl), "ex_c": region_centers(exl),
           "iv_f": instance_stats(ivl, iv_img, "invivo"), "ex_f": instance_stats(exl, ex_img, "exvivo"),
           "iv_ids": iv_ids, "ex_ids": ex_ids}
    rec["key"] = hash(rec["iv_c"].tobytes() + rec["ex_c"].tobytes())
    return sid, rec


def cands(args):
    sid, rec = args
    return sid, region_candidates(rec["iv_c"], rec["ex_c"] @ stretch_matrix().T, rec["offset"])


def window(args):
    sid, rec, modes = args
    M, score = window_register(rec["iv_c"], rec["ex_c"] @ stretch_matrix().T, rec["offset"], modes, P0, angle_win=5.0)
    return sid, (M, score, None if M is None else np.linalg.inv(stretch_matrix()) @ M)


def train_classifier():
    res = pickle.load(open(REG_TRAIN, "rb"))
    rows = pair_clf.dataset({s: (res[s][0], res[s][1]) for s in pair_clf.R})
    X = np.vstack([x[2] for x in rows if len(x[2])])
    y = np.concatenate([x[3] for x in rows if len(x[2])])
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0).fit(X, y)


if __name__ == "__main__":
    rows = list(csv.DictReader(open(SRC)))
    with Pool(8) as pool:
        recs = dict(pool.map(build, rows))
    offsets, refs = crop_offsets("hidden_test")
    for sid, rec in recs.items():
        rec["offset"] = offsets[tuple(sid.split("__"))]
    with Pool(8) as pool:
        C = dict(pool.map(cands, list(recs.items())))
    groups = {}
    for sid, rec in recs.items():
        groups.setdefault((rec["subject"], rec["ex_shape"]), []).append(sid)
    jobs = []
    for key, sids in groups.items():
        seen, voters = set(), {}
        for s in sids:
            if recs[s]["key"] not in seen:
                seen.add(recs[s]["key"])
                voters[s] = C[s]
        modes = vote_modes(voters)
        print(key, "voters", len(voters), "modes",
              [(round(m["angle"], 1), m["landing"].round(), m["n"], round(m["support"], 1)) for m in modes])
        jobs += [(s, recs[s], modes) for s in sids]
    with Pool(8) as pool:
        win = dict(pool.map(window, jobs))
    clf = train_classifier()
    out_rows, total = [], 0
    for row in rows:
        sid = row["sample_id"]
        rec = recs[sid]
        M, score, M_true = win[sid]
        mg = margin(C[sid], M, score, rec)
        pairs, X = pair_clf.candidates(rec, M_true, score)
        keep = clf.predict_proba(X)[:, 1] >= THRESHOLD if len(X) else np.zeros(0, bool)
        if mg < MIN_MARGIN:
            keep[:] = False
        chosen = [[rec["iv_ids"][i], rec["ex_ids"][j]] for (i, j), k in zip(pairs, keep) if k]
        total += len(chosen)
        a, l = pose(rec, M_true) if M_true is not None else (np.nan, np.array([np.nan, np.nan]))
        print(f"{sid[-22:]} window score {score:5.1f} margin {mg:6.1f} angle {a:6.1f} landing {np.round(l)} "
              f"pairs {len(chosen)} (old {len(json.loads(row['match_pairs']))})")
        out_rows.append({**row, "match_pairs": json.dumps(chosen)})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("saved", OUT, "pairs", total)
