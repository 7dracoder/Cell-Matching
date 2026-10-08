"""v7 test matching with an extra region gate: keep pairs if margin >= MIN_MARGIN OR cellprob z >= ZMIN.

Same poses, classifier and masks as test_apply.py (submission_v7*). The cellprob z-score is from
cp_pose_lab.cp_z on research/data/test_cp_base.npz (ex-vivo Cellpose cellprob of the test masks).
Pairs are written into the masks of MASKS_CSV (default: ../submission_v7_grow15.csv).

usage: test_cp_apply.py OUT.csv [THRESHOLD=0.025] [MIN_MARGIN=3] [ZMIN=5]
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

# Poses/pairs are computed on the ungrown v7 masks (exactly as for submission_v7.csv); the pairs are
# then written into WRITE_CSV (grow15 keeps the same ids, only boundaries differ).
os.environ.setdefault("MASKS_CSV", os.path.join(os.path.dirname(__file__), "data", "submission.csv"))
WRITE_CSV = os.environ.get("WRITE_CSV", os.path.join(os.path.dirname(__file__), "..", "submission_v7_grow15.csv"))
import test_apply as T  # noqa: E402
from common import crop_offsets  # noqa: E402
from cp_pose_lab import cp_map, cp_z  # noqa: E402
from margin_lab import margin  # noqa: E402
from vote import vote_modes  # noqa: E402
import pair_clf  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = sys.argv[1]
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.025
MIN_MARGIN = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
ZMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

if __name__ == "__main__":
    rows = list(csv.DictReader(open(T.SRC)))
    with Pool(8) as pool:
        recs = dict(pool.map(T.build, rows))
    offsets, _ = crop_offsets("hidden_test")
    for sid, rec in recs.items():
        rec["offset"] = offsets[tuple(sid.split("__"))]
    with Pool(8) as pool:
        C = dict(pool.map(T.cands, list(recs.items())))
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
        jobs += [(s, recs[s], modes) for s in sids]
    with Pool(8) as pool:
        win = dict(pool.map(T.window, jobs))
    clf = T.train_classifier()
    cp = np.load(os.environ.get("CP_NPZ", os.path.join(HERE, "data", "test_cp_base.npz")))
    write_rows = {r["sample_id"]: r for r in csv.DictReader(open(WRITE_CSV))}
    out_rows, total, nreg = [], 0, 0
    for row in rows:
        target = write_rows[row["sample_id"]]
        assert list(json.loads(target["exvivo_instances"])) == recs[row["sample_id"]]["ex_ids"]
        assert list(json.loads(target["invivo_instances"])) == recs[row["sample_id"]]["iv_ids"]
        sid = row["sample_id"]
        rec = recs[sid]
        M, score, M_true = win[sid]
        mg = margin(C[sid], M, score, rec)
        z = cp_z(cp_map(cp[f"{sid}|cp"].astype(np.float32)), rec["iv_c"], M_true) if M_true is not None else -9
        pairs, X = pair_clf.candidates(rec, M_true, score)
        keep = clf.predict_proba(X)[:, 1] >= THRESHOLD if len(X) else np.zeros(0, bool)
        gate = mg >= MIN_MARGIN or z >= ZMIN
        if not gate:
            keep[:] = False
        chosen = [[rec["iv_ids"][i], rec["ex_ids"][j]] for (i, j), k in zip(pairs, keep) if k]
        total += len(chosen)
        nreg += bool(chosen)
        print(f"{sid[-22:]} score {score:5.1f} margin {mg:6.1f} z {z:6.2f} gate {'Y' if gate else '-'} "
              f"pairs {len(chosen)} (write-src {len(json.loads(target['match_pairs']))})")
        out_rows.append({**target, "match_pairs": json.dumps(chosen)})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("saved", OUT, "pairs", total, "regions", nreg)
