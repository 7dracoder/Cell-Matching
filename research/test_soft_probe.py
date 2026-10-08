"""Test regions: wide soft-objective pose vs the v10 window pose; agreement, margin, z."""
import csv, json, os, pickle, sys
from multiprocessing import Pool
import numpy as np
os.environ.setdefault("MASKS_CSV", os.path.join(os.path.dirname(__file__), "data", "submission.csv"))
import test_apply as T
from common import crop_offsets
from vote import vote_modes
from margin_lab import margin
from cp_pose_lab import cp_map, cp_z
import wide_soft
from soft_gate_cv import choose
sys.path.insert(0, "..")
from registration import transform

wide_soft.STRETCH = [(1.0, 0.0)]

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
                seen.add(recs[s]["key"]); voters[s] = C[s]
        modes = vote_modes(voters)
        jobs += [(s, recs[s], modes) for s in sids]
    with Pool(8) as pool:
        win = dict(pool.map(T.window, jobs))
    wjobs = [(s, recs[s]["iv_c"], recs[s]["ex_c"], ([(win[s][0], win[s][1])] if win[s][0] is not None else []) +
              [(c[3], c[0]) for c in C[s]]) for s in recs]
    with Pool(8) as pool:
        Wd = dict(pool.map(wide_soft.run, wjobs))
    cp = np.load("data/test_cp_base.npz")
    out = {}
    for row in rows:
        s = row["sample_id"]; rec = recs[s]
        M, score, _ = win[s]
        mg = margin(C[s], M, score, rec)
        z = cp_z(cp_map(cp[f"{s}|cp"].astype(np.float32)), rec["iv_c"], M) if M is not None else -9
        Ms, scs, so, sm, sr = choose(rec, Wd[s])
        agree = M is not None and np.median(np.linalg.norm(transform(rec["iv_c"], M) - transform(rec["iv_c"], Ms), axis=1)) < 5
        zs = cp_z(cp_map(cp[f"{s}|cp"].astype(np.float32)), rec["iv_c"], Ms)
        v10 = mg >= 3 or z >= 5
        out[s] = dict(M_win=M, score=score, margin=mg, z=z, M_soft=Ms, soft=so, soft_margin=sm, agree=agree, z_soft=zs, v10=v10)
        print(f"{s[-22:]} v10gate {'Y' if v10 else '-'} margin {mg:6.1f} z {z:5.2f} | soft {so:5.1f} smargin {sm:5.2f} zsoft {zs:5.2f} agree {agree}")
    pickle.dump((out, recs, C, win, Wd), open("data/test_soft.pkl", "wb"))
