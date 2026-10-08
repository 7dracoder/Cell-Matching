import os, sys, pickle, time
import numpy as np
from multiprocessing import Pool
from consensus import landing_volume, consensus_modes, constrained_register, P0
from reg_lab import err

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))


def vol(sid):
    r = R[sid]
    return sid, landing_volume(r["iv_c"], r["ex_c"], r["ex_shape"], r["offset"])


def reg(args):
    sid, modes = args
    r = R[sid]
    best = None
    for m in modes:
        out = constrained_register(r["iv_c"], r["ex_c"], r["ex_shape"], r["offset"], m)
        if best is None or out[1] > best[1]:
            best = out + (m["angle"],)
    return sid, best


if __name__ == "__main__":
    started = time.time()
    with Pool(8) as pool:
        vols = dict(pool.map(vol, list(R)))
    print(f"volumes {time.time() - started:.0f}s", flush=True)
    results = {}
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        modes = consensus_modes([vols[s] for s in sids])
        # GT consensus for reference
        gt = [(np.degrees(np.arctan2(R[s]["gt_M"][1, 0], R[s]["gt_M"][0, 0])),
               R[s]["gt_M"][:, :2] @ (P0 - R[s]["offset"]) + R[s]["gt_M"][:, 2]) for s in sids if R[s]["gt_M"] is not None]
        print(subj, "modes", [(m["angle"], m["scale"], m["landing"].round(), round(m["value"], 1)) for m in modes])
        print("   GT angles", np.round(sorted(a for a, _ in gt), 1), "GT landing median", np.median([l for _, l in gt], 0).round())
        with Pool(8) as pool:
            for sid, res in pool.map(reg, [(s, modes) for s in sids]):
                results[sid] = res
        ok = sum(err(R[s], results[s][0]) < 5 for s in sids)
        print(f"   correct {ok}/{len(sids)}", flush=True)
    print("total correct", sum(err(R[s], results[s][0]) < 5 for s in R), "/", len(R), f"{time.time() - started:.0f}s")
    pickle.dump(results, open(os.path.join(HERE, "data", "reg_consensus.pkl"), "wb"))
