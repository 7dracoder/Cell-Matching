import os, pickle, time
import numpy as np
from multiprocessing import Pool
from hough import hough_register
from reg_lab import err

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))


def run(sid):
    r = R[sid]
    M, score, refined = hough_register(r["iv_c"], r["ex_c"])
    return sid, (M, score, [(s, v, m) for s, v, m in refined[:8]])


if __name__ == "__main__":
    started = time.time()
    with Pool(8) as pool:
        res = dict(pool.map(run, list(R)))
    for subj in sorted({r["subject"] for r in R.values()}):
        sids = [s for s in R if R[s]["subject"] == subj]
        print(subj, "correct", sum(err(R[s], res[s][0]) < 5 for s in sids), "/", len(sids),
              "| correct in top8", sum(any(err(R[s], m) < 5 for _, _, m in res[s][2]) for s in sids))
    print("total", sum(err(R[s], res[s][0]) < 5 for s in R), f"{time.time() - started:.0f}s")
    for s in R:
        e = err(R[s], res[s][0])
        if e >= 5:
            print(f"  BAD {s[-22:]} err {e:7.1f} score {res[s][1]:.1f} | top8 scores/err",
                  [(round(sc, 1), round(err(R[s], m), 1)) for sc, _, m in res[s][2][:4]])
    pickle.dump(res, open(os.path.join(HERE, "data", "reg_hough.pkl"), "wb"))
