"""Held-out pair F1 when poses are chosen by the soft objective, gated by soft margin."""
import pickle
import numpy as np
import pair_clf
from margin_lab import R
from reg_lab import err
from window_lab import pose

W = pickle.load(open("data/wide_soft_train.pkl", "rb"))


def choose(r, cs):
    k = int(np.argmax([c[2] for c in cs]))
    M, sc, so = cs[k]
    a, l = pose(r, M)
    alt = [c[2] for c in cs if abs(pose(r, c[0])[0] - a) > 3 or np.linalg.norm(pose(r, c[0])[1] - l) > 60]
    second = max(alt) if alt else 0.0
    return M, sc, so, so - second, so / max(second, 1e-6)


if __name__ == "__main__":
    info = {s: choose(R[s], W[s]) for s in R}
    regs = {s: (info[s][0], info[s][1]) for s in R}
    rows = pair_clf.dataset(regs)
    probs = pair_clf.loo_predict(rows)
    print("correct", sum(err(R[s], info[s][0]) < 5 for s in R))
    for s in sorted(R, key=lambda s: info[s][3]):
        print(f"   {s[-15:]} soft {info[s][2]:6.1f} margin {info[s][3]:6.2f} ratio {info[s][4]:5.2f} ok {err(R[s], info[s][0]) < 5}")
    for kind, idx, cuts in (("margin", 3, (0, 1, 2, 3, 4, 5, 6)), ("ratio", 4, (1.0, 1.1, 1.15, 1.2, 1.3, 1.4))):
        for c in cuts:
            keep = {s: info[s][idx] >= c for s in R}
            best = max((2 * sum(int(y[probs[s] >= th].sum()) for s, _, _, y in rows if keep[s]) /
                        (sum(int((probs[s] >= th).sum()) for s, _, _, y in rows if keep[s]) + pair_clf.TOTAL), th)
                       for th in np.arange(0, 0.3, 0.025))
            print(f"{kind}>={c}: F1 {best[0]:.4f} thr {best[1]:.3f} kept {sum(keep.values())} "
                  f"correct {sum(err(R[s], info[s][0]) < 5 for s in R if keep[s])}")
