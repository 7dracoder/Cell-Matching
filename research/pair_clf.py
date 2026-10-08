"""Leave-one-mouse-out pair classifier on held-out predictions; reports pair F1."""
import os, sys, pickle
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import transform  # noqa: E402

HERE = os.path.dirname(__file__)
R = pickle.load(open(os.path.join(HERE, "data", "lab.pkl"), "rb"))
TOTAL = sum(r["n_gt_pairs"] for r in R.values())


def rank(v):
    return v.argsort().argsort() / max(len(v) - 1, 1)


def candidates(r, M, score, max_d=10.0):
    """Mutual-nearest candidate pairs under M with per-pair features."""
    if M is None or len(r["ex_c"]) < 2 or len(r["iv_c"]) < 2:
        return np.zeros((0, 2), int), np.zeros((0, 0))
    P = transform(r["iv_c"], M)
    d, j = cKDTree(r["ex_c"]).query(P, k=2)
    db, ib = cKDTree(P).query(r["ex_c"], k=2)
    pairs, feats = [], []
    close5 = int(np.sum((d[:, 0] < 5) & (ib[j[:, 0], 0] == np.arange(len(P)))))
    h, w = r["ex_shape"]
    inside = np.mean((P[:, 0] > 0) & (P[:, 0] < w) & (P[:, 1] > 0) & (P[:, 1] < h))
    ex, iv = r["ex_f"], r["iv_f"]
    exr, ivr = rank(ex["contrast"]), rank(iv["contrast"])
    for i in range(len(P)):
        jj = j[i, 0]
        if d[i, 0] < max_d and ib[jj, 0] == i:
            pairs.append((i, jj))
            feats.append([d[i, 0], d[i, 1], db[jj, 1], ex["mean"][jj], ex["contrast"][jj], ex["area"][jj],
                          exr[jj], iv["mean"][i], iv["contrast"][i], iv["area"][i], ivr[i],
                          score, close5, inside, len(r["ex_c"]), len(r["iv_c"])])
    return np.array(pairs, int).reshape(-1, 2), np.array(feats)


def dataset(regs):
    rows = []
    for sid, (M, score) in regs.items():
        r = R[sid]
        pairs, X = candidates(r, M, score)
        y = np.array([(r["iv_link"][i], r["ex_link"][j]) in r["gt_pairs"] for i, j in pairs], bool)
        rows.append((sid, pairs, X, y))
    return rows


def loo_predict(rows, seed=0):
    subjects = sorted({R[s]["subject"] for s, *_ in rows})
    probs = {}
    for g in subjects:
        tr = [x for x in rows if R[x[0]]["subject"] != g and len(x[2])]
        X = np.vstack([x[2] for x in tr])
        y = np.concatenate([x[3] for x in tr])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=seed).fit(X, y)
        for sid, pairs, Xs, ys in rows:
            if R[sid]["subject"] == g:
                probs[sid] = clf.predict_proba(Xs)[:, 1] if len(Xs) else np.zeros(0)
    return probs


def f1_at(rows, probs, th):
    tp = pred = 0
    for sid, pairs, X, y in rows:
        keep = probs[sid] >= th
        tp += int(y[keep].sum())
        pred += int(keep.sum())
    return tp, pred, 2 * tp / (pred + TOTAL)


def report(name, regs):
    rows = dataset(regs)
    probs = loo_predict(rows)
    ncand = sum(len(x[1]) for x in rows)
    pos = sum(int(x[3].sum()) for x in rows)
    table = [(th,) + f1_at(rows, probs, th) for th in np.arange(0.0, 0.6, 0.025)]
    print("   ", [(round(t[0], 3), round(t[3], 3)) for t in table[:12]])
    best = max(table, key=lambda t: t[3])
    print(f"{name}: candidates {ncand} positives {pos} | best thr {best[0]:.2f} tp {best[1]} pred {best[2]} F1 {best[3]:.4f}")
    # per-mouse F1 at the best threshold
    for g in sorted({R[s]["subject"] for s in R}):
        sub = [x for x in rows if R[x[0]]["subject"] == g]
        tot = sum(R[s]["n_gt_pairs"] for s in R if R[s]["subject"] == g)
        tp = sum(int(x[3][probs[x[0]] >= best[0]].sum()) for x in sub)
        pr = sum(int((probs[x[0]] >= best[0]).sum()) for x in sub)
        print(f"    {g}: tp {tp} pred {pr} gt {tot} F1 {2 * tp / (pr + tot):.3f}")
    return rows, probs, best


if __name__ == "__main__":
    report("GT registration", {s: (r["gt_M"], 100.0) for s, r in R.items()})
    report("old ncc", {s: (r["ncc"][0], r["ncc"][1]) for s, r in R.items()})
    for name in ("window_estimated", "window_oracle"):
        res = pickle.load(open(os.path.join(HERE, "data", f"reg_{name}.pkl"), "rb"))
        report(name, {s: (res[s][0], res[s][1]) for s in R})
