"""Pairs Stage (CPU): pair classifier retrained on the new poses (Req 8).

``research/pair_clf.candidates(r, M, score)`` is a pure function of a record and
is reused unchanged: mutual-nearest pairs within 10 px under M, 16 features.
``pair_clf.dataset`` / ``loo_predict`` read the module-global ``R`` (lab.pkl), so
the same loops are re-implemented here over explicit records, with the same
estimator (``ESTIMATOR``). On lab.pkl records and the same poses, :func:`loo_probs`
equals ``pair_clf.loo_predict(pair_clf.dataset(...))`` exactly.

Records are RegionRecord dicts (``prep["heldout"][sid]`` / ``prep["test"][sid]``,
or ``lab.pkl`` entries) with at least ``iv_c, ex_c, ex_shape, iv_f, ex_f, subject``
and, for labels, ``iv_link, ex_link, gt_pairs``. A pose is ``(M, score)`` or
``{"M": M, "score": score}``; ``M`` (2x3) may be ``None`` (no candidates).

Joint Checkpoint interface consumed by :func:`compute` (``ctx.load("joint")``)::

    {"heldout": {"independent": {sid: {"M": 2x3 | None, "score": float, ...}},
                 "joint":       {sid: {...}}},
     "test":    {"independent": {sid: {...}}, "joint": {sid: {...}}}}

Pairs Checkpoint (output)::

    {"selections": ["independent", "joint"],
     "thresholds": THRESHOLDS,
     "heldout": {sel: {sid: {"subject", "pairs" (n,2) int, "probs" (n,) float,
                             "y" (n,) bool}}},     # LOO probs, model trained without sid's mouse
     "test":    {sel: {sid: {"pairs", "probs"}}},  # all-mice model on sel's held-out poses
     "meta": {...}}

Probabilities are stored raw: the region gate (verifier) and the pair threshold
(:func:`choose_threshold`) are applied later, through :func:`select`.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from hpc_unlock import checkpoint, paths
import pair_clf  # noqa: E402  (research/, on sys.path via paths; loads lab.pkl)

SELECTIONS = ("independent", "joint")
THRESHOLDS = np.round(np.arange(13) * 0.025, 3)          # 0.000 .. 0.300 (Req 8.2)
ESTIMATOR = dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                 l2_regularization=1.0)
V10_TRAIN_POSES = paths.RDATA / "reg_window_vote5.pkl"   # test_apply.REG_TRAIN default

Row = tuple  # (sid, pairs (n,2) int, X (n,16), y (n,) bool | None)


# ----------------------------------------------------------------------------
# Dataset and models
# ----------------------------------------------------------------------------

def _pose(p) -> tuple[Any, float]:
    if p is None:
        return None, 0.0
    if isinstance(p, Mapping):
        return p.get("M"), float(p.get("score", 0.0))
    M, score = p[0], p[1]
    return M, float(score)


def make_estimator(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**ESTIMATOR, random_state=seed)


def labels(r: Mapping, pairs: np.ndarray) -> np.ndarray:
    """``(iv_link[i], ex_link[j]) in gt_pairs`` per candidate (Req 8.1)."""
    return np.array([(r["iv_link"][i], r["ex_link"][j]) in r["gt_pairs"] for i, j in pairs], bool)


def dataset(records: Mapping[str, Mapping], poses: Mapping[str, Any],
            with_labels: bool = True) -> list[Row]:
    """``pair_clf.dataset`` over explicit records; rows follow ``poses`` order."""
    rows = []
    for sid, p in poses.items():
        r = records[sid]
        M, score = _pose(p)
        pairs, X = pair_clf.candidates(r, M, score)
        y = labels(r, pairs) if with_labels else None
        rows.append((sid, pairs, X, y))
    return rows


class ConstantModel:
    """Stand-in when a training set is empty or has one class (smoke runs)."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def fit(rows: Sequence[Row], seed: int = 0,
        estimator: Callable[[int], Any] = make_estimator):
    """Fit on all rows with candidates (as ``pair_clf.loo_predict`` / ``train_classifier``)."""
    tr = [x for x in rows if len(x[2])]
    if not tr:
        return ConstantModel(0.0)
    X = np.vstack([x[2] for x in tr])
    y = np.concatenate([x[3] for x in tr])
    if len(np.unique(y)) < 2:
        return ConstantModel(float(y[0]))
    return estimator(seed).fit(X, y)


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)


def loo_predict(rows: Sequence[Row], subject_of: Mapping[str, str], seed: int = 0,
                estimator: Callable[[int], Any] = make_estimator) -> dict[str, np.ndarray]:
    """Leave-one-mouse-out probabilities: each mouse scored by a model trained
    only on the other mice's rows (Req 8.1)."""
    subjects = sorted({subject_of[s] for s, *_ in rows})
    probs = {}
    for g in subjects:
        model = fit([x for x in rows if subject_of[x[0]] != g], seed, estimator)
        for sid, pairs, X, y in rows:
            if subject_of[sid] == g:
                probs[sid] = predict(model, X)
    return probs


def loo_probs(records: Mapping[str, Mapping], poses: Mapping[str, Any], seed: int = 0,
              estimator: Callable[[int], Any] = make_estimator) -> dict[str, np.ndarray]:
    """LOO probabilities for every region in ``poses`` (``sid -> (n,)``)."""
    rows = dataset(records, poses)
    return loo_predict(rows, {s: records[s]["subject"] for s in poses}, seed, estimator)


def test_model(records: Mapping[str, Mapping], poses: Mapping[str, Any], seed: int = 0):
    """Model trained on all held-out mice under ``poses`` (Req 8.5)."""
    return fit(dataset(records, poses), seed)


def v10_test_model():
    """The v10 test classifier exactly (``test_apply.train_classifier``): lab.pkl
    records under the ``reg_window_vote5.pkl`` poses, all mice, seed 0."""
    import pickle
    with open(V10_TRAIN_POSES, "rb") as f:
        res = pickle.load(f)
    return test_model(pair_clf.R, {s: (res[s][0], res[s][1]) for s in pair_clf.R})


# ----------------------------------------------------------------------------
# Selection and threshold
# ----------------------------------------------------------------------------

def select(pairs, probs, threshold: float, kept: bool) -> np.ndarray:
    """One-to-one pairs of a region (Req 8.3, 8.4).

    Empty if the region is not kept. Otherwise candidates with ``prob >= threshold``
    are accepted greedily in descending prob (ties: lower candidate index first),
    skipping any whose in-vivo or ex-vivo index is already used. Accepted pairs
    are returned in candidate order, shape ``(k, 2)``.
    """
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    probs = np.asarray(probs, float).reshape(-1)
    if len(pairs) != len(probs):
        raise ValueError(f"{len(pairs)} pairs vs {len(probs)} probabilities")
    if not kept or not len(pairs):
        return np.zeros((0, 2), int)
    used_i, used_j, take = set(), set(), []
    for k in np.argsort(-probs, kind="stable"):
        if probs[k] < threshold:
            break
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        take.append(k)
    return pairs[np.sort(np.array(take, int))] if take else np.zeros((0, 2), int)


def select_mask(pairs, probs, threshold: float, kept: bool) -> np.ndarray:
    """Boolean mask over candidates of :func:`select`'s accepted pairs."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    chosen = {tuple(p) for p in select(pairs, probs, threshold, kept).tolist()}
    return np.array([tuple(p) in chosen for p in pairs.tolist()], bool)


def threshold_table(rows: Iterable[Row], probs: Mapping[str, np.ndarray],
                    kept_by_region: Mapping[str, bool], total_gt: int,
                    thresholds: Sequence[float] = THRESHOLDS) -> list[tuple]:
    """``[(thr, tp, pred, f1)]``; F1 = 2tp / (pred + total_gt), pooled over rows.

    ``rows`` items are ``(sid, pairs, ..., y)`` (``pair_clf`` rows or
    :func:`entry_rows`); ``y`` is the candidate TP label.
    """
    rows = list(rows)
    table = []
    for th in thresholds:
        tp = pred = 0
        for row in rows:
            sid, pairs, y = row[0], row[1], np.asarray(row[-1], bool)
            m = select_mask(pairs, probs[sid], th, bool(kept_by_region[sid]))
            tp += int(y[m].sum())
            pred += int(m.sum())
        denom = pred + total_gt
        table.append((float(th), tp, pred, 2 * tp / denom if denom else 0.0))
    return table


def best_threshold(table: Sequence[tuple]) -> tuple:
    """Row with max F1; the lowest threshold on ties (Req 8.2)."""
    best = None
    for row in sorted(table, key=lambda t: t[0]):
        if best is None or row[3] > best[3]:
            best = row
    return best


def choose_threshold(rows: Iterable[Row], probs: Mapping[str, np.ndarray],
                     kept_by_region: Mapping[str, bool], total_gt: int) -> tuple:
    """``(thr, tp, pred, f1)`` over the 13 thresholds, lowest on ties (Req 8.2)."""
    return best_threshold(threshold_table(rows, probs, kept_by_region, total_gt))


def entry_rows(entries: Mapping[str, Mapping]) -> tuple[list[Row], dict[str, np.ndarray]]:
    """``(rows, probs)`` from one selection of the Checkpoint's ``heldout`` part."""
    rows = [(sid, e["pairs"], e["y"]) for sid, e in entries.items()]
    return rows, {sid: e["probs"] for sid, e in entries.items()}


# ----------------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------------

def selection_probs(heldout: Mapping[str, Mapping], test: Mapping[str, Mapping],
                    hposes: Mapping[str, Any], tposes: Mapping[str, Any],
                    seed: int = 0) -> tuple[dict, dict]:
    """Held-out LOO entries and test entries for one selection."""
    hposes = {s: hposes.get(s) for s in heldout}
    rows = dataset(heldout, hposes)
    subj = {s: heldout[s]["subject"] for s in heldout}
    probs = loo_predict(rows, subj, seed)
    hout = {sid: {"subject": subj[sid], "pairs": pairs, "probs": probs[sid], "y": y}
            for sid, pairs, X, y in rows}
    model = fit(rows, seed)
    tout = {}
    for sid, pairs, X, _ in dataset(test, {s: tposes.get(s) for s in test}, with_labels=False):
        tout[sid] = {"pairs": pairs, "probs": predict(model, X)}
    return hout, tout


def compute(cfg, ctx) -> dict:
    t0 = time.monotonic()
    prep = ctx.load("prep")
    joint = ctx.load("joint")
    out = {"selections": list(SELECTIONS), "thresholds": THRESHOLDS,
           "heldout": {}, "test": {}}
    total = sum(int(r["n_gt_pairs"]) for r in prep["heldout"].values())
    for sel in SELECTIONS:
        hout, tout = selection_probs(prep["heldout"], prep["test"],
                                     joint["heldout"][sel], joint["test"][sel])
        out["heldout"][sel], out["test"][sel] = hout, tout
        rows, probs = entry_rows(hout)
        thr, tp, pred, f1 = choose_threshold(rows, probs, {s: True for s in hout}, total)
        ncand = sum(len(e["pairs"]) for e in hout.values())
        npos = sum(int(e["y"].sum()) for e in hout.values())
        checkpoint.log_line("PAIRS_SELECTION",
                            f"{sel}: heldout={len(hout)} candidates={ncand} positives={npos} "
                            f"ungated best thr={thr:.3f} tp={tp} pred={pred} F1={f1:.4f} "
                            f"test={len(tout)} elapsed={time.monotonic() - t0:.1f}s")
    out["meta"] = {"smoke": bool(getattr(ctx, "smoke", False)), "total_gt_pairs": total,
                   "estimator": dict(ESTIMATOR, random_state=0)}
    return out
