"""Unit tests for hpc_unlock.pairs (Req 8.1-8.5)."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from hpc_unlock import pairs as P
from hpc_unlock import paths
from hpc_unlock.prep import _research_import_env


# ----------------------------------------------------------------- select
def test_select_one_to_one_by_descending_prob():
    pairs = np.array([[0, 0], [0, 1], [1, 1], [2, 2], [3, 2]])
    probs = np.array([0.5, 0.9, 0.6, 0.4, 0.4])
    out = P.select(pairs, probs, 0.0, True)
    # (0,1) first takes iv 0 and ex 1 -> (0,0), (1,1) skipped; tie at 0.4 -> lower index (2,2)
    assert out.tolist() == [[0, 1], [2, 2]]
    assert len({i for i, _ in out}) == len(out) == len({j for _, j in out})


def test_select_threshold_and_gate():
    pairs = np.array([[0, 0], [1, 1], [2, 2]])
    probs = np.array([0.1, 0.025, 0.02])
    assert P.select(pairs, probs, 0.025, True).tolist() == [[0, 0], [1, 1]]
    assert P.select(pairs, probs, 0.0, False).shape == (0, 2)
    assert P.select(np.zeros((0, 2), int), np.zeros(0), 0.0, True).shape == (0, 2)
    with pytest.raises(ValueError):
        P.select(pairs, probs[:2], 0.0, True)


# -------------------------------------------------------------- threshold
def test_thresholds_are_13_values():
    assert len(P.THRESHOLDS) == 13
    assert P.THRESHOLDS[0] == 0.0 and P.THRESHOLDS[-1] == 0.3
    assert np.allclose(np.diff(P.THRESHOLDS), 0.025)


def test_choose_threshold_lowest_on_ties_and_gate():
    # region a: positive at 0.2, negative at 0.01; region b: positive at 0.9
    rows = [("a", np.array([[0, 0], [1, 1]]), None, np.array([True, False])),
            ("b", np.array([[0, 0]]), None, np.array([True]))]
    probs = {"a": np.array([0.2, 0.01]), "b": np.array([0.9])}
    table = P.threshold_table(rows, probs, {"a": True, "b": True}, total_gt=3)
    assert [t[0] for t in table] == list(P.THRESHOLDS)
    assert table[0][1:] == (2, 3, 2 * 2 / 6)
    best = P.choose_threshold(rows, probs, {"a": True, "b": True}, 3)
    # 0.025 .. 0.200 all give tp 2, pred 2 -> F1 0.8; lowest is 0.025
    assert best == (0.025, 2, 2, 0.8)
    # gating region a away leaves only b: every threshold ties -> 0.000
    best = P.choose_threshold(rows, probs, {"a": False, "b": True}, 3)
    assert best == (0.0, 1, 1, 0.5)


# ---------------------------------------------------------------- LOO
class _Recorder:
    calls: list = []

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        _Recorder.calls.append(set(X[:, 0].astype(int).tolist()))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.5)])


def test_loo_never_trains_on_heldout_mouse():
    rng = np.random.default_rng(0)
    subj_code = {"m0": 0, "m1": 1, "m2": 2}
    subject_of, rows = {}, []
    for k in range(9):
        g = f"m{k % 3}"
        sid = f"{g}__r{k}"
        subject_of[sid] = g
        X = rng.normal(size=(5, 16))
        X[:, 0] = subj_code[g]                     # tag rows with their mouse
        rows.append((sid, np.zeros((5, 2), int), X, rng.random(5) > 0.5))
    _Recorder.calls = []
    probs = P.loo_predict(rows, subject_of, estimator=_Recorder)
    assert len(_Recorder.calls) == 3
    for g, trained in zip(sorted(subj_code), _Recorder.calls):
        assert subj_code[g] not in trained and len(trained) == 2
    assert set(probs) == set(subject_of)


# --------------------------------------------- v10 reproduction (lab.pkl)
@pytest.fixture(scope="module")
def v10_poses():
    with _research_import_env():
        import cp_pose_lab
    with open(paths.RDATA / "cp_pose_train.pkl", "rb") as f:
        S = pickle.load(f)
    out = {}
    for s in P.pair_clf.R:
        ch = cp_pose_lab.pose_choose(S[s], "score")
        out[s] = (ch[0], ch[1])
    return out


def test_loo_probs_reproduces_pair_clf(v10_poses):
    pc = P.pair_clf
    ref = pc.loo_predict(pc.dataset(v10_poses))
    got = P.loo_probs(pc.R, v10_poses)
    assert set(got) == set(ref) == set(pc.R)
    for s in ref:
        assert np.array_equal(got[s], ref[s]), s
    rows = P.dataset(pc.R, v10_poses)
    for (s1, p1, X1, y1), (s2, p2, X2, y2) in zip(rows, pc.dataset(v10_poses)):
        assert s1 == s2 and np.array_equal(p1, p2) and np.array_equal(y1, y2)
    # 0.025 on all regions equals pair_clf.f1_at (one-to-one already holds)
    tp, pred, f1 = pc.f1_at(pc.dataset(v10_poses), ref, 0.025)
    table = P.threshold_table(rows, got, {s: True for s in got}, pc.TOTAL)
    assert table[1][1:] == (tp, pred, f1)


def test_v10_test_model_matches_train_classifier():
    with _research_import_env():
        import test_apply
    ref = test_apply.train_classifier()
    got = P.v10_test_model()
    with open(paths.RDATA / "reg_window_vote5.pkl", "rb") as f:
        res = pickle.load(f)
    rows = P.dataset(P.pair_clf.R, {s: (res[s][0], res[s][1]) for s in P.pair_clf.R})
    X = np.vstack([x[2] for x in rows if len(x[2])])
    assert np.array_equal(got.predict_proba(X), ref.predict_proba(X))


# ---------------------------------------------------------------- Stage
class _Ctx:
    def __init__(self, objs):
        self.objs, self.smoke, self.workers = objs, True, 1

    def load(self, name):
        return self.objs[name]


def test_compute_shapes(v10_poses):
    R = P.pair_clf.R
    sids = sorted(R)
    by = {}
    for s in sids:
        by.setdefault(R[s]["subject"], []).append(s)
    held = [x for g in sorted(by) for x in by[g][:2]]          # 2 regions per mouse
    heldout = {s: R[s] for s in held}
    test = {"t1": {k: R[held[0]][k] for k in ("iv_c", "ex_c", "ex_shape", "iv_f", "ex_f",
                                               "subject")}}
    jp = {s: {"M": v10_poses[s][0], "score": v10_poses[s][1]} for s in held}
    joint = {"heldout": {"independent": jp, "joint": jp},
             "test": {"independent": {"t1": jp[held[0]]}, "joint": {"t1": None}}}
    out = P.compute(None, _Ctx({"prep": {"heldout": heldout, "test": test}, "joint": joint}))
    for sel in P.SELECTIONS:
        h = out["heldout"][sel]
        assert set(h) == set(held)
        for s, e in h.items():
            assert len(e["pairs"]) == len(e["probs"]) == len(e["y"])
            assert e["subject"] == R[s]["subject"]
    t = out["test"]
    assert len(t["independent"]["t1"]["probs"]) == len(t["independent"]["t1"]["pairs"]) > 0
    assert len(t["joint"]["t1"]["pairs"]) == 0                   # unregistered -> no candidates
    # same poses for both selections -> identical probabilities
    for s in held:
        assert np.array_equal(out["heldout"]["independent"][s]["probs"],
                              out["heldout"]["joint"][s]["probs"])
