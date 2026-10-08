"""Unit tests for hpc_unlock.verifier (Req 7.1-7.8, 12.5 fold gate)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from hpc_unlock import verifier as V

MICE = ("m0", "m1", "m2")
ANGLE = V.FEATURES.index("angle")


# ----------------------------------------------------------------- helpers

def cand(rng, mouse_code=0.0, z=0.0, rmargin=0.0, landing=None):
    return {"M": np.array([[1.0, 0.0, rng.uniform(-5, 5)], [0.0, 1.0, rng.uniform(-5, 5)]]),
            "refine_score": float(rng.uniform(0, 30)), "soft": float(rng.uniform(0, 20)),
            "z": float(z), "refine_margin": float(rmargin),
            "soft_margin": float(rng.uniform(-2, 5)), "angle": float(mouse_code),
            "scale": float(rng.uniform(0.9, 1.1)), "anisotropy": float(rng.uniform(0, 0.1)),
            "landing": rng.uniform(0, 300, 2) if landing is None else np.asarray(landing, float),
            "source": "gpu"}


def synth(seed=0, per_mouse=6, ok_of=None, no_gt=("m2__r5",), unreg=("m1__r4",)):
    """Synthetic prep / joint / pairs checkpoints; ``ok`` drives correctness."""
    rng = np.random.default_rng(seed)
    heldout, hent, pent = {}, {}, {}
    for g_i, g in enumerate(MICE):
        for k in range(per_mouse):
            sid = f"{g}__r{k}"
            ok = bool(rng.random() > 0.4) if ok_of is None else bool(ok_of(g, k))
            rec = {"subject": g, "group": (g, (512, 512)), "n_gt_pairs": 10,
                   "gt_M": None if sid in no_gt else np.eye(2, 3),
                   "gt_iv_c": None if sid in no_gt else np.zeros((3, 2)), "ok": ok}
            heldout[sid] = rec
            if sid in unreg:
                hent[sid] = {"M": None, "score": 0.0, "cand_index": None, "cand": None}
                n = 0
            else:
                c = cand(rng, g_i, z=rng.uniform(0, 7), rmargin=rng.uniform(-2, 5))
                hent[sid] = {"M": c["M"], "score": c["refine_score"], "cand_index": 0, "cand": c}
                n = 4
            pent[sid] = {"subject": g, "pairs": np.c_[np.arange(n), np.arange(n)],
                         "probs": rng.random(n), "y": rng.random(n) > 0.5}
    test, tent = {}, {}
    for k in range(3):
        sid = f"t__r{k}"
        test[sid] = {"subject": "tm", "group": ("tm", (512, 512))}
        c = cand(rng, 9.0)
        tent[sid] = None if k == 2 else {"M": c["M"], "score": 1.0, "cand_index": 0, "cand": c}
    prep = {"heldout": heldout, "test": test}
    joint = {"heldout": {s: hent for s in V.SELECTIONS}, "test": {s: tent for s in V.SELECTIONS}}
    pairs = {"heldout": {s: pent for s in V.SELECTIONS}}
    return prep, joint, pairs


def correct(rec, M):
    return rec["ok"]


class Recorder:
    """Fake estimator: records the mouse codes (angle column) it is trained on."""
    calls: list = []

    def fit(self, X, y):
        Recorder.calls.append(set(np.asarray(X)[:, ANGLE].astype(int).tolist()))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


# ----------------------------------------------------------------- features

def test_landing_dist_median_of_others_alone_and_unregistered():
    rng = np.random.default_rng(1)
    recs = {s: {"subject": "m0", "group": ("m0", (5, 5))} for s in "abcd"}
    recs["e"] = {"subject": "m1", "group": ("m1", (5, 5))}
    ents = {"a": {"M": np.eye(2, 3), "cand": cand(rng, landing=[0, 0])},
            "b": {"M": np.eye(2, 3), "cand": cand(rng, landing=[10, 0])},
            "c": {"M": np.eye(2, 3), "cand": cand(rng, landing=[30, 0])},
            "d": {"M": None, "cand": None},
            "e": {"M": np.eye(2, 3), "cand": cand(rng, landing=[99, 99])}}
    d = V.landing_dists(recs, ents)
    assert d["a"] == pytest.approx(20.0)          # median of (10, 30) = 20
    assert d["b"] == pytest.approx(5.0)           # median of (0, 30) = 15
    assert d["c"] == pytest.approx(25.0)          # median of (0, 10) = 5
    assert d["d"] is None and d["e"] == 0.0       # unregistered; alone in its group


def test_features_order_and_nonfinite():
    rng = np.random.default_rng(2)
    c = cand(rng, 1.0, z=4.0, rmargin=2.0)
    c["anisotropy"] = float("inf")
    x = V.region_features(c, 7.0)
    assert x.shape == (8,)
    assert x[V.FEATURES.index("landing_dist")] == 7.0
    assert x[V.FEATURES.index("z")] == 4.0 and x[V.FEATURES.index("refine_margin")] == 2.0
    assert np.isnan(x[V.FEATURES.index("anisotropy")])


def test_v10_gate():
    rng = np.random.default_rng(3)
    assert V.v10_gate(cand(rng, rmargin=3.0, z=0.0))
    assert V.v10_gate(cand(rng, rmargin=0.0, z=5.0))
    assert not V.v10_gate(cand(rng, rmargin=2.99, z=4.99))
    assert not V.v10_gate(None)


# ----------------------------------------------------------------- grid rules

def test_grid_21_rows_and_kept_rule():
    rng = np.random.default_rng(4)
    n = 12
    sids = [f"s{k}" for k in range(n)]
    probs = rng.random(n)
    v10 = rng.random(n) > 0.6
    reg = np.ones(n, bool)
    reg[0] = False
    gt = np.ones(n, bool)
    gt[1] = False
    y = rng.random(n) > 0.5
    pent = {s: {"pairs": np.array([[0, 0]]), "probs": np.array([0.5]), "y": np.array([True])}
            for s in sids}
    rows, pp = V.P.entry_rows(pent)
    grid = V.gate_grid(sids, probs, v10, reg, gt, y, rows, pp, total_gt=n)
    assert len(grid) == 21
    assert [r["tau"] for r in grid] == [round(0.05 * k, 2) for k in range(21)]
    for r in grid:
        keep = reg & (v10 | (probs >= r["tau"]))
        assert r["kept"] == keep.sum()
        assert r["correct"] == (keep & gt & y).sum() and r["wrong"] == (keep & gt & ~y).sum()
        assert r["correct"] + r["wrong"] <= r["kept"]
        assert r["tp"] == r["pred"] == keep.sum()  # one TP pair per kept region
    assert np.array_equal(V.kept_flags(probs, v10, reg, None), v10 & reg)


def _grid(wrong, f1):
    return [{"tau": float(t), "wrong": w, "f1": f} for t, w, f in zip(V.TAUS, wrong, f1)]


def test_conservative_and_aggressive_rules():
    wrong = [3] * 5 + [0] * 16
    f1 = [0.1] * 21
    f1[3] = f1[8] = 0.5                            # tie -> highest tau
    g = _grid(wrong, f1)
    assert V.conservative(g) == 0.25
    assert V.aggressive(g) == 0.40
    assert V.conservative(_grid([1] * 21, f1)) is None    # unavailable
    assert V.aggressive(_grid([1] * 21, [0.2] * 21)) == 1.0


# ----------------------------------------------------------------- models

def test_loo_never_uses_heldout_mouse_rows():
    prep, joint, _ = synth()
    T = V.table(prep["heldout"], joint["heldout"]["joint"], correct)
    Recorder.calls = []
    V.loo_probs(T, Recorder)
    assert Recorder.calls == [{1, 2}, {0, 2}, {0, 1}]


def test_single_class_fold_constant_model():
    # m1 and m2 all correct -> m0's fold is single-class -> constant prob 1
    prep, joint, _ = synth(ok_of=lambda g, k: g != "m0" or k % 2 == 0)
    T = V.table(prep["heldout"], joint["heldout"]["joint"], correct)
    p = V.loo_probs(T)
    m0 = (T["subject"] == "m0") & T["registered"]
    assert np.all(p[m0] == 1.0)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(p[~T["registered"]] == 0.0)
    # all-wrong training set -> constant 0
    prep, joint, _ = synth(ok_of=lambda g, k: g == "m0")
    T = V.table(prep["heldout"], joint["heldout"]["joint"], correct)
    assert np.all(V.loo_probs(T)[(T["subject"] == "m0") & T["registered"]] == 0.0)


def test_test_model_trained_on_all_mice():
    prep, joint, pairs = synth()
    T = V.table(prep["heldout"], joint["heldout"]["joint"], correct)
    Recorder.calls = []
    V.all_mice_model(T, Recorder)
    assert Recorder.calls == [{0, 1, 2}]
    # the Stage's test probabilities equal the all-mice model's predictions
    out = V.run(prep, joint, pairs, correct)
    Tt = V.table(prep["test"], joint["test"]["joint"], labelled=False)
    model = V.all_mice_model(T)
    want = V.predict(model, Tt["X"][Tt["registered"]])
    got = [out["selections"]["joint"]["test"][s]["prob"] for s in ("t__r0", "t__r1")]
    assert np.allclose(got, want)
    assert out["selections"]["joint"]["test"]["t__r2"]["prob"] == 0.0


# ----------------------------------------------------------------- fold gate

def test_fold_gate_excludes_mouse():
    prep, joint, pairs = synth(seed=5)
    H = prep["heldout"]
    T = V.table(H, joint["heldout"]["joint"], correct)
    pent = pairs["heldout"]["joint"]
    totals = {g: 60 for g in MICE}
    Recorder.calls = []
    V.fold_gate("m0", T, pent, totals, Recorder)
    assert Recorder.calls == [{2}, {1}]             # m1 scored by m2-only, m2 by m1-only
    ref = V.fold_gate("m0", T, pent, totals)
    assert all(r["kept"] <= int((T["subject"] != "m0").sum()) for r in ref["grid"])
    # perturbing m0's labels (pose correctness, pair labels) leaves m0's fold gate unchanged
    H2 = copy.deepcopy(H)
    for s, r in H2.items():
        if r["subject"] == "m0":
            r["ok"] = not r["ok"]
    p2 = copy.deepcopy(pent)
    for s, e in p2.items():
        if e["subject"] == "m0":
            e["y"] = ~e["y"]
    T2 = V.table(H2, joint["heldout"]["joint"], correct)
    got = V.fold_gate("m0", T2, p2, {**totals, "m0": 999})
    assert (got["conservative"], got["aggressive"]) == (ref["conservative"], ref["aggressive"])
    assert got["grid"] == ref["grid"]


# ----------------------------------------------------------------- Stage

def test_run_output_structure_and_gates():
    prep, joint, pairs = synth(seed=6)
    out = V.run(prep, joint, pairs, correct)
    assert out["feature_names"] == list(V.FEATURES) and len(out["taus"]) == 21
    for sel in V.SELECTIONS:
        r = out["selections"][sel]
        assert len(r["grid"]) == 21
        cons, aggr = r["conservative"]["tau"], r["aggressive"]["tau"]
        assert r["conservative"]["unavailable"] == (cons is None)
        assert set(r["fold"]) == set(MICE)
        assert set(r["heldout"]) == set(prep["heldout"]) and set(r["test"]) == set(prep["test"])
        gt_rows = [e for e in r["heldout"].values() if e["correct"] is not None]
        assert len(gt_rows) == len(prep["heldout"]) - 1          # region without GT
        assert r["heldout"]["m2__r5"]["correct"] is None
        for sid, e in r["heldout"].items():
            assert 0.0 <= e["prob"] <= 1.0
            reg = e["features"] is not None
            for name, tau in (("conservative", cons), ("aggressive", aggr)):
                want = e["v10"] if tau is None else (e["v10"] or e["prob"] >= tau)
                assert e["kept"][name] == (reg and want)
        u = r["heldout"]["m1__r4"]                               # unregistered
        assert u["prob"] == 0.0 and not u["v10"] and not any(u["kept"].values())
        assert u["correct"] is False
        row = {g["tau"]: g for g in r["grid"]}[aggr]
        assert row["kept"] == sum(e["kept"]["aggressive"] for e in r["heldout"].values())
