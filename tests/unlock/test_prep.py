"""Unit tests for the prep Stage helpers (task 5.1 / 5.2; Req 6.1, 6.4, 3.11)."""
from __future__ import annotations

import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hpc_unlock import checkpoint, paths, prep

RNG = np.random.default_rng(0)


def _img(shape=(32, 40), dtype=np.uint16):
    return RNG.integers(0, 4000, size=shape).astype(dtype)


# ---------------------------------------------------------------- dup_key

def test_dup_key_equal_for_byte_identical_images():
    iv, ex = _img(), _img((50, 50))
    assert prep.dup_key(iv, ex) == prep.dup_key(iv.copy(), ex.copy())
    # non-contiguous view with the same pixels hashes the same
    assert prep.dup_key(np.asfortranarray(iv), ex) == prep.dup_key(iv, ex)


def test_dup_key_differs_on_any_pixel_change_or_swap():
    iv, ex = _img(), _img((50, 50))
    base = prep.dup_key(iv, ex)
    iv2 = iv.copy()
    iv2[3, 7] += 1
    ex2 = ex.copy()
    ex2[-1, -1] ^= 1
    assert prep.dup_key(iv2, ex) != base
    assert prep.dup_key(iv, ex2) != base
    assert prep.dup_key(ex, iv) != base


def test_dup_key_includes_dtype_and_shape():
    a = np.zeros((4, 8), np.uint16)
    ex = _img((5, 5))
    assert prep.dup_key(a, ex) != prep.dup_key(a.reshape(8, 4), ex)
    assert prep.dup_key(a, ex) != prep.dup_key(a.view(np.int16), ex)


def test_dup_key_on_real_duplicate_test_regions():
    """7754ed and f05266 are pixel-identical (Req 6.4); another region is not."""
    k = {r: prep.region_dup_key("test", f"subject_78b6a7__region_{r}")
         for r in ("7754ed", "f05266", "33daed")}
    assert k["7754ed"] == k["f05266"]
    assert k["33daed"] != k["7754ed"]


# ---------------------------------------------------------------- groups

def test_group_key_is_builtin():
    g = prep.group_key(np.str_("subject_a"), (np.int64(1627), np.int64(1600)))
    assert g == ("subject_a", (1627, 1600))
    assert type(g[0]) is str and all(type(v) is int for v in g[1])


def test_group_sids_partitions_by_subject_and_canvas():
    recs = {
        "a1": {"group": prep.group_key("a", (10, 10))},
        "b1": {"group": prep.group_key("b", (10, 10))},
        "a2": {"group": prep.group_key("a", (10, 10))},
        "a3": {"group": prep.group_key("a", (12, 10))},
    }
    g = prep.group_sids(recs)
    assert g == {("a", (10, 10)): ["a1", "a2"], ("b", (10, 10)): ["b1"],
                 ("a", (12, 10)): ["a3"]}
    assert sorted(s for v in g.values() for s in v) == sorted(recs)


def test_voters_once_keeps_first_duplicate_in_order():
    keys = {"r1": "k1", "r2": "k2", "r3": "k1", "r4": "k3"}
    cands = {s: [s] for s in keys}
    v = prep.voters_once(["r1", "r2", "r3", "r4"], keys, cands)
    assert list(v) == ["r1", "r2", "r4"]


def test_smoke_heldout_two_regions_from_two_mice():
    subj = {"m2__r1": "m2", "m1__r2": "m1", "m1__r1": "m1", "m3__r1": "m3"}
    picked = prep.smoke_heldout(subj)
    assert picked == ["m1__r1", "m2__r1"]
    assert len({subj[s] for s in picked}) == 2


def test_scan_inputs_float32_and_duplicates_collapse():
    iv_c, ex_c = RNG.random((7, 2)) * 100, RNG.random((5, 2)) * 100
    recs = {
        "a": {"dup_key": "k", "iv_c": iv_c, "ex_c": ex_c, "ex_shape": (np.int64(9), 8),
              "offset": np.array([1.5, -2.0])},
        "b": {"dup_key": "k", "iv_c": iv_c, "ex_c": ex_c, "ex_shape": (9, 8),
              "offset": np.array([1.5, -2.0])},
        "c": {"dup_key": "j", "iv_c": iv_c[:3], "ex_c": ex_c, "ex_shape": (9, 8),
              "offset": np.zeros(2)},
    }
    s = prep.scan_inputs(recs)
    assert set(s) == {"k", "j"} and s["k"]["sids"] == ["a", "b"]
    for v in s.values():
        assert v["iv_c"].dtype == v["ex_c"].dtype == v["offset"].dtype == np.float32
        assert v["ex_shape"] == (9, 8) and all(type(x) is int for x in v["ex_shape"])
    np.testing.assert_array_equal(s["k"]["iv_c"], iv_c.astype(np.float32))


# ---------------------------------------------------------------- records

def _synthetic_records():
    M = np.array([[1.0, 0.1, 5.0], [-0.1, 1.0, -3.0]])
    lab = {"subject": "subject_x", "iv_shape": (np.int64(20), 30), "ex_shape": (40, 41),
           "offset": np.array([3.0, 4.0]), "iv_c": RNG.random((6, 2)), "ex_c": RNG.random((5, 2)),
           "iv_f": {"area": RNG.random(6), "prob": RNG.random(6)}, "ex_f": {"area": RNG.random(5)},
           "iv_link": np.arange(6), "ex_link": -np.ones(5, int),
           "gt_pairs": {(np.int64(1), np.int64(2))}, "n_gt_pairs": np.int64(3),
           "gt_iv_c": RNG.random((4, 2)), "gt_M": M}
    votes = [(np.float64(10.0), 2.0, np.array([300.0, 310.0]), M)]
    scored = [(M, np.float64(9.0), np.float64(1.5), "win"), (M, 10.0, 0.5, "vote")]
    h = prep.heldout_record("subject_x__region_a", lab, votes, scored, "k1")
    raw = {"subject": "subject_y", "iv_shape": (20, 30), "ex_shape": (40, 41),
           "offset": np.array([0.0, 1.0]), "iv_c": RNG.random((6, 2)), "ex_c": RNG.random((5, 2)),
           "iv_f": {"area": RNG.random(6)}, "ex_f": {"area": RNG.random(5)},
           "iv_ids": ["1", "2"], "ex_ids": ["7"]}
    t = prep.test_record("subject_y__region_b", raw, "k2")
    t["v10_window"] = {"M": M, "score": 9.0, "M_true": M}
    return h, t


def test_records_hold_only_builtins_and_arrays():
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k)
                walk(v)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v)
        else:
            assert o is None or type(o) in (str, int, float, bool) or isinstance(o, np.ndarray), \
                f"unexpected {type(o)}: {o!r}"
    h, t = _synthetic_records()
    walk(h)
    walk(t)
    assert h["group"] == ("subject_x", (40, 41)) and h["window_cands"][0][2] == "win"
    assert len(h["window_cands"]) == 1 and h["cp_bin"] == {"side": "cp_side", "key": h["sid"]}


def test_record_round_trip_without_hpc_unlock(tmp_path: Path):
    h, t = _synthetic_records()
    bins = {h["sid"]: np.ones((4, 4), np.uint8), t["sid"]: np.zeros((4, 4), np.uint8)}
    ref = checkpoint.save_npz_atomic(tmp_path / prep.CP_SIDE, **bins)
    obj = {"heldout": {h["sid"]: h}, "test": {t["sid"]: t}, "cp_side": ref,
           "test_modes": {t["group"]: [{"angle": 1.0, "landing": np.zeros(2),
                                        "support": 2.0, "n": 2}]}}
    p = tmp_path / "prep.pkl"
    checkpoint.save_atomic(p, obj)
    back = checkpoint.load(p)
    assert back["heldout"][h["sid"]]["gt_pairs"] == {(1, 2)}
    np.testing.assert_array_equal(back["test"][t["sid"]]["iv_c"], t["iv_c"])
    got = prep.load_cp_bins(back, tmp_path)
    np.testing.assert_array_equal(got[h["sid"]], bins[h["sid"]])
    code = (
        "import pickle, sys\n"
        f"o = pickle.load(open({str(p)!r}, 'rb'))\n"
        "assert not any(m.startswith('hpc_unlock') for m in sys.modules), 'hpc_unlock imported'\n"
        "r = next(iter(o['heldout'].values()))\n"
        "print(r['split'], r['group'][0], r['gt_M'].shape, len(o['test']))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["heldout", "subject_x", "(2,", "3)", "1"]


# ---------------------------------------------------------------- smoke-size run

def test_compute_smoke_two_heldout_regions(tmp_path: Path, monkeypatch):
    """Real prep on 2 held-out regions (2 mice) and no test region; reproduces the v10 inputs."""
    monkeypatch.setattr(prep, "SMOKE_TEST", 0)
    ctx = SimpleNamespace(run_dir=tmp_path, smoke=True, workers=2, load=None)
    t0 = time.monotonic()
    out = prep.compute(None, ctx)
    assert time.monotonic() - t0 < 60
    assert len(out["heldout"]) == 2 and out["test"] == {}
    assert len({r["subject"] for r in out["heldout"].values()}) == 2
    assert set(out["baseline_sha256"]) == set(prep.BASELINE_CSVS)

    p = tmp_path / "prep.pkl"
    checkpoint.save_atomic(p, out)
    back = checkpoint.load(p)
    bins = prep.load_cp_bins(back, tmp_path)
    lab = pickle.load(open(paths.RDATA / "lab.pkl", "rb"))
    cpt = pickle.load(open(paths.RDATA / "cp_pose_train.pkl", "rb"))
    for sid, rec in back["heldout"].items():
        assert rec["group"] == (lab[sid]["subject"], tuple(lab[sid]["ex_shape"]))
        np.testing.assert_array_equal(rec["iv_c"], lab[sid]["iv_c"])
        assert len(rec["cp_scored"]) == len(cpt[sid])
        for (M, s, z, src), (M0, s0, z0, src0) in zip(rec["cp_scored"], cpt[sid]):
            np.testing.assert_array_equal(M, M0)
            assert (s, z, src) == (s0, z0, src0)
        b = bins[sid]
        assert b.dtype == np.uint8 and b.shape == tuple(lab[sid]["ex_shape"])
        assert rec["dup_key"] in back["scan_inputs"]
        assert back["scan_inputs"][rec["dup_key"]]["iv_c"].dtype == np.float32
