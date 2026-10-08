"""Unit tests for hpc_unlock.checkpoint (Req 3.7-3.11)."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from hpc_unlock import checkpoint as ck


class Boom(RuntimeError):
    pass


def counting(value):
    calls = []

    def compute():
        calls.append(1)
        return value
    return compute, calls


# --- save_atomic / load -----------------------------------------------------

def test_round_trip_nested_with_arrays(tmp_path):
    obj = {"a": [1, 2.5, None, "x"], "t": (np.arange(6, dtype=np.int16).reshape(2, 3),),
           "e": np.zeros((0, 4), dtype=np.float32), "n": np.array([np.nan, 1.0])}
    p = tmp_path / "x.pkl"
    digest = ck.save_atomic(p, obj)
    assert digest == ck.sha256_file(p)
    got = ck.load(p)
    assert got["a"] == obj["a"]
    for a, b in ((got["t"][0], obj["t"][0]), (got["e"], obj["e"]), (got["n"], obj["n"])):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)


def test_save_atomic_uses_protocol_5_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "sub" / "x.pkl"
    ck.save_atomic(p, {"k": 1})
    assert p.read_bytes()[:2] == b"\x80\x05"
    assert [q.name for q in p.parent.iterdir()] == ["x.pkl"]


def test_failed_save_keeps_old_file_and_removes_tmp(tmp_path):
    p = tmp_path / "x.pkl"
    ck.save_atomic(p, {"old": True})
    with pytest.raises(Exception):
        ck.save_atomic(p, {"bad": lambda: None})  # lambdas are not picklable
    assert ck.load(p) == {"old": True}
    assert [q.name for q in tmp_path.iterdir()] == ["x.pkl"]


def test_load_raises_on_missing_and_truncated(tmp_path):
    with pytest.raises(FileNotFoundError):
        ck.load(tmp_path / "nope.pkl")
    p = tmp_path / "x.pkl"
    ck.save_atomic(p, list(range(1000)))
    p.write_bytes(p.read_bytes()[:20])
    with pytest.raises(Exception):
        ck.load(p)


# --- side files ---------------------------------------------------------------

def test_npz_side_file_round_trip_and_verification(tmp_path):
    a = np.random.default_rng(0).random((3, 4)).astype(np.float32)
    ref = ck.save_npz_atomic(tmp_path / "tiles.npz", tiles=a, ids=np.arange(3))
    assert ref == {ck.SIDE_KEY: "tiles.npz", "sha256": ck.sha256_file(tmp_path / "tiles.npz")}
    got = ck.load_npz(ref, tmp_path)
    np.testing.assert_array_equal(got["tiles"], a)
    assert got["tiles"].dtype == np.float32

    ck.save_atomic(tmp_path / "main.pkl", {"side": ref, "n": 3})
    assert ck.load(tmp_path / "main.pkl")["n"] == 3

    (tmp_path / "tiles.npz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        ck.load(tmp_path / "main.pkl")
    (tmp_path / "tiles.npz").unlink()
    with pytest.raises(FileNotFoundError):
        ck.load(tmp_path / "main.pkl")


def test_npz_requires_npz_suffix(tmp_path):
    with pytest.raises(ValueError):
        ck.save_npz_atomic(tmp_path / "x.bin", a=np.zeros(2))


# --- run_stage ---------------------------------------------------------------

def test_fresh_run_computes_saves_and_marks(tmp_path, capsys):
    compute, calls = counting({"v": 1})
    assert ck.run_stage("prep", compute, tmp_path, fingerprint="3fa9c1e2d0") == {"v": 1}
    assert calls == [1]
    rec = json.loads((tmp_path / "prep.done").read_text())
    assert set(rec) == {"stage", "sha256", "fingerprint", "finished"}
    assert rec["stage"] == "prep" and rec["fingerprint"] == "3fa9c1e2d0"
    assert rec["sha256"] == ck.sha256_file(tmp_path / "prep.pkl")
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("COMPUTE prep") and out[-1].startswith("DONE prep")


def test_second_run_reuses(tmp_path, capsys):
    ck.run_stage("prep", lambda: {"v": np.arange(3)}, tmp_path)
    capsys.readouterr()
    compute, calls = counting({"v": "new"})
    got = ck.run_stage("prep", compute, tmp_path)
    assert calls == []
    np.testing.assert_array_equal(got["v"], np.arange(3))
    assert capsys.readouterr().out.strip() == "REUSE prep"


def test_checkpoint_without_marker_recomputes(tmp_path):
    ck.save_atomic(tmp_path / "prep.pkl", "stale")
    compute, calls = counting("fresh")
    assert ck.run_stage("prep", compute, tmp_path) == "fresh"
    assert calls == [1]


@pytest.mark.parametrize("damage", ["missing", "truncated", "wrong_hash", "bad_marker",
                                    "side_missing"])
def test_inconsistent_marker_is_deleted_and_stage_recomputed(tmp_path, capsys, damage):
    def first():
        ref = ck.save_npz_atomic(tmp_path / "side.npz", a=np.ones(3))
        return {"side": ref}
    ck.run_stage("joint", first, tmp_path)
    pkl, done = tmp_path / "joint.pkl", tmp_path / "joint.done"
    if damage == "missing":
        pkl.unlink()
    elif damage == "truncated":
        pkl.write_bytes(pkl.read_bytes()[:5])
    elif damage == "wrong_hash":
        ck.save_atomic(pkl, {"other": 1})
    elif damage == "bad_marker":
        done.write_text("{not json")
    else:
        (tmp_path / "side.npz").unlink()
    capsys.readouterr()

    seen = []

    def compute():
        seen.append(done.exists())  # marker must be gone before recomputation
        return {"v": 2}
    assert ck.run_stage("joint", compute, tmp_path) == {"v": 2}
    assert seen == [False]
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("INCONSISTENT joint: ")
    assert json.loads(done.read_text())["sha256"] == ck.sha256_file(pkl)


def test_compute_exception_leaves_no_marker(tmp_path, capsys):
    def compute():
        raise Boom("preempted")
    with pytest.raises(Boom):
        ck.run_stage("verifier", compute, tmp_path)
    assert not (tmp_path / "verifier.done").exists()
    assert not (tmp_path / "verifier.pkl").exists()
    assert "STAGE_FAILED verifier: Boom: preempted" in capsys.readouterr().out
    compute2, calls = counting(7)
    assert ck.run_stage("verifier", compute2, tmp_path) == 7 and calls == [1]


def test_save_failure_leaves_no_marker(tmp_path, monkeypatch):
    def broken_dump(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pickle, "dump", broken_dump)
    with pytest.raises(OSError):
        ck.run_stage("pairs", lambda: 1, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_marker_failure_after_save_leaves_no_marker(tmp_path, monkeypatch):
    def broken_marker(*a, **k):
        raise Boom("killed before marker")
    monkeypatch.setattr(ck, "write_marker", broken_marker)
    with pytest.raises(Boom):
        ck.run_stage("pairs", lambda: 1, tmp_path)
    assert (tmp_path / "pairs.pkl").exists() and not (tmp_path / "pairs.done").exists()
    monkeypatch.undo()
    compute, calls = counting(2)
    assert ck.run_stage("pairs", compute, tmp_path) == 2 and calls == [1]
