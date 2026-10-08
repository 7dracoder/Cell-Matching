"""Unit tests for hpc_unlock/assemble.py (Req 10.1-10.8, 9.4, 12.2, 12.8)."""
from __future__ import annotations

import json
import shutil
import subprocess
import types

import numpy as np
import pytest

import run_unlock
from hpc_unlock import assemble as A
from hpc_unlock import checkpoint, paths, report, stage
from hpc_unlock.config import UnlockConfig

BASE = paths.ROOT / A.BASELINE_CSV
pytestmark = pytest.mark.skipif(not BASE.is_file(), reason="Baseline CSV not present")


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def base():
    fields, rows = A.read_rows(BASE)
    records = {r["sample_id"]: {"sid": r["sample_id"],
                                "iv_ids": list(json.loads(r["invivo_instances"])),
                                "ex_ids": list(json.loads(r["exvivo_instances"]))}
               for r in rows}
    return types.SimpleNamespace(fields=fields, rows=rows, records=records,
                                 counts=A.pair_counts(rows))


def synthetic_pairs(records, seed=0, k=5, skip=()):
    """One-to-one random index pairs per region (k or fewer)."""
    rng = np.random.default_rng(seed)
    out = {}
    for sid, rec in records.items():
        if sid in skip:
            out[sid] = []
            continue
        n = min(k, len(rec["iv_ids"]), len(rec["ex_ids"]))
        ii = rng.choice(len(rec["iv_ids"]), n, replace=False)
        jj = rng.choice(len(rec["ex_ids"]), n, replace=False)
        out[sid] = [[int(i), int(j)] for i, j in zip(ii, jj)]
    return out


def _cfg(name, accepted, test_pairs, full=0.53):
    return {"name": name, "accepted": accepted, "full": full, "f1": 0.49,
            "pq_iv": 0.7404, "pq_ex": 0.3901, "kept": 33, "kept_wrong": 1,
            "per_mouse_f1": {"m1": 0.5}, "test_pairs": test_pairs,
            "rejected_reason": None if accepted else "NO_CANDIDATE: full <= Baseline"}


def _validate_ck(configs):
    return {"baseline": {"full": 0.5186, "f1": 0.472, "pq_iv": 0.7404, "pq_ex": 0.3901,
                         "kept": 31, "kept_wrong": 2, "per_mouse_f1": {"m1": 0.47},
                         "reproduced": True,
                         "expected": {"f1": 0.472, "full": 0.5186, "tol": 0.005}},
            "configs": configs}


def _prep(base, root=paths.ROOT):
    return {"test": base.records, "baseline_sha256": A.guarded_sha256(root)}


def ok_checker(p):
    return True, "PASS fake"


# ---------------------------------------------------------------- names
def test_names_distinct_and_follow_pattern():
    names = [A.csv_name(f"{s}_{g}") for s in ("indep", "joint") for g in ("cons", "aggr")]
    assert len(set(names)) == 4
    assert A.csv_name("joint_cons") == "submission_v13_joint_cons.csv"
    assert A.SELFTRAIN_CSV == "submission_v13_selftrain.csv"
    assert A.SELFTRAIN_CSV not in names


def test_baseline_dialect_round_trips_bytes(base):
    assert A.render_rows(base.fields, base.rows) == BASE.read_bytes()


# ---------------------------------------------------------------- registration-only
def test_registration_only_keeps_baseline_fields(base, tmp_path):
    tp = synthetic_pairs(base.records)
    out = A.assemble(_prep(base), _validate_ck([_cfg("joint_cons", True, tp),
                                                _cfg("indep_aggr", False, tp, 0.51)]),
                     root=tmp_path, checker=ok_checker, write_report=False)
    path = tmp_path / "submission_v13_joint_cons.csv"
    assert out["candidates"] == [str(path)]
    assert not (tmp_path / "submission_v13_indep_aggr.csv").exists()
    fields, rows = A.read_rows(path)
    assert fields == base.fields
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in base.rows]
    for got, ref in zip(rows, base.rows):
        for k in ("sample_id", "invivo_instances", "exvivo_instances"):
            assert got[k] == ref[k]
        pairs = json.loads(got["match_pairs"])
        iv, ex = json.loads(got["invivo_instances"]), json.loads(got["exvivo_instances"])
        assert all(a in iv and b in ex for a, b in pairs)
        assert len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs})
        rec = base.records[got["sample_id"]]
        assert pairs == [[rec["iv_ids"][i], rec["ex_ids"][j]]
                         for i, j in tp[got["sample_id"]]]
    # Writing the Baseline's own pairs back reproduces the Baseline bytes
    same = [dict(r, match_pairs=ref["match_pairs"]) for r, ref in zip(rows, base.rows)]
    assert A.render_rows(fields, same) == BASE.read_bytes()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_id_mismatch_and_non_one_to_one_are_rejected(base, tmp_path):
    sid = next(iter(base.records))
    recs = {**base.records, sid: {**base.records[sid],
                                  "iv_ids": base.records[sid]["iv_ids"][::-1]}}
    with pytest.raises(A.CandidateError, match="in-vivo IDs"):
        A.registration_rows(base.rows, recs, {})
    with pytest.raises(A.CandidateError, match="more than one pair"):
        A.registration_rows(base.rows, base.records, {sid: [[0, 0], [0, 1]]})
    with pytest.raises(A.CandidateError, match="out of range"):
        A.registration_rows(base.rows, base.records, {sid: [[10 ** 6, 0]]})
    # Inside assemble a bad candidate is recorded as rejected, not written
    out = A.assemble(_prep(base), _validate_ck([_cfg("joint_aggr", True,
                                                     {sid: [[0, 0], [1, 0]]})]),
                     root=tmp_path, checker=ok_checker, write_report=False)
    assert out["candidates"] == [] and out["rejected"][0]["name"] == "joint_aggr"
    assert not (tmp_path / "submission_v13_joint_aggr.csv").exists()


def test_format_checker_failure_deletes_csv(base, tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 1, "", "ValueError: Overlapping masks")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    tp = synthetic_pairs(base.records)
    out = A.assemble(_prep(base), _validate_ck([_cfg("indep_cons", True, tp)]),
                     root=tmp_path, write_report=True)
    argv, kw = calls[0]
    assert argv[1] == "validate_submission.py" and argv[2].endswith("submission_v13_indep_cons.csv")
    assert kw["cwd"] == str(paths.ROOT)
    assert not (tmp_path / "submission_v13_indep_cons.csv").exists()
    assert out["candidates"] == []
    assert out["rejected"][0]["name"] == "indep_cons"
    assert "Overlapping masks" in out["rejected"][0]["reason"]
    rep = out["report"]
    assert rep["status"] == report.NO_CANDIDATE
    assert any(r["name"] == "indep_cons" and "Format_Checker failed" in r["reason"]
               for r in rep["rejected"])
    assert (tmp_path / report.JSON_NAME).is_file() and (tmp_path / report.MD_NAME).is_file()


def test_real_format_checker_passes_candidate(base, tmp_path):
    tp = synthetic_pairs(base.records, seed=3)
    out = A.assemble(_prep(base), _validate_ck([_cfg("joint_cons", True, tp)]),
                     root=tmp_path, write_report=False)
    assert out["rejected"] == [], out["rejected"]
    assert (tmp_path / "submission_v13_joint_cons.csv").is_file()


# ---------------------------------------------------------------- hash guard
def test_baseline_hash_guard(base, tmp_path):
    broot = tmp_path / "broot"
    broot.mkdir()
    for name in A.GUARDED_CSVS:
        shutil.copy(paths.ROOT / name, broot / name)
    prep = _prep(base, broot)
    A.check_baseline(prep["baseline_sha256"], broot)
    with open(broot / A.GUARDED_CSVS[1], "ab") as f:
        f.write(b"\n")
    with pytest.raises(A.BaselineChanged, match="submission_v7_grow15.csv"):
        A.check_baseline(prep["baseline_sha256"], broot)
    out_root = tmp_path / "out"
    out_root.mkdir()
    with pytest.raises(A.BaselineChanged):
        A.assemble(prep, _validate_ck([_cfg("joint_cons", True, {})]), root=out_root,
                   baseline_root=broot, checker=ok_checker)
    assert not list(out_root.iterdir())          # nothing written


def test_compute_logs_baseline_changed_and_exits_1(base, tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    prep = {"test": base.records,
            "baseline_sha256": {n: "0" * 64 for n in A.GUARDED_CSVS}}
    checkpoint.run_stage("prep", lambda: prep, run_dir)
    checkpoint.run_stage("validate", lambda: _validate_ck([]), run_dir)
    ctx = stage.StageContext(name="assemble", run_dir=run_dir, smoke=True, workers=1)
    with pytest.raises(SystemExit) as e:
        A.compute(UnlockConfig(disable_selftrain=True), ctx)
    assert e.value.code == 1
    assert "BASELINE_CSV_CHANGED" in capsys.readouterr().out


# ---------------------------------------------------------------- unlock table
def test_unlock_table_newly_unlocked_rule(base, tmp_path):
    zero = [s for s, c in base.counts.items() if c == 0]
    paired = [s for s, c in base.counts.items() if c > 0]
    tp = synthetic_pairs(base.records, skip={zero[0], paired[0]})
    out = A.assemble(_prep(base), _validate_ck([_cfg("joint_aggr", True, tp)]),
                     root=tmp_path, checker=ok_checker, write_report=False)
    table = {r["sid"]: r for r in out["unlock"]["joint_aggr"]}
    assert len(table) == len(base.rows) == 29
    for sid, r in table.items():
        assert r["baseline_pairs"] == base.counts[sid]
        assert r["candidate_pairs"] == len(tp[sid])
        assert r["newly_unlocked"] == (r["baseline_pairs"] == 0 and r["candidate_pairs"] >= 1)
    assert table[zero[0]]["newly_unlocked"] is False      # 0 -> 0
    assert table[zero[1]]["newly_unlocked"] is True       # 0 -> k
    assert table[paired[0]]["newly_unlocked"] is False    # k -> 0
    ranked = out["report"]["ranking"][0]
    assert ranked["name"] == "joint_aggr" and ranked["unlock_table"]


# ---------------------------------------------------------------- report-only
def _failure(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"status": "BASELINE_REPRODUCTION_FAILED",
           "measured": {"full": 0.50, "f1": 0.44, "pq_iv": 0.74, "pq_ex": 0.39,
                        "kept": 30, "kept_wrong": 3, "tp": 1, "pred": 2},
           "expected": {"f1": 0.472, "full": 0.5186, "tol": 0.005}}
    (run_dir / A.FAILURE_JSON).write_text(json.dumps(rec))


def test_report_only_writes_no_csv(tmp_path):
    run_dir, root = tmp_path / "run", tmp_path / "root"
    root.mkdir()
    _failure(run_dir)
    out = A.report_only(run_dir, root=root)
    assert out["candidates"] == []
    assert sorted(p.name for p in root.iterdir()) == sorted([report.JSON_NAME, report.MD_NAME])
    rep = json.loads((root / report.JSON_NAME).read_text())
    assert rep["status"] == report.NO_CANDIDATE
    assert rep["baseline_reproduction"]["status"] == "failed"
    assert "Baseline reproduction failure" in (root / report.MD_NAME).read_text()


def test_report_only_through_driver_no_marker(tmp_path, monkeypatch):
    root, hpc = tmp_path / "root", tmp_path / "hpc"
    root.mkdir()
    monkeypatch.setattr(paths, "ROOT", root)
    monkeypatch.setattr(paths, "HPC", hpc)
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    cfg = UnlockConfig()
    fp = cfg.fingerprint()
    cfg.save(hpc / fp)
    _failure(hpc / fp)
    assert run_unlock.main(["assemble", "--run", fp, "--report-only"]) == 0
    assert (root / report.JSON_NAME).is_file()
    assert not list(root.glob("*.csv"))
    assert not checkpoint.marker_path(hpc / fp, "assemble").exists()
    assert not checkpoint.checkpoint_path(hpc / fp, "assemble").exists()
    assert run_unlock.main(["prep", "--run", fp, "--report-only"]) == 2


# ---------------------------------------------------------------- self-training
def test_selftrain_disabled_never_loads_selftrain(tmp_path):
    loaded = []
    ctx = types.SimpleNamespace(load=lambda n: loaded.append(n))
    st, state = A._selftrain_input(UnlockConfig(disable_selftrain=True), ctx)
    assert st is None and state == {"status": "disabled"} and loaded == []

    def missing(n):
        raise stage.DependencyError(f"{n} has no done-marker")
    st, state = A._selftrain_input(UnlockConfig(), types.SimpleNamespace(load=missing))
    assert st is None and state["status"] == "skipped"


def test_selftrain_writer_keeps_invivo(base, tmp_path):
    sids = list(base.records)[:2]
    labels = {}
    for n, sid in enumerate(sids):
        lab = np.zeros((40, 40), np.int32)
        for k in range(1, 4 + n):
            lab[k * 5:k * 5 + 3, 2:6] = k
        labels[sid] = lab
    run_dir = tmp_path / "run"
    side = checkpoint.save_npz_atomic(run_dir / "st_labels.npz", **labels)
    tp = {sids[0]: [[0, 2], [1, 0]], sids[1]: [[3, 3]]}
    st = {"accepted": True, "full": 0.55, "f1": 0.5, "pq_iv": 0.74, "pq_ex": 0.41,
          "kept": 33, "kept_wrong": 1, "per_mouse_f1": {}, "compared_to": 0.53,
          "test_labels_side": side, "test_pairs": tp}
    out = A.assemble(_prep(base), _validate_ck([]), root=tmp_path, run_dir=run_dir,
                     selftrain=st, checker=ok_checker, write_report=False)
    path = tmp_path / A.SELFTRAIN_CSV
    assert out["candidates"] == [str(path)]
    _, rows = A.read_rows(path)
    for got, ref in zip(rows, base.rows):
        assert got["sample_id"] == ref["sample_id"]
        assert got["invivo_instances"] == ref["invivo_instances"]
        if got["sample_id"] not in labels:
            assert got["exvivo_instances"] == ref["exvivo_instances"]
            assert got["match_pairs"] == "[]"
    r0 = rows[0]
    ex0 = json.loads(r0["exvivo_instances"])
    assert list(ex0) == ["EXP_000001", "EXP_000002", "EXP_000003"]
    iv0 = base.records[sids[0]]["iv_ids"]
    assert json.loads(r0["match_pairs"]) == [[iv0[0], "EXP_000003"], [iv0[1], "EXP_000001"]]
    assert out["report"]["selftrain"]["status"] == "accepted"

    # Not strictly better than the comparison target -> no CSV (12.7)
    path.unlink()
    out = A.assemble(_prep(base), _validate_ck([]), root=tmp_path, run_dir=run_dir,
                     selftrain={**st, "full": 0.53}, checker=ok_checker, write_report=False)
    assert out["candidates"] == [] and not path.exists()
    assert out["report"]["selftrain"]["status"] == "no_candidate"


# ---------------------------------------------------------------- smoke (13.5)
def test_smoke_candidate_equals_baseline(base, tmp_path):
    some = dict(list(base.records.items())[:3])
    rows = A.smoke_rows(base.rows, some)
    assert A.render_rows(base.fields, rows) == BASE.read_bytes()
    out = tmp_path / "hpc_smoke" / A.SMOKE_CSV
    assert A.write_smoke_candidate(out, {"test": some}, checker=ok_checker) == "PASS fake"
    assert out.read_bytes() == BASE.read_bytes()


def test_smoke_candidate_rejected_is_deleted(base, tmp_path):
    out = tmp_path / A.SMOKE_CSV
    with pytest.raises(A.CandidateError, match="Format_Checker failed"):
        A.write_smoke_candidate(out, {"test": {}}, checker=lambda p: (False, "rc=1 bad"))
    assert not out.exists()


def test_assemble_without_csvs_writes_nothing(base, tmp_path):
    tp = synthetic_pairs(base.records, seed=4)
    out = A.assemble(_prep(base), _validate_ck([_cfg("joint_cons", True, tp)]),
                     root=tmp_path, checker=ok_checker, write_csvs=False)
    assert out["candidates"] == [] and out["rejected"] == []
    assert not list(tmp_path.glob("*.csv"))
    assert len(out["unlock"]["joint_cons"]) == len(base.rows)
    assert (tmp_path / report.JSON_NAME).is_file()
