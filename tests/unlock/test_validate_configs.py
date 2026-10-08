"""Baseline reproduction, configurations and acceptance (Req 9.3-9.6, 8.5)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle

import numpy as np
import pytest

from hpc_unlock import checkpoint, paths, prep, validate

PQ = {"pq_iv": 0.7404, "pq_ex": 0.3901}
REAL_INPUTS = [paths.RDATA / n for n in ("lab.pkl", "vote_cands.pkl", "cp_pose_train.pkl",
                                         "heldout_labels.npz")]
SKIP_REAL = os.environ.get("UNLOCK_SKIP_REALDATA") == "1"
real_data = pytest.mark.skipif(SKIP_REAL or not all(p.is_file() for p in REAL_INPUTS),
                               reason="research/data inputs missing or UNLOCK_SKIP_REALDATA=1")


# ------------------------------------------------------------ acceptance rule
def test_acceptance_is_strict_and_unrounded():
    b = 0.5186251234
    assert not validate.accept(b, b)
    assert validate.accept(b + 1e-12, b)
    assert round(b - 1e-9, 4) == round(b, 4)
    assert not validate.accept(b - 1e-9, b)        # equal at 4 decimals, still rejected
    assert validate.rejected_reason(b + 1e-12, b) is None
    reason = validate.rejected_reason(b, b)
    assert reason.startswith(validate.NO_CANDIDATE) and repr(b) in reason


def test_reproduction_tolerance():
    assert validate.reproduction_ok(0.472, 0.5186)
    assert validate.reproduction_ok(0.4769, 0.5137)
    assert not validate.reproduction_ok(0.4775, 0.5186)
    assert not validate.reproduction_ok(0.472, 0.5240)


# ------------------------------------------------------------ synthetic configs
def _region(subject, n_gt, gt_pairs):
    k = 6
    return {"subject": subject, "iv_link": np.arange(k), "ex_link": np.arange(k),
            "gt_pairs": set(gt_pairs), "n_gt_pairs": n_gt}


def _synthetic():
    """A kept (all gates), B dropped (all gates), C kept only by the aggressive gate."""
    records = {"A": _region("m1", 2, {(0, 0), (1, 1)}),
               "B": _region("m2", 5, {(k, k) for k in range(5)}),
               "C": _region("m3", 1, {(0, 0)})}
    entries = {
        # thr 0.125+ keeps only the TP -> best on kept regions
        "A": {"subject": "m1", "pairs": np.array([[0, 0], [2, 3]]),
              "probs": np.array([0.30, 0.10]), "y": np.array([True, False])},
        # five TPs at 0.05: would pull the threshold down to <= 0.05 if B counted
        "B": {"subject": "m2", "pairs": np.array([[k, k] for k in range(5)]),
              "probs": np.full(5, 0.05), "y": np.ones(5, bool)},
        "C": {"subject": "m3", "pairs": np.array([[0, 0]]),
              "probs": np.array([0.20]), "y": np.array([True])},
    }
    test = {"t1": {"pairs": np.array([[0, 1], [1, 0], [2, 2]]),
                   "probs": np.array([0.5, 0.11, 0.2])},
            "t2": {"pairs": np.array([[0, 0]]), "probs": np.array([0.9])}}
    pairs_ck = {"heldout": {sel: entries for sel in ("independent", "joint")},
                "test": {sel: test for sel in ("independent", "joint")}}
    cons = {"A": True, "B": False, "C": False}
    aggr = {"A": True, "B": False, "C": True}
    correct = {"A": True, "B": True, "C": False}
    sel_out = {
        "heldout": {s: {"kept": {"conservative": cons[s], "aggressive": aggr[s]},
                        "correct": correct[s]} for s in records},
        "test": {"t1": {"kept": {"conservative": True, "aggressive": True}},
                 "t2": {"kept": {"conservative": False, "aggressive": True}}},
        "conservative": {"tau": 0.4, "unavailable": False},
        "aggressive": {"tau": 0.2},
    }
    verifier_ck = {"selections": {"independent": sel_out, "joint": sel_out}}
    return records, pairs_ck, verifier_ck


def test_config_threshold_uses_only_kept_regions():
    records, pairs_ck, verifier_ck = _synthetic()
    res = {c["name"]: c for c in validate.evaluate_configs(records, pairs_ck, verifier_ck,
                                                           PQ, 0.0)}
    assert [c[0] for c in validate.CONFIGS] == list(res)
    c = res["indep_cons"]
    # kept = {A}: thr 0.125 (pred 1, tp 1); with B counted it would be <= 0.05
    assert c["pair_threshold"] == pytest.approx(0.125)
    assert (c["tp"], c["pred"], c["kept"], c["kept_wrong"]) == (1, 1, 1, 0)
    assert c["f1"] == pytest.approx(2 * 1 / (1 + 8))
    assert c["per_mouse_f1"] == pytest.approx({"m1": 2 / 3, "m2": 0.0, "m3": 0.0})
    assert c["full"] == pytest.approx(0.25 * (0.7404 + 0.3901) + 0.5 * c["f1"])
    # dropped-region probabilities / labels never move the threshold
    pairs_ck["heldout"]["independent"]["B"]["probs"][:] = 0.29
    again = validate.evaluate_configs(records, pairs_ck, verifier_ck, PQ, 0.0)[0]
    assert again["pair_threshold"] == c["pair_threshold"] and again["f1"] == c["f1"]

    a = res["indep_aggr"]                       # kept = {A, C}; C is a wrong pose
    assert (a["kept"], a["kept_wrong"], a["tau"]) == (2, 1, 0.2)
    assert a["pair_threshold"] == pytest.approx(0.125)
    assert (a["tp"], a["pred"]) == (2, 2)
    # test pairs: config threshold and the config's test gate, one-to-one
    assert c["test_pairs"] == {"t1": [[0, 1], [2, 2]], "t2": []}
    assert a["test_pairs"] == {"t1": [[0, 1], [2, 2]], "t2": [[0, 0]]}
    assert c["tau"] == 0.4 and not c["conservative_unavailable"]


def test_config_acceptance_against_baseline_unrounded():
    records, pairs_ck, verifier_ck = _synthetic()
    full = validate.evaluate_configs(records, pairs_ck, verifier_ck, PQ, 0.0)[0]["full"]
    tie = validate.evaluate_configs(records, pairs_ck, verifier_ck, PQ, full)[0]
    assert not tie["accepted"] and tie["compared_to"] == full
    assert tie["rejected_reason"].startswith(validate.NO_CANDIDATE)
    above = validate.evaluate_configs(records, pairs_ck, verifier_ck, PQ, full - 1e-12)[0]
    assert above["accepted"] and above["rejected_reason"] is None


# ------------------------------------------------------------ failure path
def test_reproduction_failure_writes_json_and_no_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "heldout_pq",
                        lambda records: {**PQ, "n_regions": 47, "per_region": {}})
    bad = {"full": 0.4901, "f1": 0.40, "pq_iv": 0.7404, "pq_ex": 0.3901, "tp": 400,
           "pred": 861, "kept": 31, "kept_wrong": 2, "per_mouse_f1": {}}
    monkeypatch.setattr(validate, "baseline_heldout", lambda records, pq: dict(bad))
    ck = {"heldout": {}, "test": {}}
    with pytest.raises(SystemExit) as e:
        checkpoint.run_stage("validate", lambda: validate.run(ck, {}, {}, run_dir=tmp_path),
                             tmp_path, "fp")
    assert e.value.code == 1
    assert not checkpoint.marker_path(tmp_path, "validate").exists()
    assert not checkpoint.checkpoint_path(tmp_path, "validate").exists()
    rec = json.loads((tmp_path / validate.FAILURE_JSON).read_text())
    assert rec["status"] == "BASELINE_REPRODUCTION_FAILED"
    assert rec["measured"]["f1"] == 0.40 and rec["measured"]["full"] == 0.4901
    assert rec["expected"] == validate.EXPECTED


def test_smoke_skips_reproduction_check(tmp_path, monkeypatch):
    records, pairs_ck, verifier_ck = _synthetic()
    monkeypatch.setattr(validate, "heldout_pq",
                        lambda r: {**PQ, "n_regions": len(r), "per_region": {}})
    monkeypatch.setattr(validate, "baseline_heldout",
                        lambda r, pq: {"full": 0.1, "f1": 0.0, "pq_iv": 0.7404,
                                       "pq_ex": 0.3901, "tp": 0, "pred": 0, "kept": 0,
                                       "kept_wrong": 0, "per_mouse_f1": {}})
    out = validate.run({"heldout": records, "test": {}}, pairs_ck, verifier_ck,
                       smoke=True, run_dir=tmp_path)
    assert out["baseline"]["reproduced"] is None
    assert not (tmp_path / validate.FAILURE_JSON).exists()
    assert [c["name"] for c in out["configs"]] == [c[0] for c in validate.CONFIGS]
    assert all(c["accepted"] for c in out["configs"])          # every config > 0.1


# ------------------------------------------------------------ real data
@pytest.fixture(scope="module")
def heldout_records():
    lab = pickle.loads((paths.RDATA / "lab.pkl").read_bytes())
    votes = pickle.loads((paths.RDATA / "vote_cands.pkl").read_bytes())
    scored = pickle.loads((paths.RDATA / "cp_pose_train.pkl").read_bytes())
    return {s: prep.heldout_record(s, lab[s], votes[s], scored[s], "k") for s in sorted(lab)}


@real_data
def test_baseline_reproduces_v10_heldout(heldout_records):
    pq = validate.heldout_pq(heldout_records)
    assert pq["n_regions"] == 47
    assert pq["pq_iv"] == pytest.approx(0.7404, abs=5e-5)
    assert pq["pq_ex"] == pytest.approx(0.3901, abs=5e-5)
    base = validate.baseline_heldout(heldout_records, pq)
    assert (base["tp"], base["pred"]) == (472, 861)
    assert base["f1"] == pytest.approx(0.4720, abs=1e-12)
    assert abs(base["f1"] - 0.472) <= 0.005
    assert abs(base["full"] - 0.5186) <= 0.005
    assert base["full"] == pytest.approx(0.5186, abs=1e-4)
    assert validate.reproduction_ok(base["f1"], base["full"])
    assert base["kept"] == 31
    assert set(base["per_mouse_f1"]) == {r["subject"] for r in heldout_records.values()}


@real_data
@pytest.mark.skipif(not (paths.RDATA / "test_cp_base.npz").is_file()
                    or not validate.V10_CSV.is_file(), reason="test inputs missing")
def test_baseline_test_pairs_reproduce_v10_csv():
    prep._research()
    with mp.get_context("fork").Pool(min(8, os.cpu_count() or 1),
                                     initializer=prep._init_worker) as pool:
        recs, bins, _ = prep._build_test(pool, False)
    got = validate.baseline_test_pairs(recs, bins)
    match = validate.compare_test_pairs(got, validate.csv_index_pairs(recs))
    assert match["regions"] == 29
    assert match["n_mismatch"] == 0, match["mismatched"]
    assert (match["pairs"], match["regions_with_pairs"]) == (316, 11)
