"""Unit tests for hpc_unlock.report (Req 11, 9.4, 9.6, 12.2, 12.7, 12.11, 12.12)."""
from __future__ import annotations

import json

from hpc_unlock import report


def _baseline(**kw):
    b = {"full": 0.51862, "f1": 0.47213, "pq_iv": 0.6012, "pq_ex": 0.4300,
         "kept": 30, "kept_wrong": 2, "reproduced": True,
         "per_mouse_f1": {"m1": 0.41, "m2": 0.52, "m3": 0.47}}
    b.update(kw)
    return b


def _cfg(name, full, f1, kept_wrong=0, accepted=True, **kw):
    c = {"name": name, "accepted": accepted, "full": full, "f1": f1,
         "pq_iv": 0.6012, "pq_ex": 0.4300, "kept": 35, "kept_wrong": kept_wrong,
         "per_mouse_f1": {"m1": 0.5, "m2": 0.6, "m3": 0.55},
         "csv": f"/x/submission_v13_{name}.csv" if accepted else None,
         "rejected_reason": None if accepted else "full <= Baseline full",
         "unlock_table": [{"sid": "s1", "baseline_pairs": 0,
                           "candidate_pairs": 3, "newly_unlocked": True},
                          {"sid": "s2", "baseline_pairs": 4,
                           "candidate_pairs": 4, "newly_unlocked": False}]}
    c.update(kw)
    return c


def _state(configs, **kw):
    s = {"baseline": _baseline(), "configs": configs,
         "selftrain": {"status": "disabled"},
         "diagnostics": {
             "pose_search": {"sigma": 2.0, "n_with_gt": 46,
                             "correct_in_candidates": 40, "correct_first": 31},
             "gpu_scan": {"n_with_gt": 46, "correct_in_raw": 38,
                          "correct_in_merged": 41},
             "joint": {"total": {"joint": 33, "independent": 29},
                       "per_mouse": {"m1": {"joint": 10, "independent": 9}}}}}
    s.update(kw)
    return s


def _all_scores(rep):
    rows = [rep["baseline"], *rep["ranking"]]
    vals = [r[k] for r in rows for k in report.SCORE_KEYS]
    vals += [r["full"] for r in rep["rejected"]]
    return [v for v in vals if v is not None]


def test_ranking_order_and_tiebreaks():
    cfgs = [_cfg("a", 0.55, 0.50, kept_wrong=1),
            _cfg("b", 0.56, 0.49),
            _cfg("c", 0.55, 0.51, kept_wrong=3),
            _cfg("d", 0.55, 0.50, kept_wrong=0),
            _cfg("r", 0.50, 0.40, accepted=False)]
    rep = report.build(_state(cfgs))
    assert [c["name"] for c in rep["ranking"]] == ["b", "c", "d", "a"]
    assert [c["rank"] for c in rep["ranking"]] == [1, 2, 3, 4]
    assert rep["status"] == "accepted"
    assert rep["recommendation"]["finals"] == ["submission_v13_b.csv",
                                               "submission_v10_cpgate.csv"]
    note = rep["recommendation"]["note"]
    assert "0.48893" in note and "(c)" in note
    assert [r["name"] for r in rep["rejected"]] == ["r"]


def test_no_candidate_lists_rejected_and_fallback_finals():
    cfgs = [_cfg("x", 0.5101, 0.46, accepted=False),
            _cfg("y", 0.5150, 0.47, accepted=False)]
    rep = report.build(_state(cfgs))
    assert rep["status"] == report.NO_CANDIDATE
    assert rep["ranking"] == []
    assert rep["recommendation"]["finals"] == ["submission_v10_cpgate.csv",
                                               "submission_v7_grow15.csv"]
    assert [(r["name"], r["full"]) for r in rep["rejected"]] == [
        ("x", 0.5101), ("y", 0.5150)]
    assert all(r["compared_to"] == 0.51862 for r in rep["rejected"])
    md = report.render_md(rep)
    assert "NO_CANDIDATE" in md
    assert "| x | 0.5101 | 0.5186 |" in md and "| y | 0.5150 | 0.5186 |" in md


def test_disclaimer_in_both_outputs():
    rep = report.build(_state([_cfg("a", 0.55, 0.5)]))
    assert "held-out gains are estimates" in rep["disclaimer"]
    assert "0.65" in rep["disclaimer"] and "not guaranteed" in rep["disclaimer"]
    assert rep["disclaimer"] in report.render_md(rep)


def test_md_contains_every_json_score(tmp_path):
    cfgs = [_cfg("a", 0.55123, 0.50987), _cfg("b", 0.53333, 0.49111),
            _cfg("r", 0.50017, 0.41, accepted=False)]
    rep = report.build(_state(cfgs))
    jp, mp = report.write(rep, tmp_path)
    assert jp.name == "hpc_unlock_report.json" and mp.name == "hpc_unlock_report.md"
    loaded = json.loads(jp.read_text())
    md = mp.read_text()
    assert loaded == json.loads(json.dumps(rep))
    assert [c["name"] for c in loaded["ranking"]] == ["a", "b"]
    for v in _all_scores(loaded):
        assert f"{v:.4f}" in md
    # ranking rows appear in the same order in the Markdown
    assert md.index("| 1 | a |") < md.index("| 2 | b |")
    assert not list(tmp_path.glob("*.tmp-*"))


def test_baseline_and_candidate_counts_reported():
    rep = report.build(_state([_cfg("a", 0.55, 0.5, kept_wrong=1)]))
    md = report.render_md(rep)
    assert "| - | Baseline (submission_v10_cpgate.csv) | 0.5186 | 0.4721 | " \
           "0.6012 | 0.4300 | 30 | 2 |" in md
    assert "| 1 | a | 0.5500 | 0.5000 | 0.6012 | 0.4300 | 35 | 1 |" in md
    assert "m1 0.5000" in md  # per-mouse F1 (9.6)
    assert "Unlock table: a (1 newly unlocked)" in md


def test_baseline_reproduction_failure_blocks_candidates():
    st = _state([_cfg("a", 0.60, 0.55)],
                baseline=_baseline(reproduced=False, full=0.50, f1=0.44))
    rep = report.build(st)
    assert rep["baseline_reproduction"]["status"] == "failed"
    assert rep["baseline_reproduction"]["measured_full"] == 0.50
    assert rep["ranking"] == [] and rep["status"] == report.NO_CANDIDATE
    md = report.render_md(rep)
    assert "Baseline reproduction failure" in md and "0.4400" in md


def test_smoke_reproduction_is_na():
    st = _state([], baseline=_baseline(reproduced=None), run={"smoke": True})
    assert report.build(st)["baseline_reproduction"]["status"] == "n/a (smoke)"


def test_selftrain_statuses():
    dis = report.build(_state([]))
    assert dis["selftrain"]["status"] == "disabled"
    assert "disabled" in report.render_md(dis)

    nc = report.build(_state([], selftrain={
        "status": "no_candidate", "full": 0.5111, "compared_to": 0.51862,
        "no_confident_mice": ["m2"]}))
    md = report.render_md(nc)
    assert "NO_CANDIDATE: self-trained held-out full 0.5111 <= compared 0.5186" in md
    assert "no confident regions" in md and "m2" in md

    ncr = report.build(_state([], selftrain={"status": "no_confident_regions"}))
    assert "NO_CANDIDATE: no confident regions" in report.render_md(ncr)

    acc = report.build(_state([_cfg("a", 0.55, 0.50)], selftrain={
        "status": "accepted", "full": 0.58, "f1": 0.52, "pq_iv": 0.6, "pq_ex": 0.5,
        "kept": 35, "kept_wrong": 0, "compared_to": 0.55,
        "csv": "/x/submission_v13_selftrain.csv"}))
    assert [c["name"] for c in acc["ranking"]] == ["selftrain", "a"]
    assert acc["recommendation"]["finals"][0] == "submission_v13_selftrain.csv"


def test_diagnostics_rendered():
    md = report.render_md(report.build(_state([])))
    assert "sigma 2.0 px" in md and "40/46" in md and "31/46" in md
    assert "raw scan candidates 38/46" in md and "41/46" in md
    assert "joint 33, independent 29" in md and "m1: joint 10, independent 9" in md
