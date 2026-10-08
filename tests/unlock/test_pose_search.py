"""Tests for hpc_unlock/pose_search.py (Req 4.3-4.10, 5.7, 5.10)."""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hpc_unlock import checkpoint, paths, prep
from hpc_unlock import pose_search as ps
from hpc_unlock import soft as S

R = ps.research()
TINY = ps.HoughGrid(angles=(-10.0, 0.0, 10.0), scales=(0.93, 1.0), stretches=((1.0, 0.0),), top=6)


def _tiny_params(sigma=2.5, grid=TINY):
    return {**ps.params(SimpleNamespace(sigma=sigma), smoke=True), "hough": grid.as_dict()}


def _M(a00=1.0, a01=0.0, a10=0.0, a11=1.0, tx=0.0, ty=0.0):
    return np.array([[a00, a01, tx], [a10, a11, ty]], float)


@pytest.fixture(scope="module")
def lab():
    with open(paths.RDATA / "lab.pkl", "rb") as f:
        L = pickle.load(f)
    with open(paths.RDATA / "vote_cands.pkl", "rb") as f:
        V = pickle.load(f)
    with open(paths.RDATA / "cp_pose_train.pkl", "rb") as f:
        C = pickle.load(f)
    return L, V, C


def _record(lab, sid, n_win=None, n_vote=None):
    L, V, C = lab
    scored = list(C[sid])
    if n_win is not None:
        scored = ([c for c in scored if c[3] == "win"][:n_win]
                  + [c for c in scored if c[3] == "vote"][:n_vote])
    return prep.heldout_record(sid, L[sid], V[sid], scored, key=f"key-{sid}")


def _cp_bin(sid):
    with np.load(paths.RDATA / "heldout_labels.npz") as z:
        cp = z[f"{sid}|exvivo|prob"].astype(np.float32)
    return R.cp_pose_lab.cp_map(cp).astype(np.uint8)


def _gpu_cand(M, score=1.0):
    ang, scale, _ = S.decompose(M)
    return {"M": np.asarray(M, float), "score": float(score), "angle": ang, "scale": scale,
            "stretch_k": 1.0, "stretch_dir": None,
            "translation": (float(M[0, 2]), float(M[1, 2])), "landing": (0.0, 0.0)}


# ------------------------------------------------------------------ dedup
def test_dedup_keeps_higher_soft_within_tolerance_inclusive():
    a = _M()
    b = _M(a00=1.005, tx=4.0)                # |dA| <= 0.01, |dt| = 4 exactly: duplicates
    assert ps.dedup([a, b], [1.0, 2.0]) == [1]
    assert ps.dedup([a, b], [2.0, 1.0]) == [0]
    c = _M(a00=1.5)                          # |dA| = 0.5 exactly (representable): inclusive
    assert ps.dedup([a, c], [1.0, 2.0], lin_tol=0.5) == [1]


def test_dedup_tie_keeps_earlier_source():
    a, b = _M(), _M(ty=1.0)
    assert ps.dedup([a, b], [3.0, 3.0]) == [0]


def test_dedup_outside_tolerance_keeps_both_in_soft_order():
    a = _M()
    lin = _M(a01=0.0101)                     # linear part just outside 0.01
    tr = _M(tx=3.0, ty=2.7)                  # |dt| = 4.04 > 4
    assert ps.dedup([a, lin, tr], [1.0, 3.0, 2.0]) == [1, 2, 0]
    assert ps.dedup([], []) == []


def test_dedup_dropped_candidate_has_kept_duplicate_not_lower():
    rng = np.random.default_rng(0)
    Ms = [_M(1 + rng.uniform(-0.01, 0.01), tx=rng.uniform(0, 6)) for _ in range(40)]
    softs = rng.integers(0, 5, 40).astype(float)
    kept = ps.dedup(Ms, softs)
    assert [softs[i] for i in kept] == sorted((softs[i] for i in kept), reverse=True)
    for i in set(range(40)) - set(kept):
        assert any(np.abs(Ms[i][:, :2] - Ms[k][:, :2]).max() <= 0.01
                   and np.linalg.norm(Ms[i][:, 2] - Ms[k][:, 2]) <= 4 and softs[k] >= softs[i]
                   for k in kept)


def test_merge_source_order_and_labels():
    scored = [(_M(tx=3), 5.0, 1.0, "vote"), (_M(tx=2), 4.0, 1.0, "win")]
    m = ps.merge([(_M(), 1.0)], [(_M(tx=1), 2.0)], scored)
    assert [c[2] for c in m] == ["hough", "gpu", "window", "vote"]
    assert [c[1] for c in m] == [1.0, 2.0, 4.0, 5.0]
    with pytest.raises(ValueError):
        ps.merge([], [], [(_M(), 1.0, 0.0, "other")])


# ------------------------------------------------------------- Hough
def test_hough_refined_matches_wide_candidates(lab, monkeypatch):
    """wide_soft.wide_candidates == wide_soft's own dedup applied to hough_refined."""
    sid = sorted(lab[0])[0]
    r = lab[0][sid]
    grid = ps.HoughGrid((-5.0, 0.0, 5.0), (0.95, 1.01), ((1.0, 0.0), (1.08, 45.0)), 9)
    monkeypatch.setattr(R.wide_soft, "ANGLES", np.asarray(grid.angles))
    monkeypatch.setattr(R.wide_soft, "SCALES", np.asarray(grid.scales))
    ref = R.wide_soft.wide_candidates(r["iv_c"], r["ex_c"], top=grid.top,
                                      stretches=list(grid.stretches))
    out = []
    for Rm, sc in ps.hough_refined(r["iv_c"], r["ex_c"], grid):
        if any(np.abs(Rm - o[0]).max() < 1e-3 or (np.abs(Rm[:, :2] - o[0][:, :2]).max() < 0.01
               and np.linalg.norm(Rm[:, 2] - o[0][:, 2]) < 4) for o in out):
            continue
        out.append((Rm, sc))
    assert len(out) == len(ref) > 0
    for (M, s), (M0, s0) in zip(out, ref):
        np.testing.assert_array_equal(M, M0)
        assert s == s0


def test_production_grid_is_wide_soft_grid():
    ps.check_wide_soft_grid()
    g = ps.hough_grid(False)
    assert g.angles[0] == -35 and g.angles[-1] == 35 and len(g.stretches) == 9


# ------------------------------------------------------------ features
def test_features_match_soft_margin_lab_and_cp_z(lab):
    L, V, _ = lab
    sids = prep.smoke_heldout({s: L[s]["subject"] for s in L})
    for sid in sids:
        rec = _record(lab, sid, n_win=3, n_vote=2)
        cp_bin = _cp_bin(sid)
        gt = rec["gt_M"] if rec["gt_M"] is not None else rec["cp_scored"][0][0]
        gpu = [_gpu_cand(gt + np.array([[0, 0, 3.0], [0, 0, -2.0]])),
               _gpu_cand(_M(tx=50.0, ty=40.0))]
        out = ps.region_candidates(rec, cp_bin, gpu, _tiny_params(2.0))
        cands = out["cands"]
        assert out["n"] == len(cands) > 0
        assert sum(out["n_by_source"].values()) == 6 + 2 + 5   # top=6 Hough peaks + 2 + 5
        softs = [c["soft"] for c in cands]
        assert softs == sorted(softs, reverse=True)
        Ms = [c["M"] for c in cands]
        for c in cands:
            M = c["M"]
            assert c["source"] in ps.SOURCES
            assert c["soft"] == R.wide_soft.soft(rec["iv_c"], rec["ex_c"], M, 2.0)
            assert c["z"] == R.cp_pose_lab.cp_z(cp_bin, rec["iv_c"], M)
            assert c["refine_margin"] == R.margin_lab.margin(V[sid], M, c["refine_score"], L[sid])
            ang, land = S.pose(M, L[sid]["offset"])
            assert c["angle"] == pytest.approx(ang) and np.allclose(c["landing"], land)
            assert c["scale"] == pytest.approx(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        np.testing.assert_allclose([c["soft_margin"] for c in cands],
                                   S.soft_margins(Ms, softs, L[sid]["offset"]))
        for i in range(len(Ms)):            # no two kept candidates are duplicates
            for j in range(i):
                assert not (np.abs(Ms[i][:, :2] - Ms[j][:, :2]).max() <= 0.01
                            and np.linalg.norm(Ms[i][:, 2] - Ms[j][:, 2]) <= 4)
        win = {(round(c["refine_score"], 9)) for c in cands if c["source"] == "window"}
        assert win <= {round(s, 9) for _, s, _, src in rec["cp_scored"] if src == "win"}


# ---------------------------------------------------------- sigma check
def _no_load(name):
    raise AssertionError(f"ctx.load({name!r}) called before sigma validation")


@pytest.mark.parametrize("sigma", [1.4, 2.6, float("nan"), None])
def test_invalid_sigma_stops_before_any_region(sigma, capsys, tmp_path):
    cfg = SimpleNamespace(sigma=sigma)
    assert ps.validate_config(cfg)
    ctx = SimpleNamespace(run_dir=tmp_path, smoke=True, workers=1, load=_no_load)
    with pytest.raises(ValueError, match="sigma"):
        ps.compute(cfg, ctx)
    assert "SIGMA_INVALID" in capsys.readouterr().out


def test_valid_sigma_bounds():
    assert ps.validate_config(SimpleNamespace(sigma=1.5)) == []
    assert ps.validate_config(SimpleNamespace(sigma=2.5)) == []


# ------------------------------------------------------- Stage compute
def _ctx(tmp_path, heldout, bins, gpu, workers):
    side = checkpoint.save_npz_atomic(tmp_path / prep.CP_SIDE, **bins)
    P = {"heldout": heldout, "test": {}, "cp_side": side}
    deps = {"prep": P, "gpu_scan": {"by_sid": gpu}}
    return SimpleNamespace(run_dir=tmp_path, smoke=True, workers=workers, load=deps.__getitem__)


def test_zero_candidate_region_recorded(tmp_path, lab):
    L, V, C = lab
    sid = sorted(L)[0]
    rec = prep.heldout_record("empty_region", L[sid], [], [], key="k")
    rec.update(iv_c=np.zeros((2, 2)), ex_c=np.zeros((1, 2)), gt_M=None, gt_iv_c=None)
    ctx = _ctx(tmp_path, {"empty_region": rec}, {"empty_region": np.zeros((8, 8), np.uint8)},
               {"empty_region": {"cands": [], "kept": 0}}, workers=1)
    out = ps.compute(SimpleNamespace(sigma=2.5), ctx)
    assert out["heldout"]["empty_region"]["cands"] == []
    assert out["heldout"]["empty_region"]["n"] == 0
    assert out["zero_candidates"] == {"heldout": ["empty_region"], "test": []}
    assert out["n_zero"] == 1 and out["diagnostics"]["n_with_gt"] == 0


def test_compute_smoke_two_heldout_regions(tmp_path, lab):
    """2 held-out regions (2 mice), fake gpu_scan with the GT pose and a wrong pose."""
    L, V, C = lab
    sids = prep.smoke_heldout({s: L[s]["subject"] for s in L})
    heldout, bins, gpu = {}, {}, {}
    for sid in sids:
        heldout[sid] = _record(lab, sid)
        bins[sid] = _cp_bin(sid)
        gt = heldout[sid]["gt_M"]
        cands = [_gpu_cand(_M(tx=-80.0, ty=120.0), 2.0)]
        if gt is not None:
            cands.insert(0, _gpu_cand(gt, 3.0))
        gpu[sid] = {"cands": cands, "kept": len(cands)}
    ctx = _ctx(tmp_path, heldout, bins, gpu, workers=2)
    t0 = time.monotonic()
    out = ps.compute(SimpleNamespace(sigma=2.5), ctx)
    assert time.monotonic() - t0 < 90

    n_gt = sum(heldout[s]["gt_M"] is not None for s in sids)
    d = out["diagnostics"]
    assert d["n_with_gt"] == n_gt and d["sigma"] == out["sigma"] == 2.5
    assert d["correct_in_raw_gpu"] == n_gt          # the GT pose itself is a raw candidate
    assert d["correct_in_candidates"] == n_gt       # refined GT pose stays correct
    assert 0 <= d["correct_first"] <= n_gt
    assert list(out["heldout"]) == sids and out["test"] == {}
    assert out["params"]["hough"]["stretches"] == [(1.0, 0.0)]
    for sid in sids:
        reg = out["heldout"][sid]
        assert reg["n"] == len(reg["cands"]) > 0
        assert reg["n_by_source"]["gpu"] == len(gpu[sid]["cands"])
        assert reg["n_by_source"]["window"] + reg["n_by_source"]["vote"] == len(C[sid])
        softs = [c["soft"] for c in reg["cands"]]
        assert softs == sorted(softs, reverse=True)
        first = d["per_region"].get(sid)
        if first is not None:
            ok0 = R.reg_lab.err(heldout[sid], reg["cands"][0]["M"]) < 5
            assert first["first"] == ok0
    p = tmp_path / "pose_search.pkl"
    checkpoint.save_atomic(p, out)
    back = checkpoint.load(p)
    np.testing.assert_array_equal(back["heldout"][sids[0]]["cands"][0]["M"],
                                  out["heldout"][sids[0]]["cands"][0]["M"])
