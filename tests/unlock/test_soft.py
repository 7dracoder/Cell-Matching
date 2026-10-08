"""Unit tests for hpc_unlock/soft.py (Req 4.4)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from hpc_unlock import paths, soft as S

import wide_soft  # noqa: E402  (research/ on sys.path via hpc_unlock.paths)


def _similarity(theta_deg, scale, tx=0.0, ty=0.0):
    t = np.radians(theta_deg)
    return np.array([[scale * np.cos(t), -scale * np.sin(t), tx],
                     [scale * np.sin(t), scale * np.cos(t), ty]])


def _random_affine(rng):
    A = _similarity(rng.uniform(-35, 35), rng.uniform(0.85, 1.13))[:, :2]
    A = A @ (np.eye(2) + rng.normal(0, 0.05, (2, 2)))
    return np.c_[A, rng.uniform(-400, 400, 2)]


# ---------------------------------------------------------------- imports
def test_does_not_import_objective_probe_or_soft_gate_cv():
    tree = ast.parse(Path(S.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "objective_probe" not in names
    assert "soft_gate_cv" not in names


# ---------------------------------------------------------------- soft
def test_soft_equals_wide_soft():
    rng = np.random.default_rng(0)
    ex_c = rng.uniform(0, 600, (300, 2))
    iv_c = ex_c[:200] + rng.normal(0, 1.5, (200, 2))
    for sigma in (1.5, 2.0, 2.5):
        for _ in range(5):
            M = _similarity(rng.uniform(-3, 3), rng.uniform(0.98, 1.02), *rng.uniform(-5, 5, 2))
            assert S.soft(iv_c, ex_c, M, sigma) == wide_soft.soft(iv_c, ex_c, M, sigma)


def test_soft_identity_counts_exact_matches():
    pts = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    assert S.soft(pts, pts, np.c_[np.eye(2), np.zeros(2)]) == pytest.approx(3.0)


# ---------------------------------------------------------------- pose / landing
def test_pose_matches_window_lab_pose():
    rdata = paths.RDATA
    if not ((rdata / "lab.pkl").is_file() and (rdata / "reg_hough.pkl").is_file()):
        pytest.skip("window_lab data pickles not present")
    with mock.patch.object(sys, "argv", ["window_lab"]):
        import window_lab
    rng = np.random.default_rng(1)
    for _ in range(10):
        M = _random_affine(rng)
        off = rng.uniform(-200, 800, 2)
        a_ref, l_ref = window_lab.pose({"offset": off}, M)
        a, l = S.pose(M, off)
        assert a == a_ref
        np.testing.assert_array_equal(l, l_ref)


def test_landing_formula():
    M = _similarity(0.0, 1.0, 10.0, -20.0)
    np.testing.assert_allclose(S.landing(M, (100.0, 50.0)), [210.0, 230.0])


# ---------------------------------------------------------------- decomposition
def test_decompose_similarity():
    a, s, an = S.decompose(_similarity(12.0, 1.05, 3.0, 4.0))
    assert a == pytest.approx(12.0)
    assert s == pytest.approx(1.05)
    assert an == pytest.approx(0.0, abs=1e-12)


def test_decompose_stretch():
    M = np.c_[np.diag([1.08, 1.0]), np.zeros(2)]
    a, s, an = S.decompose(M)
    assert a == pytest.approx(0.0)
    assert s == pytest.approx(np.sqrt(1.08))
    assert an == pytest.approx(0.08)


@settings(max_examples=100, deadline=None)
@given(theta=st.floats(-35, 35), scale=st.floats(0.8, 1.2), k=st.floats(0.9, 1.1),
       phi=st.floats(0, 180))
def test_decompose_scale_and_anisotropy_of_rotated_stretch(theta, scale, k, phi):
    # Rotation-scale times a stretch k along phi has singular values scale*{k, 1}.
    A = _similarity(theta, scale)[:, :2] @ wide_soft.stretch(k, phi)
    _, s, an = S.decompose(np.c_[A, np.zeros(2)])
    assert s == pytest.approx(scale * np.sqrt(k), rel=1e-9)
    assert an == pytest.approx(max(k, 1 / k) - 1, abs=1e-9)


# ---------------------------------------------------------------- soft margin
def test_soft_margin_none_clearly_different_uses_zero_second():
    # No clearly different alternative: the best alternative counts as 0 (choose convention).
    Ms = [_similarity(0.0, 1.0), _similarity(2.0, 1.0, 5.0, 5.0)]
    np.testing.assert_array_equal(S.soft_margins(Ms, [10.0, 8.0], (0.0, 0.0)), [10.0, 8.0])
    assert S.soft_margin(0, Ms, [10.0, 8.0], (0.0, 0.0)) == 10.0
    assert S.soft_margins([], [], (0.0, 0.0)).shape == (0,)


def test_soft_margin_angle_and_landing_thresholds():
    off = (0.0, 0.0)
    base = _similarity(0.0, 1.0)
    Ms = [base,
          _similarity(3.5, 1.0),             # angle > 3 deg: clearly different
          _similarity(0.0, 1.0, 61.0, 0.0),  # landing > 60 px: clearly different
          _similarity(2.9, 1.0, 0.0, 0.0)]   # close in angle; landing moves ~21 px (< 60)
    softs = [20.0, 12.0, 15.0, 19.0]
    m = S.soft_margins(Ms, softs, off)
    assert m[0] == pytest.approx(20.0 - 15.0)
    assert m[1] == pytest.approx(12.0 - 20.0)   # best of {0, 2, 3}: all clearly different from 3.5 deg
    assert m[2] == pytest.approx(15.0 - 20.0)
    for i in range(len(Ms)):
        assert S.soft_margin(i, Ms, softs, off) == pytest.approx(m[i])


def _choose_margin(r, cs):
    """research/soft_gate_cv.choose margin, inlined (that module loads data at import)."""
    def pose(M):
        return np.degrees(np.arctan2(M[1, 0], M[0, 0])), M[:, :2] @ (S.P0 - r["offset"]) + M[:, 2]
    k = int(np.argmax([c[2] for c in cs]))
    a, l = pose(cs[k][0])
    alt = [c[2] for c in cs if abs(pose(c[0])[0] - a) > 3 or np.linalg.norm(pose(c[0])[1] - l) > 60]
    return k, cs[k][2] - (max(alt) if alt else 0.0)


def test_soft_margin_of_top_candidate_matches_choose():
    rng = np.random.default_rng(2)
    for _ in range(20):
        n = int(rng.integers(1, 12))
        Ms = [_random_affine(rng) for _ in range(n)]
        softs = list(rng.uniform(0, 100, n))
        off = rng.uniform(0, 500, 2)
        k, ref = _choose_margin({"offset": off}, [(M, 0.0, s) for M, s in zip(Ms, softs)])
        assert S.soft_margins(Ms, softs, off)[k] == pytest.approx(ref)
