"""Unit tests for the gpu_scan grid and FFT kernel (Req 5.1-5.4). CPU torch only."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hpc_unlock import gpu_scan as gs

DEV = "cpu"


# ------------------------------------------------------------------ grids
def test_angle_grid_default():
    a = gs.angle_grid(0.5)
    assert len(a) == 141
    assert a[0] == -35.0 and a[-1] == 35.0
    assert np.allclose(np.diff(a), 0.5)


def test_scale_grid_default():
    s = gs.scale_grid(0.01)
    assert len(s) == 29
    assert s[0] == 0.85 and s[-1] == 1.13
    assert np.allclose(np.diff(s), 0.01)


@pytest.mark.parametrize("step", [0.3, 0.25, 0.007])
def test_grid_steps_never_exceed_request(step):
    for g, lo, hi in ((gs.angle_grid(step), -35.0, 35.0), (gs.scale_grid(step), 0.85, 1.13)):
        assert g[0] == lo and g[-1] == hi
        assert np.diff(g).max() <= step + 1e-12


def test_stretch_hypotheses():
    h = gs.stretch_hypotheses()
    assert len(h) == 17 and len(set(h)) == 17
    assert h[0] == (1.0, None)
    assert all(phi is not None and k != 1.0 for k, phi in h[1:])
    assert {k for k, _ in h[1:]} == {0.92, 0.96, 1.04, 1.08}
    assert {phi for _, phi in h[1:]} == {0.0, 45.0, 90.0, 135.0}
    mats = [gs.stretch_matrix(k, phi) for k, phi in h]
    assert np.allclose(mats[0], np.eye(2))
    assert len({m.round(12).tobytes() for m in mats}) == 17


def test_stretch_matrix_formula():
    k, phi = 1.08, 45.0
    u = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    assert np.allclose(gs.stretch_matrix(k, phi), np.eye(2) + (k - 1) * np.outer(u, u))
    with pytest.raises(ValueError):
        gs.stretch_matrix(1.08, None)


def test_full_grid_linear_parts():
    g = gs.full_grid()
    assert len(g) == 141 * 29 * 17 == 69513
    rng = np.random.default_rng(0)
    for p in rng.integers(0, len(g), 20):
        k, phi = g.stretch_of(p)
        ia, is_, ik = p // (29 * 17), (p // 17) % 29, p % 17
        assert g.angle[p] == g.angles[ia] and g.scale[p] == g.scales[is_] and g.stretch_idx[p] == ik
        want = g.scale[p] * gs.rotation(g.angle[p]) @ gs.stretch_matrix(k, phi)
        assert np.allclose(g.A[p], want)
    # k = 1: angle recovered exactly as window_lab.pose does (atan2(A10, A00))
    p0 = int(np.flatnonzero(g.stretch_idx == 0)[123])
    assert math.isclose(math.degrees(math.atan2(g.A[p0, 1, 0], g.A[p0, 0, 0])), g.angle[p0], abs_tol=1e-9)


# ------------------------------------------------------------- fast_size
def _is_fast(m):
    for p in (2, 3, 5):
        while m % p == 0:
            m //= p
    return m == 1


def test_fast_size():
    for n in range(1, 2000):
        m = gs.fast_size(n)
        assert m >= n and _is_fast(m)
        assert all(not _is_fast(j) for j in range(n, m))
    assert gs.fast_size(1350) == 1350 and gs.fast_size(1351) == 1440


# ------------------------------------------------------ numeric reference
def _ref_splat(pts, h, w):
    out = np.zeros((h, w))
    for x, y in pts:
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dy, dx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                           (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= y0 + dy < h and 0 <= x0 + dx < w:
                out[y0 + dy, x0 + dx] += wt
    return out


def _ref_corr(E, T, s, R, ty_range, tx_range):
    """Direct discrete correlation: sum_v sum_u E[v] T[u] g(v - u - t)."""
    ev = np.argwhere(E != 0)
    tu = np.argwhere(T != 0)
    out = np.zeros((len(ty_range), len(tx_range)))
    for a, ty in enumerate(ty_range):
        for b, tx in enumerate(tx_range):
            acc = 0.0
            for vy, vx in ev:
                for uy, ux in tu:
                    dy, dx = vy - uy - ty, vx - ux - tx
                    if abs(dy) <= R and abs(dx) <= R:
                        acc += E[vy, vx] * T[uy, ux] * math.exp(-(dx * dx + dy * dy) / (2 * s * s))
            out[a, b] = acc
    return out


def test_fft_matches_direct_correlation_32x32():
    rng = np.random.default_rng(7)
    H = W = 62                                   # (Hg, Wg) = (32, 32) at cell 2
    cell, sigma = 2.0, 1.5
    # points near the top-left edge so that negative shifts carry real mass
    ex = np.r_[rng.uniform(0, 61.9, (8, 2)), [[0.3, 0.7], [1.2, 60.5]]].astype(np.float32)
    iv = rng.uniform(0, 30, (6, 2)).astype(np.float32)
    grid = gs.linear_parts([-12.0, 7.5], [0.9, 1.1], [(1.0, None), (1.08, 45.0)])
    tshape = gs.template_shape(iv, grid.A, cell)
    spec = gs.ex_spectrum(ex, (H, W), sigma, cell, tshape, DEV)
    assert (spec.Hg, spec.Wg) == (32, 32)
    assert spec.Hf >= spec.Hg + spec.Ht + 2 * spec.R and gs.fast_size(spec.Hf) == spec.Hf
    res = gs.correlate_batch(spec, torch.from_numpy(iv), torch.from_numpy(grid.A))
    assert res.corr.dtype == torch.float32 and res.corr.shape == (len(grid), spec.Hf, spec.Wf)

    s = sigma * math.sqrt(2) / cell
    E = _ref_splat(ex.astype(np.float64) / cell, spec.Hg, spec.Wg)
    # every linear shift with possible overlap, including negative ones
    ty_range = np.arange(-(spec.Ht - 1) - spec.R, spec.Hg + spec.R)
    tx_range = np.arange(-(spec.Wt - 1) - spec.R, spec.Wg + spec.R)
    assert len(ty_range) <= spec.Hf and len(tx_range) <= spec.Wf      # no aliasing
    ky, kx = np.mod(ty_range, spec.Hf), np.mod(tx_range, spec.Wf)
    assert np.array_equal(spec.sy.numpy()[ky], ty_range) and np.array_equal(spec.sx.numpy()[kx], tx_range)
    for b in (0, 5):                             # one k = 1 pose, one stretched pose
        G = iv.astype(np.float64) @ grid.A[b].T / cell
        o = 2 - np.floor(G.min(0))
        assert np.array_equal(res.o[b].numpy(), o.astype(np.int64))
        T = _ref_splat(G + o, spec.Ht, spec.Wt)
        ref = _ref_corr(E, T, s, spec.R, ty_range, tx_range)
        got = res.corr[b].double().numpy()[np.ix_(ky, kx)]
        assert np.abs(got - ref).max() <= 1e-4 * ref.max()
        neg = ref[ty_range < 0][:, tx_range < 0]
        assert neg.max() > 0.05 * ref.max()     # negative shifts are actually exercised
        # indices outside the linear range hold (numerically) nothing: no wraparound
        rest = np.ones((spec.Hf, spec.Wf), bool)
        rest[np.ix_(ky, kx)] = False
        if rest.any():
            assert np.abs(res.corr[b].numpy()[rest]).max() <= 1e-4 * ref.max()


def test_valid_mask_matches_field_centre_rule():
    rng = np.random.default_rng(3)
    iv = rng.uniform(0, 40, (12, 2)).astype(np.float32)
    ex = rng.uniform(0, 50, (12, 2)).astype(np.float32)
    grid = gs.linear_parts([-20.0, 15.0], [0.9], [(1.0, None)])
    H, W, cell = 50, 70, 2.0
    spec = gs.ex_spectrum(ex, (H, W), 2.0, cell, gs.template_shape(iv, grid.A, cell), DEV)
    res = gs.correlate_batch(spec, torch.from_numpy(iv), torch.from_numpy(grid.A))
    mask = gs.valid_mask(spec, res.o, res.centre).numpy()
    for b in range(len(grid)):
        m = (iv.astype(np.float64) @ grid.A[b].T).mean(0)
        o = res.o[b].numpy()
        cx = m[0] + cell * (spec.sx.numpy() + o[0])
        cy = m[1] + cell * (spec.sy.numpy() + o[1])
        want = ((cy >= 0) & (cy < H))[:, None] & ((cx >= 0) & (cx < W))[None, :]
        assert np.array_equal(mask[b], want)
        assert mask[b].any()


def test_empty_point_sets():
    grid = gs.linear_parts([0.0], [1.0], [(1.0, None)])
    spec = gs.ex_spectrum(np.zeros((0, 2)), (40, 40), 2.0, 2.0, (6, 6), DEV)
    res = gs.correlate_batch(spec, torch.zeros(0, 2), torch.from_numpy(grid.A))
    assert float(res.corr.abs().max()) == 0.0
    assert not gs.valid_mask(spec, res.o, res.centre).any()


# ----------------------------------------------------------- planted pose
def _spaced_points(rng, n, lo, hi, min_d):
    pts = []
    while len(pts) < n:
        p = rng.uniform(lo, hi, 2)
        if all(np.hypot(*(p - q)) >= min_d for q in pts):
            pts.append(p)
    return np.array(pts)


def test_planted_pose_is_the_peak():
    rng = np.random.default_rng(11)
    iv = _spaced_points(rng, 25, 0, 100, 10.0)
    grid = gs.linear_parts([-10.0, -5.0, 0.0, 5.0, 10.0], [0.95, 1.0, 1.05],
                           [(1.0, None), (1.08, 45.0)])
    p_true = int(np.flatnonzero((grid.angle == 5.0) & (grid.scale == 1.05) & (grid.stretch_idx == 1))[0])
    tau = np.array([70.3, 41.7])                                   # (x, y) px
    H, W, cell, sigma = 180, 240, 2.0, 2.0
    ex = iv @ grid.A[p_true].T + tau
    assert ex.min() >= 0 and (ex[:, 0] < W).all() and (ex[:, 1] < H).all()

    spec = gs.ex_spectrum(ex, (H, W), sigma, cell, gs.template_shape(iv, grid.A, cell), DEV)
    res = gs.correlate_batch(spec, torch.from_numpy(iv.astype(np.float32)), torch.from_numpy(grid.A))
    mask = gs.valid_mask(spec, res.o, res.centre)
    scores = torch.where(mask, res.corr, torch.tensor(-np.inf))
    flat = int(torch.argmax(scores))
    b, rem = divmod(flat, spec.Hf * spec.Wf)
    ky, kx = divmod(rem, spec.Wf)
    assert b == p_true
    t_px = gs.translation_px(spec, res.o[b], int(spec.sy[ky]), int(spec.sx[kx]))
    assert np.linalg.norm(t_px - tau) <= 2.0
    M = gs.pose_matrix(grid.A[b], t_px)
    assert np.abs(iv @ M[:, :2].T + M[:, 2] - ex).max() <= 2.0
