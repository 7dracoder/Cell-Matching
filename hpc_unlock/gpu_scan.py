"""GPU dense pose scan: grid, FFT kernel, peak selection and the Stage (Req 5, 3.5, 3.6, 13.2).

This module holds the scan grid, the batched FFT kernel, local-maximum peak
selection and the ``gpu_scan`` Stage ``compute`` (loads only ``prep.pkl``).

Conventions
-----------
* Points are ``(x, y)`` in px (as ``cellmatch.region_centers``). A pose is the
  2x3 matrix ``M = [A | tau]`` acting as ``A @ p + tau`` (``registration.transform``).
* Grid maps are indexed ``[row = y, col = x]``. Grid coordinates are px / ``cell``.
* The ex-vivo grid ``E`` has shape ``(Hg, Wg) = (ceil(H/c) + 1, ceil(W/c) + 1)``,
  so the bilinear neighbours of every centroid inside ``[0, W) x [0, H)`` fit.
* For a pose with linear part ``A`` the template offset is, per axis,
  ``o = 2 - floor(min_i (A p_i) / c)`` (int64, ``(ox, oy)``), and the in-vivo
  points are splatted into the template ``T`` at grid coordinates
  ``g_i = A p_i / c + o`` (so every ``g_i >= 2``). The design text writes the
  same offset as ``floor(min) - 2`` with the opposite sign; the sign here is the
  one that makes the translation formula below hold literally.
* Placing ``T`` at integer grid shift ``t = (tx, ty)`` puts point ``i`` at
  ``E`` position ``g_i + t``, i.e. the in-vivo field gets the px translation::

      tau = c * (t + o)            # M = [A | c * (t + o)]

* ``corr[b, ky, kx]`` holds the score of shift ``t = (sx[kx], sy[ky])`` where
  ``sx`` / ``sy`` are the signed shift vectors of :class:`ExSpectrum`. Array
  index ``k`` maps to ``t = k`` for ``k < Hg + R`` and to ``t = k - Hf`` otherwise,
  so negative shifts (template origin left of / above the canvas) never wrap.

Score
-----
With ``s = sigma * sqrt(2) / c`` (the two per-map Gaussians of ``sigma / c``
combined into one) and the sampled, peak-normalised kernel
``g(d) = exp(-|d|^2 / (2 s^2))`` truncated to ``|d|_inf <= R = ceil(4 s)``::

    corr[t] = sum_v sum_u E[v] * T[u] * g(v - u - t)
            ~ sum_i sum_j exp(-|A p_i + tau - q_j|^2 / (4 sigma^2))

The kernel is applied once per region to ``E`` in the Fourier domain (FFT of the
sampled kernel, not the analytic transfer function, so the result equals the
direct discrete correlation exactly up to float32 rounding). Padded FFT sizes are
``Hf = fast_size(Hg + Ht + 2R)`` and ``Wf = fast_size(Wg + Wt + 2R)``. The FFT
path is float32 only.

Device-agnostic torch (CUDA on Burst, CPU in tests and smoke). Imports none of
``registration``, ``sklearn``, ``validate``, ``assemble``.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F_nn

from .checkpoint import log_line

ANGLE_RANGE = (-35.0, 35.0)
SCALE_RANGE = (0.85, 1.13)
STRETCH_FACTORS = (0.92, 0.96, 1.04, 1.08)
STRETCH_DIRECTIONS = (0.0, 45.0, 90.0, 135.0)
MARGIN = 2                         # empty grid cells before the first template point
DTYPE = torch.float32

Stretch = tuple  # (k: float, phi: float | None)


# ------------------------------------------------------------------- grid
def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    if not (isinstance(step, (int, float)) and math.isfinite(step) and step > 0):
        raise ValueError(f"step={step!r} must be a positive number")
    n = math.ceil((hi - lo) / step - 1e-9) + 1     # 0.28 / 0.01 = 28.000000000000004
    return np.linspace(lo, hi, n)                  # both endpoints exact


def angle_grid(step: float = 0.5) -> np.ndarray:
    """``linspace(-35, 35, ceil(70 / step) + 1)`` degrees (141 at 0.5)."""
    return _grid(*ANGLE_RANGE, step)


def scale_grid(step: float = 0.01) -> np.ndarray:
    """``linspace(0.85, 1.13, ceil(0.28 / step) + 1)`` (29 at 0.01)."""
    return _grid(*SCALE_RANGE, step)


def stretch_hypotheses() -> list[Stretch]:
    """17 ``(k, phi)``: ``(1.0, None)`` first, then k x phi in degrees."""
    return [(1.0, None)] + [(k, phi) for k in STRETCH_FACTORS for phi in STRETCH_DIRECTIONS]


def stretch_matrix(k: float, phi: Optional[float]) -> np.ndarray:
    """``S(k, phi) = I + (k - 1) u u^T``, ``u = (cos phi, sin phi)`` (as ``wide_soft.stretch``).

    ``phi=None`` (k = 1.00, direction not applicable) gives the identity.
    """
    if phi is None:
        if k != 1.0:
            raise ValueError(f"stretch k={k} needs a direction")
        return np.eye(2)
    u = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    return np.eye(2) + (k - 1.0) * np.outer(u, u)


def rotation(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


@dataclass(frozen=True)
class ScanGrid:
    """All grid poses in C order over ``(angle, scale, stretch)``.

    Pose ``p`` has ``angle_idx = p // (n_s * n_k)``, ``scale_idx = (p // n_k) % n_s``,
    ``stretch_idx = p % n_k`` and linear part ``A[p] = s R(theta) S(k, phi)``.
    """
    angles: np.ndarray             # (n_a,) degrees
    scales: np.ndarray             # (n_s,)
    stretches: tuple               # n_k x (k, phi | None)
    A: np.ndarray                  # (P, 2, 2) float64
    angle: np.ndarray              # (P,) degrees
    scale: np.ndarray              # (P,)
    stretch_idx: np.ndarray        # (P,) int64 into ``stretches``

    def __len__(self) -> int:
        return len(self.A)

    def stretch_of(self, p: int) -> Stretch:
        return self.stretches[int(self.stretch_idx[p])]


def linear_parts(angles: Sequence[float], scales: Sequence[float],
                 stretches: Sequence[Stretch]) -> ScanGrid:
    """Linear parts ``s * R(theta) * S(k, phi)`` for every grid combination."""
    angles = np.asarray(angles, float)
    scales = np.asarray(scales, float)
    stretches = tuple((float(k), None if phi is None else float(phi)) for k, phi in stretches)
    R = np.stack([rotation(a) for a in angles])                    # (n_a, 2, 2)
    S = np.stack([stretch_matrix(k, phi) for k, phi in stretches])  # (n_k, 2, 2)
    RS = np.einsum("aij,kjl->akil", R, S)                          # (n_a, n_k, 2, 2)
    A = scales[None, :, None, None, None] * RS[:, None]            # (n_a, n_s, n_k, 2, 2)
    n_a, n_s, n_k = len(angles), len(scales), len(stretches)
    ia, is_, ik = np.meshgrid(np.arange(n_a), np.arange(n_s), np.arange(n_k), indexing="ij")
    return ScanGrid(angles=angles, scales=scales, stretches=stretches,
                    A=A.reshape(-1, 2, 2), angle=angles[ia.ravel()],
                    scale=scales[is_.ravel()], stretch_idx=ik.ravel().astype(np.int64))


def full_grid(angle_step: float = 0.5, scale_step: float = 0.01) -> ScanGrid:
    """The production grid: 141 x 29 x 17 = 69,513 poses at the default steps."""
    return linear_parts(angle_grid(angle_step), scale_grid(scale_step), stretch_hypotheses())


# ------------------------------------------------------------- sizes
def fast_size(n: int) -> int:
    """Smallest ``m >= n`` of the form ``2^a 3^b 5^c`` (cuFFT / pocketfft fast sizes)."""
    m = max(1, int(n))
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def ex_grid_shape(shape: tuple, cell: float) -> tuple[int, int]:
    """``(Hg, Wg) = (ceil(H/c) + 1, ceil(W/c) + 1)`` for canvas ``shape = (H, W)``."""
    H, W = shape[:2]
    return math.ceil(H / cell) + 1, math.ceil(W / cell) + 1


def kernel_radius(sigma: float, cell: float) -> int:
    """``R = ceil(4 s)`` with ``s = sigma * sqrt(2) / cell``."""
    return int(math.ceil(4.0 * sigma * math.sqrt(2.0) / cell))


def template_shape(iv_c: np.ndarray, A: np.ndarray, cell: float, chunk: int = 2048) -> tuple[int, int]:
    """Common template ``(Ht, Wt)`` for every linear part in ``A`` (P, 2, 2).

    With ``ext`` the largest per-pose extent ``max_i - min_i`` of ``A p_i / c``,
    ``Ht = floor(ext_y) + 6`` and ``Wt = floor(ext_x) + 6``: the points sit at
    ``g >= 2`` and their bilinear neighbours at ``<= floor(ext) + 5`` (one cell of
    slack covers float32 rounding on the device).
    """
    iv = np.asarray(iv_c, np.float64).reshape(-1, 2)
    A = np.asarray(A, np.float64).reshape(-1, 2, 2)
    if len(iv) == 0 or len(A) == 0:
        return 6, 6
    ext = np.zeros(2)
    for s in range(0, len(A), chunk):
        P = np.einsum("pij,nj->pni", A[s:s + chunk], iv) / cell    # (p, N, 2) xy
        ext = np.maximum(ext, (P.max(1) - P.min(1)).max(0))
    return int(math.floor(ext[1])) + 6, int(math.floor(ext[0])) + 6


# ----------------------------------------------------------- splats
def splat(points: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bilinear splat of grid-coordinate points into zero maps.

    ``points``: ``(N, 2)`` or ``(B, N, 2)`` float ``(x, y)`` in grid units.
    Returns ``(h, w)`` or ``(B, h, w)`` float32 maps; point ``(x, y)`` adds
    ``(1 - fx)(1 - fy)`` at ``[y0, x0]``, ``fx (1 - fy)`` at ``[y0, x0 + 1]``,
    ``(1 - fx) fy`` at ``[y0 + 1, x0]`` and ``fx fy`` at ``[y0 + 1, x0 + 1]``.
    Corners outside the map are dropped. Accumulates with ``index_put_``.
    """
    single = points.dim() == 2
    pts = points[None] if single else points
    B, N = pts.shape[:2]
    h, w = int(shape[0]), int(shape[1])
    out = torch.zeros(B * h * w, dtype=DTYPE, device=pts.device)
    if N > 0:
        pts = pts.to(DTYPE)
        x0 = torch.floor(pts[..., 0])
        y0 = torch.floor(pts[..., 1])
        fx = pts[..., 0] - x0
        fy = pts[..., 1] - y0
        x0 = x0.long()
        y0 = y0.long()
        base = (torch.arange(B, device=pts.device) * (h * w))[:, None]
        for dy, dx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                           (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            yy, xx = y0 + dy, x0 + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            idx = (base + yy * w + xx)[ok]
            out.index_put_((idx,), wt[ok], accumulate=True)
    out = out.view(B, h, w)
    return out[0] if single else out


def gauss_kernel_map(s: float, R: int, Hf: int, Wf: int, device) -> torch.Tensor:
    """Sampled kernel ``exp(-|d|^2 / (2 s^2))``, ``|d|_inf <= R``, centred circularly at 0."""
    d = torch.arange(-R, R + 1, dtype=torch.float64)
    g1 = torch.exp(-d ** 2 / (2.0 * s * s))
    K = torch.zeros(Hf, Wf, dtype=torch.float64)
    iy = torch.remainder(torch.arange(-R, R + 1), Hf)
    ix = torch.remainder(torch.arange(-R, R + 1), Wf)
    K[iy[:, None], ix[None, :]] = g1[:, None] * g1[None, :]        # separable
    return K.to(device=device, dtype=DTYPE)


def signed_shifts(n_fft: int, n_ex: int, R: int, device=None) -> torch.Tensor:
    """Grid shift per correlation index: ``k`` if ``k < n_ex + R`` else ``k - n_fft`` (int64)."""
    k = torch.arange(n_fft, device=device)
    return torch.where(k < n_ex + R, k, k - n_fft)


# ------------------------------------------------------- ex spectrum
@dataclass
class ExSpectrum:
    """Per-region ex-vivo side of the correlation (built once, reused for every batch)."""
    F: torch.Tensor                # (Hf, Wf // 2 + 1) complex64: rfft2(E) * rfft2(kernel)
    H: int                         # canvas height (px)
    W: int                         # canvas width (px)
    cell: float
    sigma: float
    Hg: int
    Wg: int
    Ht: int
    Wt: int
    R: int
    Hf: int
    Wf: int
    sy: torch.Tensor               # (Hf,) int64 signed ty per row index
    sx: torch.Tensor               # (Wf,) int64 signed tx per column index

    @property
    def device(self):
        return self.F.device


def ex_spectrum(ex_c: np.ndarray, shape: tuple, sigma: float, cell: float,
                tshape: tuple[int, int], device="cpu") -> ExSpectrum:
    """Splat ``ex_c / cell`` into ``E`` and fold in the ``sigma * sqrt(2)`` Gaussian.

    ``shape`` is the ex-vivo canvas ``(H, W)`` in px; ``tshape`` the common
    template ``(Ht, Wt)`` from :func:`template_shape`.
    """
    H, W = int(shape[0]), int(shape[1])
    Hg, Wg = ex_grid_shape((H, W), cell)
    Ht, Wt = int(tshape[0]), int(tshape[1])
    s = sigma * math.sqrt(2.0) / cell
    R = kernel_radius(sigma, cell)
    Hf, Wf = fast_size(Hg + Ht + 2 * R), fast_size(Wg + Wt + 2 * R)
    q = torch.as_tensor(np.asarray(ex_c, np.float32).reshape(-1, 2), device=device) / cell
    E = splat(q, (Hg, Wg))
    F = torch.fft.rfft2(E, s=(Hf, Wf)) * torch.fft.rfft2(gauss_kernel_map(s, R, Hf, Wf, device))
    return ExSpectrum(F=F, H=H, W=W, cell=float(cell), sigma=float(sigma), Hg=Hg, Wg=Wg,
                      Ht=Ht, Wt=Wt, R=R, Hf=Hf, Wf=Wf,
                      sy=signed_shifts(Hf, Hg, R, device), sx=signed_shifts(Wf, Wg, R, device))


# --------------------------------------------------------- correlation
class ScanBatch(NamedTuple):
    corr: torch.Tensor             # (B, Hf, Wf) float32, see module docstring
    o: torch.Tensor                # (B, 2) int64 (ox, oy); tau = cell * (t + o)
    centre: torch.Tensor           # (B, 2) float32 mean_i(A p_i) in px (NaN if no points)


def correlate_batch(spec: ExSpectrum, iv_c: torch.Tensor, A_batch: torch.Tensor) -> ScanBatch:
    """Correlation maps of ``B`` linear parts against the region's ``E``.

    ``iv_c``: ``(N, 2)`` px ``(x, y)``; ``A_batch``: ``(B, 2, 2)``. Both are moved to
    ``spec.device`` as float32. ``corr[b, ky, kx]`` is the score of pose
    ``M = [A_b | cell * (t + o_b)]`` with ``t = (spec.sx[kx], spec.sy[ky])``.
    """
    dev, c = spec.device, spec.cell
    A = torch.as_tensor(A_batch, dtype=DTYPE, device=dev).reshape(-1, 2, 2)
    p = torch.as_tensor(iv_c, dtype=DTYPE, device=dev).reshape(-1, 2)
    B, N = A.shape[0], p.shape[0]
    if N == 0:
        corr = torch.zeros(B, spec.Hf, spec.Wf, dtype=DTYPE, device=dev)
        return ScanBatch(corr, torch.zeros(B, 2, dtype=torch.long, device=dev),
                         torch.full((B, 2), float("nan"), dtype=DTYPE, device=dev))
    Ap = torch.einsum("bij,nj->bni", A, p)                         # (B, N, 2) px
    centre = Ap.mean(1)
    G = Ap / c
    o = MARGIN - torch.floor(G.min(1).values).long()               # (B, 2)
    g = G + o[:, None, :].to(DTYPE)                                # template coords >= 2
    top = torch.floor(g).long().amax(dim=(0, 1)) + 1               # (x, y) highest corner index
    if int(top[0]) >= spec.Wt or int(top[1]) >= spec.Ht:
        raise ValueError(f"template {spec.Ht}x{spec.Wt} too small for this batch "
                         f"(needs {int(top[1]) + 1}x{int(top[0]) + 1}); "
                         "build tshape with template_shape over the same iv_c and grid")
    T = splat(g, (spec.Ht, spec.Wt))                               # (B, Ht, Wt)
    FT = torch.fft.rfft2(T, s=(spec.Hf, spec.Wf))
    corr = torch.fft.irfft2(spec.F[None] * FT.conj(), s=(spec.Hf, spec.Wf))
    return ScanBatch(corr, o, centre)


def valid_mask(spec: ExSpectrum, o: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    """``(B, Hf, Wf)`` bool: field centre ``centre + cell * (t + o)`` inside ``[0, W) x [0, H)``."""
    c = spec.cell
    cx = centre[:, 0:1] + c * (spec.sx[None, :] + o[:, 0:1]).to(DTYPE)    # (B, Wf)
    cy = centre[:, 1:2] + c * (spec.sy[None, :] + o[:, 1:2]).to(DTYPE)    # (B, Hf)
    vx = (cx >= 0) & (cx < spec.W)
    vy = (cy >= 0) & (cy < spec.H)
    return vy[:, :, None] & vx[:, None, :]


def translation_px(spec: ExSpectrum, o, ty: int, tx: int) -> np.ndarray:
    """``tau = cell * (t + o)`` as float64 ``(x, y)`` for one pose's ``o`` and shift."""
    o = np.asarray(o.cpu() if torch.is_tensor(o) else o, np.float64).reshape(2)
    return spec.cell * (np.array([tx, ty], np.float64) + o)


def pose_matrix(A: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """2x3 float64 pose ``[A | tau]``."""
    return np.c_[np.asarray(A, np.float64).reshape(2, 2), np.asarray(tau, np.float64).reshape(2)]


# ===================================================================== Stage
NEEDS_GPU = True                   # stage.py contract: GPU Stage, checks visibility itself

PEAKS_PER_POSE = 4                 # top-m local maxima per pose copied to the host
SEP_ANGLE = 3.0                    # deg: skip if |d angle| <= 3 ...
SEP_LANDING = 20.0                 # px:  ... and |d landing| <= 20 of a kept candidate
P0 = np.array([300.0, 300.0])      # landing reference point (window_lab.pose / soft.landing)
BATCH_MIN, BATCH_MAX = 16, 1024
CPU_BATCH = 64
MEM_FRACTION = 0.6
BYTES_PER_CELL = 12                # ~3 real float32 buffers of Hf x Wf per pose

SMOKE_SCALES = (0.85, 1.0, 1.13)
SMOKE_STRETCHES = ((1.0, None),)


def smoke_grid() -> ScanGrid:
    """5 angles (``linspace(-35, 35, 5)``, includes 0) x 3 scales (incl. 1.0) x k = 1 (Req 13.2)."""
    return linear_parts(np.linspace(*ANGLE_RANGE, 5), SMOKE_SCALES, SMOKE_STRETCHES)


# --------------------------------------------------------------- batching
def batch_from_free(free_bytes: float, Hf: int, Wf: int) -> int:
    """``floor(0.6 * free / (12 * Hf * Wf))`` clamped to ``[16, 1024]``."""
    b = int(MEM_FRACTION * float(free_bytes) // (BYTES_PER_CELL * int(Hf) * int(Wf)))
    return int(min(max(b, BATCH_MIN), BATCH_MAX))


def batch_size(spec: ExSpectrum) -> int:
    """Poses per batch: from free device memory on CUDA, ``CPU_BATCH`` on CPU."""
    dev = torch.device(spec.device)
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        return batch_from_free(free, spec.Hf, spec.Wf)
    return CPU_BATCH


# ------------------------------------------------------------------ peaks
class BatchPeaks(NamedTuple):
    score: np.ndarray              # (n,) float64
    b: np.ndarray                  # (n,) int64 pose index within the batch
    ty: np.ndarray                 # (n,) int64 signed grid shift (row)
    tx: np.ndarray                 # (n,) int64 signed grid shift (column)


def batch_peaks(spec: ExSpectrum, res: ScanBatch, m: int = PEAKS_PER_POSE) -> BatchPeaks:
    """Top-``m`` local maxima per pose over the valid translations.

    A local maximum is a valid shift where ``corr == max_pool2d(corr, 3, 1, 1)``
    (invalid shifts set to -inf first) and ``corr > 0``. Only the ``B * m``
    top-k entries are copied to the host.
    """
    neg = torch.tensor(-math.inf, dtype=DTYPE, device=res.corr.device)
    s = torch.where(valid_mask(spec, res.o, res.centre), res.corr, neg)
    pooled = F_nn.max_pool2d(s[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    s = torch.where((s == pooled) & (s > 0), s, neg)
    B = s.shape[0]
    flat = s.reshape(B, -1)
    val, idx = torch.topk(flat, min(int(m), flat.shape[1]), dim=1)
    val = val.cpu().numpy().astype(np.float64)
    idx = idx.cpu().numpy()
    bb, jj = np.nonzero(np.isfinite(val))
    k = idx[bb, jj]
    sy = spec.sy.cpu().numpy()
    sx = spec.sx.cpu().numpy()
    return BatchPeaks(val[bb, jj], bb.astype(np.int64), sy[k // spec.Wf], sx[k % spec.Wf])


class RegionPeaks(NamedTuple):
    score: np.ndarray              # (n,) float64
    pose: np.ndarray               # (n,) int64 grid pose index
    tau: np.ndarray                # (n, 2) float64 translation (x, y) px
    landing: np.ndarray            # (n, 2) float64 A (P0 - offset) + tau


def region_spectrum(inp: dict, grid: ScanGrid, sigma: float, cell: float, device) -> ExSpectrum:
    """Ex-vivo spectrum with the common template shape over the whole grid."""
    tshape = template_shape(inp["iv_c"], grid.A, cell)
    return ex_spectrum(inp["ex_c"], inp["ex_shape"], sigma, cell, tshape, device)


def scan_peaks(spec: ExSpectrum, inp: dict, grid: ScanGrid, batch: int,
               m: int = PEAKS_PER_POSE) -> RegionPeaks:
    """All per-pose local maxima of one region, batch by batch."""
    dev = spec.device
    iv_t = torch.as_tensor(np.asarray(inp["iv_c"], np.float32).reshape(-1, 2), device=dev)
    A_t = torch.as_tensor(grid.A, dtype=DTYPE, device=dev)
    lever = grid.A @ (P0 - np.asarray(inp["offset"], np.float64).reshape(2))   # (P, 2)
    batch = max(1, int(batch))
    scores, poses, taus = [], [], []
    with torch.inference_mode():
        for s0 in range(0, len(grid), batch):
            res = correlate_batch(spec, iv_t, A_t[s0:s0 + batch])
            pk = batch_peaks(spec, res, m)
            o = res.o.cpu().numpy().astype(np.float64)
            del res
            scores.append(pk.score)
            poses.append(s0 + pk.b)
            taus.append(spec.cell * (np.stack([pk.tx, pk.ty], 1).astype(np.float64) + o[pk.b]))
    score = np.concatenate(scores) if scores else np.zeros(0)
    pose = np.concatenate(poses).astype(np.int64) if poses else np.zeros(0, np.int64)
    tau = np.concatenate(taus).reshape(-1, 2) if taus else np.zeros((0, 2))
    return RegionPeaks(score, pose, tau, lever[pose] + tau)


def select_peaks(score: np.ndarray, angle: np.ndarray, landing: np.ndarray, k: int,
                 sep_angle: float = SEP_ANGLE, sep_landing: float = SEP_LANDING,
                 chunk: int = 4096) -> np.ndarray:
    """Greedy separation: indices of kept peaks, in kept (descending score) order.

    Peaks are visited by descending score (ties: input order). A peak is
    skipped if some kept peak has ``|d angle| <= sep_angle`` and
    ``|d landing| <= sep_landing``. Stops at ``k`` kept peaks.
    """
    score = np.asarray(score, np.float64).ravel()
    angle = np.asarray(angle, np.float64).ravel()
    landing = np.asarray(landing, np.float64).reshape(-1, 2)
    k = int(k)
    if k <= 0 or len(score) == 0:
        return np.zeros(0, np.int64)
    order = np.argsort(-score, kind="stable")
    ka, kl = np.empty(k), np.empty((k, 2))
    kept: list[int] = []
    for c0 in range(0, len(order), chunk):
        idx = order[c0:c0 + chunk]
        n = len(kept)
        if n:   # kept only grows: a conflict now stays a conflict
            close = ((np.abs(angle[idx, None] - ka[None, :n]) <= sep_angle)
                     & (np.hypot(landing[idx, None, 0] - kl[None, :n, 0],
                                 landing[idx, None, 1] - kl[None, :n, 1]) <= sep_landing))
            idx = idx[~close.any(1)]
        for i in idx:
            n = len(kept)
            if n and np.any((np.abs(ka[:n] - angle[i]) <= sep_angle)
                            & (np.hypot(kl[:n, 0] - landing[i, 0],
                                        kl[:n, 1] - landing[i, 1]) <= sep_landing)):
                continue
            ka[n], kl[n] = angle[i], landing[i]
            kept.append(int(i))
            if len(kept) == k:
                return np.asarray(kept, np.int64)
    return np.asarray(kept, np.int64)


def make_candidates(grid: ScanGrid, pk: RegionPeaks, kept: np.ndarray) -> list[dict]:
    """``ScanCandidate`` plain dicts (builtins + float64 arrays) for the kept peaks."""
    out = []
    for i in kept:
        p = int(pk.pose[i])
        k, phi = grid.stretch_of(p)
        tau, land = pk.tau[i], pk.landing[i]
        out.append({"M": pose_matrix(grid.A[p], tau), "score": float(pk.score[i]),
                    "angle": float(grid.angle[p]), "scale": float(grid.scale[p]),
                    "stretch_k": float(k), "stretch_dir": None if phi is None else float(phi),
                    "translation": (float(tau[0]), float(tau[1])),
                    "landing": (float(land[0]), float(land[1]))})
    return out


# ------------------------------------------------------------------ Stage
def torch_info() -> dict:
    avail = bool(torch.cuda.is_available())
    dev = "cpu"
    if avail:
        try:
            dev = torch.cuda.get_device_name(0)
        except Exception as e:  # noqa: BLE001 - only informational
            dev = f"unknown ({type(e).__name__})"
    return {"version": str(torch.__version__), "cuda": torch.version.cuda,
            "available": avail, "device": dev}


def _free_cache(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def scan_region(inp: dict, grid: ScanGrid, sigma: float, cell: float, device,
                name: str) -> tuple[RegionPeaks, int]:
    """Scan one region; on CUDA OOM retry once at half the batch size.

    Returns ``(peaks, batch used)``. A second OOM propagates.
    """
    spec = region_spectrum(inp, grid, sigma, cell, device)
    batch = batch_size(spec)
    cause = None
    try:
        return scan_peaks(spec, inp, grid, batch), batch
    except torch.cuda.OutOfMemoryError as e:
        cause = f"{type(e).__name__}: {e}"
    # retry outside the except block so the failed batch's tensors are released
    half = max(1, batch // 2)
    log_line("SCAN_OOM_RETRY", f"sid={name} batch={batch} -> {half}: {cause}")
    _free_cache(device)
    return scan_peaks(spec, inp, grid, half), half


def grid_settings(grid: ScanGrid, cfg, smoke: bool) -> dict:
    return {"angles": [float(a) for a in grid.angles], "scales": [float(s) for s in grid.scales],
            "stretches": [(float(k), None if phi is None else float(phi)) for k, phi in grid.stretches],
            "n_poses": len(grid), "cell_px": float(cfg.scan_cell_px), "sigma": float(cfg.sigma),
            "k": int(cfg.scan_k), "peaks_per_pose": PEAKS_PER_POSE, "sep_angle": SEP_ANGLE,
            "sep_landing": SEP_LANDING, "p0": [float(v) for v in P0], "smoke": bool(smoke)}


def compute(cfg, ctx) -> dict:
    """Stage entry: scan every unique region of ``prep.pkl`` and keep up to K peaks.

    Returns ``{"by_sid": {sid: {"cands": [...], "kept": int}}, "grid": {...},
    "torch": {...}}``. Duplicate regions (same ``dup_key``) are scanned once and
    the result is copied under every ID.
    """
    smoke = bool(getattr(ctx, "smoke", False))
    info = torch_info()
    log_line("TORCH_INFO", f"version={info['version']} cuda={info['cuda']} "
                           f"available={info['available']} device={info['device']}")
    if not smoke and not info["available"]:
        log_line("NO_GPU_VISIBLE", "gpu_scan: torch.cuda.is_available() is False; "
                                   "no region processed")
        raise SystemExit(1)
    device = torch.device("cpu") if smoke else torch.device("cuda")
    grid = smoke_grid() if smoke else full_grid(cfg.scan_angle_step, cfg.scan_scale_step)
    sigma, cell, K = float(cfg.sigma), float(cfg.scan_cell_px), int(cfg.scan_k)

    inputs = ctx.load("prep")["scan_inputs"]
    log_line("SCAN_START", f"unique_regions={len(inputs)} poses={len(grid)} device={device} "
                           f"sigma={sigma} cell={cell} k={K} smoke={smoke}")
    t0 = time.monotonic()
    by_sid: dict[str, dict] = {}
    for inp in inputs.values():
        sids = [str(s) for s in inp["sids"]]
        name = ",".join(sids)
        t1 = time.monotonic()
        try:
            pk, batch = scan_region(inp, grid, sigma, cell, device, name)
            kept = select_peaks(pk.score, grid.angle[pk.pose], pk.landing, K)
            cands = make_candidates(grid, pk, kept)
        except Exception as e:  # noqa: BLE001 - incl. torch.cuda.OutOfMemoryError
            log_line("REGION_FAILED", f"{name}: {type(e).__name__}: {e}")
            raise SystemExit(1)
        entry = {"cands": cands, "kept": len(cands)}
        for j, sid in enumerate(sids):
            by_sid[sid] = entry if j == 0 else copy.deepcopy(entry)
        log_line("SCAN", f"sid={name} poses={len(grid)} maxima={len(pk.score)} kept={len(cands)} "
                         f"batch={batch} elapsed={time.monotonic() - t1:.1f}s "
                         f"total={time.monotonic() - t0:.1f}s")
        del pk
        _free_cache(device)
    return {"by_sid": by_sid, "grid": grid_settings(grid, cfg, smoke), "torch": info}
