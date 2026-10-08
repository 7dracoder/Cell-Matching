"""Soft_Score, Soft_Margin, landing and pose decomposition (Req 4.4).

- ``soft`` is ``research/wide_soft.soft``: the sum over mutual-nearest centroid
  pairs of exp(-d^2 / 2 sigma^2).
- ``pose`` / ``landing`` follow ``research/window_lab.pose``: angle in degrees
  from atan2(M[1,0], M[0,0]) and landing ``A (P0 - offset) + t`` with
  ``P0 = (300, 300)``. Re-implemented here so that ``window_lab`` (which loads
  pickles at import time) is not imported.
- ``soft_margins`` is ``research/soft_gate_cv.choose``'s margin generalised to
  every candidate: soft minus the best soft among clearly different
  candidates (|dtheta| > 3 deg or |dlanding| > 60 px). As in ``choose``, that
  best alternative is taken as 0 when none exist, so the margin is then the
  candidate's own soft.

This module does not import ``objective_probe`` or ``soft_gate_cv``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from hpc_unlock import paths  # noqa: F401  (puts ROOT and research/ on sys.path)
from wide_soft import soft as _wide_soft  # noqa: E402

P0 = np.array([300.0, 300.0])
ANGLE_TOL = 3.0     # deg, "clearly different" angle threshold
LANDING_TOL = 60.0  # px, "clearly different" landing threshold


def soft(iv_c: np.ndarray, ex_c: np.ndarray, M: np.ndarray, sigma: float = 2.5) -> float:
    """Soft_Score of pose ``M`` (2x3) mapping in-vivo to ex-vivo centroids."""
    return _wide_soft(iv_c, ex_c, np.asarray(M, float), sigma)


def angle(M: np.ndarray) -> float:
    """Rotation angle in degrees, ``atan2(M[1,0], M[0,0])``."""
    M = np.asarray(M, float)
    return float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


def landing(M: np.ndarray, offset) -> np.ndarray:
    """Where the in-vivo reference point P0 lands: ``A (P0 - offset) + t``."""
    M = np.asarray(M, float)
    return M[:, :2] @ (P0 - np.asarray(offset, float)) + M[:, 2]


def pose(M: np.ndarray, offset) -> tuple[float, np.ndarray]:
    """``(angle_deg, landing)`` exactly as ``window_lab.pose(r, M)`` with ``r['offset'] = offset``."""
    return angle(M), landing(M, offset)


def decompose(M: np.ndarray) -> tuple[float, float, float]:
    """``(angle_deg, scale, anisotropy)`` of the linear part A.

    scale = sqrt(|det A|); anisotropy = sigma1 / sigma2 - 1 from the SVD of A
    (0 for a similarity, inf for a singular A).
    """
    A = np.asarray(M, float)[:, :2]
    s = np.linalg.svd(A, compute_uv=False)
    aniso = float(s[0] / s[1] - 1.0) if s[1] > 0 else float("inf")
    return angle(M), float(np.sqrt(abs(np.linalg.det(A)))), aniso


def clearly_different(p: tuple[float, np.ndarray], q: tuple[float, np.ndarray]) -> bool:
    """True if poses differ by more than 3 deg in angle or 60 px in landing."""
    return bool(abs(p[0] - q[0]) > ANGLE_TOL
                or np.linalg.norm(np.asarray(p[1]) - np.asarray(q[1])) > LANDING_TOL)


def soft_margins(Ms: Sequence[np.ndarray], softs: Sequence[float], offset) -> np.ndarray:
    """Soft_Margin for every candidate of one region.

    ``margin[i] = softs[i] - second_i`` with ``second_i`` the best soft among
    candidates clearly different from i, or 0 when there are none
    (``soft_gate_cv.choose`` convention).
    """
    softs = np.asarray(softs, float)
    poses = [pose(M, offset) for M in Ms]
    out = np.zeros(len(poses))
    for i, p in enumerate(poses):
        alt = [softs[j] for j, q in enumerate(poses) if j != i and clearly_different(p, q)]
        out[i] = softs[i] - (max(alt) if alt else 0.0)
    return out


def soft_margin(i: int, Ms: Sequence[np.ndarray], softs: Sequence[float], offset) -> float:
    """Soft_Margin of candidate ``i`` (see ``soft_margins``)."""
    p = pose(Ms[i], offset)
    alt = [softs[j] for j in range(len(Ms))
           if j != i and clearly_different(p, pose(Ms[j], offset))]
    return float(softs[i] - (max(alt) if alt else 0.0))
