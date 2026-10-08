"""Pose_Search Stage (CPU): wide Hough + stretch, GPU scan peaks, window and vote
candidates, ``registration.refine``, dedup by Soft_Score and per-candidate features
(Req 4.1-4.10, 5.7, 5.10).

Per region, four candidate sources are concatenated in this order:

* ``hough``  - the raw peaks of ``research/wide_soft.wide_candidates`` (similarity
  Hough over angles -35..35 step 1 deg and scales 0.85..1.13 step 0.02, for the
  k = 1 hypothesis plus stretch 0.92 / 1.08 at 0/45/90/135 deg), each refined with
  ``registration.refine``, in descending vote order. ``wide_candidates`` drops a
  refined peak that duplicates an earlier (higher-vote) one; here every refined
  peak is kept and the duplicate rule is applied once, globally, by Soft_Score
  (Req 4.3). Its own rule (``|dM| < 1e-3`` or linear part < 0.01 and translation
  < 4 px) is contained in the global one.
* ``gpu``    - every kept GPU_Pose_Scan candidate (``gpu_scan.pkl``), refined (4.6).
* ``window`` / ``vote`` - the record's ``cp_scored`` list (held-out: the exact
  ``cp_pose_train.pkl`` entries behind v10; test: ``cp_pose_lab.all_window`` plus
  ``test_apply.cands``), which are already refined (4.5).

Dedup (4.3): candidates are visited by Soft_Score, descending (stable, so ties
keep the earlier source and then the earlier index); a candidate is dropped if a
kept one has ``max |dA| <= 0.01`` over the 2x2 linear part and ``|dt| <= 4 px``.

Features per kept candidate (4.4): ``refine_score``, ``soft`` (``soft.soft``),
``z`` (``cp_pose_lab.cp_z``), ``refine_margin`` (``margin_lab.margin`` against the
region's vote candidates, the v10 calibration), ``soft_margin`` (``soft.soft_margins``
over the merged list), ``angle`` / ``scale`` / ``anisotropy`` (``soft.decompose``),
``landing`` (``soft.landing``) and ``source``.

Checkpoint (plain builtins and NumPy arrays)::

    {"heldout": {sid: {"cands": [cand, ...], "n": int, "n_by_source": {...},
                       "n_kept_by_source": {...}}},
     "test": {...}, "sigma": float, "params": {...},
     "zero_candidates": {"heldout": [sid], "test": [sid]}, "n_zero": int,
     "diagnostics": {"n_with_gt", "correct_in_raw_gpu", "correct_in_candidates",
                     "correct_first", "sigma", "per_region": {sid: {...}}}}
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from hpc_unlock import checkpoint, paths, prep  # paths: ROOT and research/ on sys.path
from hpc_unlock import soft as S
from hpc_unlock.config import SIGMA_RANGE

SOURCES = ("hough", "gpu", "window", "vote")
SRC_NAME = {"win": "window", "vote": "vote"}       # cp_scored labels -> source names
DUP_LINEAR = 0.01                                  # max |dA| over the linear part
DUP_TRANSLATION = 4.0                              # px, |dt|
CORRECT_ERR = 5.0                                  # px, reg_lab.err < 5 = Correct_Pose
MIN_POINTS = 3                                     # Hough needs >= 3 centroids per side

# Production Hough grid (Req 4.1, 4.2): exactly wide_soft's module globals.
HOUGH_ANGLES = tuple(float(a) for a in np.arange(-35, 35.1, 1.0))
HOUGH_SCALES = tuple(float(s) for s in np.arange(0.85, 1.131, 0.02))
HOUGH_STRETCH = ((1.0, 0.0),) + tuple((k, phi) for k in (0.92, 1.08) for phi in (0, 45, 90, 135))
HOUGH_TOP = 150
HOUGH_PER_POSE = 3

# Smoke grid: k = 1 only, coarse steps (Req 13).
SMOKE_ANGLES = tuple(float(a) for a in np.arange(-35, 35.1, 5.0))
SMOKE_SCALES = tuple(float(s) for s in np.arange(0.85, 1.131, 0.04))
SMOKE_STRETCH = ((1.0, 0.0),)
SMOKE_TOP = 40


@dataclass(frozen=True)
class HoughGrid:
    angles: tuple
    scales: tuple
    stretches: tuple            # ((k, phi_deg), ...); k = 1 first
    top: int                    # raw peaks kept for k = 1 (top // 3 per stretch)

    def as_dict(self) -> dict:
        return {"angles": list(self.angles), "scales": list(self.scales),
                "stretches": [(float(k), float(p)) for k, p in self.stretches],
                "top": int(self.top), "per_pose": HOUGH_PER_POSE}


def hough_grid(smoke: bool = False) -> HoughGrid:
    if smoke:
        return HoughGrid(SMOKE_ANGLES, SMOKE_SCALES, SMOKE_STRETCH, SMOKE_TOP)
    return HoughGrid(HOUGH_ANGLES, HOUGH_SCALES, HOUGH_STRETCH, HOUGH_TOP)


# ----------------------------------------------------------------------------
# Research imports (lazy; window_lab / test_apply parse sys.argv at import)
# ----------------------------------------------------------------------------

_R: SimpleNamespace | None = None


def research() -> SimpleNamespace:
    global _R
    if _R is None:
        with prep._research_import_env():
            import cp_pose_lab
            import hough
            import margin_lab
            import reg_lab
            import registration
            import wide_soft
        _R = SimpleNamespace(cp_pose_lab=cp_pose_lab, hough=hough, margin_lab=margin_lab,
                             reg_lab=reg_lab, registration=registration, wide_soft=wide_soft)
    return _R


def check_wide_soft_grid() -> None:
    """The production grid must equal ``wide_soft``'s (fails loudly if research code drifts)."""
    ws = research().wide_soft
    if not (np.array_equal(ws.ANGLES, HOUGH_ANGLES) and np.array_equal(ws.SCALES, HOUGH_SCALES)
            and [tuple(map(float, s)) for s in ws.STRETCH] == [tuple(map(float, s))
                                                               for s in HOUGH_STRETCH]):
        raise RuntimeError("wide_soft ANGLES / SCALES / STRETCH differ from pose_search's grid")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

def sigma_errors(sigma) -> list[str]:
    lo, hi = SIGMA_RANGE
    ok = (isinstance(sigma, (int, float)) and not isinstance(sigma, bool)
          and np.isfinite(sigma) and lo <= sigma <= hi)
    return [] if ok else [f"sigma={sigma!r} is invalid; expected a number in [{lo}, {hi}] px"]


def validate_config(cfg) -> list[str]:
    """stage.py contract: Stage-specific config errors (Req 4.8)."""
    return sigma_errors(getattr(cfg, "sigma", None))


def params(cfg, smoke: bool) -> dict:
    """Every parameter of the Stage; the same dict is used for all regions (Req 4.7)."""
    return {"sigma": float(cfg.sigma), "hough": hough_grid(smoke).as_dict(),
            "dup_linear": DUP_LINEAR, "dup_translation": DUP_TRANSLATION,
            "soft_margin_angle": S.ANGLE_TOL, "soft_margin_landing": S.LANDING_TOL,
            "correct_err": CORRECT_ERR, "smoke": bool(smoke)}


# ----------------------------------------------------------------------------
# Candidate sources
# ----------------------------------------------------------------------------

def hough_raw(iv_c: np.ndarray, ex_c: np.ndarray, grid: HoughGrid) -> list[tuple[int, np.ndarray]]:
    """Raw ``(votes, M)`` peaks, exactly as the first half of ``wide_soft.wide_candidates``."""
    R = research()
    angles, scales = np.asarray(grid.angles, float), np.asarray(grid.scales, float)
    raw = []
    for k, phi in grid.stretches:
        Sm = R.wide_soft.stretch(k, phi)
        for votes, _a, _s, M in R.hough.hough_candidates(
                iv_c @ Sm.T, ex_c, angles=angles, scales=scales, per_pose=HOUGH_PER_POSE,
                keep=grid.top if k == 1.0 else grid.top // 3):
            raw.append((votes, np.c_[M[:, :2] @ Sm, M[:, 2]]))
    raw.sort(key=lambda c: c[0], reverse=True)
    return raw


def hough_refined(iv_c: np.ndarray, ex_c: np.ndarray, grid: HoughGrid) -> list[tuple[np.ndarray, float]]:
    """Every raw Hough peak refined with ``registration.refine`` (vote order, no dedup)."""
    if len(iv_c) < MIN_POINTS or len(ex_c) < MIN_POINTS:
        return []
    refine = research().registration.refine
    return [refine(iv_c, ex_c, M) for _, M in hough_raw(iv_c, ex_c, grid)]


def gpu_refined(iv_c: np.ndarray, ex_c: np.ndarray, cands: Sequence[Mapping]) -> list[tuple[np.ndarray, float]]:
    """Every kept GPU_Pose_Scan candidate refined with ``registration.refine`` (4.6)."""
    if len(iv_c) < MIN_POINTS or len(ex_c) < MIN_POINTS:
        return [(np.asarray(c["M"], float), 0.0) for c in cands]
    refine = research().registration.refine
    return [refine(iv_c, ex_c, np.asarray(c["M"], float)) for c in cands]


def merge(hough: Sequence, gpu: Sequence, scored: Sequence) -> list[tuple[np.ndarray, float, str]]:
    """``(M, refine_score, source)`` in source order Hough, GPU, window, vote.

    ``scored`` is the record's ``cp_scored`` list ``[(M, score, z, src)]``; its
    window entries come before its vote entries.
    """
    win = [(np.asarray(M, float), float(s), "window") for M, s, _z, src in scored if src == "win"]
    vote = [(np.asarray(M, float), float(s), "vote") for M, s, _z, src in scored if src == "vote"]
    other = {src for *_, src in scored} - set(SRC_NAME)
    if other:
        raise ValueError(f"unknown cp_scored source(s): {sorted(other)}")
    return ([(np.asarray(M, float), float(s), "hough") for M, s in hough]
            + [(np.asarray(M, float), float(s), "gpu") for M, s in gpu] + win + vote)


def dedup(Ms: Sequence[np.ndarray], softs: Sequence[float],
          lin_tol: float = DUP_LINEAR, t_tol: float = DUP_TRANSLATION) -> list[int]:
    """Indices kept by the duplicate rule, in descending Soft_Score order.

    Visits candidates by Soft_Score, descending; ties keep the lower index (the
    earlier source). A candidate is dropped if a kept one has every linear-part
    entry within ``lin_tol`` and translation within ``t_tol`` px (Req 4.3).
    """
    n = len(Ms)
    if n == 0:
        return []
    M = np.asarray(Ms, float).reshape(n, 2, 3)
    A = M[:, :, :2].reshape(n, 4)
    t = M[:, :, 2]
    order = np.argsort(-np.asarray(softs, float), kind="stable")
    kA, kt = np.empty((n, 4)), np.empty((n, 2))
    kept: list[int] = []
    for i in order:
        m = len(kept)
        if m and np.any((np.abs(kA[:m] - A[i]).max(1) <= lin_tol)
                        & (np.linalg.norm(kt[:m] - t[i], axis=1) <= t_tol)):
            continue
        kA[m], kt[m] = A[i], t[i]
        kept.append(int(i))
    return kept


# ----------------------------------------------------------------------------
# One region
# ----------------------------------------------------------------------------

def region_candidates(rec: Mapping, cp_bin: np.ndarray, gpu_cands: Sequence[Mapping],
                      prm: Mapping) -> dict:
    """Merged, deduplicated and featured candidate list of one region."""
    R = research()
    sigma = float(prm["sigma"])
    grid = HoughGrid(tuple(prm["hough"]["angles"]), tuple(prm["hough"]["scales"]),
                     tuple(tuple(s) for s in prm["hough"]["stretches"]), int(prm["hough"]["top"]))
    iv_c = np.asarray(rec["iv_c"]).reshape(-1, 2)
    ex_c = np.asarray(rec["ex_c"]).reshape(-1, 2)
    offset = np.asarray(rec["offset"], float).reshape(2)

    merged = merge(hough_refined(iv_c, ex_c, grid), gpu_refined(iv_c, ex_c, gpu_cands),
                   rec["cp_scored"])
    n_by = {s: sum(c[2] == s for c in merged) for s in SOURCES}
    if len(iv_c) == 0 or len(ex_c) == 0:
        softs = [0.0] * len(merged)
    else:
        softs = [S.soft(iv_c, ex_c, M, sigma) for M, _, _ in merged]
    keep = dedup([c[0] for c in merged], softs)
    Ms = [merged[i][0] for i in keep]
    kept_soft = [float(softs[i]) for i in keep]
    smargin = S.soft_margins(Ms, kept_soft, offset) if Ms else np.zeros(0)
    vote = rec["vote_cands"]
    rmr = {"offset": offset}
    cands = []
    for j, i in enumerate(keep):
        M, score, src = merged[i]
        ang, scale, aniso = S.decompose(M)
        cands.append({
            "M": np.asarray(M, float),
            "refine_score": float(score),
            "soft": kept_soft[j],
            "z": float(R.cp_pose_lab.cp_z(cp_bin, iv_c, M)),
            "refine_margin": float(R.margin_lab.margin(vote, M, score, rmr)),
            "soft_margin": float(smargin[j]),
            "angle": float(ang), "scale": float(scale), "anisotropy": float(aniso),
            "landing": S.landing(M, offset),
            "source": src,
        })
    return {"cands": cands, "n": len(cands), "n_by_source": n_by,
            "n_kept_by_source": {s: sum(c["source"] == s for c in cands) for s in SOURCES}}


def correctness(rec: Mapping, out: Mapping, gpu_cands: Sequence[Mapping]) -> dict | None:
    """Correct_Pose flags for a held-out region with a GT transform, else None (Req 4.10, 5.10)."""
    if rec.get("gt_M") is None or rec.get("gt_iv_c") is None:
        return None
    err = research().reg_lab.err
    raw = [err(rec, np.asarray(c["M"], float)) for c in gpu_cands]
    errs = [err(rec, c["M"]) for c in out["cands"]]
    ok = [e < CORRECT_ERR for e in errs]
    return {"raw_gpu": bool(any(e < CORRECT_ERR for e in raw)),
            "candidates": bool(any(ok)),
            "first": bool(ok[0]) if ok else False,
            "first_correct_rank": next((k for k, v in enumerate(ok) if v), None),
            "best_err": float(min(errs)) if errs else float("inf"),
            "best_raw_gpu_err": float(min(raw)) if raw else float("inf")}


# ----------------------------------------------------------------------------
# Pool workers (fork: the parent sets _STATE and imports research modules first)
# ----------------------------------------------------------------------------

_STATE: dict = {}


def _job(key: tuple[str, str]) -> tuple[str, str, dict, dict | None, float]:
    split, sid = key
    t0 = time.monotonic()
    rec = _STATE["prep"][split][sid]
    gpu = _STATE["gpu"][sid]["cands"]
    out = region_candidates(rec, _STATE["bins"][sid], gpu, _STATE["params"])
    diag = correctness(rec, out, gpu) if split == "heldout" else None
    return split, sid, out, diag, time.monotonic() - t0


def diagnostics(per_region: Mapping[str, dict | None], sigma: float) -> dict:
    """Run_Report counts over the held-out regions with a GT transform."""
    gt = {s: d for s, d in per_region.items() if d is not None}
    return {"n_with_gt": len(gt),
            "correct_in_raw_gpu": sum(d["raw_gpu"] for d in gt.values()),
            "correct_in_candidates": sum(d["candidates"] for d in gt.values()),
            "correct_first": sum(d["first"] for d in gt.values()),
            "sigma": float(sigma), "per_region": dict(gt)}


def compute(cfg, ctx) -> dict:
    """Stage entry: ``ctx`` has ``run_dir``, ``smoke``, ``workers``, ``load``."""
    errors = sigma_errors(getattr(cfg, "sigma", None))
    if errors:                                         # Req 4.8: before any region
        for e in errors:
            checkpoint.log_line("SIGMA_INVALID", f"pose_search: {e}")
        raise ValueError(f"pose_search: {errors[0]}")
    smoke = bool(getattr(ctx, "smoke", False))
    workers = max(1, int(getattr(ctx, "workers", 1) or 1))
    prm = params(cfg, smoke)
    t0 = time.monotonic()

    research()                                         # import before fork
    if not smoke:
        check_wide_soft_grid()
    P = ctx.load("prep")
    gpu = ctx.load("gpu_scan")["by_sid"]
    bins = prep.load_cp_bins(P, ctx.run_dir)
    keys = [(split, sid) for split in ("heldout", "test") for sid in P[split]]
    missing = [sid for _, sid in keys if sid not in gpu]
    if missing:
        raise KeyError(f"gpu_scan has no entry for {len(missing)} region(s): {missing[:5]}")
    checkpoint.log_line("POSE_SEARCH_START",
                        f"regions={len(keys)} workers={workers} sigma={prm['sigma']} "
                        f"angles={len(prm['hough']['angles'])} scales={len(prm['hough']['scales'])} "
                        f"stretches={len(prm['hough']['stretches'])} smoke={smoke}")

    _STATE.update(prep=P, gpu=gpu, bins=bins, params=prm)
    results: dict = {}
    try:
        if workers == 1 or len(keys) <= 1:
            it = map(_job, keys)
            for res in it:
                results[res[:2]] = res
                _log_region(res, len(results), len(keys), t0)
        else:
            with mp.get_context("fork").Pool(workers, initializer=prep._init_worker) as pool:
                for res in pool.imap_unordered(_job, keys, chunksize=1):
                    results[res[:2]] = res
                    _log_region(res, len(results), len(keys), t0)
    finally:
        _STATE.clear()

    out = {"heldout": {}, "test": {}}
    zero = {"heldout": [], "test": []}
    per_region = {}
    for split, sid in keys:                            # prep order
        _, _, reg, diag, _ = results[(split, sid)]
        out[split][sid] = reg
        if reg["n"] == 0:                              # Req 4.9
            zero[split].append(sid)
            checkpoint.log_line("ZERO_CANDIDATES", f"{split} {sid}")
        if split == "heldout":
            per_region[sid] = diag
    diag = diagnostics(per_region, prm["sigma"])
    n_zero = len(zero["heldout"]) + len(zero["test"])
    checkpoint.log_line("POSE_SEARCH_DONE",
                        f"heldout={len(out['heldout'])} test={len(out['test'])} zero={n_zero} "
                        f"gt={diag['n_with_gt']} raw_gpu={diag['correct_in_raw_gpu']} "
                        f"candidates={diag['correct_in_candidates']} first={diag['correct_first']} "
                        f"sigma={prm['sigma']} elapsed={time.monotonic() - t0:.1f}s")
    return {**out, "sigma": prm["sigma"], "params": prm, "zero_candidates": zero,
            "n_zero": n_zero, "diagnostics": diag}


def _log_region(res, done: int, total: int, t0: float) -> None:
    split, sid, reg, diag, dt = res
    extra = "" if diag is None else (f" raw_gpu={int(diag['raw_gpu'])} "
                                     f"cand={int(diag['candidates'])} first={int(diag['first'])}")
    checkpoint.log_line("POSE_REGION", f"{split} {sid} n={reg['n']} by_source={reg['n_by_source']}"
                                       f"{extra} {dt:.1f}s [{done}/{total}] "
                                       f"total={time.monotonic() - t0:.1f}s")
