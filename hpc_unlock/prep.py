"""Prep Stage (CPU, precedes gpu_scan): region records, duplicate keys, scan inputs.

Builds one RegionRecord per region as a plain dict of builtins and NumPy arrays,
so ``prep.pkl`` loads without importing ``hpc_unlock`` (design "Data Models"):

* held-out (47 regions): fields from ``research/data/lab.pkl``; window and vote
  candidates exactly as behind v10 (``cp_pose_train.pkl`` / ``vote_cands.pkl``).
* test (29 regions): ``test_apply.build`` on ``research/data/submission.csv`` (the
  ungrown v7 masks whose IDs match v7_grow15 and v10), ``offset`` from
  ``common.crop_offsets("hidden_test")``, vote candidates from ``test_apply.cands``,
  vote modes per ``(subject, ex_shape)`` group with duplicates voting once (the
  ``test_cp_apply.py`` recipe), window candidates from ``cp_pose_lab.all_window``
  and the v10 window pose from ``test_apply.window``.

Per record: ``dup_key`` (SHA-1 of the decoded in-vivo + ex-vivo pixels, Req 6.4),
``group = (subject, ex_shape)`` and a reference to ``cp_bin`` (``cp_pose_lab.cp_map``
of the ex-vivo cellprob, uint8). The cp_bin arrays (about 2.6 MB each) live in the
side file ``prep_cp_bin.npz`` next to ``prep.pkl``; use :func:`load_cp_bins`.

Record centroids keep the dtype of their source (float64) so the v10 recipe
reproduces bit-for-bit; the GPU scan inputs are float32 copies.
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hpc_unlock import checkpoint, paths

SPLIT_DIRS = {"heldout": "training", "test": "hidden_test"}
MASKS_CSV = paths.RDATA / "submission.csv"
BASELINE_CSVS = ("submission_v10_cpgate.csv", "submission_v7_grow15.csv")
CP_SIDE = "prep_cp_bin.npz"
SMOKE_HELDOUT = 2      # held-out regions in smoke, one per mouse (Req 13)
SMOKE_TEST = 1         # test regions in smoke (0 or 1)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

csv.field_size_limit(sys.maxsize)


# ----------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------------

def dup_key(iv_img: np.ndarray, ex_img: np.ndarray) -> str:
    """SHA-1 over the raw pixel bytes of both images (dtype and shape included).

    Equal exactly when both images are pixel-identical (Req 6.4).
    """
    h = hashlib.sha1()
    for img in (iv_img, ex_img):
        a = np.ascontiguousarray(img)
        h.update(f"{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def centroid_key(iv_c: np.ndarray, ex_c: np.ndarray) -> str:
    """Deterministic version of ``test_apply``'s centroid hash (for consistency checks)."""
    return hashlib.sha1(np.ascontiguousarray(iv_c).tobytes()
                        + np.ascontiguousarray(ex_c).tobytes()).hexdigest()


def group_key(subject: str, ex_shape: Sequence[int]) -> tuple[str, tuple[int, int]]:
    """Mouse/canvas group ``(subject, ex_shape)`` with builtin ints."""
    return (str(subject), (int(ex_shape[0]), int(ex_shape[1])))


def group_sids(records: Mapping[str, Mapping]) -> dict[tuple, list[str]]:
    """``group -> [sid]`` in the iteration order of ``records``."""
    out: dict[tuple, list[str]] = {}
    for sid, rec in records.items():
        out.setdefault(rec["group"], []).append(sid)
    return out


def voters_once(sids: Iterable[str], keys: Mapping[str, str],
                cands: Mapping[str, Any]) -> dict[str, Any]:
    """First region per duplicate key votes; later duplicates are skipped (order kept)."""
    seen, voters = set(), {}
    for s in sids:
        if keys[s] not in seen:
            seen.add(keys[s])
            voters[s] = cands[s]
    return voters


def smoke_heldout(subject_of: Mapping[str, str], n: int = SMOKE_HELDOUT) -> list[str]:
    """First region (sorted) of each of the first ``n`` mice (sorted)."""
    by_subj: dict[str, list[str]] = {}
    for sid in sorted(subject_of):
        by_subj.setdefault(subject_of[sid], []).append(sid)
    return [by_subj[s][0] for s in sorted(by_subj)[:n]]


def scan_inputs(records: Mapping[str, Mapping]) -> dict[str, dict]:
    """One GPU scan input per unique ``dup_key`` (float32; first region's centroids)."""
    out: dict[str, dict] = {}
    for sid, rec in records.items():
        k = rec["dup_key"]
        if k in out:
            out[k]["sids"].append(sid)
            continue
        out[k] = {"iv_c": np.asarray(rec["iv_c"], np.float32).reshape(-1, 2),
                  "ex_c": np.asarray(rec["ex_c"], np.float32).reshape(-1, 2),
                  "ex_shape": tuple(int(v) for v in rec["ex_shape"]),
                  "offset": np.asarray(rec["offset"], np.float32).reshape(2),
                  "sids": [sid]}
    return out


def baseline_sha256(root: str | Path = paths.ROOT) -> dict[str, str]:
    """SHA-256 of the Baseline CSVs, checked again at the end of the chain (Req 10.5)."""
    return {name: checkpoint.sha256_file(Path(root) / name) for name in BASELINE_CSVS}


def load_cp_bins(prep: Mapping, run_dir: str | Path) -> dict[str, np.ndarray]:
    """``sid -> cp_bin`` (uint8) from the side file referenced by ``prep["cp_side"]``."""
    arrays = checkpoint.load_npz(prep["cp_side"], run_dir)
    out = {}
    for split in ("heldout", "test"):
        for sid, rec in prep[split].items():
            out[sid] = arrays[rec["cp_bin"]["key"]]
    return out


def image_dir(split: str, sid: str) -> Path:
    subj, reg = sid.split("__")
    return paths.DATA / SPLIT_DIRS[split] / subj / reg


def region_dup_key(split: str, sid: str) -> str:
    from cellmatch import read_image
    d = image_dir(split, sid)
    return dup_key(read_image(d / "invivo.tif"), read_image(d / "exvivo.tif"))


def _f(x) -> float:
    return float(x)


def _vote_list(cands) -> list[tuple]:
    """vote_cands entries ``(score, angle, landing, M)`` with builtin scalars."""
    return [(_f(c[0]), _f(c[1]), np.asarray(c[2], float), np.asarray(c[3], float)) for c in cands]


def _scored_list(scored) -> list[tuple]:
    """``(M, refine_score, z, src)`` entries with builtin scalars."""
    return [(np.asarray(M, float), _f(s), _f(z), str(src)) for M, s, z, src in scored]


def _modes_plain(modes) -> list[dict]:
    return [{"angle": _f(m["angle"]), "landing": np.asarray(m["landing"], float),
             "support": _f(m["support"]), "n": int(m["n"])} for m in modes]


def heldout_record(sid: str, r: Mapping, vote_cands, cp_scored, key: str) -> dict:
    """RegionRecord dict for a held-out region from its ``lab.pkl`` entry."""
    scored = _scored_list(cp_scored)
    return {
        "sid": sid, "split": "heldout", "subject": str(r["subject"]),
        "iv_shape": tuple(int(v) for v in r["iv_shape"]),
        "ex_shape": tuple(int(v) for v in r["ex_shape"]),
        "offset": np.asarray(r["offset"], float).reshape(2),
        "iv_c": np.asarray(r["iv_c"]), "ex_c": np.asarray(r["ex_c"]),
        "iv_f": {k: np.asarray(v) for k, v in r["iv_f"].items()},
        "ex_f": {k: np.asarray(v) for k, v in r["ex_f"].items()},
        "iv_ids": None, "ex_ids": None,
        "cp_bin": {"side": "cp_side", "key": sid},
        "dup_key": key, "group": group_key(r["subject"], r["ex_shape"]),
        "centroid_key": centroid_key(r["iv_c"], r["ex_c"]),
        "iv_link": np.asarray(r["iv_link"]), "ex_link": np.asarray(r["ex_link"]),
        "gt_pairs": {(int(a), int(b)) for a, b in r["gt_pairs"]},
        "n_gt_pairs": int(r["n_gt_pairs"]),
        "gt_iv_c": None if r.get("gt_iv_c") is None else np.asarray(r["gt_iv_c"]),
        "gt_M": None if r.get("gt_M") is None else np.asarray(r["gt_M"], float),
        "vote_cands": _vote_list(vote_cands),
        "window_cands": [(M, s, "win") for M, s, _, src in scored if src == "win"],
        "cp_scored": scored,           # exact v10 candidates: input of cp_pose_lab.pose_choose
        "v10_window": None,
    }


def test_record(sid: str, rec: Mapping, key: str) -> dict:
    """RegionRecord dict for a test region from ``test_apply.build`` output + offset."""
    return {
        "sid": sid, "split": "test", "subject": str(rec["subject"]),
        "iv_shape": tuple(int(v) for v in rec["iv_shape"]),
        "ex_shape": tuple(int(v) for v in rec["ex_shape"]),
        "offset": np.asarray(rec["offset"], float).reshape(2),
        "iv_c": np.asarray(rec["iv_c"]), "ex_c": np.asarray(rec["ex_c"]),
        "iv_f": {k: np.asarray(v) for k, v in rec["iv_f"].items()},
        "ex_f": {k: np.asarray(v) for k, v in rec["ex_f"].items()},
        "iv_ids": [str(i) for i in rec["iv_ids"]], "ex_ids": [str(i) for i in rec["ex_ids"]],
        "cp_bin": {"side": "cp_side", "key": sid},
        "dup_key": key, "group": group_key(rec["subject"], rec["ex_shape"]),
        "centroid_key": centroid_key(rec["iv_c"], rec["ex_c"]),
        "iv_link": None, "ex_link": None, "gt_pairs": None, "n_gt_pairs": None,
        "gt_iv_c": None, "gt_M": None,
        "vote_cands": [], "window_cands": [], "cp_scored": [], "v10_window": None,
    }


# ----------------------------------------------------------------------------
# Research imports
# ----------------------------------------------------------------------------

_RESEARCH: SimpleNamespace | None = None


@contextmanager
def _research_import_env():
    """``test_apply`` / ``window_lab`` parse ``sys.argv`` at import (``float(argv[2])``),
    which breaks under ``run_unlock.py prep --run <fp>``; hide argv while importing.
    ``test_apply`` reads ``MASKS_CSV`` / ``STRETCH`` at import: force the v10 values."""
    argv = sys.argv
    sys.argv = [argv[0] if argv else "prep"]
    os.environ["MASKS_CSV"] = str(MASKS_CSV)
    os.environ.pop("STRETCH", None)
    try:
        yield
    finally:
        sys.argv = argv


def _research() -> SimpleNamespace:
    global _RESEARCH
    if _RESEARCH is None:
        with _research_import_env():
            import common
            import cp_pose_lab
            import test_apply
            import vote
        if Path(test_apply.SRC).resolve() != MASKS_CSV.resolve():
            raise RuntimeError(f"test_apply.SRC is {test_apply.SRC}, expected {MASKS_CSV}")
        if test_apply.STRETCH:
            raise RuntimeError(f"test_apply.STRETCH must be empty, got {test_apply.STRETCH!r}")
        _RESEARCH = SimpleNamespace(common=common, cp_pose_lab=cp_pose_lab,
                                    test_apply=test_apply, vote=vote)
    return _RESEARCH


# ----------------------------------------------------------------------------
# Pool workers (fork: research modules are imported in the parent first)
# ----------------------------------------------------------------------------

def _init_worker() -> None:
    for v in THREAD_VARS:
        os.environ[v] = "1"
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass


def _heldout_job(sid: str) -> tuple[str, str, np.ndarray]:
    R = _research()
    with np.load(paths.RDATA / "heldout_labels.npz") as z:
        cp = z[f"{sid}|exvivo|prob"].astype(np.float32)
    return sid, region_dup_key("heldout", sid), R.cp_pose_lab.cp_map(cp).astype(np.uint8)


def _test_build_job(row: dict) -> tuple[str, dict, str]:
    R = _research()
    sid, rec = R.test_apply.build(row)
    rec.pop("key", None)  # salted Python hash; replaced by dup_key / centroid_key
    return sid, rec, region_dup_key("test", sid)


def _test_window_job(args) -> tuple:
    """v10 window pose, all window candidates, cp_bin and cp z for every candidate."""
    sid, rec, modes = args
    R = _research()
    _, v10 = R.test_apply.window((sid, rec, modes))
    win = R.cp_pose_lab.all_window(rec["iv_c"], rec["ex_c"], rec["offset"], modes)
    with np.load(paths.RDATA / "test_cp_base.npz") as z:
        cp = z[f"{sid}|cp"].astype(np.float32)
    binmap = R.cp_pose_lab.cp_map(cp).astype(np.uint8)
    cands = win + [(c[3], c[0], "vote") for c in rec["vote_cands"]]   # cp_pose_lab.process order
    scored = [(M, s, R.cp_pose_lab.cp_z(binmap, rec["iv_c"], M), src) for M, s, src in cands]
    return sid, v10, win, scored, binmap


# ----------------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------------

def _pickle_load(path: Path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_heldout(pool, smoke: bool) -> tuple[dict, dict]:
    lab = _pickle_load(paths.RDATA / "lab.pkl")
    votes = _pickle_load(paths.RDATA / "vote_cands.pkl")
    scored = _pickle_load(paths.RDATA / "cp_pose_train.pkl")
    sids = sorted(lab)
    if smoke:
        sids = smoke_heldout({s: lab[s]["subject"] for s in sids})
    recs, bins = {}, {}
    for sid, key, binmap in pool.map(_heldout_job, sids):
        recs[sid] = heldout_record(sid, lab[sid], votes[sid], scored[sid], key)
        bins[sid] = binmap
    return recs, bins


def _build_test(pool, smoke: bool) -> tuple[dict, dict, dict]:
    R = _research()
    with open(MASKS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if smoke:
        rows = rows[:SMOKE_TEST]
    if not rows:
        return {}, {}, {}
    built = pool.map(_test_build_job, rows)                  # CSV row order kept
    offsets, _ = R.common.crop_offsets("hidden_test")
    raw, keys = {}, {}
    for sid, rec, key in built:
        rec["offset"] = offsets[tuple(sid.split("__"))]
        raw[sid], keys[sid] = rec, key
    C = dict(pool.map(R.test_apply.cands, list(raw.items())))
    for sid in raw:
        raw[sid]["vote_cands"] = C[sid]
    recs = {sid: test_record(sid, raw[sid], keys[sid]) for sid in raw}

    test_modes, jobs = {}, []
    for g, sids in group_sids(recs).items():
        voters = voters_once(sids, keys, C)
        cvoters = voters_once(sids, {s: recs[s]["centroid_key"] for s in sids}, C)
        if list(voters) != list(cvoters):
            checkpoint.log_line("DUP_KEY_MISMATCH",
                                f"{g}: pixel voters {list(voters)} centroid voters {list(cvoters)}")
        modes = R.vote.vote_modes(voters)
        test_modes[g] = _modes_plain(modes)
        checkpoint.log_line("TEST_GROUP", f"{g} regions={len(sids)} voters={len(voters)} "
                                          f"modes={len(modes)}")
        jobs += [(s, raw[s], modes) for s in sids]

    bins = {}
    for sid, v10, win, scored, binmap in pool.map(_test_window_job, jobs):
        M, score, M_true = v10
        rec = recs[sid]
        rec["vote_cands"] = _vote_list(C[sid])
        rec["window_cands"] = [(np.asarray(m, float), _f(s), "win") for m, s, _ in win]
        rec["cp_scored"] = _scored_list(scored)
        rec["v10_window"] = {"M": None if M is None else np.asarray(M, float),
                             "score": _f(score),
                             "M_true": None if M_true is None else np.asarray(M_true, float)}
        bins[sid] = binmap
    return recs, bins, test_modes


def compute(cfg, ctx) -> dict:
    """Stage entry point: ``ctx`` has ``run_dir``, ``smoke``, ``workers``, ``load``."""
    import multiprocessing as mp

    t0 = time.monotonic()
    run_dir = Path(ctx.run_dir)
    smoke = bool(getattr(ctx, "smoke", False))
    workers = max(1, int(getattr(ctx, "workers", 1)))
    _research()                                                # import before fork
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        heldout, hbins = _build_heldout(pool, smoke)
        checkpoint.log_line("PREP_HELDOUT", f"regions={len(heldout)} "
                                            f"elapsed={time.monotonic() - t0:.1f}s")
        test, tbins, test_modes = _build_test(pool, smoke)
        checkpoint.log_line("PREP_TEST", f"regions={len(test)} "
                                         f"elapsed={time.monotonic() - t0:.1f}s")

    bins = {**hbins, **tbins}
    if len(bins) != len(hbins) + len(tbins):
        raise RuntimeError("sample id collision between held-out and test")
    cp_side = checkpoint.save_npz_atomic(run_dir / CP_SIDE, **bins)
    scans = scan_inputs({**heldout, **test})
    n_dup = sum(len(v["sids"]) > 1 for v in scans.values())
    checkpoint.log_line("PREP_DONE", f"heldout={len(heldout)} test={len(test)} "
                                     f"scan_inputs={len(scans)} duplicate_keys={n_dup} "
                                     f"elapsed={time.monotonic() - t0:.1f}s")
    return {
        "heldout": heldout,
        "test": test,
        "scan_inputs": scans,
        "test_modes": test_modes,
        "baseline_sha256": baseline_sha256(),
        "cp_side": cp_side,
        "meta": {"smoke": smoke, "masks_csv": str(MASKS_CSV.relative_to(paths.ROOT)),
                 "workers": workers},
    }
