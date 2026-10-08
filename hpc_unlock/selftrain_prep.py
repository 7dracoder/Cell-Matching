"""Self-training preprocessing Stage (CPU): pseudo-labels, 96-px tiles, images (Req 12).

Runs on ``n2c48m24`` before ``selftrain_gpu`` (12.10). The GPU job then only
loads this Stage's Checkpoint and side file: it reads no TIFF and builds no label.

Jobs
----
One job per held-out mouse m (leave-one-mouse-out evaluation, 12.5) and one
``"test"`` job (12.9). Each job has

* inference regions: every region of mouse m, or all test regions;
* training regions: the regions kept by the conservative gate that end up with
  at least one pseudo-label cell.

Gate, poses and pairs (12.3, 12.5)
----------------------------------
Selection: ``joint`` (the default source of self-training poses); ``independent``
if ``joint`` is missing from a Checkpoint, or if validation accepted
``indep_cons`` but not ``joint_cons``.

* Held-out mouse m. Kept = the verifier's ``fold_conservative`` flag (fold tau
  chosen without m; v10 gate only when the fold tau is unavailable). Pair
  probabilities = the pairs Stage's LOO probabilities (model trained without m).
  The pair threshold is chosen on the other two mice only
  (:func:`fold_pair_threshold`): their pair probabilities come from models
  trained on the mice outside ``{m, o}`` and their kept flags from the nested
  verifier probabilities behind m's fold gate (``verifier.loo_probs(...,
  exclude=[m])``). Nothing of m's ground truth (masks, pairs, transform) enters
  m's pseudo-labels.
* Test. Kept = the full conservative gate (``kept["conservative"]``, the v10 gate
  when conservative is unavailable). Pair probabilities = the all-mice test
  model; threshold = the validated ``<sel>_cons`` configuration's pair threshold
  (recomputed with ``pairs.choose_threshold`` if that configuration is absent).

A Matched_Invivo_Instance is the in-vivo index of a pair chosen by
``pairs.select`` under that threshold and gate.

Pseudo-labels (:func:`pseudo_labels`, Property 22)
--------------------------------------------------
Predicted ex-vivo masks: held-out ``heldout_labels.npz[<sid>|exvivo]`` (ungrown),
test ``submission.csv`` ``exvivo_instances`` (ungrown v7) decoded with
``cellmatch.rle_to_labels``. No ground-truth mask is read. Matched in-vivo
centroids are projected through the region's pose M (``iv_c @ M[:, :2].T + M[:, 2]``,
``(x, y)`` canvas pixels). Then

1. every predicted instance whose centroid is within ``match_radius`` (<=) of a
   projected point is retained with its original pixels;
2. every projected point with no predicted centroid within ``match_radius``
   seeds a filled disk of ``seed_radius`` (pixel centres ``(x, y)`` with
   ``(x - cx)^2 + (y - cy)^2 <= r^2``), clipped to the canvas, painted only on
   pixels that are background in the predicted map and not taken by an earlier
   seed; a seed with no paintable pixel produces no instance;
3. every other predicted instance is excluded;
4. IDs are consecutive: retained instances 1..k in ascending original ID, then
   seeds in projected-point order.

Tiles (the ``pseudo_cv_colab.make_tiles`` recipe)
-------------------------------------------------
Per training region: the image normalised with ``pipeline.percentile_normalize``
over the whole image, up to 100 (smoke: 8) 96-px crops centred on pseudo-label
centroids chosen without replacement (one ``default_rng(TILE_SEED)`` per job,
regions in sorted order), crop origin clipped to the canvas, crop labels
relabelled by ``np.unique(return_inverse)``. Canvases smaller than 96 px are
zero-padded.

Side file ``selftrain_tiles.npz`` (``checkpoint.save_npz_atomic``)::

    "image|<sid>"        raw ex-vivo image, float32 (every inference region, once)
    "tiles|<job>|image"  (n, 96, 96) float32 normalised tiles
    "tiles|<job>|label"  (n, 96, 96) int32 tile labels
    "tiles|<job>|sid"    (n,) str, source region of each tile

All 76 images (at most 1627^2) as float32 take about 0.8 GB, under the 32 GB
Stage memory, so no float16 packing is used; the total is logged.

Checkpoint::

    {"selection": "joint" | "independent",
     "jobs": {job: {"split", "regions" (inference), "train_regions", "kept_regions",
                    "n_tiles", "no_confident" (bool), "reason" (str | None),
                    "pair_threshold", "per_region": {sid: {...counts}}}},
     "tiles_side": side-file reference, "keys": {...key patterns},
     "no_confident_regions": {"mice": [...], "test": bool},
     "params": {...}}

``no_confident`` is True when the gate keeps 0 regions of the job (reason
"no confident regions", 12.11 / 12.12), or when the kept regions yield no
pseudo-label cell (reason "no pseudo-labels in kept regions"); the GPU Stage
skips such jobs.

Invalid radii (outside [1, 20] px) stop the Stage before any Checkpoint is
loaded or any label is built: the error is logged and ``SystemExit(1)`` leaves
no done-marker (12.13).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from hpc_unlock import paths  # noqa: F401  (puts ROOT and research/ on sys.path)
from hpc_unlock import checkpoint
from hpc_unlock import pairs as P
from hpc_unlock import verifier as V

TILE = 96
TILES_PER_REGION = 100
SMOKE_TILES_PER_REGION = 8
TILE_SEED = 27
RADIUS_RANGE = (1.0, 20.0)
SIDE_NAME = "selftrain_tiles.npz"
TEST_JOB = "test"
LABELS_NPZ = paths.RDATA / "heldout_labels.npz"
MASKS_CSV = paths.RDATA / "submission.csv"
NO_CONFIDENT = "no confident regions"
NO_PSEUDO = "no pseudo-labels in kept regions"
SELECTIONS = ("joint", "independent")   # preference order

csv.field_size_limit(sys.maxsize)


# ----------------------------------------------------------------------------
# Radii (Req 12.13)
# ----------------------------------------------------------------------------

def radius_errors(match_radius: Any, seed_radius: Any) -> list[str]:
    """One error per radius outside [1, 20] px (or not a finite number)."""
    lo, hi = RADIUS_RANGE
    errors = []
    for name, v in (("match_radius", match_radius), ("seed_radius", seed_radius)):
        ok = (isinstance(v, (int, float)) and not isinstance(v, bool)
              and math.isfinite(v) and lo <= v <= hi)
        if not ok:
            errors.append(f"{name}={v!r} is invalid; expected a number in "
                          f"[{lo:g}, {hi:g}] px")
    return errors


def validate_config(cfg) -> list[str]:
    """Stage-specific config errors (called by ``stage.execute`` before compute)."""
    return radius_errors(getattr(cfg, "match_radius", None), getattr(cfg, "seed_radius", None))


def check_radii(match_radius: Any, seed_radius: Any,
                log: Callable[..., str] = checkpoint.log_line) -> None:
    """Log every invalid radius and ``SystemExit(1)``; nothing is built."""
    errors = radius_errors(match_radius, seed_radius)
    if errors:
        for err in errors:
            log("CONFIG_INVALID", f"selftrain_prep: {err}")
        log("STAGE_ABORTED", "selftrain_prep: invalid radius, no pseudo-label built")
        raise SystemExit(1)


# ----------------------------------------------------------------------------
# Pseudo-labels (Req 12.3, Property 22)
# ----------------------------------------------------------------------------

def instance_centroids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, centroids)``: present non-zero IDs (ascending) and their ``(x, y)`` means."""
    labels = np.asarray(labels)
    flat = labels.ravel().astype(np.int64)
    if flat.size == 0 or flat.max() <= 0:
        return np.zeros(0, np.int64), np.zeros((0, 2))
    if flat.min() < 0:
        raise ValueError("label map has negative IDs")
    n = int(flat.max()) + 1
    yy, xx = np.indices(labels.shape)
    count = np.bincount(flat, minlength=n)
    sx = np.bincount(flat, weights=xx.ravel(), minlength=n)
    sy = np.bincount(flat, weights=yy.ravel(), minlength=n)
    ids = np.flatnonzero(count)
    ids = ids[ids > 0]
    c = np.stack([sx[ids] / count[ids], sy[ids] / count[ids]], axis=1)
    return ids.astype(np.int64), c


def project(iv_c: np.ndarray, M: np.ndarray) -> np.ndarray:
    """``registration.transform``: ``(x, y) @ M[:, :2].T + M[:, 2]``."""
    pts = np.asarray(iv_c, float).reshape(-1, 2)
    M = np.asarray(M, float).reshape(2, 3)
    return pts @ M[:, :2].T + M[:, 2]


def _paint_disk(out: np.ndarray, free: np.ndarray, cx: float, cy: float, r: float,
                value: int) -> int:
    """Paint ``value`` on free pixels of the canvas-clipped disk; return the pixel count."""
    H, W = out.shape
    x0, x1 = max(0, math.ceil(cx - r)), min(W - 1, math.floor(cx + r))
    y0, y1 = max(0, math.ceil(cy - r)), min(H - 1, math.floor(cy + r))
    if x0 > x1 or y0 > y1:
        return 0
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & free[y0:y1 + 1, x0:x1 + 1]
    n = int(m.sum())
    if n:
        out[y0:y1 + 1, x0:x1 + 1][m] = value
        free[y0:y1 + 1, x0:x1 + 1][m] = False
    return n


def pseudo_labels(pred: np.ndarray, proj: np.ndarray, match_radius: float,
                  seed_radius: float) -> tuple[np.ndarray, dict]:
    """Retain / seed / exclude rule (module docstring). Returns ``(labels int32, info)``.

    ``info``: ``retained`` / ``seeded`` / ``excluded`` / ``seeds_empty`` counts,
    ``n_pred``, ``n_proj`` and ``n_pseudo`` (= max label).
    """
    pred = np.asarray(pred)
    if pred.ndim != 2:
        raise ValueError(f"predicted label map must be 2-D, got {pred.shape}")
    proj = np.asarray(proj, float).reshape(-1, 2)
    proj = proj[np.isfinite(proj).all(axis=1)]
    ids, cent = instance_centroids(pred)
    if len(proj) and len(ids):
        d_ex, _ = cKDTree(proj).query(cent, k=1)
        keep = d_ex <= match_radius
        d_pr, _ = cKDTree(cent).query(proj, k=1)
        seeds = proj[d_pr > match_radius]
    else:
        keep = np.zeros(len(ids), bool)
        seeds = proj
    lut = np.zeros(int(pred.max()) + 1 if pred.size else 1, np.int32)
    kept_ids = ids[keep]
    lut[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
    out = lut[np.clip(pred, 0, None).astype(np.int64)] if pred.size else pred.astype(np.int32)
    free = pred == 0
    nxt = len(kept_ids) + 1
    empty = 0
    for cx, cy in seeds:
        if _paint_disk(out, free, float(cx), float(cy), float(seed_radius), nxt):
            nxt += 1
        else:
            empty += 1
    info = {"n_pred": int(len(ids)), "n_proj": int(len(proj)),
            "retained": int(len(kept_ids)), "seeded": int(nxt - 1 - len(kept_ids)),
            "seeds_empty": int(empty), "excluded": int(len(ids) - len(kept_ids)),
            "n_pseudo": int(nxt - 1)}
    return out.astype(np.int32), info


# ----------------------------------------------------------------------------
# Selection, gate, matched in-vivo instances (Req 12.3, 12.5)
# ----------------------------------------------------------------------------

def _configs(validate_ck: Mapping | None) -> list[Mapping]:
    return list((validate_ck or {}).get("configs", []) or [])


def _config_for(validate_ck: Mapping | None, sel: str, gate: str = "conservative"):
    for c in _configs(validate_ck):
        if c.get("selection") == sel and c.get("gate") == gate:
            return c
    return None


def choose_selection(joint_ck: Mapping, pairs_ck: Mapping, verifier_ck: Mapping,
                     validate_ck: Mapping | None = None) -> str:
    """``joint`` unless missing, or unless ``indep_cons`` was accepted and ``joint_cons`` not."""
    avail = [s for s in SELECTIONS
             if s in joint_ck.get("heldout", {}) and s in pairs_ck.get("heldout", {})
             and s in verifier_ck.get("selections", {})]
    if not avail:
        raise ValueError("no pose selection available in joint / pairs / verifier")
    if "joint" in avail and "independent" in avail:
        jc = _config_for(validate_ck, "joint")
        ic = _config_for(validate_ck, "independent")
        if jc is not None and ic is not None and not jc.get("accepted") and ic.get("accepted"):
            return "independent"
    return avail[0]


def matched_invivo(entry: Mapping | None, threshold: float, kept: bool) -> np.ndarray:
    """In-vivo indices of the pairs ``pairs.select`` picks (sorted, unique)."""
    if entry is None:
        return np.zeros(0, int)
    sel = P.select(entry["pairs"], entry["probs"], threshold, kept)
    return np.unique(sel[:, 0]).astype(int)


def fold_pair_threshold(m: str, heldout: Mapping[str, Mapping], hposes: Mapping[str, Any],
                        ver_sel: Mapping, correct: Callable[[Mapping, np.ndarray], bool],
                        pair_estimator: Callable[[int], Any] = P.make_estimator,
                        ver_estimator: Callable[[], Any] = V.make_model) -> dict:
    """Pair threshold for held-out mouse ``m`` chosen on the other mice only.

    Each other mouse o: pair probabilities from a model trained on the mice
    outside ``{m, o}``; kept = ``registered & (v10 | p_ver >= fold tau of m)`` with
    the nested verifier probabilities of m's fold gate. Neither depends on m.
    """
    m = str(m)
    others = {s: r for s, r in heldout.items() if str(r["subject"]) != m}
    tau = (ver_sel.get("fold", {}).get(m) or {}).get("conservative")
    if not others:
        return {"threshold": float(P.THRESHOLDS[0]), "tau": tau, "tp": 0, "pred": 0,
                "f1": 0.0, "n_other": 0}
    rows = P.dataset(others, {s: hposes.get(s) for s in others})
    pprobs = P.loo_predict(rows, {s: others[s]["subject"] for s in others},
                           estimator=pair_estimator)
    T = V.table(heldout, hposes, correct)
    vprobs = V.loo_probs(T, ver_estimator, exclude=[m])
    kept_all = V.kept_flags(vprobs, T["v10"], T["registered"], tau)
    kept = {s: bool(k) for s, k in zip(T["sids"], kept_all) if s in others}
    total = sum(int(r.get("n_gt_pairs") or 0) for r in others.values())
    thr, tp, pred, f1 = P.choose_threshold(rows, pprobs, kept, total)
    return {"threshold": float(thr), "tau": tau, "tp": int(tp), "pred": int(pred),
            "f1": float(f1), "n_other": len(others)}


def plan(prep: Mapping, joint_ck: Mapping, pairs_ck: Mapping, verifier_ck: Mapping,
         validate_ck: Mapping | None = None,
         correct: Callable[[Mapping, np.ndarray], bool] | None = None,
         pair_estimator: Callable[[int], Any] = P.make_estimator,
         ver_estimator: Callable[[], Any] = V.make_model) -> dict:
    """Per job: inference regions, kept regions with pose and matched in-vivo indices.

    ``{"selection", "jobs": {job: {"split", "regions", "kept": {sid: {"M",
    "matched"}}, "pair_threshold", "threshold_info"}}}``.
    """
    heldout, test = prep["heldout"], prep.get("test", {}) or {}
    sel = choose_selection(joint_ck, pairs_ck, verifier_ck, validate_ck)
    ver = verifier_ck["selections"][sel]
    hposes = joint_ck["heldout"][sel]
    if correct is None and any(V.has_gt(r) for r in heldout.values()):
        from hpc_unlock import joint as J
        correct = J.default_correct()
    jobs: dict[str, dict] = {}
    for m in sorted({str(r["subject"]) for r in heldout.values()}):
        regions = sorted(s for s, r in heldout.items() if str(r["subject"]) == m)
        info = fold_pair_threshold(m, heldout, hposes, ver, correct,
                                   pair_estimator, ver_estimator)
        thr = info["threshold"]
        kept = {}
        for sid in regions:
            e = hposes.get(sid) or {}
            k = bool(ver["heldout"][sid]["kept"]["fold_conservative"])
            if not k or e.get("M") is None:
                continue
            kept[sid] = {"M": np.asarray(e["M"], float),
                         "matched": matched_invivo(pairs_ck["heldout"][sel].get(sid), thr, True)}
        jobs[m] = {"split": "heldout", "regions": regions, "kept": kept,
                   "pair_threshold": thr, "threshold_info": info}
    if test:
        tposes = joint_ck.get("test", {}).get(sel, {})
        tentries = pairs_ck.get("test", {}).get(sel, {})
        cfg_c = _config_for(validate_ck, sel)
        if cfg_c is not None:
            thr = float(cfg_c["pair_threshold"])
            src = cfg_c.get("name")
        else:
            rows, probs = P.entry_rows(pairs_ck["heldout"][sel])
            kept_h = {s: bool(ver["heldout"][s]["kept"]["conservative"]) for s in heldout}
            total = sum(int(r.get("n_gt_pairs") or 0) for r in heldout.values())
            thr = float(P.choose_threshold(rows, probs, kept_h, total)[0])
            src = "recomputed"
        kept = {}
        for sid in sorted(test):
            e = tposes.get(sid) or {}
            k = bool(((ver.get("test", {}).get(sid) or {}).get("kept") or {}).get("conservative"))
            if not k or e.get("M") is None:
                continue
            kept[sid] = {"M": np.asarray(e["M"], float),
                         "matched": matched_invivo(tentries.get(sid), thr, True)}
        jobs[TEST_JOB] = {"split": "test", "regions": sorted(test), "kept": kept,
                          "pair_threshold": thr, "threshold_info": {"source": src}}
    return {"selection": sel, "jobs": jobs}


def job_pseudo_labels(job: Mapping, records: Mapping[str, Mapping],
                      load_pred: Callable[[str, str, Mapping], np.ndarray],
                      match_radius: float, seed_radius: float) -> dict:
    """``sid -> (labels, info)`` for every kept region of a planned job."""
    out = {}
    for sid, k in sorted(job["kept"].items()):
        rec = records[sid]
        pred = load_pred(job["split"], sid, rec)
        idx = np.asarray(k["matched"], int)
        iv_c = np.asarray(rec["iv_c"], float).reshape(-1, 2)
        proj = project(iv_c[idx], k["M"]) if len(idx) else np.zeros((0, 2))
        lab, info = pseudo_labels(pred, proj, match_radius, seed_radius)
        info["n_matched"] = int(len(idx))
        out[sid] = (lab, info)
    return out


# ----------------------------------------------------------------------------
# Tiles
# ----------------------------------------------------------------------------

def _normalize(image: np.ndarray) -> np.ndarray:
    import pipeline  # noqa: E402  (imports torch; only needed for tiles)
    return pipeline.percentile_normalize(image)


def _pad_to_tile(a: np.ndarray, tile: int) -> np.ndarray:
    ph, pw = max(0, tile - a.shape[0]), max(0, tile - a.shape[1])
    return np.pad(a, ((0, ph), (0, pw))) if ph or pw else a


def make_tiles(image: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
               per_region: int = TILES_PER_REGION, tile: int = TILE,
               normalize: Callable[[np.ndarray], np.ndarray] = _normalize
               ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Up to ``per_region`` crops centred on pseudo-label centroids (make_tiles recipe)."""
    image = np.asarray(image)
    labels = np.asarray(labels)
    if image.shape != labels.shape:
        raise ValueError(f"image {image.shape} vs labels {labels.shape}")
    _, centers = instance_centroids(labels)
    if not len(centers) or per_region <= 0:
        return [], []
    norm = _pad_to_tile(np.asarray(normalize(image), np.float32), tile)
    lab = _pad_to_tile(labels, tile)
    half = tile // 2
    chosen = rng.choice(len(centers), min(per_region, len(centers)), replace=False)
    imgs, labs = [], []
    for x, y in centers[chosen]:
        y0 = int(np.clip(round(y) - half, 0, norm.shape[0] - tile))
        x0 = int(np.clip(round(x) - half, 0, norm.shape[1] - tile))
        crop = lab[y0:y0 + tile, x0:x0 + tile]
        _, inv = np.unique(crop, return_inverse=True)
        imgs.append(np.ascontiguousarray(norm[y0:y0 + tile, x0:x0 + tile], np.float32))
        labs.append(inv.reshape(tile, tile).astype(np.int32))
    return imgs, labs


# ----------------------------------------------------------------------------
# Default loaders
# ----------------------------------------------------------------------------

class PredictedExLoader:
    """Predicted (ungrown) ex-vivo label maps; never ground truth."""

    def __init__(self, labels_npz: str | Path = LABELS_NPZ, masks_csv: str | Path = MASKS_CSV):
        self.labels_npz = Path(labels_npz)
        self.masks_csv = Path(masks_csv)
        self._csv: dict[str, str] | None = None

    def _rows(self) -> dict[str, str]:
        if self._csv is None:
            with open(self.masks_csv, newline="") as f:
                self._csv = {r["sample_id"]: r["exvivo_instances"] for r in csv.DictReader(f)}
        return self._csv

    def __call__(self, split: str, sid: str, rec: Mapping) -> np.ndarray:
        if split == "heldout":
            with np.load(self.labels_npz) as z:
                return z[f"{sid}|exvivo"].astype(np.int32)
        from cellmatch import rle_to_labels
        shape = tuple(int(v) for v in rec["ex_shape"])
        lab, _ = rle_to_labels(json.loads(self._rows()[sid]), shape)
        return lab.astype(np.int32)


def read_exvivo(split: str, sid: str) -> np.ndarray:
    """Raw ex-vivo image as float32."""
    from cellmatch import read_image
    from hpc_unlock import prep as _prep
    return np.asarray(read_image(_prep.image_dir(split, sid) / "exvivo.tif"), np.float32)


# ----------------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------------

def build(prep: Mapping, joint_ck: Mapping, pairs_ck: Mapping, verifier_ck: Mapping,
          validate_ck: Mapping | None, match_radius: float, seed_radius: float,
          load_pred: Callable[[str, str, Mapping], np.ndarray] | None = None,
          load_image: Callable[[str, str], np.ndarray] = read_exvivo,
          per_region: int = TILES_PER_REGION,
          correct: Callable[[Mapping, np.ndarray], bool] | None = None,
          pair_estimator: Callable[[int], Any] = P.make_estimator,
          ver_estimator: Callable[[], Any] = V.make_model,
          normalize: Callable[[np.ndarray], np.ndarray] = _normalize,
          log: Callable[..., str] = checkpoint.log_line) -> tuple[dict, dict]:
    """``(checkpoint_without_side_ref, side_arrays)``. Radii are checked first."""
    check_radii(match_radius, seed_radius, log)
    load_pred = PredictedExLoader() if load_pred is None else load_pred
    pl = plan(prep, joint_ck, pairs_ck, verifier_ck, validate_ck, correct,
              pair_estimator, ver_estimator)
    arrays: dict[str, np.ndarray] = {}
    jobs_out: dict[str, dict] = {}
    for job, J in pl["jobs"].items():
        records = prep["heldout"] if J["split"] == "heldout" else prep["test"]
        pseudo = job_pseudo_labels(J, records, load_pred, match_radius, seed_radius)
        rng = np.random.default_rng(TILE_SEED)
        t_img, t_lab, t_sid = [], [], []
        per = {}
        for sid in J["regions"]:
            img = load_image(J["split"], sid)
            key = f"image|{sid}"
            if key not in arrays:
                arrays[key] = np.ascontiguousarray(img, np.float32)
            row = {"kept": sid in J["kept"], "n_tiles": 0}
            if sid in pseudo:
                lab, info = pseudo[sid]
                if lab.shape != img.shape:
                    raise ValueError(f"{sid}: label map {lab.shape} vs image {img.shape}")
                row.update(info)
                if info["n_pseudo"]:
                    ims, lbs = make_tiles(img, lab, rng, per_region, TILE, normalize)
                    t_img += ims
                    t_lab += lbs
                    t_sid += [sid] * len(ims)
                    row["n_tiles"] = len(ims)
            per[sid] = row
        train = sorted(s for s, r in per.items() if r["n_tiles"] > 0)
        if not J["kept"]:
            reason = NO_CONFIDENT
        elif not train:
            reason = NO_PSEUDO
        else:
            reason = None
        arrays[f"tiles|{job}|image"] = (np.stack(t_img).astype(np.float32) if t_img
                                        else np.zeros((0, TILE, TILE), np.float32))
        arrays[f"tiles|{job}|label"] = (np.stack(t_lab).astype(np.int32) if t_lab
                                        else np.zeros((0, TILE, TILE), np.int32))
        arrays[f"tiles|{job}|sid"] = np.array(t_sid, dtype=str) if t_sid else np.zeros(0, "<U1")
        jobs_out[job] = {"split": J["split"], "regions": list(J["regions"]),
                         "train_regions": train, "kept_regions": sorted(J["kept"]),
                         "n_tiles": len(t_img), "no_confident": reason is not None,
                         "reason": reason, "pair_threshold": float(J["pair_threshold"]),
                         "threshold_info": J["threshold_info"], "per_region": per}
        tot = {k: sum(int(r.get(k, 0)) for r in per.values())
               for k in ("retained", "seeded", "excluded", "n_matched")}
        log("SELFTRAIN_JOB", f"{job}: regions={len(J['regions'])} kept={len(J['kept'])} "
                             f"train={len(train)} tiles={len(t_img)} "
                             f"pair_thr={J['pair_threshold']:.3f} matched={tot['n_matched']} "
                             f"retained={tot['retained']} seeded={tot['seeded']} "
                             f"excluded={tot['excluded']}"
                             + (f" NO_CONFIDENT ({reason})" if reason else ""))
    mice = [j for j, v in jobs_out.items() if v["split"] == "heldout" and v["no_confident"]]
    test_nc = bool(jobs_out.get(TEST_JOB, {}).get("no_confident", False))
    n_img = sum(1 for k in arrays if k.startswith("image|"))
    img_bytes = sum(a.nbytes for k, a in arrays.items() if k.startswith("image|"))
    log("SELFTRAIN_IMAGES", f"images={n_img} float32 bytes={img_bytes} "
                            f"({img_bytes / 2**30:.2f} GiB)")
    ck = {"selection": pl["selection"], "jobs": jobs_out,
          "keys": {"image": "image|<sid>", "tiles": "tiles|<job>|image",
                   "labels": "tiles|<job>|label", "tile_sid": "tiles|<job>|sid"},
          "no_confident_regions": {"mice": mice, "test": test_nc},
          "params": {"match_radius": float(match_radius), "seed_radius": float(seed_radius),
                     "tile": TILE, "tiles_per_region": int(per_region), "tile_seed": TILE_SEED,
                     "image_dtype": "float32", "image_bytes": int(img_bytes),
                     "selection": pl["selection"]}}
    return ck, arrays


def compute(cfg, ctx) -> dict:
    t0 = time.monotonic()
    # Req 12.13: radii first, before any Checkpoint is loaded or any label is built.
    check_radii(cfg.match_radius, cfg.seed_radius)
    prep = ctx.load("prep")
    joint_ck = ctx.load("joint")
    pairs_ck = ctx.load("pairs")
    verifier_ck = ctx.load("verifier")
    validate_ck = ctx.load("validate")
    smoke = bool(getattr(ctx, "smoke", False))
    per = SMOKE_TILES_PER_REGION if smoke else TILES_PER_REGION
    ck, arrays = build(prep, joint_ck, pairs_ck, verifier_ck, validate_ck,
                       cfg.match_radius, cfg.seed_radius, per_region=per)
    ck["tiles_side"] = checkpoint.save_npz_atomic(ctx.path(SIDE_NAME), **arrays)
    ck["params"]["smoke"] = smoke
    checkpoint.log_line("SELFTRAIN_PREP_DONE",
                        f"selection={ck['selection']} "
                        f"no_confident_mice={ck['no_confident_regions']['mice']} "
                        f"test_no_confident={ck['no_confident_regions']['test']} "
                        f"elapsed={time.monotonic() - t0:.1f}s")
    return ck
