"""Self-training pairs Stage (CPU): re-pair on the self-trained ex-vivo masks,
held-out score and acceptance (Req 12.5-12.8, 12.11, 12.12).

Runs on ``n2c48m24`` after ``selftrain_gpu`` (12.6). Inputs: ``prep``, ``joint``,
``verifier``, ``validate``, ``selftrain_prep`` and ``selftrain_gpu``.

selftrain_gpu interface (read here)::

    {"jobs": {job: {"status": "done" | "skipped_no_confident" | "skipped_smoke",
                    "flows_side": npz ref (only when done),
                    "regions": [sid, ...],
                    "decode_grid": [(cellprob, flow), ...]}}}

Jobs: one per held-out mouse (job name = subject id) and ``"test"``. The job npz
holds ``<sid>|dp`` (float16), ``<sid>|cp`` (float16 ex-vivo cellprob at image
resolution), ``<sid>|n`` (int32) and ``<sid>|labels|<k>`` (uint16) for decode
setting ``k`` (the k-th entry of ``decode_grid``).

Poses and gate
--------------
Selection ``sel = selftrain_prep["selection"]``. Held-out poses
``joint["heldout"][sel]``, test poses ``joint["test"][sel]``. Region gate: the
verifier's conservative gate of that selection
(``verifier["selections"][sel]["heldout"|"test"][sid]["kept"]["conservative"]``),
the gate the registration-only ``<sel>_cons`` configuration used. Only the
ex-vivo masks change; poses, in-vivo masks and gate do not.

Region records per decode setting
---------------------------------
For every region of a mouse whose job is ``done`` and every decode setting k,
the label map is relabelled to consecutive IDs (order kept), then
``ex_c = cellmatch.region_centers(lab)``,
``ex_f = lab_build.instance_stats(lab, ex image, "exvivo", prob=cp)``,
``ex_link = validate.gt_link(lab, GT ex)`` and
``pq_ex = validate.region_pq(lab, GT ex)``. GT is used only for these labels and
scores. The work runs per region in a fork ``Pool(workers)``.

A mouse whose job is ``skipped_no_confident`` keeps its Baseline records and
Baseline per-region PQ_ex (``validate["pq"]["per_region"]``) everywhere (12.12).

Decode setting choice (no label leakage, Property 15)
-----------------------------------------------------
For held-out mouse m and each setting k, the score is the held-out full score
over the *other* mice only: their records at setting k (Baseline for
no-confident mice), pair probabilities from ``pairs.loo_predict`` over those
records (each other mouse o is scored by a model trained on the mice outside
``{m, o}``), the pair threshold from ``pairs.choose_threshold`` over their kept
regions, ``full = 0.25*PQ_iv + 0.25*PQ_ex + 0.5*F1`` over their regions. m's
setting is the best k (lowest k on ties). Nothing of m's ground truth enters
m's choice. The test setting is the best k of the same score over all three
mice at a uniform setting (``per_setting`` table).

Held-out score (12.5)
---------------------
All 47 records, each mouse at its own chosen setting (Baseline masks for
no-confident mice); ``pairs.loo_predict`` (mouse m scored by a model trained on
the other mice's rows), threshold from ``pairs.choose_threshold`` over the kept
regions, ``pairs.select``, ``validate.score_selection`` with the Baseline
in-vivo PQ (``validate["pq"]``) and the new mean PQ_ex.

Acceptance (12.7, 12.8, Property 19)
------------------------------------
Target = the highest full of the *accepted* registration-only configurations in
``validate["configs"]``, else ``validate["baseline"]["full"]``. Accepted iff
``full > target`` (unrounded).

Test pairs (12.6, 12.8)
-----------------------
Only when accepted and the test job is ``done``: the test job's labels at the
test setting, relabelled; ``ex_c`` / ``ex_f`` from those (ungrown) labels and the
test ``exvivo.tif`` (the v10 convention: pairs from ungrown masks, grown masks
written, IDs unchanged); with ``cfg.st_grow15`` the labels are then grown once
with ``size_lab.grow(lab, prob=cp, frac=0.15)`` (the v7_grow15 ring grow). The
pair model is trained on all 47 held-out records at the test setting
(``pairs.test_model`` rule, ``pairs.fit`` on ``pairs.dataset``), the threshold
is the one chosen at that setting, the gate is the test conservative gate.
``pairs.select`` gives one-to-one index pairs ``(i, j)``: ``i`` into the record's
``iv_ids``, ``j`` into the new labels (label ``j + 1``). Test labels (int32,
grown when ``st_grow15``) go to ``selftrain_test_labels.npz`` (key ``<sid>``).

Statuses
--------
* ``"skipped"``: a selftrain_gpu job is ``skipped_smoke`` (no fine-tune in smoke).
* ``"no_confident_regions"``: the test job is ``skipped_no_confident`` (12.11;
  this also covers "every job no-confident").
* ``"accepted"`` / ``"no_candidate"`` otherwise.

Checkpoint (the interface ``assemble.py`` reads, plus diagnostics)::

    {"accepted", "status", "full", "f1", "pq_iv", "pq_ex", "kept", "kept_wrong",
     "per_mouse_f1", "compared_to", "compared_to_name", "reason",
     "no_confident_mice", "test_labels_side" (ref | None),
     "test_pairs": {sid: [[i, j], ...]}, "selection", "pair_threshold",
     "tp", "pred", "decode_by_mouse": {mouse: {...}}, "per_setting": [...],
     "test_setting": {...} | None, "per_region": {sid: {...}}, "meta": {...}}
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hpc_unlock import paths  # noqa: F401  (puts ROOT and research/ on sys.path)
from hpc_unlock import checkpoint
from hpc_unlock import pairs as P
from hpc_unlock import validate as VAL
from hpc_unlock import selftrain_prep as SP
from cellmatch import region_centers  # noqa: E402
from lab_build import instance_stats  # noqa: E402

DONE = "done"
SKIP_NO_CONFIDENT = "skipped_no_confident"
SKIP_SMOKE = "skipped_smoke"
JOB_STATUSES = (DONE, SKIP_NO_CONFIDENT, SKIP_SMOKE)
TEST_JOB = SP.TEST_JOB
TEST_LABELS_NPZ = "selftrain_test_labels.npz"
GROW_FRAC = 0.15
NO_CONFIDENT = SP.NO_CONFIDENT
NO_CANDIDATE = VAL.NO_CANDIDATE


# ----------------------------------------------------------------------------
# Injectable data sources
# ----------------------------------------------------------------------------

def default_grow(lab: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """One 15 % cellprob-ranked ring grow (``research/size_lab.grow``, v7_grow15)."""
    from size_lab import grow  # noqa: E402
    return grow(lab, prob=prob, frac=GROW_FRAC)


class GroundTruthEx:
    """GT ex-vivo label maps from ``train_ground_truth.csv`` (scoring / labels only)."""

    def __init__(self, truth: Mapping | None = None):
        self._truth = truth

    def __call__(self, sid: str, shape: tuple[int, int]) -> np.ndarray:
        R = VAL._research()
        if self._truth is None:
            self._truth = R.common.load_truth()
        lab, _ = R.common.label_map(self._truth[sid]["exvivo_instances"], tuple(shape))
        return lab


@dataclass
class Sources:
    """Everything that touches files; tests pass in-memory stand-ins."""
    read_image: Callable[[str, str], np.ndarray]          # (split, sid) -> ex-vivo image
    gt_ex: Callable[[str, tuple], np.ndarray]             # (sid, shape) -> GT ex labels
    job_arrays: Callable[[str, Mapping], Mapping]         # (job, gpu job entry) -> arrays
    grow: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_grow


def default_sources(run_dir: str | Path) -> Sources:
    def job_arrays(job: str, entry: Mapping) -> dict:
        return checkpoint.load_npz(entry["flows_side"], run_dir)
    return Sources(read_image=SP.read_exvivo, gt_ex=GroundTruthEx(), job_arrays=job_arrays)


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------

def relabel(lab: np.ndarray) -> np.ndarray:
    """Consecutive IDs 1..n in ascending original-ID order (background 0), int32."""
    lab = np.asarray(lab)
    if lab.size == 0:
        return lab.astype(np.int32)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(int(lab.max()) + 1 if lab.max() > 0 else 1, np.int32)
    lut[ids.astype(np.int64)] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[np.clip(lab, 0, None).astype(np.int64)]


def label_settings(arrays: Mapping, sid: str) -> list[int]:
    """Decode-setting keys ``k`` present as ``<sid>|labels|<k>`` (sorted)."""
    pre = f"{sid}|labels|"
    return sorted(int(k[len(pre):]) for k in arrays if k.startswith(pre))


def region_features(lab: np.ndarray, image: np.ndarray, cp: np.ndarray,
                    gt: np.ndarray | None) -> dict:
    """``ex_c``, ``ex_f`` (and ``ex_link``, ``pq_ex`` when GT is given) of one label map."""
    lab = np.asarray(lab)
    if lab.shape != np.asarray(image).shape or lab.shape != np.asarray(cp).shape:
        raise ValueError(f"labels {lab.shape} vs image {np.shape(image)} vs cp {np.shape(cp)}")
    out = {"ex_c": region_centers(lab),
           "ex_f": instance_stats(lab, image, "exvivo", prob=np.asarray(cp, np.float32)),
           "n": int(lab.max()) if lab.size else 0}
    if gt is not None:
        if np.shape(gt) != lab.shape:
            raise ValueError(f"labels {lab.shape} vs GT {np.shape(gt)}")
        out["ex_link"] = VAL.gt_link(lab, gt)
        out["pq_ex"] = float(VAL.region_pq(lab, gt)[0])
    return out


def comparison_target(validate_ck: Mapping) -> tuple[float, str]:
    """Best accepted registration-only full, else the measured Baseline full (12.7)."""
    acc = [c for c in validate_ck.get("configs", []) or [] if c.get("accepted")]
    if acc:
        best = max(acc, key=lambda c: float(c["full"]))
        return float(best["full"]), str(best.get("name"))
    return float(validate_ck["baseline"]["full"]), "baseline"


def best_setting(scores: Mapping[int, float]) -> int | None:
    """Highest score; the lowest setting key on ties."""
    best = None
    for k in sorted(scores):
        if best is None or scores[k] > scores[best]:
            best = k
    return best


def _pairs_list(sel) -> list[list[int]]:
    return [[int(i), int(j)] for i, j in np.asarray(sel, int).reshape(-1, 2)]


# ----------------------------------------------------------------------------
# Pool over regions (fork; state shared through a module global)
# ----------------------------------------------------------------------------

_G: dict = {}


def _heldout_region(item: tuple) -> tuple[str, dict]:
    job, sid, settings = item
    arrays, src = _G["arrays"][job], _G["sources"]
    image = src.read_image("heldout", sid)
    cp = np.asarray(arrays[f"{sid}|cp"], np.float32)
    gt = src.gt_ex(sid, tuple(np.shape(image)))
    return sid, {k: region_features(relabel(arrays[f"{sid}|labels|{k}"]), image, cp, gt)
                 for k in settings}


def _test_region(item: tuple) -> tuple[str, dict, np.ndarray]:
    sid, k, grow15 = item
    arrays, src = _G["arrays"][TEST_JOB], _G["sources"]
    image = src.read_image("test", sid)
    cp = np.asarray(arrays[f"{sid}|cp"], np.float32)
    lab = relabel(arrays[f"{sid}|labels|{k}"])
    feats = region_features(lab, image, cp, None)
    if grow15:
        lab = np.asarray(src.grow(lab, cp))
    return sid, feats, lab.astype(np.int32)


def _map(fn: Callable, items: Sequence, workers: int) -> list:
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from hpc_unlock import prep as _prep
    ctx = mp.get_context("fork")
    with ctx.Pool(min(int(workers), len(items)), initializer=_prep._init_worker) as pool:
        return pool.map(fn, items, chunksize=1)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def records_for(base: Mapping[str, Mapping], feats: Mapping[str, Mapping[int, dict]],
                base_pq_ex: Mapping[str, float], choice: Mapping[str, int | None],
                mice: Sequence[str] | None = None) -> tuple[dict, dict]:
    """Records and per-region PQ_ex with each mouse's ex masks at ``choice[mouse]``.

    ``None`` (or a mouse without features) keeps the Baseline record and PQ_ex.
    Only regions of ``mice`` are returned (all when ``None``).
    """
    recs, pq_ex = {}, {}
    for sid in sorted(base):
        r = base[sid]
        g = str(r["subject"])
        if mice is not None and g not in mice:
            continue
        k = choice.get(g)
        if k is None or sid not in feats:
            recs[sid], pq_ex[sid] = r, float(base_pq_ex[sid])
        else:
            f = feats[sid][k]
            recs[sid] = {**r, "ex_c": f["ex_c"], "ex_f": f["ex_f"], "ex_link": f["ex_link"]}
            pq_ex[sid] = float(f["pq_ex"])
    return recs, pq_ex


def score(records: Mapping[str, Mapping], poses: Mapping[str, Any],
          kept: Mapping[str, bool], correct: Mapping[str, bool | None],
          pq_iv: Mapping[str, float], pq_ex: Mapping[str, float],
          estimator: Callable[[int], Any] = P.make_estimator, seed: int = 0) -> dict:
    """LOO pairs + threshold + full score over exactly ``records`` (Req 8.1, 8.2, 9.1)."""
    sids = sorted(records)
    if not sids:
        raise ValueError("no regions to score")
    rows = P.dataset(records, {s: poses.get(s) for s in sids})
    probs = P.loo_predict(rows, {s: str(records[s]["subject"]) for s in sids}, seed, estimator)
    kept_s = {s: bool(kept.get(s, False)) for s in sids}
    total = int(sum(int(records[s]["n_gt_pairs"]) for s in sids))
    thr = float(P.choose_threshold(rows, probs, kept_s, total)[0])
    selected = {sid: P.select(pairs, probs[sid], thr, kept_s[sid]) for sid, pairs, _X, _y in rows}
    pq = {"pq_iv": float(np.mean([pq_iv[s] for s in sids])),
          "pq_ex": float(np.mean([pq_ex[s] for s in sids]))}
    m = VAL.score_selection(records, selected, pq, kept_s, correct)
    m.update(pair_threshold=thr, selected=selected, regions=len(sids))
    return m


def _summary(m: Mapping) -> dict:
    return {k: m[k] for k in ("full", "f1", "pq_iv", "pq_ex", "tp", "pred", "kept",
                              "kept_wrong", "pair_threshold", "per_mouse_f1", "regions")}


# ----------------------------------------------------------------------------
# Stage core
# ----------------------------------------------------------------------------

def _job_status(jobs: Mapping, job: str) -> str:
    if job not in jobs:
        raise ValueError(f"selftrain_gpu has no job {job!r}")
    st = str(jobs[job].get("status"))
    if st not in JOB_STATUSES:
        raise ValueError(f"selftrain_gpu job {job!r} has unknown status {st!r}")
    return st


def _decode(grid: Sequence, settings: Sequence[int], k: int | None):
    if k is None or not grid:
        return None
    pos = list(settings).index(k)
    return [float(v) for v in grid[pos]]


def _base_result(target: float, target_name: str, sel: str | None, **kw) -> dict:
    out = {"accepted": False, "status": None, "full": None, "f1": None, "pq_iv": None,
           "pq_ex": None, "kept": None, "kept_wrong": None, "per_mouse_f1": {},
           "compared_to": float(target), "compared_to_name": target_name, "reason": None,
           "no_confident_mice": [], "test_labels_side": None, "test_pairs": {},
           "selection": sel, "pair_threshold": None, "tp": None, "pred": None,
           "decode_by_mouse": {}, "per_setting": [], "test_setting": None,
           "per_region": {}, "meta": {}}
    out.update(kw)
    return out


def run(prep: Mapping, joint_ck: Mapping, verifier_ck: Mapping, validate_ck: Mapping,
        st_prep: Mapping, st_gpu: Mapping, sources: Sources, *, grow15: bool = True,
        workers: int = 1, smoke: bool = False,
        estimator: Callable[[int], Any] = P.make_estimator,
        log: Callable[..., str] = checkpoint.log_line) -> tuple[dict, dict]:
    """``(checkpoint without side ref, test label arrays)``; arrays only when accepted."""
    t0 = time.monotonic()
    sel = st_prep["selection"]
    target, target_name = comparison_target(validate_ck)
    jobs = st_gpu.get("jobs", {}) or {}
    meta = {"smoke": bool(smoke), "grow15": bool(grow15), "workers": int(workers)}

    if any(str(j.get("status")) == SKIP_SMOKE for j in jobs.values()) or (smoke and not any(
            str(j.get("status")) == DONE for j in jobs.values())):
        reason = "selftrain_gpu skipped (smoke: no fine-tune); no self-trained candidate"
        log("SELFTRAIN_PAIRS_SKIPPED", reason)
        return _base_result(target, target_name, sel, status="skipped", reason=reason,
                            meta=meta), {}

    heldout = prep["heldout"]
    test = prep.get("test", {}) or {}
    mice = sorted({str(r["subject"]) for r in heldout.values()})
    status = {m: _job_status(jobs, m) for m in mice}
    done = [m for m in mice if status[m] == DONE]
    no_conf = [m for m in mice if status[m] == SKIP_NO_CONFIDENT]
    test_status = _job_status(jobs, TEST_JOB) if test else SKIP_NO_CONFIDENT

    ver = verifier_ck["selections"][sel]
    hposes = joint_ck["heldout"][sel]
    kept = {s: bool(ver["heldout"][s]["kept"]["conservative"]) for s in heldout}
    correct = {s: ver["heldout"][s].get("correct") for s in heldout}
    per_pq = validate_ck["pq"]["per_region"]
    pq_iv = {s: float(per_pq[s]["iv"]) for s in heldout}
    base_pq_ex = {s: float(per_pq[s]["ex"]) for s in heldout}

    # ---- rebuild ex features for every region of every done mouse
    grid, settings = None, None
    feats: dict[str, dict[int, dict]] = {}
    if done:
        arrays = {m: sources.job_arrays(m, jobs[m]) for m in done}
        items = []
        for m in done:
            g = [tuple(v) for v in jobs[m].get("decode_grid") or []]
            regions = sorted(s for s, r in heldout.items() if str(r["subject"]) == m)
            for sid in regions:
                ks = label_settings(arrays[m], sid)
                if not ks:
                    raise ValueError(f"job {m}: no label maps for {sid}")
                if settings is None:
                    settings, grid = ks, g
                if ks != settings or g != grid:
                    raise ValueError(f"job {m} / {sid}: decode settings {ks} {g} differ "
                                     f"from {settings} {grid}")
                items.append((m, sid, ks))
        if grid and len(grid) != len(settings):
            raise ValueError(f"decode_grid has {len(grid)} entries, labels {settings}")
        _G.update(arrays=arrays, sources=sources)
        try:
            feats = dict(_map(_heldout_region, items, workers))
        finally:
            _G.clear()
        del arrays
        log("SELFTRAIN_PAIRS_FEATURES", f"mice={done} regions={len(items)} settings={settings} "
                                        f"elapsed={time.monotonic() - t0:.1f}s")

    # ---- decode setting per mouse, chosen on the other mice only
    decode_by_mouse: dict[str, dict] = {}
    for m in mice:
        if m not in done:
            decode_by_mouse[m] = {"setting": None, "decode": None, "baseline_masks": True,
                                  "reason": NO_CONFIDENT, "scores": {}}
            continue
        others = [o for o in mice if o != m]
        scores = {}
        for k in settings:
            if others:
                recs, pqx = records_for(heldout, feats, base_pq_ex,
                                        {o: k for o in others if o in done}, others)
                scores[k] = float(score(recs, hposes, kept, correct, pq_iv, pqx,
                                        estimator)["full"])
            else:
                scores[k] = 0.0
        k_m = best_setting(scores)
        decode_by_mouse[m] = {"setting": k_m, "decode": _decode(grid, settings, k_m),
                              "baseline_masks": False, "reason": None, "scores": scores}
        log("SELFTRAIN_PAIRS_FOLD", f"{m}: setting={k_m} decode={_decode(grid, settings, k_m)} "
                                    + " ".join(f"k{k}={v:.4f}" for k, v in scores.items()))

    # ---- uniform settings over all three mice (test choice + diagnostics)
    per_setting = []
    for k in settings or []:
        recs, pqx = records_for(heldout, feats, base_pq_ex, {m: k for m in done})
        s = _summary(score(recs, hposes, kept, correct, pq_iv, pqx, estimator))
        s.update(setting=k, decode=_decode(grid, settings, k))
        per_setting.append(s)
        log("SELFTRAIN_PAIRS_SETTING", f"k={k} decode={s['decode']} full={s['full']:.4f} "
                                       f"f1={s['f1']:.4f} pq_ex={s['pq_ex']:.4f} "
                                       f"thr={s['pair_threshold']:.3f}")

    # ---- held-out score with each mouse's own setting (12.5, 12.12)
    choice = {m: decode_by_mouse[m]["setting"] for m in mice}
    recs, pqx = records_for(heldout, feats, base_pq_ex, choice)
    final = score(recs, hposes, kept, correct, pq_iv, pqx, estimator)
    per_region = {s: {"subject": str(heldout[s]["subject"]), "pq_iv": pq_iv[s],
                      "pq_ex": pqx[s], "baseline_pq_ex": base_pq_ex[s],
                      "setting": choice[str(heldout[s]["subject"])], "kept": kept[s],
                      "n_pairs": int(len(final["selected"][s]))} for s in sorted(heldout)}
    full = float(final["full"])
    accepted = VAL.accept(full, target)
    res = _base_result(
        target, target_name, sel, full=full, f1=float(final["f1"]),
        pq_iv=float(final["pq_iv"]), pq_ex=float(final["pq_ex"]), kept=int(final["kept"]),
        kept_wrong=int(final["kept_wrong"]), per_mouse_f1=dict(final["per_mouse_f1"]),
        no_confident_mice=no_conf, pair_threshold=float(final["pair_threshold"]),
        tp=int(final["tp"]), pred=int(final["pred"]), decode_by_mouse=decode_by_mouse,
        per_setting=per_setting, per_region=per_region,
        meta=dict(meta, decode_grid=[list(map(float, g)) for g in grid or []],
                  settings=list(settings or []), job_status=dict(status, test=test_status)))
    log("SELFTRAIN_PAIRS_HELDOUT", f"full={full!r} f1={final['f1']:.4f} pq_iv={final['pq_iv']:.4f} "
                                   f"pq_ex={final['pq_ex']:.4f} kept={final['kept']} "
                                   f"kept_wrong={final['kept_wrong']} thr={final['pair_threshold']:.3f} "
                                   f"no_confident_mice={no_conf} compared_to={target!r} "
                                   f"({target_name})")

    if test_status == SKIP_NO_CONFIDENT:
        res.update(status="no_confident_regions", accepted=False, reason=NO_CONFIDENT)
        log("NO_CANDIDATE", f"selftrain: {NO_CONFIDENT} (test job skipped)")
        return res, {}
    if not accepted:
        res.update(status="no_candidate", accepted=False,
                   reason=f"{NO_CANDIDATE}: self-trained held-out full {full!r} <= "
                          f"compared {target!r} ({target_name})")
        log("NO_CANDIDATE", f"selftrain: full={full!r} compared_to={target!r} ({target_name})")
        return res, {}

    # ---- test pairs (12.6, 12.8)
    if per_setting:
        best = max(per_setting, key=lambda s: (s["full"], -s["setting"]))
        k_t, thr_t = best["setting"], float(best["pair_threshold"])
    else:
        k_t, thr_t = None, float(final["pair_threshold"])
    t_arrays = sources.job_arrays(TEST_JOB, jobs[TEST_JOB])
    t_regions = sorted(jobs[TEST_JOB].get("regions") or test)
    t_settings = label_settings(t_arrays, t_regions[0]) if t_regions else []
    if k_t is None:
        k_t = t_settings[0]
    if k_t not in t_settings:
        raise ValueError(f"test job has no labels for setting {k_t} (has {t_settings})")
    _G.update(arrays={TEST_JOB: t_arrays}, sources=sources)
    try:
        out = _map(_test_region, [(sid, k_t, bool(grow15)) for sid in t_regions], workers)
    finally:
        _G.clear()
    del t_arrays
    recs_k, _ = records_for(heldout, feats, base_pq_ex, {m: k_t for m in done})
    model = P.fit(P.dataset(recs_k, {s: hposes.get(s) for s in sorted(recs_k)}), 0, estimator)
    tposes = joint_ck.get("test", {}).get(sel, {})
    tver = ver.get("test", {})
    test_pairs, labels = {}, {}
    for sid, f, lab in out:
        rec = {**test[sid], "ex_c": f["ex_c"], "ex_f": f["ex_f"]}
        (_, cand, X, _), = P.dataset({sid: rec}, {sid: tposes.get(sid)}, with_labels=False)
        k_ok = bool(((tver.get(sid) or {}).get("kept") or {}).get("conservative"))
        test_pairs[sid] = _pairs_list(P.select(cand, P.predict(model, X), thr_t, k_ok))
        labels[sid] = lab
    res.update(status="accepted", accepted=True, reason=None, test_pairs=test_pairs,
               test_setting={"setting": k_t, "decode": _decode(grid, settings, k_t)
                             if settings else None, "pair_threshold": thr_t,
                             "grow15": bool(grow15), "regions": len(t_regions)})
    log("SELFTRAIN_ACCEPTED", f"full={full!r} > {target!r} ({target_name}); test setting={k_t} "
                              f"thr={thr_t:.3f} grow15={bool(grow15)} "
                              f"pairs={sum(len(v) for v in test_pairs.values())} "
                              f"regions_with_pairs={sum(bool(v) for v in test_pairs.values())} "
                              f"elapsed={time.monotonic() - t0:.1f}s")
    return res, labels


def compute(cfg, ctx) -> dict:
    prep = ctx.load("prep")
    joint_ck = ctx.load("joint")
    verifier_ck = ctx.load("verifier")
    validate_ck = ctx.load("validate")
    st_prep = ctx.load("selftrain_prep")
    st_gpu = ctx.load("selftrain_gpu")
    res, labels = run(prep, joint_ck, verifier_ck, validate_ck, st_prep, st_gpu,
                      default_sources(ctx.run_dir), grow15=bool(cfg.st_grow15),
                      workers=int(getattr(ctx, "workers", 1) or 1),
                      smoke=bool(getattr(ctx, "smoke", False)))
    if res["accepted"]:
        res["test_labels_side"] = checkpoint.save_npz_atomic(ctx.path(TEST_LABELS_NPZ), **labels)
    return res
