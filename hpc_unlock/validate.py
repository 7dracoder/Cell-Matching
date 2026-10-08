"""Held-out validation: PQ / pair F1 / full score (Req 9.1, 9.2).

Metric layer
------------
* ``region_pq``  per-region panoptic quality via ``cellmatch.pq_score``
  (a prediction matches a GT instance when IoU > 0.75).
* ``mean_pq``    PQ averaged over regions (47 held-out regions).
* ``gt_link``    predicted instance index (0-based) -> GT instance index
  (0-based) at IoU > 0.75, -1 otherwise (``research/lab_build.gt_link``).
* ``pair_f1``    pooled pair F1. TP, FP and FN are summed over regions first.
  A predicted pair (i, j) is a TP only if both masks are PQ TPs
  (``iv_link[i] >= 0`` and ``ex_link[j] >= 0``) and
  ``(iv_link[i], ex_link[j])`` is a GT pair. FN = TOTAL - TP, where TOTAL is
  every GT pair (``n_gt_pairs``, 1,139 on the held-out set), so
  ``F1 = 2TP / (2TP + FP + FN) = 2TP / (pred + TOTAL)``, exactly the
  ``research/lab_eval.score`` formula.
* ``full_score`` ``0.25*PQ_iv + 0.25*PQ_ex + 0.5*F1``.

Baseline reproduction (Req 9.3, 9.4)
------------------------------------
The exact v10 recipe (``research/paired_shrink_cv.py::pipeline``) on the prep
held-out records: ``cp_pose_lab.pose_choose(rec["cp_scored"], "score")`` gives
``(M, refine_score, z)``; the region is kept iff
``margin_lab.margin(rec["vote_cands"], M, score, rec) >= 3 or z >= 5``; pair
probabilities come from ``pairs.dataset`` / ``pairs.loo_predict`` (equal to
``pair_clf.dataset`` / ``loo_predict``) and pairs with ``prob >= 0.025`` in kept
regions are selected. PQ_iv / PQ_ex are measured from ``heldout_labels.npz``
(``<sid>|invivo``, ``<sid>|exvivo``, ungrown) against the GT label maps, once,
and reused by every registration-only configuration (masks are unchanged).
Expected: F1 0.472, full 0.5186 (= 0.25*(0.7404 + 0.3901) + 0.5*0.472), each
within 0.005. Outside the tolerance the Stage logs
``BASELINE_REPRODUCTION_FAILED``, writes ``validate_failure.json`` atomically
into the run directory and raises ``SystemExit(1)`` before any marker. Smoke
runs skip the check (``reproduced = None``).

Baseline test pairs reproduce ``submission_v10_cpgate.csv`` (``test_cp_apply.py``):
pose ``v10_window["M_true"]``, gate ``margin(vote_cands, M, score) >= 3`` or
``cp_z(cp_bin, iv_c, M_true) >= 5``, classifier ``pairs.v10_test_model()``,
threshold 0.03, candidates ``pair_clf.candidates``. They are compared per
region with the CSV's ``match_pairs`` (``v10_csv_match``).

Configurations (Req 9.5, 9.6, 8.5)
----------------------------------
``indep_cons``, ``indep_aggr``, ``joint_cons``, ``joint_aggr`` =
{independent, joint} x {conservative, aggressive}. Kept regions come from the
verifier, the pair threshold from ``pairs.choose_threshold`` over the kept
regions only (13 values, lowest on ties), the held-out pairs from
``pairs.select`` on the pairs Stage's LOO probabilities. A configuration is
accepted iff ``full > baseline_full`` (unrounded). Test pairs use the pairs
Stage's all-mice probabilities, the configuration's threshold and its test gate.

Checkpoint (``validate.pkl``; names follow ``report.py``'s state schema)::

    {"baseline": {"full", "f1", "pq_iv", "pq_ex", "tp", "pred", "kept",
                  "kept_wrong", "per_mouse_f1", "pair_threshold",
                  "reproduced" (bool | None), "expected", "regions",
                  "test_pairs": {sid: [[i, j], ...]}, "v10_csv_match": {...}},
     "configs": [ConfigResult dicts],
     "pq": {"pq_iv", "pq_ex", "n_regions", "per_region": {sid: {"iv", "ex"}}},
     "meta": {...}}
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from hpc_unlock import paths  # noqa: F401  (puts ROOT and research/ on sys.path)
from hpc_unlock import checkpoint
from hpc_unlock import pairs as P
from hpc_unlock import prep as _prep
from cellmatch import pq_score  # noqa: E402
from lab_build import gt_link as _research_gt_link  # noqa: E402

PQ_IOU = 0.75
W_PQ_IV, W_PQ_EX, W_F1 = 0.25, 0.25, 0.5

Pair = tuple[int, int]


# --------------------------------------------------------------------------- PQ
def region_pq(pred: np.ndarray, gt: np.ndarray) -> tuple[float, int, int, int]:
    """``(pq, tp, fp, fn)`` for one region's label maps (IoU > 0.75 rule)."""
    return pq_score(np.asarray(pred), np.asarray(gt), cutoff=PQ_IOU)


def mean_pq(pred_by_region: Mapping[str, np.ndarray],
            gt_by_region: Mapping[str, np.ndarray]) -> float:
    """PQ computed per region, then averaged over regions (Req 9.1).

    Both mappings must cover exactly the same region IDs.
    """
    if set(pred_by_region) != set(gt_by_region):
        missing = sorted(set(pred_by_region) ^ set(gt_by_region))
        raise ValueError(f"prediction / GT region sets differ: {missing[:5]}")
    if not gt_by_region:
        raise ValueError("no regions to score")
    return float(np.mean([region_pq(pred_by_region[s], gt_by_region[s])[0]
                          for s in sorted(gt_by_region)]))


# -------------------------------------------------------------------- pair F1
def gt_link(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Predicted index (0-based) -> GT index (0-based) at IoU > 0.75, else -1."""
    return _research_gt_link(np.asarray(pred), np.asarray(gt))


@dataclass(frozen=True)
class PairScore:
    tp: int
    fp: int
    fn: int
    f1: float

    @property
    def pred(self) -> int:
        return self.tp + self.fp


def region_pair_tp(pairs: Iterable[Pair], iv_link: Sequence[int],
                   ex_link: Sequence[int], gt_pairs: set[Pair]) -> int:
    """Count TP pairs in one region (both masks PQ TPs and a GT pair)."""
    tp = 0
    for i, j in pairs:
        a, b = int(iv_link[i]), int(ex_link[j])
        tp += a >= 0 and b >= 0 and (a, b) in gt_pairs
    return tp


def pair_f1(pred_pairs_by_region: Mapping[str, Iterable[Pair]],
            iv_links: Mapping[str, Sequence[int]],
            ex_links: Mapping[str, Sequence[int]],
            gt_pairs: Mapping[str, set[Pair]],
            total_gt_pairs: int | Mapping[str, int]) -> PairScore:
    """Pooled pair F1 over regions (Req 9.2).

    ``total_gt_pairs`` is either the total GT pair count (e.g. 1,139) or a
    per-region ``n_gt_pairs`` mapping, which is summed (use a subset of regions
    for per-mouse F1). Regions absent from ``pred_pairs_by_region`` contribute
    only FN. F1 is 0.0 when there are neither predictions nor GT pairs.
    """
    total = (int(sum(total_gt_pairs.values())) if isinstance(total_gt_pairs, Mapping)
             else int(total_gt_pairs))
    tp = pred = 0
    for sid, pairs in pred_pairs_by_region.items():
        pairs = list(pairs)
        tp += region_pair_tp(pairs, iv_links[sid], ex_links[sid], gt_pairs[sid])
        pred += len(pairs)
    if tp > total:
        raise ValueError(f"TP {tp} exceeds GT pair total {total}")
    denom = pred + total
    return PairScore(tp, pred - tp, total - tp, 2 * tp / denom if denom else 0.0)


# ---------------------------------------------------------------------- full
def full_score(pq_iv: float, pq_ex: float, f1: float) -> float:
    """``0.25*PQ_iv + 0.25*PQ_ex + 0.5*F1`` (Req 9.1)."""
    return W_PQ_IV * pq_iv + W_PQ_EX * pq_ex + W_F1 * f1


# =========================================================================
# Baseline reproduction, configurations and acceptance (Req 9.3-9.6, 8.5)
# =========================================================================

EXPECTED = {"f1": 0.472, "full": 0.5186, "tol": 0.005}
BASELINE_PAIR_THR = 0.025        # paired_shrink_cv.THR (held-out v10)
V10_TEST_PAIR_THR = 0.03         # threshold behind submission_v10_cpgate.csv
V10_MARGIN = 3.0
V10_Z = 5.0
CORRECT_ERR = 5.0                # reg_lab.err < 5 px = Correct_Pose
GATES = ("conservative", "aggressive")
CONFIGS = (("indep_cons", "independent", "conservative"),
           ("indep_aggr", "independent", "aggressive"),
           ("joint_cons", "joint", "conservative"),
           ("joint_aggr", "joint", "aggressive"))
FAILURE_JSON = "validate_failure.json"
LABELS_NPZ = paths.RDATA / "heldout_labels.npz"
V10_CSV = paths.ROOT / "submission_v10_cpgate.csv"
NO_CANDIDATE = "NO_CANDIDATE"

csv.field_size_limit(sys.maxsize)

_RESEARCH: SimpleNamespace | None = None


def _research() -> SimpleNamespace:
    """Research modules behind the v10 recipe (imported with argv hidden)."""
    global _RESEARCH
    if _RESEARCH is None:
        with _prep._research_import_env():
            import common
            import cp_pose_lab
            import margin_lab
            import reg_lab
        _RESEARCH = SimpleNamespace(common=common, cp_pose_lab=cp_pose_lab,
                                    margin_lab=margin_lab, reg_lab=reg_lab)
    return _RESEARCH


@dataclass
class ConfigResult:
    """One evaluated registration-only configuration (plain dict via ``asdict``)."""
    name: str
    selection: str
    gate: str
    tau: float | None
    conservative_unavailable: bool
    pair_threshold: float
    full: float
    f1: float
    pq_iv: float
    pq_ex: float
    tp: int
    pred: int
    per_mouse_f1: dict
    kept: int
    kept_wrong: int
    accepted: bool
    compared_to: float
    rejected_reason: str | None
    test_pairs: dict = field(default_factory=dict)


# -------------------------------------------------------------- small rules
def accept(full: float, baseline_full: float) -> bool:
    """Acceptance (Req 9.5): strictly greater, compared unrounded."""
    return float(full) > float(baseline_full)


def rejected_reason(full: float, baseline_full: float) -> str | None:
    if accept(full, baseline_full):
        return None
    return (f"{NO_CANDIDATE}: held-out full {float(full)!r} <= "
            f"Baseline full {float(baseline_full)!r}")


def reproduction_ok(f1: float, full: float, expected: Mapping = EXPECTED) -> bool:
    """Both measured values within ``tol`` of the expected Baseline (Req 9.3)."""
    tol = float(expected["tol"])
    return (abs(float(f1) - float(expected["f1"])) <= tol
            and abs(float(full) - float(expected["full"])) <= tol)


def v10_gate(margin: float, z: float) -> bool:
    return bool(float(margin) >= V10_MARGIN or float(z) >= V10_Z)


def _pairs_list(sel) -> list[list[int]]:
    return [[int(i), int(j)] for i, j in np.asarray(sel, int).reshape(-1, 2)]


def default_correct() -> Callable[[Mapping, Any], bool | None]:
    """``reg_lab.err(rec, M) < 5`` for regions with a GT affine, else None."""
    err = _research().reg_lab.err

    def correct(rec: Mapping, M) -> bool | None:
        if rec.get("gt_M") is None or rec.get("gt_iv_c") is None:
            return None
        if M is None:
            return False
        return bool(err(rec, np.asarray(M, float)) < CORRECT_ERR)
    return correct


# ---------------------------------------------------------------- PQ (once)
def heldout_pq(records: Mapping[str, Mapping], labels_npz: str | Path = LABELS_NPZ,
               truth: Mapping | None = None) -> dict:
    """Per-region and mean PQ of the ungrown held-out masks (Req 9.1).

    Predictions: ``heldout_labels.npz[<sid>|invivo]`` / ``[<sid>|exvivo]``;
    GT: ``common.label_map`` of ``train_ground_truth.csv``.
    """
    if not records:
        raise ValueError("no held-out regions to score")
    R = _research()
    truth = R.common.load_truth() if truth is None else truth
    per = {}
    with np.load(labels_npz) as z:
        for sid in sorted(records):
            out = {}
            for side, key in (("invivo", "iv"), ("exvivo", "ex")):
                pred = z[f"{sid}|{side}"]
                gt, _ = R.common.label_map(truth[sid][f"{side}_instances"], pred.shape)
                out[key] = float(region_pq(pred, gt)[0])
            per[sid] = out
    return {"pq_iv": float(np.mean([v["iv"] for v in per.values()])),
            "pq_ex": float(np.mean([v["ex"] for v in per.values()])),
            "n_regions": len(per), "per_region": per}


# ------------------------------------------------------------ scoring
def score_selection(records: Mapping[str, Mapping], selected: Mapping[str, Any],
                    pq: Mapping, kept: Mapping[str, bool],
                    correct: Mapping[str, bool | None]) -> dict:
    """Pooled and per-mouse F1, full score, kept and kept-wrong (Req 9.1, 9.2, 9.6).

    ``selected``: ``sid -> (k, 2)`` pairs for every held-out region (empty if dropped).
    """
    sids = list(records)
    links_iv = {s: records[s]["iv_link"] for s in sids}
    links_ex = {s: records[s]["ex_link"] for s in sids}
    gtp = {s: records[s]["gt_pairs"] for s in sids}
    n_gt = {s: int(records[s]["n_gt_pairs"]) for s in sids}
    sel = {s: [tuple(p) for p in _pairs_list(selected.get(s, []))] for s in sids}
    total = pair_f1(sel, links_iv, links_ex, gtp, n_gt)
    per_mouse = {}
    for g in sorted({str(records[s]["subject"]) for s in sids}):
        ms = [s for s in sids if str(records[s]["subject"]) == g]
        per_mouse[g] = pair_f1({s: sel[s] for s in ms}, links_iv, links_ex, gtp,
                               {s: n_gt[s] for s in ms}).f1
    pq_iv, pq_ex = float(pq["pq_iv"]), float(pq["pq_ex"])
    return {"full": full_score(pq_iv, pq_ex, total.f1), "f1": total.f1,
            "pq_iv": pq_iv, "pq_ex": pq_ex, "tp": total.tp, "pred": total.pred,
            "per_mouse_f1": per_mouse,
            "kept": int(sum(bool(kept.get(s, False)) for s in sids)),
            "kept_wrong": int(sum(bool(kept.get(s, False)) and correct.get(s) is False
                                  for s in sids))}


# ------------------------------------------------------------ Baseline
def baseline_heldout(records: Mapping[str, Mapping], pq: Mapping,
                     correct: Callable[[Mapping, Any], bool | None] | None = None,
                     threshold: float = BASELINE_PAIR_THR) -> dict:
    """The exact v10 held-out recipe on prep held-out records (Req 9.3)."""
    R = _research()
    correct = default_correct() if correct is None else correct
    poses, kept, regions, ok = {}, {}, {}, {}
    for sid, rec in records.items():
        ch = R.cp_pose_lab.pose_choose(rec["cp_scored"], "score")
        if ch is None:
            poses[sid] = (None, 0.0)
            mg, z = -99.0, -99.0
        else:
            poses[sid] = (ch[0], float(ch[1]))
            mg = float(R.margin_lab.margin(rec["vote_cands"], ch[0], ch[1], rec))
            z = float(ch[2])
        kept[sid] = ch is not None and v10_gate(mg, z)
        ok[sid] = correct(rec, poses[sid][0])
        regions[sid] = {"margin": mg, "z": z, "kept": bool(kept[sid]), "correct": ok[sid]}
    rows = P.dataset(records, poses)
    probs = P.loo_predict(rows, {s: records[s]["subject"] for s in records})
    selected = {sid: P.select(pairs, probs[sid], threshold, kept[sid])
                for sid, pairs, _X, _y in rows}
    for sid in regions:
        regions[sid]["n_pairs"] = int(len(selected[sid]))
    out = score_selection(records, selected, pq, kept, ok)
    out.update(pair_threshold=float(threshold), regions=regions)
    return out


def baseline_test_pairs(records: Mapping[str, Mapping], cp_bins: Mapping[str, np.ndarray],
                        model=None, threshold: float = V10_TEST_PAIR_THR) -> dict:
    """v10 test pairs (``test_cp_apply.py``): ``sid -> [[i, j], ...]`` index pairs."""
    R = _research()
    model = P.v10_test_model() if model is None else model
    out = {}
    for sid, rec in records.items():
        w = rec.get("v10_window") or {}
        M, score, M_true = w.get("M"), float(w.get("score", 0.0)), w.get("M_true")
        mg = float(R.margin_lab.margin(rec["vote_cands"], M, score, rec))
        z = (float(R.cp_pose_lab.cp_z(cp_bins[sid], rec["iv_c"], M_true))
             if M_true is not None else -9.0)
        pairs, X = P.pair_clf.candidates(rec, M_true, score)
        out[sid] = _pairs_list(P.select(pairs, P.predict(model, X), threshold,
                                        v10_gate(mg, z)))
    return out


def csv_index_pairs(records: Mapping[str, Mapping], csv_path: str | Path = V10_CSV) -> dict:
    """``match_pairs`` of a submission CSV as index pairs into the records' IDs."""
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["sample_id"]
            if sid not in records:
                continue
            iv = {str(k): n for n, k in enumerate(records[sid]["iv_ids"])}
            ex = {str(k): n for n, k in enumerate(records[sid]["ex_ids"])}
            out[sid] = [[iv[str(a)], ex[str(b)]] for a, b in json.loads(row["match_pairs"])]
    return out


def compare_test_pairs(got: Mapping[str, list], ref: Mapping[str, list]) -> dict:
    """Per-region set comparison of reproduced vs CSV test pairs."""
    sids = sorted(set(got) | set(ref))
    bad = [s for s in sids if {tuple(p) for p in got.get(s, [])}
           != {tuple(p) for p in ref.get(s, [])}]
    return {"available": True, "regions": len(sids), "mismatched": bad,
            "n_mismatch": len(bad), "equal": not bad,
            "pairs": int(sum(len(v) for v in got.values())),
            "pairs_csv": int(sum(len(v) for v in ref.values())),
            "regions_with_pairs": int(sum(bool(v) for v in got.values())),
            "regions_with_pairs_csv": int(sum(bool(v) for v in ref.values()))}


# ------------------------------------------------------------ configurations
def evaluate_configs(records: Mapping[str, Mapping], pairs_ck: Mapping, verifier_ck: Mapping,
                     pq: Mapping, baseline_full: float,
                     configs: Sequence[tuple[str, str, str]] = CONFIGS) -> list[dict]:
    """ConfigResult dicts for every configuration (Req 9.5, 9.6, 8.2, 8.5)."""
    total = int(sum(int(r["n_gt_pairs"]) for r in records.values()))
    out = []
    for name, sel, gate in configs:
        ver = verifier_ck["selections"][sel]
        hv, entries = ver["heldout"], pairs_ck["heldout"][sel]
        kept = {s: bool(hv[s]["kept"][gate]) for s in records}
        correct = {s: hv[s].get("correct") for s in records}
        rows, probs = P.entry_rows({s: entries[s] for s in records})
        thr = float(P.choose_threshold(rows, probs, kept, total)[0])
        selected = {s: P.select(entries[s]["pairs"], entries[s]["probs"], thr, kept[s])
                    for s in records}
        m = score_selection(records, selected, pq, kept, correct)
        tau = ver["conservative"]["tau"] if gate == "conservative" else ver["aggressive"]["tau"]
        unavailable = gate == "conservative" and bool(ver["conservative"].get("unavailable",
                                                                              tau is None))
        tv, tentries = ver.get("test", {}), pairs_ck.get("test", {}).get(sel, {})
        test_pairs = {s: _pairs_list(P.select(e["pairs"], e["probs"], thr,
                                              bool(tv[s]["kept"][gate])))
                      for s, e in tentries.items()}
        res = ConfigResult(
            name=name, selection=sel, gate=gate,
            tau=None if tau is None else float(tau), conservative_unavailable=unavailable,
            pair_threshold=thr, full=m["full"], f1=m["f1"], pq_iv=m["pq_iv"],
            pq_ex=m["pq_ex"], tp=m["tp"], pred=m["pred"], per_mouse_f1=m["per_mouse_f1"],
            kept=m["kept"], kept_wrong=m["kept_wrong"],
            accepted=accept(m["full"], baseline_full), compared_to=float(baseline_full),
            rejected_reason=rejected_reason(m["full"], baseline_full),
            test_pairs=test_pairs)
        out.append(asdict(res))
    return out


# ------------------------------------------------------------ failure
def write_failure(run_dir: str | Path, baseline: Mapping, expected: Mapping = EXPECTED) -> Path:
    """``validate_failure.json`` (read by assemble / report), written atomically."""
    rec = {"status": "BASELINE_REPRODUCTION_FAILED",
           "measured": {k: baseline.get(k) for k in ("full", "f1", "pq_iv", "pq_ex",
                                                     "kept", "kept_wrong", "tp", "pred")},
           "expected": dict(expected)}
    data = (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode()
    p = Path(run_dir) / FAILURE_JSON
    checkpoint._write_atomic(p, lambda f: f.write(data))
    return p


def _fmt_metrics(m: Mapping) -> str:
    pm = " ".join(f"{g[-6:]}={v:.4f}" for g, v in m["per_mouse_f1"].items())
    return (f"full={m['full']:.4f} f1={m['f1']:.4f} pq_iv={m['pq_iv']:.4f} "
            f"pq_ex={m['pq_ex']:.4f} tp={m['tp']} pred={m['pred']} kept={m['kept']} "
            f"kept_wrong={m['kept_wrong']} per_mouse_f1[{pm}]")


# ------------------------------------------------------------ Stage
def run(prep_ck: Mapping, pairs_ck: Mapping, verifier_ck: Mapping,
        cp_bins: Mapping[str, np.ndarray] | None = None, smoke: bool = False,
        run_dir: str | Path = ".", v10_csv: str | Path | None = V10_CSV,
        test_model=None) -> dict:
    """The whole Stage on in-memory Checkpoints."""
    t0 = time.monotonic()
    heldout, test = prep_ck["heldout"], prep_ck.get("test", {})
    pq = heldout_pq(heldout)
    checkpoint.log_line("VALIDATE_PQ", f"regions={pq['n_regions']} pq_iv={pq['pq_iv']:.4f} "
                                       f"pq_ex={pq['pq_ex']:.4f}")
    base = baseline_heldout(heldout, pq)
    checkpoint.log_line("BASELINE", _fmt_metrics(base))
    if smoke:
        base["reproduced"] = None
    else:
        base["reproduced"] = reproduction_ok(base["f1"], base["full"])
        if not base["reproduced"]:
            checkpoint.log_line("BASELINE_REPRODUCTION_FAILED",
                                f"f1={base['f1']!r} full={base['full']!r} "
                                f"expected f1={EXPECTED['f1']} full={EXPECTED['full']} "
                                f"tol={EXPECTED['tol']}")
            p = write_failure(run_dir, base)
            checkpoint.log_line("VALIDATE_FAILURE_WRITTEN", str(p))
            raise SystemExit(1)
    base["expected"] = dict(EXPECTED)

    if test:
        base["test_pairs"] = baseline_test_pairs(test, cp_bins or {}, test_model)
        if v10_csv is not None and Path(v10_csv).is_file():
            match = compare_test_pairs(base["test_pairs"], csv_index_pairs(test, v10_csv))
        else:
            match = {"available": False, "reason": f"{v10_csv} not found"}
    else:
        base["test_pairs"] = {}
        match = {"available": False, "reason": "no test regions"}
    base["v10_csv_match"] = match
    if match.get("available"):
        tag = "V10_TEST_PAIRS_MATCH" if match["equal"] else "V10_TEST_PAIRS_MISMATCH"
        checkpoint.log_line(tag, f"regions={match['regions']} pairs={match['pairs']} "
                                 f"csv={match['pairs_csv']} mismatched={match['mismatched']}")

    configs = evaluate_configs(heldout, pairs_ck, verifier_ck, pq, base["full"])
    for c in configs:
        checkpoint.log_line(
            "CONFIG", f"{c['name']}: tau={c['tau']} pair_thr={c['pair_threshold']:.3f} "
                      f"{_fmt_metrics(c)} accepted={c['accepted']}"
                      + ("" if c["accepted"] else f" ({c['rejected_reason']})")
                      + f" test_pairs={sum(len(v) for v in c['test_pairs'].values())}")
    checkpoint.log_line("VALIDATE_DONE", f"accepted={[c['name'] for c in configs if c['accepted']]} "
                                         f"elapsed={time.monotonic() - t0:.1f}s")
    return {"baseline": base, "configs": configs, "pq": pq,
            "meta": {"smoke": bool(smoke), "expected": dict(EXPECTED),
                     "baseline_pair_threshold": BASELINE_PAIR_THR,
                     "v10_test_pair_threshold": V10_TEST_PAIR_THR}}


def compute(cfg, ctx) -> dict:
    prep_ck = ctx.load("prep")
    ctx.load("joint")                 # dependency check (poses enter through pairs / verifier)
    pairs_ck = ctx.load("pairs")
    verifier_ck = ctx.load("verifier")
    cp_bins = _prep.load_cp_bins(prep_ck, ctx.run_dir) if prep_ck.get("test") else {}
    return run(prep_ck, pairs_ck, verifier_ck, cp_bins=cp_bins,
               smoke=bool(getattr(ctx, "smoke", False)), run_dir=ctx.run_dir)
