"""Unit tests for the metric layer of hpc_unlock.validate (Req 9.1, 9.2)."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from hpc_unlock import paths, validate


def research_score(pairs_by_region, records, total):
    """Copy of ``research/lab_eval.score`` (that module loads lab.pkl at import)."""
    tp = pred = 0
    for sid, pairs in pairs_by_region.items():
        r = records[sid]
        tp += sum((r["iv_link"][i], r["ex_link"][j]) in r["gt_pairs"] for i, j in pairs)
        pred += len(pairs)
    return tp, pred, 2 * tp / (pred + total)


# ------------------------------------------------------------------- PQ
def _hand_maps():
    gt = np.zeros((4, 4), np.int32)
    gt[:, :2] = 1                  # 8 px
    gt[:2, 3] = 2                  # 2 px
    pred = gt.copy()
    pred[3, 1] = 0                 # pred 1: 7 px inside GT 1 -> IoU 7/8 = 0.875 (TP)
    pred[1, 3] = 0                 # pred 2: 1 px inside GT 2 -> IoU 1/2 = 0.5 (FP + FN)
    return pred, gt


def test_region_pq_hand_example():
    pred, gt = _hand_maps()
    pq, tp, fp, fn = validate.region_pq(pred, gt)
    assert (tp, fp, fn) == (1, 1, 1)
    assert pq == pytest.approx(0.875 / (1 + 0.5 * 2))


def test_region_pq_threshold_is_strict():
    gt = np.zeros((1, 4), np.int32)
    gt[0, :4] = 1
    pred = np.zeros((1, 4), np.int32)
    pred[0, :3] = 1                # IoU exactly 0.75 is not a match
    assert validate.region_pq(pred, gt)[1:] == (0, 1, 1)


def test_mean_pq_averages_per_region():
    pred, gt = _hand_maps()
    empty = np.zeros((4, 4), np.int32)
    preds = {"a": pred, "b": gt, "c": empty}
    gts = {"a": gt, "b": gt, "c": gt}
    # per region: 0.4375, 1.0, 0.0 (not pooled)
    assert validate.mean_pq(preds, gts) == pytest.approx((0.4375 + 1.0 + 0.0) / 3)
    with pytest.raises(ValueError):
        validate.mean_pq({"a": pred}, gts)


def test_gt_link_hand_example():
    pred, gt = _hand_maps()
    assert validate.gt_link(pred, gt).tolist() == [0, -1]
    assert validate.gt_link(np.zeros((4, 4), np.int32), gt).tolist() == []


# -------------------------------------------------------------- pair F1
def _hand_regions():
    iv_links = {"A": [0, -1, 1], "B": [0]}
    ex_links = {"A": [1, 0, -1], "B": [0]}
    gt_pairs = {"A": {(0, 1), (2, 0)}, "B": {(0, 0)}}
    n_gt = {"A": 3, "B": 1}        # A has one GT pair whose masks no prediction matched
    pred = {"A": [(0, 0),          # -> (0, 1): TP
                  (1, 1),          # in-vivo mask is not a PQ TP
                  (2, 1)],         # -> (1, 0): both TPs but not a GT pair
            "B": [(0, 0)]}         # TP
    return pred, iv_links, ex_links, gt_pairs, n_gt


def test_pair_f1_hand_example():
    pred, ivl, exl, gtp, n_gt = _hand_regions()
    s = validate.pair_f1(pred, ivl, exl, gtp, n_gt)
    assert (s.tp, s.fp, s.fn, s.pred) == (2, 2, 2, 4)
    assert s.f1 == pytest.approx(0.5)
    assert validate.pair_f1(pred, ivl, exl, gtp, 4) == s


def test_pair_f1_is_pooled_and_counts_unpredicted_regions_as_fn():
    pred, ivl, exl, gtp, n_gt = _hand_regions()
    s = validate.pair_f1({"B": pred["B"]}, ivl, exl, gtp, n_gt)
    # B alone would have F1 1.0; pooled over A's 3 GT pairs it is 2/(1+4)
    assert (s.tp, s.fp, s.fn) == (1, 0, 3)
    assert s.f1 == pytest.approx(2 / 5)
    assert validate.pair_f1({}, ivl, exl, gtp, 0).f1 == 0.0


def test_pair_f1_matches_research_formula_and_tp_fp_fn():
    rng = np.random.default_rng(0)
    for _ in range(50):
        records, pred, n_gt = {}, {}, {}
        for r in range(int(rng.integers(1, 5))):
            niv, nex, ngi, nge = (int(x) for x in rng.integers(1, 8, 4))
            iv_link = rng.integers(-1, ngi, niv)
            ex_link = rng.integers(-1, nge, nex)
            gpairs = {(int(a), int(b)) for a, b in zip(rng.permutation(ngi), rng.permutation(nge))
                      if rng.random() < 0.7}
            sid = f"s{r}"
            records[sid] = {"iv_link": iv_link, "ex_link": ex_link, "gt_pairs": gpairs}
            n_gt[sid] = len(gpairs) + int(rng.integers(0, 3))
            k = int(rng.integers(0, min(niv, nex) + 1))
            pred[sid] = list(zip(rng.permutation(niv)[:k].tolist(), rng.permutation(nex)[:k].tolist()))
        total = sum(n_gt.values())
        s = validate.pair_f1(pred, {k: v["iv_link"] for k, v in records.items()},
                             {k: v["ex_link"] for k, v in records.items()},
                             {k: v["gt_pairs"] for k, v in records.items()}, total)
        tp, npred, f1 = research_score(pred, records, total)
        assert (s.tp, s.pred) == (tp, npred)
        assert s.f1 == pytest.approx(f1, abs=1e-15)
        if s.tp + s.fp + s.fn:
            assert s.f1 == pytest.approx(2 * s.tp / (2 * s.tp + s.fp + s.fn), abs=1e-15)


LAB = paths.RDATA / "lab.pkl"


@pytest.mark.skipif(not LAB.is_file(), reason="research/data/lab.pkl not present")
def test_pair_f1_on_heldout_records_matches_research():
    records = pickle.loads(LAB.read_bytes())
    assert len(records) == 47
    n_gt = {s: r["n_gt_pairs"] for s, r in records.items()}
    assert sum(n_gt.values()) == 1139
    # every predicted pair whose linked GT instances form a GT pair (oracle matching)
    pred = {}
    for s, r in records.items():
        inv_iv = {int(g): i for i, g in enumerate(r["iv_link"]) if g >= 0}
        inv_ex = {int(g): j for j, g in enumerate(r["ex_link"]) if g >= 0}
        pred[s] = [(inv_iv[a], inv_ex[b]) for a, b in sorted(r["gt_pairs"])
                   if a in inv_iv and b in inv_ex]
        pred[s] += [(0, 0)] if len(r["iv_link"]) and len(r["ex_link"]) else []
    s = validate.pair_f1(pred, {k: r["iv_link"] for k, r in records.items()},
                         {k: r["ex_link"] for k, r in records.items()},
                         {k: r["gt_pairs"] for k, r in records.items()}, n_gt)
    tp, npred, f1 = research_score(pred, records, 1139)
    assert (s.tp, s.pred) == (tp, npred)
    assert s.f1 == pytest.approx(f1, abs=1e-15)


# ----------------------------------------------------------------- full
def test_full_score_weights():
    assert validate.full_score(0.7404, 0.3901, 0.472) == pytest.approx(0.518625)
    assert validate.full_score(1.0, 0.0, 0.0) == pytest.approx(0.25)
    assert validate.full_score(0.0, 0.0, 1.0) == pytest.approx(0.5)
