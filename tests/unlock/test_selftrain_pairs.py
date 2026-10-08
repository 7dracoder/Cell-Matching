"""Tests for hpc_unlock.selftrain_pairs (Req 12.5-12.8, 12.11, 12.12; Properties 15, 19)."""
from __future__ import annotations

import copy
import zlib

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hpc_unlock import selftrain_pairs as SPR
from hpc_unlock import validate as VAL

MICE = ("m0", "m1", "m2")
H = W = 72
GRID = [(-0.5, 0.15), (0.0, 0.4)]
SEL = "joint"
CENTERS = np.array([(x, y) for y in range(10, 64, 12) for x in range(10, 64, 12)], float)


def fast_estimator(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))


def disks(points, shape=(H, W), r=3, ids=None):
    lab = np.zeros(shape, np.int32)
    yy, xx = np.indices(shape)
    for n, (x, y) in enumerate(points):
        m = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) & (lab == 0)
        lab[m] = n + 1 if ids is None else ids[n]
    return lab


GOOD = disks(CENTERS)                       # == GT
BAD = disks(CENTERS + 3.0)                  # IoU < 0.75 with GT, still within 10 px
BASE = disks(CENTERS[:-8])                  # Baseline: some cells missed


def _image(sid):
    return np.random.default_rng(zlib.crc32(sid.encode())).uniform(100, 4000, (H, W)).astype(
        np.float32)


def _cp(sid):
    return np.random.default_rng(zlib.crc32(sid.encode()) + 1).normal(0, 2, (H, W)).astype(
        np.float16)


def synth(per_mouse=2, n_test=2, good_k=None, status=None, test_status="done",
          test_ids_gap=True):
    """prep / joint / verifier / validate / selftrain checkpoints + in-memory sources.

    ``good_k[m]``: setting whose labels equal m's GT (the other setting is shifted).
    """
    good_k = good_k or {"m0": 1, "m1": 0, "m2": 0}
    status = dict({m: "done" for m in MICE}, **(status or {}))
    n = len(CENTERS)
    heldout, test, hp, tp_, hv, tv, per_pq = {}, {}, {}, {}, {}, {}, {}
    arrays = {job: {} for job in (*MICE, "test")}
    gt_maps = {}
    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    rng = np.random.default_rng(0)

    def feats(lab, sid):
        return SPR.region_features(lab, _image(sid), _cp(sid).astype(np.float32), None)

    for m in MICE:
        for k in range(per_mouse):
            sid = f"{m}__r{k}"
            f = feats(BASE, sid)
            heldout[sid] = {
                "sid": sid, "subject": m, "split": "heldout", "ex_shape": (H, W),
                "iv_c": CENTERS.copy(), "ex_c": f["ex_c"], "ex_f": f["ex_f"],
                "iv_f": {kk: rng.random(n) for kk in ("mean", "contrast", "area")},
                "iv_link": np.arange(n), "ex_link": VAL.gt_link(BASE, GOOD),
                "gt_pairs": {(i, i) for i in range(n)}, "n_gt_pairs": n}
            gt_maps[sid] = GOOD.copy()
            hp[sid] = {"M": M.copy(), "score": 20.0}
            hv[sid] = {"kept": {"conservative": True, "aggressive": True}, "correct": True}
            per_pq[sid] = {"iv": 0.8, "ex": float(VAL.region_pq(BASE, GOOD)[0])}
            arrays[m][f"{sid}|cp"] = _cp(sid)
            for kk in range(len(GRID)):
                arrays[m][f"{sid}|labels|{kk}"] = (GOOD if kk == good_k[m] else BAD).astype(
                    np.uint16)
    perm = np.random.default_rng(1).permutation(n)
    for k in range(n_test):
        sid = f"t__r{k}"
        f = feats(BASE, sid)
        test[sid] = {"sid": sid, "subject": "tm", "split": "test", "ex_shape": (H, W),
                     "iv_c": CENTERS.copy(), "ex_c": f["ex_c"], "ex_f": f["ex_f"],
                     "iv_f": {kk: rng.random(n) for kk in ("mean", "contrast", "area")},
                     "iv_ids": [f"IV_{i}" for i in range(n)], "ex_ids": None}
        tp_[sid] = {"M": M.copy(), "score": 20.0}
        tv[sid] = {"kept": {"conservative": k == 0, "aggressive": True}}
        arrays["test"][f"{sid}|cp"] = _cp(sid)
        ids = (perm + 1) * (3 if test_ids_gap else 1)        # shuffled, with gaps
        arrays["test"][f"{sid}|labels|0"] = disks(CENTERS, ids=ids).astype(np.uint16)
        arrays["test"][f"{sid}|labels|1"] = disks(CENTERS + 3.0, ids=ids).astype(np.uint16)

    prep = {"heldout": heldout, "test": test}
    joint = {"heldout": {SEL: hp}, "test": {SEL: tp_}}
    verifier = {"selections": {SEL: {"heldout": hv, "test": tv}}}
    validate = {"baseline": {"full": 0.0},
                "configs": [{"name": "joint_cons", "full": 0.9, "accepted": False}],
                "pq": {"pq_iv": 0.8, "per_region": per_pq}}
    st_prep = {"selection": SEL}
    jobs = {}
    for m in MICE:
        jobs[m] = {"status": status[m], "regions": sorted(s for s in heldout if s[:2] == m),
                   "decode_grid": GRID}
        if status[m] == "done":
            jobs[m]["flows_side"] = {"__side_npz__": f"{m}.npz"}
    jobs["test"] = {"status": test_status, "regions": sorted(test), "decode_grid": GRID}
    if test_status == "done":
        jobs["test"]["flows_side"] = {"__side_npz__": "test.npz"}
    calls = []

    def job_arrays(job, entry):
        calls.append(job)
        assert entry["status"] == "done"
        return arrays[job]

    sources = SPR.Sources(read_image=lambda split, sid: _image(sid),
                          gt_ex=lambda sid, shape: gt_maps[sid], job_arrays=job_arrays)
    return SimpleData(prep, joint, verifier, validate, st_prep, {"jobs": jobs}, sources,
                      gt_maps, arrays, calls)


class SimpleData:
    def __init__(self, prep, joint, verifier, validate, st_prep, st_gpu, sources, gt_maps,
                 arrays, calls):
        self.prep, self.joint, self.verifier, self.validate = prep, joint, verifier, validate
        self.st_prep, self.st_gpu, self.sources = st_prep, st_gpu, sources
        self.gt_maps, self.arrays, self.calls = gt_maps, arrays, calls

    def run(self, **kw):
        kw.setdefault("estimator", fast_estimator)
        kw.setdefault("log", lambda *a, **k: "")
        return SPR.run(self.prep, self.joint, self.verifier, self.validate, self.st_prep,
                       self.st_gpu, self.sources, **kw)


# --------------------------------------------------------------------------- helpers
def test_relabel_consecutive_keeps_order():
    lab = np.array([[0, 7, 7], [3, 0, 9]], np.uint16)
    out = SPR.relabel(lab)
    assert out.dtype == np.int32
    assert out.tolist() == [[0, 2, 2], [1, 0, 3]]


def test_comparison_target_best_accepted_else_baseline():
    v = {"baseline": {"full": 0.5},
         "configs": [{"name": "a", "full": 0.61, "accepted": True},
                     {"name": "b", "full": 0.70, "accepted": False},
                     {"name": "c", "full": 0.63, "accepted": True}]}
    assert SPR.comparison_target(v) == (0.63, "c")
    v["configs"] = [dict(c, accepted=False) for c in v["configs"]]
    assert SPR.comparison_target(v) == (0.5, "baseline")


# --------------------------------------------------------------- Property 15 (12.5)
# Feature: hpc-registration-unlock, Property 15 (no label leakage from the evaluated mouse)
def test_decode_setting_for_mouse_chosen_without_its_ground_truth():
    d = synth()
    res, _ = d.run()
    # m0's own GT prefers setting 1, but its choice comes from m1 / m2 (setting 0)
    assert res["decode_by_mouse"]["m0"]["setting"] == 0
    assert res["decode_by_mouse"]["m0"]["decode"] == list(GRID[0])

    p = synth()
    for sid, r in p.prep["heldout"].items():
        if r["subject"] == "m0":                               # perturb only m0's GT
            p.gt_maps[sid] = BAD.copy()
            r["gt_pairs"] = {(i, (i + 1) % len(CENTERS)) for i in range(len(CENTERS))}
            r["iv_link"] = np.roll(r["iv_link"], 3)
            r["n_gt_pairs"] = 7
    res2, _ = p.run()
    assert res2["decode_by_mouse"]["m0"] == res["decode_by_mouse"]["m0"]
    # the perturbation is effective: m0's own score changes
    assert res2["per_region"]["m0__r0"]["pq_ex"] != res["per_region"]["m0__r0"]["pq_ex"]
    assert res2["full"] != res["full"]


# ------------------------------------------------------------------- 12.12
def test_no_confident_mouse_uses_baseline_ex_masks():
    d = synth(status={"m2": "skipped_no_confident"})
    res, _ = d.run()
    assert res["no_confident_mice"] == ["m2"]
    assert "m2" not in d.calls                                  # its job is never read
    assert res["decode_by_mouse"]["m2"]["setting"] is None
    for sid, row in res["per_region"].items():
        if row["subject"] == "m2":
            assert row["pq_ex"] == row["baseline_pq_ex"]
            assert row["setting"] is None
    want = np.mean([r["pq_ex"] for r in res["per_region"].values()])
    assert res["pq_ex"] == pytest.approx(want)
    assert res["pq_iv"] == pytest.approx(0.8)


# --------------------------------------------------------- Property 19 (12.7, 12.8)
# Feature: hpc-registration-unlock, Property 19 (acceptance is strict and unrounded)
def test_strict_comparison_against_target():
    d = synth()
    res, _ = d.run()
    full = res["full"]
    assert res["accepted"] and res["status"] == "accepted"

    d.validate["configs"] = [{"name": "joint_cons", "full": full, "accepted": True},
                             {"name": "indep_cons", "full": full - 0.1, "accepted": True}]
    eq, labels = d.run()
    assert eq["status"] == "no_candidate" and not eq["accepted"] and labels == {}
    assert eq["compared_to"] == full and eq["compared_to_name"] == "joint_cons"
    assert eq["test_pairs"] == {} and NO_CANDIDATE_IN(eq["reason"])

    d.validate["configs"][0]["full"] = float(np.nextafter(full, -np.inf))
    lo, _ = d.run()
    assert lo["accepted"] and lo["status"] == "accepted"

    d.validate["configs"] = [{"name": "joint_cons", "full": full + 1, "accepted": False}]
    d.validate["baseline"]["full"] = full                        # no accepted config
    base, _ = d.run()
    assert base["compared_to_name"] == "baseline" and base["status"] == "no_candidate"


def NO_CANDIDATE_IN(reason):
    return isinstance(reason, str) and reason.startswith("NO_CANDIDATE")


# ------------------------------------------------------------------- test pairs
def _centroid(lab, k):
    yy, xx = np.nonzero(lab == k)
    return np.array([xx.mean(), yy.mean()])


def test_test_pairs_index_new_labels_one_to_one():
    d = synth()
    res, labels = d.run(grow15=False)
    assert res["accepted"] and res["test_setting"]["setting"] == 0
    assert set(labels) == set(d.prep["test"])
    pairs = res["test_pairs"]
    assert pairs["t__r0"], "kept test region gets pairs"
    assert pairs["t__r1"] == []                                  # gate drops region 1
    for sid, pp in pairs.items():
        lab = labels[sid]
        assert lab.dtype == np.int32
        assert sorted(np.unique(lab[lab > 0]).tolist()) == list(range(1, int(lab.max()) + 1))
        ii = [i for i, _ in pp]
        jj = [j for _, j in pp]
        assert len(set(ii)) == len(ii) and len(set(jj)) == len(jj)
        for i, j in pp:                                          # label j + 1 sits at iv cell i
            assert np.linalg.norm(_centroid(lab, j + 1) - CENTERS[i]) < 0.5


def test_grow15_only_when_flag():
    d = synth()
    on, lab_on = d.run(grow15=True)
    off, lab_off = d.run(grow15=False)
    assert on["test_pairs"] == off["test_pairs"]                 # pairs from ungrown masks
    for sid in lab_off:
        raw = SPR.relabel(d.arrays["test"][f"{sid}|labels|0"])
        assert np.array_equal(lab_off[sid], raw)
        assert (lab_on[sid] > 0).sum() > (raw > 0).sum()
        grown = SPR.default_grow(raw, d.arrays["test"][f"{sid}|cp"].astype(np.float32))
        assert np.array_equal(lab_on[sid], grown)
        same = raw > 0
        assert np.array_equal(lab_on[sid][same], raw[same])      # IDs unchanged
    assert on["test_setting"]["grow15"] and not off["test_setting"]["grow15"]


# ------------------------------------------------------------------- statuses
def test_smoke_skipped_path():
    d = synth(status={m: "skipped_smoke" for m in MICE}, test_status="skipped_smoke")
    res, labels = d.run(smoke=True)
    assert res["status"] == "skipped" and not res["accepted"] and labels == {}
    assert d.calls == []
    assert res["test_labels_side"] is None and res["test_pairs"] == {}


def test_test_job_without_confident_regions():
    d = synth(test_status="skipped_no_confident")
    res, labels = d.run()
    assert res["status"] == "no_confident_regions" and not res["accepted"]
    assert res["reason"] == "no confident regions" and labels == {}
    assert "test" not in d.calls


def test_fork_pool_matches_serial():
    d = synth()
    a, la = d.run(workers=1)
    b, lb = d.run(workers=2)
    assert a["full"] == b["full"] and a["test_pairs"] == b["test_pairs"]
    assert all(np.array_equal(la[s], lb[s]) for s in la)


def test_compute_writes_test_labels_side(tmp_path):
    d = synth()

    class Ctx:
        run_dir, smoke, workers = tmp_path, False, 1

        def load(self, name):
            return {"prep": d.prep, "joint": d.joint, "verifier": d.verifier,
                    "validate": d.validate, "selftrain_prep": d.st_prep,
                    "selftrain_gpu": d.st_gpu}[name]

        def path(self, f):
            return tmp_path / f

    class Cfg:
        st_grow15 = False

    orig_sources = SPR.default_sources
    SPR.default_sources = lambda run_dir: d.sources
    try:
        res = SPR.compute(Cfg(), Ctx())
    finally:
        SPR.default_sources = orig_sources
    assert res["accepted"]
    from hpc_unlock import checkpoint
    labels = checkpoint.load_npz(res["test_labels_side"], tmp_path)
    assert res["test_labels_side"]["__side_npz__"] == SPR.TEST_LABELS_NPZ
    assert set(labels) == set(d.prep["test"]) and all(v.dtype == np.int32 for v in labels.values())
    copy.deepcopy(res)                                           # plain, copyable data
