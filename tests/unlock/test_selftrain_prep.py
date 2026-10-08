"""Unit tests for hpc_unlock.selftrain_prep (Req 12.3, 12.5, 12.10-12.13)."""
from __future__ import annotations

import copy
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hpc_unlock import pairs as P
from hpc_unlock import selftrain_prep as S
from hpc_unlock.config import UnlockConfig

MICE = ("m0", "m1", "m2")
H = W = 160


# --------------------------------------------------------------- synthetic data
def _disk_map(points, shape, r=3):
    lab = np.zeros(shape, np.int32)
    yy, xx = np.indices(shape)
    k = 0
    for x, y in points:
        m = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) & (lab == 0)
        if m.any():
            k += 1
            lab[m] = k
    return lab


def _cand(rng, M, code):
    return {"M": M, "refine_score": float(rng.uniform(5, 30)), "soft": float(rng.uniform(0, 20)),
            "z": float(rng.uniform(0, 7)), "refine_margin": float(rng.uniform(-2, 5)),
            "soft_margin": float(rng.uniform(-2, 5)), "angle": float(code),
            "scale": float(rng.uniform(0.95, 1.05)), "anisotropy": float(rng.uniform(0, 0.05)),
            "landing": rng.uniform(0, 300, 2), "source": "gpu"}


def synth(seed=0, per_mouse=3, n_test=2):
    """prep / joint / pairs (real LOO code) / fake verifier + predicted ex maps."""
    rng = np.random.default_rng(seed)
    heldout, test, hent, tent, pred = {}, {}, {}, {}, {}

    def region(sid, subject, split, code):
        n = 25
        iv_c = rng.uniform(15, 140, (n, 2))
        M = np.array([[1.0, 0.0, rng.uniform(-5, 5)], [0.0, 1.0, rng.uniform(-5, 5)]])
        proj = S.project(iv_c, M)
        # predicted ex cells: most in-vivo cells visible (jittered), plus distractors
        vis = proj[: n - 6] + rng.normal(0, 0.7, (n - 6, 2))
        extra = rng.uniform(5, 155, (6, 2))
        lab = _disk_map(np.vstack([vis, extra]), (H, W))
        _, ex_c = S.instance_centroids(lab)
        rec = {"sid": sid, "split": split, "subject": subject, "ex_shape": (H, W),
               "group": (subject, (H, W)), "iv_c": iv_c, "ex_c": ex_c,
               "iv_f": {k: rng.random(n) for k in ("mean", "contrast", "area")},
               "ex_f": {k: rng.random(len(ex_c)) for k in ("mean", "contrast", "area")}}
        if split == "heldout":
            rec.update(iv_link=np.arange(n), ex_link=np.arange(len(ex_c)),
                       gt_pairs={(i, i) for i in range(min(n, len(ex_c)) - 6)},
                       n_gt_pairs=n - 6, gt_M=M.copy(), gt_iv_c=iv_c.copy(),
                       ok=bool(rng.random() > 0.3))
        c = _cand(rng, M, code)
        entry = {"M": M, "score": c["refine_score"], "cand_index": 0, "cand": c}
        return rec, entry, lab

    for g_i, g in enumerate(MICE):
        for k in range(per_mouse):
            sid = f"{g}__r{k}"
            heldout[sid], hent[sid], pred[sid] = region(sid, g, "heldout", g_i)
    for k in range(n_test):
        sid = f"t__r{k}"
        test[sid], tent[sid], pred[sid] = region(sid, "tm", "test", 9)
    prep = {"heldout": heldout, "test": test}
    joint = {"heldout": {"joint": hent}, "test": {"joint": tent}}
    return prep, joint, pred


def pairs_ck(prep, joint):
    hout, tout = P.selection_probs(prep["heldout"], prep["test"],
                                   joint["heldout"]["joint"], joint["test"]["joint"])
    return {"heldout": {"joint": hout}, "test": {"joint": tout}}


def fake_verifier(prep, fold_kept=lambda sid: True, test_kept=lambda sid: True, tau=0.5):
    hv = {s: {"subject": r["subject"],
              "kept": {"conservative": True, "aggressive": True,
                       "fold_conservative": bool(fold_kept(s))}}
          for s, r in prep["heldout"].items()}
    tv = {s: {"kept": {"conservative": bool(test_kept(s)), "aggressive": True}}
          for s in prep["test"]}
    return {"selections": {"joint": {
        "heldout": hv, "test": tv, "fold": {m: {"conservative": tau} for m in MICE},
        "conservative": {"tau": tau, "unavailable": False}, "aggressive": {"tau": tau}}}}


VALIDATE = {"configs": [{"name": "joint_cons", "selection": "joint", "gate": "conservative",
                         "pair_threshold": 0.05, "accepted": True}]}


def correct(rec, M):
    return rec["ok"]


def loaders(pred, shape=(H, W)):
    def load_pred(split, sid, rec):
        return pred[sid].copy()

    def load_image(split, sid):
        r = np.random.default_rng(zlib.crc32(sid.encode()))
        return r.uniform(100, 4000, shape).astype(np.float32)
    return load_pred, load_image


# --------------------------------------------------------------- pseudo-label rule
def test_pseudo_label_rule_retain_seed_exclude_clip_background():
    pred = np.zeros((20, 20), np.int32)
    pred[2:5, 2:5] = 4            # centroid (3, 3)   -> retained
    pred[14:17, 14:17] = 7        # centroid (15, 15) -> excluded (no point near)
    pred[8:11, 12:15] = 9         # centroid (13, 9)  -> excluded (2 px > 1.5 from the seed)
    proj = np.array([[3.5, 3.0],      # retains instance 4
                     [15.0, 9.0],     # seed; disk overlaps instance 9's pixels
                     [-1.0, 10.0],    # seed partly outside the canvas -> clipped
                     [-10.0, -10.0]])  # seed fully outside -> no instance
    out, info = S.pseudo_labels(pred, proj, match_radius=1.5, seed_radius=2.0)

    assert out.dtype == np.int32 and out.shape == pred.shape
    assert sorted(np.unique(out).tolist()) == [0, 1, 2, 3]          # consecutive from 1
    assert info == {"n_pred": 3, "n_proj": 4, "retained": 1, "seeded": 2, "seeds_empty": 1,
                    "excluded": 2, "n_pseudo": 3}
    # retained instance keeps its original pixels
    assert np.array_equal(out == 1, pred == 4)
    # excluded instances are gone and their pixels never repainted
    assert not out[pred == 7].any() and not out[pred == 9].any()
    # seeds: inside their disk, only on background pixels, clipped to the canvas
    yy, xx = np.indices(pred.shape)
    for lab, (cx, cy) in ((2, (15.0, 9.0)), (3, (-1.0, 10.0))):
        m = out == lab
        assert m.any()
        assert np.all((xx[m] - cx) ** 2 + (yy[m] - cy) ** 2 <= 4.0)
        assert np.all(pred[m] == 0)
        disk_free = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 4.0) & (pred == 0)
        assert np.array_equal(m, disk_free)                          # fully painted


def test_pseudo_labels_no_points_or_no_predictions():
    pred = np.zeros((10, 10), np.int32)
    pred[1:3, 1:3] = 5
    out, info = S.pseudo_labels(pred, np.zeros((0, 2)), 6, 6)
    assert not out.any() and info["excluded"] == 1 and info["n_pseudo"] == 0
    out, info = S.pseudo_labels(np.zeros((10, 10), np.int32), [[5.0, 5.0]], 6, 2)
    assert info["seeded"] == 1 and out.max() == 1 and int((out == 1).sum()) == 13


# --------------------------------------------------------------- radii (12.13)
@pytest.mark.parametrize("mr,sr", [(0.5, 6.0), (6.0, 21.0), (float("nan"), 6.0)])
def test_invalid_radius_stops_before_any_label(mr, sr, capsys):
    def boom(*a, **k):
        raise AssertionError("label work started")
    with pytest.raises(SystemExit) as e:
        S.build(None, None, None, None, None, mr, sr, load_pred=boom, load_image=boom)
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "CONFIG_INVALID" in out and "STAGE_ABORTED" in out
    assert S.validate_config(SimpleNamespace(match_radius=mr, seed_radius=sr))


def test_compute_checks_radii_before_loading(tmp_path):
    class Ctx:
        run_dir, smoke = tmp_path, False

        def load(self, name):
            raise AssertionError(f"loaded {name}")
    with pytest.raises(SystemExit):
        S.compute(UnlockConfig(match_radius=25.0), Ctx())
    assert not list(Path(tmp_path).iterdir())
    assert S.validate_config(UnlockConfig()) == []


# --------------------------------------------------------------- fold gate / leakage (12.5)
def _m0_maps(prep, joint, pred, ver):
    pl = S.plan(prep, joint, pairs_ck(prep, joint), ver, VALIDATE, correct)
    load_pred, _ = loaders(pred)
    res = S.job_pseudo_labels(pl["jobs"]["m0"], prep["heldout"], load_pred, 6.0, 6.0)
    return pl, res


def test_heldout_pseudo_labels_ignore_own_ground_truth():
    prep, joint, pred = synth(1)
    ver = fake_verifier(prep)
    pl1, res1 = _m0_maps(prep, joint, pred, ver)
    assert sum(info["n_pseudo"] for _, info in res1.values()) > 0

    prep2 = copy.deepcopy(prep)
    rng = np.random.default_rng(99)
    for sid, r in prep2["heldout"].items():
        if r["subject"] != "m0":
            continue
        r["gt_M"] = r["gt_M"] + rng.normal(0, 20, (2, 3))
        r["iv_link"] = rng.permutation(r["iv_link"])
        r["ex_link"] = rng.permutation(r["ex_link"])
        r["gt_pairs"] = {(int(a), int(b)) for a, b in rng.integers(0, 25, (15, 2))}
        r["n_gt_pairs"] = 40
        r["ok"] = not r["ok"]
    # the perturbation does reach the other mice's LOO pair models ...
    p1 = pairs_ck(prep, joint)["heldout"]["joint"]["m1__r0"]["probs"]
    p2 = pairs_ck(prep2, joint)["heldout"]["joint"]["m1__r0"]["probs"]
    assert not np.allclose(p1, p2)
    # ... but not m0's threshold, matched instances or pseudo-label maps
    pl2, res2 = _m0_maps(prep2, joint, pred, ver)
    assert pl1["jobs"]["m0"]["pair_threshold"] == pl2["jobs"]["m0"]["pair_threshold"]
    assert res1.keys() == res2.keys()
    for sid in res1:
        assert np.array_equal(pl1["jobs"]["m0"]["kept"][sid]["matched"],
                              pl2["jobs"]["m0"]["kept"][sid]["matched"])
        assert np.array_equal(res1[sid][0], res2[sid][0])


def test_fold_gate_selects_heldout_regions_and_full_gate_test():
    prep, joint, pred = synth(2)
    ver = fake_verifier(prep, fold_kept=lambda s: s != "m1__r1",
                        test_kept=lambda s: s == "t__r0")
    # the full held-out gate is ignored for held-out jobs
    for s in ver["selections"]["joint"]["heldout"].values():
        s["kept"]["conservative"] = False
    pl = S.plan(prep, joint, pairs_ck(prep, joint), ver, VALIDATE, correct)
    assert sorted(pl["jobs"]["m1"]["kept"]) == ["m1__r0", "m1__r2"]
    assert pl["jobs"]["m1"]["regions"] == ["m1__r0", "m1__r1", "m1__r2"]
    assert sorted(pl["jobs"]["test"]["kept"]) == ["t__r0"]
    assert pl["jobs"]["test"]["pair_threshold"] == 0.05
    assert pl["selection"] == "joint"


# --------------------------------------------------------------- build / no confident / tiles
def test_build_no_confident_flags_and_tiles():
    prep, joint, pred = synth(3)
    ver = fake_verifier(prep, fold_kept=lambda s: not s.startswith("m2"),
                        test_kept=lambda s: False)
    load_pred, load_image = loaders(pred)
    ck, arrays = S.build(prep, joint, pairs_ck(prep, joint), ver, VALIDATE, 6.0, 6.0,
                         load_pred=load_pred, load_image=load_image, per_region=8,
                         correct=correct)
    assert ck["no_confident_regions"] == {"mice": ["m2"], "test": True}
    assert ck["jobs"]["m2"]["no_confident"] and ck["jobs"]["m2"]["reason"] == S.NO_CONFIDENT
    assert ck["jobs"]["test"]["reason"] == S.NO_CONFIDENT
    assert ck["jobs"]["test"]["regions"] == ["t__r0", "t__r1"]
    for job in ("m0", "m1"):
        j = ck["jobs"][job]
        assert not j["no_confident"] and j["n_tiles"] > 0
        assert j["n_tiles"] <= 8 * len(j["train_regions"])
        img, lab = arrays[f"tiles|{job}|image"], arrays[f"tiles|{job}|label"]
        assert img.shape == lab.shape == (j["n_tiles"], 96, 96)
        assert img.dtype == np.float32 and lab.dtype == np.int32
        assert set(arrays[f"tiles|{job}|sid"].tolist()) == set(j["train_regions"])
    assert arrays["tiles|test|image"].shape == (0, 96, 96)
    # every inference image once, raw float32
    for split in ("heldout", "test"):
        for sid in prep[split]:
            a = arrays[f"image|{sid}"]
            assert a.dtype == np.float32 and a.shape == (H, W)
            assert np.array_equal(a, load_image(split, sid))


def test_make_tiles_shapes_cap_and_padding():
    lab = _disk_map(np.random.default_rng(0).uniform(5, 140, (30, 2)), (130, 150))
    img = np.random.default_rng(1).uniform(0, 1000, lab.shape).astype(np.float32)
    ims, lbs = S.make_tiles(img, lab, np.random.default_rng(0), per_region=10)
    assert len(ims) == len(lbs) == 10
    for a, b in zip(ims, lbs):
        assert a.shape == b.shape == (96, 96) and a.dtype == np.float32 and b.dtype == np.int32
        u = np.unique(b)
        assert np.array_equal(u, np.arange(len(u)))                # relabelled 0..k
    small = _disk_map([[20.0, 15.0]], (40, 50))
    ims, lbs = S.make_tiles(np.ones((40, 50), np.float32), small, np.random.default_rng(0))
    assert len(ims) == 1 and ims[0].shape == (96, 96) and lbs[0].max() == 1


def test_choose_selection_fallbacks():
    j = {"heldout": {"joint": {}, "independent": {}}}
    p = {"heldout": {"joint": {}, "independent": {}}}
    v = {"selections": {"joint": {}, "independent": {}}}
    assert S.choose_selection(j, p, v, None) == "joint"
    val = {"configs": [{"selection": "joint", "gate": "conservative", "accepted": False},
                       {"selection": "independent", "gate": "conservative", "accepted": True}]}
    assert S.choose_selection(j, p, v, val) == "independent"
    v2 = {"selections": {"independent": {}}}
    assert S.choose_selection(j, p, v2, None) == "independent"
