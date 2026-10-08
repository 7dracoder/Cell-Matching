"""Tests for hpc_unlock.joint (Req 6.1-6.7; Properties 12-15)."""
from __future__ import annotations

import copy
import itertools

import numpy as np
from hypothesis import given, settings, strategies as st

from hpc_unlock import joint as J

SEEDS = st.integers(0, 2 ** 31 - 1)


# ----------------------------------------------------------------- helpers

def rand_cand(rng, soft_choices=None):
    th = np.deg2rad(rng.uniform(-20, 20))
    s = rng.uniform(0.9, 1.1)
    M = np.array([[s * np.cos(th), -s * np.sin(th), rng.uniform(-50, 50)],
                  [s * np.sin(th), s * np.cos(th), rng.uniform(-50, 50)]])
    soft = float(rng.choice(soft_choices)) if soft_choices is not None else float(rng.uniform(0, 10))
    return {"M": M, "refine_score": float(rng.uniform(0, 1)), "soft": soft,
            "soft_margin": float(rng.uniform(-1, 3)), "landing": rng.uniform(0, 300, 2),
            "source": "hough"}


def rand_vars(rng, n, size_lo, size_hi, ties=False):
    choices = [1.0, 2.0, 3.0] if ties else None
    out = []
    for _ in range(n):
        k = int(rng.integers(size_lo, size_hi + 1))
        cands = [rand_cand(rng, choices) for _ in range(k)]
        out.append({"sids": [f"s{len(out)}"], "dup_key": f"d{len(out)}",
                    "offset": rng.uniform(0, 1500, 2), "index": list(range(k)), "cands": cands})
    return out


def lowest_argmax(var):
    soft = [c["soft"] for c in var["cands"]]
    return soft.index(max(soft))


def transform(p, M):
    return p @ M[:, :2].T + M[:, 2]


def correct(rec, M):
    g = rec["gt_iv_c"]
    return bool(np.median(np.linalg.norm(transform(g, M) - transform(g, rec["gt_M"]), axis=1)) < 5)


# --------------------------------------------------------- Property 12

@settings(max_examples=60, deadline=None)
@given(SEEDS, st.integers(1, 7), st.booleans())
def test_lambda_zero_is_independent_argmax(seed, n, ties):
    """**Validates: Requirements 6.3** (Property 12)"""
    rng = np.random.default_rng(seed)
    vs = rand_vars(rng, n, 1, 10, ties=ties)
    c = J.Problem(vs).solve(0.0)
    assert c == [lowest_argmax(v) for v in vs]


@settings(max_examples=40, deadline=None)
@given(SEEDS, st.sampled_from([0.0, 2.0, 20.0, 1e6]), st.booleans())
def test_single_variable_group_is_independent(seed, lam, ties):
    """**Validates: Requirements 6.3** (Property 12)"""
    rng = np.random.default_rng(seed)
    vs = rand_vars(rng, 1, 1, 10, ties=ties)
    assert J.Problem(vs).solve(lam) == [lowest_argmax(vs[0])]


# --------------------------------------------------------- Property 13

@settings(max_examples=60, deadline=None)
@given(SEEDS, st.integers(2, 4), st.sampled_from([0.5, 2.0, 5.0, 20.0, 200.0]))
def test_exhaustive_equals_brute_force(seed, n, lam):
    """**Validates: Requirements 6.2** (Property 13)"""
    rng = np.random.default_rng(seed)
    vs = rand_vars(rng, n, 1, 4)
    P = J.Problem(vs)
    c = P.solve(lam)
    assert all(0 <= k < len(v["cands"]) for k, v in zip(c, vs))
    brute = max(P.objective(x, lam) for x in itertools.product(*[range(s) for s in P.sizes]))
    assert np.isclose(P.objective(c, lam), brute, rtol=0, atol=1e-9)
    assert P.objective(c, lam) >= P.objective(P.independent(), lam) - 1e-12


@settings(max_examples=15, deadline=None)
@given(SEEDS, st.sampled_from([2.0, 10.0, 50.0]))
def test_icm_never_worse_than_init(seed, lam):
    """**Validates: Requirements 6.2** (Property 13)"""
    rng = np.random.default_rng(seed)
    vs = rand_vars(rng, 6, 9, 10)               # >= 9^6 > 1e5 -> ICM path
    P = J.Problem(vs)
    assert np.prod(P.sizes) > J.EXHAUSTIVE_MAX
    c = P.solve(lam)
    assert all(0 <= k < len(v["cands"]) for k, v in zip(c, vs))
    init = P.independent()
    assert P.objective(c, lam) >= P.objective(init, lam) - 1e-12
    assert P.icm(init, lam) is not None and P.solve(lam) == c       # deterministic


def test_objective_matches_formula():
    rng = np.random.default_rng(0)
    vs = rand_vars(rng, 3, 2, 3)
    P = J.Problem(vs)
    off = np.array([v["offset"] for v in vs])
    e = np.exp(-np.linalg.norm(off[:, None] - off[None], axis=-1) / J.RHO)
    np.fill_diagonal(e, 0)
    w = e / e.sum(1, keepdims=True)
    c = [1, 0, 1]
    pen = 0.0
    for u, v in itertools.combinations(range(3), 2):
        a, b = vs[u]["cands"][c[u]], vs[v]["cands"][c[v]]
        psi = (min(1, np.sum((a["landing"] - b["landing"]) ** 2) / 60 ** 2)
               + min(1, np.sum((a["M"][:, :2] - b["M"][:, :2]) ** 2) / 0.06 ** 2))
        pen += 0.5 * (w[u, v] + w[v, u]) * psi
    expect = sum(vs[u]["cands"][c[u]]["soft"] for u in range(3)) - 5.0 * pen
    assert np.isclose(P.objective(c, 5.0), expect, atol=1e-12)


# ------------------------------------------------- run(): synthetic Stage

def synth(seed, n_mice=3, per_group=(3, 4), gt=True):
    """prep / pose_search dicts for synthetic mice; each mouse has 2 canvas groups."""
    rng = np.random.default_rng(seed)
    prep, pose = {"heldout": {}, "test": {}}, {"heldout": {}, "test": {}}
    for split, prefix in (("heldout", "m"), ("test", "t")):
        for m in range(n_mice):
            subj = f"{prefix}{m}"
            for gi, shape in enumerate([(700, 1000), (1600, 1600)]):
                for r in range(int(rng.integers(per_group[0], per_group[1] + 1))):
                    sid = f"{subj}_g{gi}_r{r}"
                    cands = sorted([rand_cand(rng) for _ in range(int(rng.integers(1, 12)))],
                                   key=lambda c: -c["soft"])
                    rec = {"sid": sid, "subject": subj, "ex_shape": shape,
                           "group": (subj, shape), "dup_key": sid,
                           "offset": rng.uniform(0, 1500, 2), "gt_iv_c": None, "gt_M": None}
                    if gt and split == "heldout":
                        rec["gt_iv_c"] = rng.uniform(0, 500, (20, 2))
                        k = int(rng.integers(len(cands)))
                        rec["gt_M"] = cands[k]["M"] + rng.normal(0, 0.001, (2, 3))
                    prep[split][sid] = rec
                    pose[split][sid] = {"cands": cands, "n": len(cands)}
    return prep, pose


LAMS = (0.0, 2.0, 5.0, 10.0, 20.0)


def test_run_interface_duplicates_and_unregistered():
    """**Validates: Requirements 6.1, 6.4, 6.6, 6.7** (Properties 13, 14)"""
    prep, pose = synth(1)
    # duplicate of a test region (same dup_key, same group) and an empty region
    src = "t0_g0_r0"
    prep["test"]["dup"] = dict(prep["test"][src], sid="dup")
    pose["test"]["dup"] = copy.deepcopy(pose["test"][src])
    prep["test"]["empty"] = dict(prep["test"]["t0_g0_r1"], sid="empty", dup_key="e")
    pose["test"]["empty"] = {"cands": [], "n": 0}
    prep["heldout"]["hempty"] = dict(prep["heldout"]["m0_g0_r0"], sid="hempty", dup_key="he")
    pose["heldout"]["hempty"] = {"cands": [], "n": 0}
    out = J.run(prep, pose, LAMS, 10, correct)

    assert set(out["unregistered"]) == {"empty", "hempty"}
    for split in J.SPLITS:
        for sel in ("independent", "joint"):
            assert set(out[split][sel]) == set(prep[split])
            for sid, e in out[split][sel].items():
                if sid in ("empty", "hempty"):
                    assert e == {"M": None, "score": 0.0, "cand_index": None, "cand": None}
                    continue
                cands = pose[split][sid]["cands"]
                assert e["cand_index"] < min(10, len(cands))
                assert np.array_equal(e["cand"]["M"], cands[e["cand_index"]]["M"])
                assert np.array_equal(e["M"], cands[e["cand_index"]]["M"])
                assert e["score"] == cands[e["cand_index"]]["refine_score"]
            assert np.array_equal(out["test"][sel]["dup"]["M"], out["test"][sel][src]["M"])
    # independent = per-region argmax soft
    for sid, e in out["heldout"]["independent"].items():
        if sid != "hempty":
            assert e["cand_index"] == lowest_argmax(pose["heldout"][sid])
    # lambda and diagnostics
    assert set(out["lambda"]["heldout"]) == {"m0", "m1", "m2"}
    assert out["lambda"]["test"] in LAMS
    d = out["diagnostics"]
    for sel in ("joint", "independent"):
        assert d["total"][sel] == sum(v[sel] for v in d["per_mouse"].values())
        n = sum(correct(prep["heldout"][s], e["M"]) for s, e in out["heldout"][sel].items()
                if e["M"] is not None)
        assert d["total"][sel] == n


def test_unregistered_excluded_from_consistency():
    """An empty-list region changes nothing for the others (Req 6.6)."""
    prep, pose = synth(2, gt=False)
    a = J.run(prep, pose, LAMS, 10, correct)
    prep["test"]["empty"] = dict(prep["test"]["t1_g1_r0"], sid="empty", dup_key="e",
                                 offset=np.array([10.0, 10.0]))
    pose["test"]["empty"] = {"cands": [], "n": 0}
    b = J.run(prep, pose, LAMS, 10, correct)
    assert b["test"]["joint"]["empty"]["M"] is None
    for sid, e in a["test"]["joint"].items():
        assert b["test"]["joint"][sid]["cand_index"] == e["cand_index"]


@settings(max_examples=30, deadline=None)
@given(SEEDS)
def test_grouping_partition_and_duplicates(seed):
    """**Validates: Requirements 6.1, 6.4** (Property 14)"""
    rng = np.random.default_rng(seed)
    recs, cands = {}, {}
    keys = [f"k{i}" for i in range(int(rng.integers(1, 6)))]
    for i in range(int(rng.integers(1, 15))):
        sid = f"r{i}"
        subj, shape = str(rng.choice(["a", "b"])), [(5, 5), (7, 9)][int(rng.integers(2))]
        key = str(rng.choice(keys)) + subj + str(shape)        # identical bytes imply same canvas
        recs[sid] = {"subject": subj, "ex_shape": shape, "group": (subj, shape),
                     "dup_key": key, "offset": rng.uniform(0, 100, 2)}
        cands[sid] = [rand_cand(rng) for _ in range(int(rng.integers(0, 4)))]
    groups, unreg = J.build_groups(recs, cands, 10)
    seen = [s for g, vs in groups.items() for v in vs for s in v["sids"]] + unreg
    assert sorted(seen) == sorted(recs)                          # exactly one group each
    for g, vs in groups.items():
        for v in vs:
            for s in v["sids"]:
                assert g == (recs[s]["subject"], recs[s]["ex_shape"])
    prep = {"heldout": {}, "test": recs}
    pose = {"heldout": {}, "test": {s: {"cands": c} for s, c in cands.items()}}
    # duplicates share the representative's list, so give every member the same list
    first = {}
    for s, r in recs.items():
        first.setdefault(r["dup_key"], s)
        pose["test"][s] = pose["test"][first[r["dup_key"]]]
    out = J.run(prep, pose, LAMS, 10, correct)
    for sel in ("independent", "joint"):
        for s, r in recs.items():
            a, b = out["test"][sel][s]["M"], out["test"][sel][first[r["dup_key"]]]["M"]
            assert (a is None and b is None) or np.array_equal(a, b)


# --------------------------------------------------------- Property 15

@settings(max_examples=10, deadline=None)
@given(SEEDS, st.integers(0, 2))
def test_lambda_never_uses_evaluated_mouse_gt(seed, m):
    """**Validates: Requirements 6.5** (Property 15)"""
    prep, pose = synth(seed, per_group=(3, 7))
    mouse = f"m{m}"
    a = J.run(prep, pose, LAMS, 10, correct)
    rng = np.random.default_rng(seed + 1)
    for sid, rec in prep["heldout"].items():
        if rec["subject"] == mouse:
            k = int(rng.integers(len(pose["heldout"][sid]["cands"])))
            rec["gt_M"] = pose["heldout"][sid]["cands"][k]["M"] + rng.normal(0, 0.5, (2, 3))
    b = J.run(prep, pose, LAMS, 10, correct)
    assert a["lambda"]["heldout"][mouse] == b["lambda"]["heldout"][mouse]
    for sid, rec in prep["heldout"].items():
        if rec["subject"] == mouse:
            assert a["heldout"]["joint"][sid]["cand_index"] == b["heldout"]["joint"][sid]["cand_index"]


def test_choose_lambda_ties_to_smaller():
    counts = {0.0: {"a": 1, "b": 2}, 2.0: {"a": 3, "b": 0}, 5.0: {"a": 2, "b": 1}}
    assert J.choose_lambda(counts, ["a", "b"], [5.0, 2.0, 0.0]) == 0.0
    assert J.choose_lambda(counts, ["a"], [0.0, 2.0, 5.0]) == 2.0
    assert J.choose_lambda(counts, [], [0.0, 2.0, 5.0]) == 0.0
