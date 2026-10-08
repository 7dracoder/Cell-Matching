"""Joint Stage (CPU): per mouse/canvas joint pose selection with ICM (Req 6.1-6.7).

Groups (6.1): held-out and test regions are grouped by ``(subject, ex_shape)``
(the prep record's ``group``). Inside a group, regions with the same ``dup_key``
(pixel-identical in-vivo and ex-vivo images) collapse into one variable; its
candidate list is the first member's and its pick is copied to every member
(6.4). A region whose candidate list is empty is unregistered: it gets no pose
and is left out of the consistency term (6.6).

Objective (6.2), per group, with ``L_u`` = the top ``joint_top_l`` candidates of
variable u by Soft_Score::

    J(c) = sum_u soft(c_u) - lam * sum_{u<v} wbar_uv * psi(c_u, c_v)
    psi(a, b) = min(1, |land_a - land_b|^2 / TAU_L^2) + min(1, |A_a - A_b|_F^2 / TAU_A^2)
    w_uv      = exp(-|off_u - off_v| / RHO) / sum_{v' != u} exp(-|off_u - off_v'| / RHO)
    wbar_uv   = (w_uv + w_vu) / 2

``w_uv`` is normalised per u (row-normalised), so it is not symmetric; the pair
weight is its symmetric part, which makes J independent of the variable order
(it equals ``sum_{u<v} w_uv psi`` whenever ``w`` is symmetric) and makes the ICM
update below an exact coordinate maximisation of J.

Solver: if ``prod |L_u| <= EXHAUSTIVE_MAX`` all assignments are enumerated in
lexicographic order (ties: first). Otherwise ICM: start at the per-variable
argmax of soft (ties: lowest index); sweep the variables in descending
Soft_Margin of their current pick (ties: lower variable index), setting
``c_u <- argmax_c soft(c) - lam * sum_v wbar_uv psi(c, c_v)`` (ties: lowest
index); stop after a sweep without change or after MAX_SWEEPS sweeps. Then
restart from every candidate of the RESTART_VARS highest-Soft_Margin variables
(that variable held fixed during the first sweep) and keep the run with the
largest J (ties: earliest run). At ``lam = 0`` or for a single variable, the
result is the independent argmax (6.3).

lambda (6.5): for held-out mouse m, lambda in ``cfg.joint_lambdas`` maximises the
Correct_Pose count on the other two mice (ties: smaller lambda). The solutions
use predicted candidates only; ground truth only scores lambda, and m's own GT
never enters m's lambda or m's selection. Test lambda maximises the count over
all held-out mice.

Checkpoint::

    {"heldout": {"independent": {sid: entry}, "joint": {sid: entry}},
     "test":    {"independent": {sid: entry}, "joint": {sid: entry}},
     "lambda":  {"heldout": {mouse: lam}, "test": lam},
     "unregistered": [sid, ...],
     "diagnostics": {"total": {"joint": int, "independent": int},
                     "per_mouse": {mouse: {"joint": int, "independent": int}},
                     "counts_by_lambda": {mouse: {lam: int}}, "n_with_gt": int,
                     "groups": {"heldout": int, "test": int}, ...}}

    entry = {"M": (2, 3) array | None, "score": refine_score (0.0 if unregistered),
             "cand_index": index into pose_search's candidate list | None,
             "cand": candidate dict | None}
"""
from __future__ import annotations

import itertools
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from hpc_unlock import checkpoint

TAU_L = 60.0              # px, landing scale
TAU_A = 0.06              # linear-part scale (Frobenius)
RHO = 400.0               # px, mosaic offset proximity
MAX_SWEEPS = 50
RESTART_VARS = 3
EXHAUSTIVE_MAX = 100_000
CORRECT_ERR = 5.0         # px, reg_lab.err < 5 = Correct_Pose
SPLITS = ("heldout", "test")


# ----------------------------------------------------------------------------
# Groups and variables
# ----------------------------------------------------------------------------

def _group_of(rec: Mapping) -> tuple:
    g = rec.get("group")
    if g is None:
        g = (rec["subject"], rec["ex_shape"])
    subj, shape = g
    return (str(subj), (int(shape[0]), int(shape[1])))


def top_list(cands: Sequence[Mapping], top_l: int) -> list[int]:
    """Indices of the top ``top_l`` candidates by soft, descending (stable)."""
    if not cands:
        return []
    soft = np.array([float(c["soft"]) for c in cands])
    order = np.argsort(-soft, kind="stable")
    return [int(i) for i in order[:top_l]]


def build_groups(records: Mapping[str, Mapping], cands: Mapping[str, Sequence[Mapping]],
                 top_l: int) -> tuple[dict, list[str]]:
    """``(groups, unregistered)``.

    ``groups[key]`` is a list of variables in first-appearance order, each
    ``{"sids", "dup_key", "offset", "index" (into the sid's candidate list),
    "cands" (the L_u candidate dicts)}``. ``unregistered`` lists the sids whose
    variable has an empty candidate list (record order).
    """
    groups: dict = {}
    by_key: dict = {}
    for sid, rec in records.items():
        g = _group_of(rec)
        k = (g, rec["dup_key"])
        if k in by_key:
            by_key[k]["sids"].append(sid)
            continue
        idx = top_list(cands[sid], top_l)
        var = {"sids": [sid], "dup_key": rec["dup_key"],
               "offset": np.asarray(rec["offset"], float).reshape(2),
               "index": idx, "cands": [cands[sid][i] for i in idx]}
        by_key[k] = var
        groups.setdefault(g, []).append(var)
    empty = {s for var in by_key.values() if not var["cands"] for s in var["sids"]}
    unregistered = [s for s in records if s in empty]
    groups = {g: [v for v in vs if v["cands"]] for g, vs in groups.items()}
    return {g: vs for g, vs in groups.items() if vs}, unregistered


# ----------------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------------

def psi_matrix(ca: Sequence[Mapping], cb: Sequence[Mapping]) -> np.ndarray:
    """``psi(a, b)`` for every a in ``ca``, b in ``cb``."""
    la = np.array([np.asarray(c["landing"], float).reshape(2) for c in ca])
    lb = np.array([np.asarray(c["landing"], float).reshape(2) for c in cb])
    Aa = np.array([np.asarray(c["M"], float)[:, :2].ravel() for c in ca])
    Ab = np.array([np.asarray(c["M"], float)[:, :2].ravel() for c in cb])
    dl = ((la[:, None, :] - lb[None, :, :]) ** 2).sum(-1)
    dA = ((Aa[:, None, :] - Ab[None, :, :]) ** 2).sum(-1)
    return np.minimum(1.0, dl / TAU_L ** 2) + np.minimum(1.0, dA / TAU_A ** 2)


def offset_weights(offsets: np.ndarray) -> np.ndarray:
    """Symmetric pair weights ``wbar`` (zero diagonal) from row-normalised ``w``."""
    off = np.asarray(offsets, float).reshape(-1, 2)
    n = len(off)
    if n < 2:
        return np.zeros((n, n))
    d = np.linalg.norm(off[:, None, :] - off[None, :, :], axis=-1)
    e = np.exp(-d / RHO)
    np.fill_diagonal(e, 0.0)
    w = e / e.sum(1, keepdims=True)
    return 0.5 * (w + w.T)


class Problem:
    """One group's joint selection problem (variables with non-empty L_u)."""

    def __init__(self, variables: Sequence[Mapping]):
        self.n = len(variables)
        self.soft = [np.array([float(c["soft"]) for c in v["cands"]]) for v in variables]
        self.margin = [np.array([float(c.get("soft_margin", 0.0)) for c in v["cands"]])
                       for v in variables]
        self.sizes = [len(s) for s in self.soft]
        self.W = offset_weights(np.array([v["offset"] for v in variables]).reshape(-1, 2))
        self.psi = {}
        for u in range(self.n):
            for v in range(u + 1, self.n):
                P = psi_matrix(variables[u]["cands"], variables[v]["cands"])
                self.psi[(u, v)] = P
                self.psi[(v, u)] = P.T

    def objective(self, c: Sequence[int], lam: float) -> float:
        J = float(sum(self.soft[u][c[u]] for u in range(self.n)))
        if lam == 0:
            return J
        pen = 0.0
        for u in range(self.n):
            for v in range(u + 1, self.n):
                pen += self.W[u, v] * self.psi[(u, v)][c[u], c[v]]
        return J - lam * pen

    def independent(self) -> list[int]:
        return [int(np.argmax(s)) for s in self.soft]

    def local(self, u: int, c: Sequence[int], lam: float) -> np.ndarray:
        val = self.soft[u].copy()
        for v in range(self.n):
            if v != u:
                val -= lam * self.W[u, v] * self.psi[(u, v)][:, c[v]]
        return val

    def icm(self, init: Sequence[int], lam: float, skip_first: int | None = None) -> list[int]:
        c = list(init)
        for sweep in range(MAX_SWEEPS):
            keys = [-self.margin[u][c[u]] for u in range(self.n)]
            order = sorted(range(self.n), key=lambda u: (keys[u], u))
            changed = False
            for u in order:
                if sweep == 0 and u == skip_first:
                    continue
                k = int(np.argmax(self.local(u, c, lam)))
                if k != c[u]:
                    c[u], changed = k, True
            if not changed:
                break
        return c

    def exhaustive(self, lam: float) -> list[int]:
        grids = np.indices(self.sizes).reshape(self.n, -1).T     # lexicographic order
        J = np.zeros(len(grids))
        for u in range(self.n):
            J += self.soft[u][grids[:, u]]
        if lam != 0:
            pen = np.zeros(len(grids))
            for u in range(self.n):
                for v in range(u + 1, self.n):
                    pen += self.W[u, v] * self.psi[(u, v)][grids[:, u], grids[:, v]]
            J = J - lam * pen
        return [int(k) for k in grids[int(np.argmax(J))]]

    def solve(self, lam: float) -> list[int]:
        lam = float(lam)
        if lam < 0:
            raise ValueError(f"joint: lambda must be >= 0, got {lam}")
        init = self.independent()
        if lam == 0 or self.n <= 1:                    # Req 6.3: empty pair sum
            return init
        if int(np.prod([float(s) for s in self.sizes])) <= EXHAUSTIVE_MAX:
            return self.exhaustive(lam)
        best = self.icm(init, lam)
        best_J = self.objective(best, lam)
        m0 = [self.margin[u][init[u]] for u in range(self.n)]
        top = sorted(range(self.n), key=lambda u: (-m0[u], u))[:RESTART_VARS]
        for u in top:
            for k in range(self.sizes[u]):
                start = list(init)
                start[u] = k
                c = self.icm(start, lam, skip_first=u)
                J = self.objective(c, lam)
                if J > best_J:
                    best, best_J = c, J
        return best


# ----------------------------------------------------------------------------
# Selections
# ----------------------------------------------------------------------------

def _entry(var: Mapping | None, k: int | None) -> dict:
    if var is None or k is None:
        return {"M": None, "score": 0.0, "cand_index": None, "cand": None}
    cand = var["cands"][k]
    return {"M": np.asarray(cand["M"], float).copy(), "score": float(cand["refine_score"]),
            "cand_index": int(var["index"][k]), "cand": cand}


def select_split(records: Mapping[str, Mapping], groups: Mapping, problems: Mapping,
                 lam_of: Callable[[tuple], float]) -> dict:
    """``{sid: entry}`` in record order; ``lam_of(group_key)`` gives each group's lambda."""
    out = {}
    for g, variables in groups.items():
        c = problems[g].solve(lam_of(g))
        for var, k in zip(variables, c):
            for sid in var["sids"]:
                out[sid] = _entry(var, k)
    return {sid: out.get(sid, _entry(None, None)) for sid in records}


def default_correct() -> Callable[[Mapping, np.ndarray], bool]:
    from hpc_unlock import pose_search
    err = pose_search.research().reg_lab.err
    return lambda rec, M: bool(err(rec, np.asarray(M, float)) < CORRECT_ERR)


def _has_gt(rec: Mapping) -> bool:
    return rec.get("gt_M") is not None and rec.get("gt_iv_c") is not None


def choose_lambda(counts: Mapping[float, Mapping[str, int]], mice: Sequence[str],
                  lambdas: Sequence[float]) -> float:
    """Largest summed count over ``mice``; ties to the smaller lambda."""
    best, best_n = None, None
    for lam in sorted(float(x) for x in lambdas):
        n = sum(counts[lam].get(m, 0) for m in mice)
        if best_n is None or n > best_n:
            best, best_n = lam, n
    return best


def run(prep: Mapping, pose: Mapping, lambdas: Sequence[float], top_l: int,
        correct: Callable[[Mapping, np.ndarray], bool] | None = None) -> dict:
    """The whole Stage on in-memory checkpoints (pure apart from ``correct``)."""
    grid = sorted({float(x) for x in lambdas})         # lambda candidates (cfg)
    if not grid or grid[0] < 0:
        raise ValueError(f"joint: lambdas must be a non-empty set of values >= 0, got {lambdas}")
    solve_at = sorted(set(grid) | {0.0})               # 0 = independent selection
    structs, unreg = {}, []
    for split in SPLITS:
        recs = prep.get(split, {})
        cands = {}
        for sid in recs:
            if sid not in pose.get(split, {}):
                raise KeyError(f"joint: pose_search has no entry for {split} {sid}")
            cands[sid] = list(pose[split][sid]["cands"])
        groups, u = build_groups(recs, cands, int(top_l))
        problems = {g: Problem(vs) for g, vs in groups.items()}
        structs[split] = (recs, groups, problems)
        unreg += u

    # Held-out solutions for every lambda in the grid; GT only scores them.
    recs, groups, problems = structs["heldout"]
    mice = sorted({str(r["subject"]) for r in recs.values()})
    gt_sids = [s for s, r in recs.items() if _has_gt(r)]
    if gt_sids and correct is None:
        correct = default_correct()
    sols, counts = {}, {}
    for lam in solve_at:
        sel = select_split(recs, groups, problems, lambda g, lam=lam: lam)
        sols[lam] = sel
        cnt = {m: 0 for m in mice}
        for s in gt_sids:
            if sel[s]["M"] is not None and correct(recs[s], sel[s]["M"]):
                cnt[str(recs[s]["subject"])] += 1
        counts[lam] = cnt

    lam_mouse = {m: choose_lambda(counts, [o for o in mice if o != m], grid) for m in mice}
    h_joint = {s: sols[lam_mouse[str(r["subject"])]][s] for s, r in recs.items()}
    h_indep = sols[0.0]
    lam_test = choose_lambda(counts, mice, grid)

    trecs, tgroups, tproblems = structs["test"]
    t_indep = select_split(trecs, tgroups, tproblems, lambda g: 0.0)
    t_joint = select_split(trecs, tgroups, tproblems, lambda g: lam_test)

    per_mouse = {m: {"joint": int(counts[lam_mouse[m]][m]), "independent": int(counts[0.0][m])}
                 for m in mice}
    diag = {"total": {"joint": sum(v["joint"] for v in per_mouse.values()),
                      "independent": sum(v["independent"] for v in per_mouse.values())},
            "per_mouse": per_mouse,
            "counts_by_lambda": {m: {lam: int(counts[lam][m]) for lam in grid} for m in mice},
            "n_with_gt": len(gt_sids),
            "groups": {sp: len(structs[sp][1]) for sp in SPLITS},
            "variables": {sp: sum(len(v) for v in structs[sp][1].values()) for sp in SPLITS},
            "lambdas": grid}
    return {"heldout": {"independent": h_indep, "joint": h_joint},
            "test": {"independent": t_indep, "joint": t_joint},
            "lambda": {"heldout": lam_mouse, "test": lam_test},
            "unregistered": unreg, "diagnostics": diag}


def compute(cfg, ctx) -> dict:
    t0 = time.monotonic()
    prep = ctx.load("prep")
    pose = ctx.load("pose_search")
    out = run(prep, pose, cfg.joint_lambdas, cfg.joint_top_l)
    d = out["diagnostics"]
    for sid in out["unregistered"]:
        checkpoint.log_line("UNREGISTERED", sid)
    for m, v in d["per_mouse"].items():
        checkpoint.log_line("JOINT_MOUSE", f"{m} lambda={out['lambda']['heldout'][m]} "
                                           f"joint={v['joint']} independent={v['independent']} "
                                           f"by_lambda={d['counts_by_lambda'][m]}")
    checkpoint.log_line("JOINT_DONE", f"gt={d['n_with_gt']} joint={d['total']['joint']} "
                                      f"independent={d['total']['independent']} "
                                      f"test_lambda={out['lambda']['test']} groups={d['groups']} "
                                      f"unregistered={len(out['unregistered'])} "
                                      f"elapsed={time.monotonic() - t0:.1f}s")
    return out
