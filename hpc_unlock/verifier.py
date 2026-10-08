"""Verifier Stage (CPU): learned Pose_Verifier and region gate (Req 7.1-7.8).

Features per region (7.1), for the pose chosen by the Joint Stage (the candidate
dict ``entry["cand"]`` of pose_search)::

    soft, soft_margin, z, refine_margin, landing_dist, angle, scale, anisotropy

``landing_dist`` is the distance from the region's landing to the median landing
of the *other* registered regions of its ``(subject, ex_shape)`` group under the
same selection (0 if it is alone). Groups never span mice and only predicted
poses enter it, so no ground truth is used. Non-finite values (e.g. an infinite
anisotropy) become NaN and are imputed by the pipeline's median imputer.

Label (7.2): ``reg_lab.err(rec, M) < 5`` for regions with a GT transform. The
region without ``gt_M`` is left out of training and out of the kept counts, but
it is still gated and its pairs still enter the pooled F1. An unregistered region
(no candidate) has no features, prob 0, fails the v10 gate and is never kept.

Model (7.3): ``SimpleImputer(median) -> StandardScaler -> LogisticRegression(C=0.5)``
(design.md), trained leave-one-mouse-out for held-out probabilities and on all
three mice for test. A training set with one class (or none) is replaced by a
constant model with that class's probability (0 if empty) and logged.

Gate grid (7.4-7.8): for every tau in ``TAUS`` (0.00 .. 1.00, 21 values) a region
is kept iff ``v10 or prob >= tau`` with ``v10 = refine_margin >= 3 or z >= 5``.
Each row has kept / correct / wrong counts (GT regions) and the pooled pair F1 at
the pair threshold chosen by ``pairs.choose_threshold`` on that kept set.
Conservative = lowest tau with zero kept wrong, else ``unavailable`` (tau None,
v10 gate only). Aggressive = max F1, highest tau on ties.

Fold gate (Req 12.5): :func:`fold_gate` for mouse m uses only the rows of the two
other mice, with verifier probabilities from a nested leave-one-mouse-out that
never trains on m (each other mouse o is scored by a model fitted on the mice
outside {m, o}). The fold conservative tau therefore does not depend on m's
labels. The fold aggressive tau's F1 uses the pairs Checkpoint's LOO pair
probabilities, whose models for the other mice were trained with m's pair labels.

Checkpoint::

    {"selections": {sel: {
        "heldout": {sid: {"subject", "prob", "v10", "correct" (bool | None),
                          "features" (8,) | None,
                          "kept": {"conservative", "aggressive", "fold_conservative"}}},
        "test":    {sid: {"prob", "v10", "features",
                          "kept": {"conservative", "aggressive"}}},
        "grid": [{"tau", "kept", "correct", "wrong", "f1", "pair_threshold", "tp", "pred"}],
        "conservative": {"tau": float | None, "unavailable": bool},
        "aggressive": {"tau": float},
        "fold": {mouse: {"conservative": float | None, "unavailable": bool,
                         "aggressive": float}}}},
     "feature_names": [...], "taus": [...], "model": {...}}
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hpc_unlock import checkpoint
from hpc_unlock import joint as J
from hpc_unlock import pairs as P

SELECTIONS = P.SELECTIONS
FEATURES = ("soft", "soft_margin", "z", "refine_margin", "landing_dist",
            "angle", "scale", "anisotropy")
TAUS = np.round(np.arange(21) * 0.05, 2)                 # 0.00 .. 1.00 (Req 7.4)
V10_MARGIN = 3.0
V10_Z = 5.0
MODEL = {"imputer": "median", "scaler": "standard", "C": 0.5, "max_iter": 1000}


# ----------------------------------------------------------------------------
# Features, v10 gate, labels
# ----------------------------------------------------------------------------

def _cand(entry) -> Mapping | None:
    if entry is None:
        return None
    if entry.get("M") is None:
        return None
    return entry.get("cand")


def v10_gate(cand: Mapping | None) -> bool:
    """``refine_margin >= 3 or z >= 5`` of the chosen candidate (False if none)."""
    if cand is None:
        return False
    return bool(float(cand["refine_margin"]) >= V10_MARGIN or float(cand["z"]) >= V10_Z)


def landing_dists(records: Mapping[str, Mapping], entries: Mapping[str, Any]) -> dict:
    """``sid -> distance to the median landing of the group's other registered
    regions`` (0 if alone; None if the region is unregistered)."""
    land, group = {}, {}
    for sid, rec in records.items():
        c = _cand(entries.get(sid))
        if c is not None:
            land[sid] = np.asarray(c["landing"], float).reshape(2)
            group.setdefault(J._group_of(rec), []).append(sid)
    out = {sid: None for sid in records}
    for sids in group.values():
        for s in sids:
            others = [land[o] for o in sids if o != s]
            if not others:
                out[s] = 0.0
                continue
            med = np.median(np.array(others), axis=0)
            out[s] = float(np.linalg.norm(land[s] - med))
    return out


def region_features(cand: Mapping, landing_dist: float) -> np.ndarray:
    """The 8 features of :data:`FEATURES` (non-finite -> NaN)."""
    vals = {k: cand[k] for k in FEATURES if k != "landing_dist"}
    vals["landing_dist"] = landing_dist
    x = np.array([float(vals[k]) for k in FEATURES], float)
    x[~np.isfinite(x)] = np.nan
    return x


def has_gt(rec: Mapping) -> bool:
    return rec.get("gt_M") is not None and rec.get("gt_iv_c") is not None


def table(records: Mapping[str, Mapping], entries: Mapping[str, Any],
          correct: Callable[[Mapping, np.ndarray], bool] | None = None,
          labelled: bool = True) -> dict:
    """Per-split arrays in record order.

    ``{"sids", "subject", "X" (n, 8) (NaN rows if unregistered), "registered",
    "v10", "has_gt", "y" (bool; False where no GT)}``.
    """
    sids = list(records)
    dist = landing_dists(records, entries)
    n = len(sids)
    X = np.full((n, len(FEATURES)), np.nan)
    reg = np.zeros(n, bool)
    v10 = np.zeros(n, bool)
    gt = np.zeros(n, bool)
    y = np.zeros(n, bool)
    for k, sid in enumerate(sids):
        rec = records[sid]
        entry = entries.get(sid)
        c = _cand(entry)
        if c is not None:
            X[k] = region_features(c, dist[sid])
            reg[k] = True
            v10[k] = v10_gate(c)
        if labelled and has_gt(rec):
            gt[k] = True
            y[k] = bool(c is not None and correct(rec, np.asarray(entry["M"], float)))
    return {"sids": sids, "subject": np.array([str(records[s].get("subject")) for s in sids]),
            "X": X, "registered": reg, "v10": v10, "has_gt": gt, "y": y}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def make_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=MODEL["C"], max_iter=MODEL["max_iter"]))


def fit(X: np.ndarray, y: np.ndarray, estimator: Callable[[], Any] = make_model,
        note: str = ""):
    """Fitted model, or ``pairs.ConstantModel`` for an empty / single-class set (logged)."""
    y = np.asarray(y, bool)
    if len(y) == 0 or len(np.unique(y)) < 2:
        p = float(y[0]) if len(y) else 0.0
        checkpoint.log_line("VERIFIER_CONSTANT", f"{note} n={len(y)} constant prob={p}".strip())
        return P.ConstantModel(p)
    return estimator().fit(np.asarray(X, float), y)


def predict(model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if not len(X):
        return np.zeros(0)
    return np.clip(model.predict_proba(X)[:, 1], 0.0, 1.0)


def _train_mask(T: Mapping) -> np.ndarray:
    return T["registered"] & T["has_gt"]


def loo_probs(T: Mapping, estimator: Callable[[], Any] = make_model,
              exclude: Iterable[str] = ()) -> np.ndarray:
    """Leave-one-mouse-out probabilities for every row of ``T`` (Req 7.3).

    Rows of mouse g are scored by a model trained on the GT rows of the mice
    outside ``{g} | exclude``. Rows of excluded mice and unregistered rows get 0.
    """
    excl = {str(m) for m in exclude}
    probs = np.zeros(len(T["sids"]))
    train = _train_mask(T)
    for g in sorted(set(T["subject"].tolist()) - excl):
        rows = T["registered"] & (T["subject"] == g)
        if not rows.any():
            continue
        tr = train & ~np.isin(T["subject"], sorted(excl | {g}))
        model = fit(T["X"][tr], T["y"][tr], estimator,
                    note=f"heldout={g} exclude={sorted(excl)}")
        probs[rows] = predict(model, T["X"][rows])
    return probs


def all_mice_model(T: Mapping, estimator: Callable[[], Any] = make_model):
    """Model trained on every held-out GT row (the test model, Req 7.3)."""
    tr = _train_mask(T)
    return fit(T["X"][tr], T["y"][tr], estimator, note="test all-mice")


# ----------------------------------------------------------------------------
# Gate grid and selection rules
# ----------------------------------------------------------------------------

def kept_flags(probs, v10, registered, tau: float | None) -> np.ndarray:
    """``registered & (v10 | prob >= tau)``; ``tau None`` = v10 gate only (Req 7.7, 7.8)."""
    v10 = np.asarray(v10, bool)
    reg = np.asarray(registered, bool)
    if tau is None:
        return v10 & reg
    return reg & (v10 | (np.asarray(probs, float) >= float(tau)))


def gate_grid(sids: Sequence[str], probs, v10, registered, has_gt, y,
              pair_rows: Sequence, pair_probs: Mapping[str, np.ndarray], total_gt: int,
              taus: Sequence[float] = TAUS) -> list[dict]:
    """One row per tau: kept, kept correct, kept wrong (GT regions), pooled pair F1."""
    gt = np.asarray(has_gt, bool)
    y = np.asarray(y, bool)
    rows = []
    for tau in taus:
        keep = kept_flags(probs, v10, registered, float(tau))
        by = dict(zip(sids, keep.tolist()))
        thr, tp, pred, f1 = P.choose_threshold(pair_rows, pair_probs,
                                               {s: by.get(s, False) for s, *_ in pair_rows},
                                               total_gt)
        rows.append({"tau": float(tau), "kept": int(keep.sum()),
                     "correct": int((keep & gt & y).sum()),
                     "wrong": int((keep & gt & ~y).sum()),
                     "f1": float(f1), "pair_threshold": float(thr),
                     "tp": int(tp), "pred": int(pred)})
    return rows


def conservative(grid: Sequence[Mapping]) -> float | None:
    """Lowest tau with zero kept wrong poses; None = unavailable (Req 7.5, 7.8)."""
    ok = [r["tau"] for r in grid if r["wrong"] == 0]
    return min(ok) if ok else None


def aggressive(grid: Sequence[Mapping]) -> float:
    """Tau with the highest F1; the highest tau on ties (Req 7.6)."""
    best = None
    for r in sorted(grid, key=lambda r: r["tau"]):
        if best is None or r["f1"] >= best["f1"]:
            best = r
    return best["tau"]


def _subset(T: Mapping, mask: np.ndarray) -> dict:
    return {"sids": [s for s, m in zip(T["sids"], mask) if m],
            **{k: T[k][mask] for k in ("subject", "X", "registered", "v10", "has_gt", "y")}}


def fold_gate(m: str, T: Mapping, pair_entries: Mapping[str, Mapping],
              total_by_mouse: Mapping[str, int],
              estimator: Callable[[], Any] = make_model) -> dict:
    """Gate taus for held-out mouse ``m`` from the other two mice only (Req 12.5).

    Probabilities of each other mouse come from a model trained without m and
    without that mouse; the grid, counts and pair F1 use only the other mice's
    rows and GT pair totals.
    """
    m = str(m)
    probs = loo_probs(T, estimator, exclude=[m])
    other = T["subject"] != m
    S = _subset(T, other)
    p = probs[other]
    rows, pprobs = P.entry_rows({s: e for s, e in pair_entries.items()
                                 if str(e.get("subject")) != m})
    total = sum(int(v) for g, v in total_by_mouse.items() if str(g) != m)
    grid = gate_grid(S["sids"], p, S["v10"], S["registered"], S["has_gt"], S["y"],
                     rows, pprobs, total)
    cons = conservative(grid)
    return {"conservative": cons, "unavailable": cons is None,
            "aggressive": aggressive(grid), "grid": grid}


# ----------------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------------

def run_selection(heldout: Mapping[str, Mapping], test: Mapping[str, Mapping],
                  hentries: Mapping[str, Any], tentries: Mapping[str, Any],
                  pair_entries: Mapping[str, Mapping],
                  correct: Callable[[Mapping, np.ndarray], bool] | None,
                  estimator: Callable[[], Any] = make_model) -> dict:
    """Verifier output of one selection (see the module docstring)."""
    H = table(heldout, hentries, correct)
    Tt = table(test, tentries, labelled=False)
    probs = loo_probs(H, estimator)
    total_by_mouse: dict = {}
    for rec in heldout.values():
        g = str(rec.get("subject"))
        total_by_mouse[g] = total_by_mouse.get(g, 0) + int(rec.get("n_gt_pairs", 0))
    rows, pprobs = P.entry_rows(pair_entries)
    grid = gate_grid(H["sids"], probs, H["v10"], H["registered"], H["has_gt"], H["y"],
                     rows, pprobs, sum(total_by_mouse.values()))
    cons, aggr = conservative(grid), aggressive(grid)

    mice = sorted(set(H["subject"].tolist()))
    fold = {}
    for m in mice:
        f = fold_gate(m, H, pair_entries, total_by_mouse, estimator)
        fold[m] = {"conservative": f["conservative"], "unavailable": f["unavailable"],
                   "aggressive": f["aggressive"]}

    model = all_mice_model(H, estimator)
    tprobs = np.zeros(len(Tt["sids"]))
    if Tt["registered"].any():
        tprobs[Tt["registered"]] = predict(model, Tt["X"][Tt["registered"]])

    k_cons = kept_flags(probs, H["v10"], H["registered"], cons)
    k_aggr = kept_flags(probs, H["v10"], H["registered"], aggr)
    hout = {}
    for k, sid in enumerate(H["sids"]):
        g = str(H["subject"][k])
        k_fold = kept_flags(probs[k:k + 1], H["v10"][k:k + 1], H["registered"][k:k + 1],
                            fold[g]["conservative"])[0]
        hout[sid] = {"subject": g, "prob": float(probs[k]), "v10": bool(H["v10"][k]),
                     "correct": bool(H["y"][k]) if H["has_gt"][k] else None,
                     "features": H["X"][k].copy() if H["registered"][k] else None,
                     "kept": {"conservative": bool(k_cons[k]), "aggressive": bool(k_aggr[k]),
                              "fold_conservative": bool(k_fold)}}
    tk_cons = kept_flags(tprobs, Tt["v10"], Tt["registered"], cons)
    tk_aggr = kept_flags(tprobs, Tt["v10"], Tt["registered"], aggr)
    tout = {sid: {"prob": float(tprobs[k]), "v10": bool(Tt["v10"][k]),
                  "features": Tt["X"][k].copy() if Tt["registered"][k] else None,
                  "kept": {"conservative": bool(tk_cons[k]), "aggressive": bool(tk_aggr[k])}}
            for k, sid in enumerate(Tt["sids"])}
    return {"heldout": hout, "test": tout, "grid": grid,
            "conservative": {"tau": cons, "unavailable": cons is None},
            "aggressive": {"tau": aggr}, "fold": fold}


def run(prep: Mapping, joint_ck: Mapping, pairs_ck: Mapping,
        correct: Callable[[Mapping, np.ndarray], bool] | None = None,
        estimator: Callable[[], Any] = make_model,
        selections: Sequence[str] = SELECTIONS) -> dict:
    """The whole Stage on in-memory Checkpoints (pure apart from ``correct``)."""
    heldout, test = prep["heldout"], prep.get("test", {})
    if correct is None and any(has_gt(r) for r in heldout.values()):
        correct = J.default_correct()
    out = {}
    for sel in selections:
        out[sel] = run_selection(heldout, test, joint_ck["heldout"][sel],
                                 joint_ck.get("test", {}).get(sel, {}),
                                 pairs_ck["heldout"][sel], correct, estimator)
    return {"selections": out, "feature_names": list(FEATURES),
            "taus": [float(t) for t in TAUS], "model": dict(MODEL)}


def compute(cfg, ctx) -> dict:
    t0 = time.monotonic()
    out = run(ctx.load("prep"), ctx.load("joint"), ctx.load("pairs"))
    for sel, r in out["selections"].items():
        g = {row["tau"]: row for row in r["grid"]}
        cons, aggr = r["conservative"]["tau"], r["aggressive"]["tau"]
        if cons is None:
            checkpoint.log_line("CONSERVATIVE_UNAVAILABLE",
                                f"{sel}: no gate threshold has zero kept wrong poses; "
                                f"v10 gate only")
        c = g.get(cons)
        a = g[aggr]
        checkpoint.log_line(
            "VERIFIER_SELECTION",
            f"{sel}: conservative={cons} "
            + (f"kept={c['kept']} wrong={c['wrong']} f1={c['f1']:.4f} " if c else "")
            + f"aggressive={aggr} kept={a['kept']} correct={a['correct']} wrong={a['wrong']} "
              f"f1={a['f1']:.4f} pair_thr={a['pair_threshold']:.3f} "
              f"fold={ {m: f['conservative'] for m, f in r['fold'].items()} } "
              f"test_kept_cons={sum(e['kept']['conservative'] for e in r['test'].values())} "
              f"test_kept_aggr={sum(e['kept']['aggressive'] for e in r['test'].values())}")
    checkpoint.log_line("VERIFIER_DONE", f"elapsed={time.monotonic() - t0:.1f}s")
    return out
