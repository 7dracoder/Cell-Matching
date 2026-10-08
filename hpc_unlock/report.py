"""Run_Report: ``hpc_unlock_report.json`` and ``hpc_unlock_report.md`` (Req 11).

``build(state)`` turns the plain-dict run state into one report dict;
``render_md(report)`` and ``json.dumps(report)`` both come from that dict, so
the two files always list the same ranking and the same scores (11.1).
``write(report, root)`` writes both files temp-then-rename.

Imports only the standard library (plus ``hpc_unlock.checkpoint``, which is
also standard library only), so a plain Python kernel can import it.

Input schema of ``state`` (plain dicts; scores are unrounded floats, counts
are ints; any score may be ``None`` when it was not measured)::

    {
      "run": {                      # optional
        "fingerprint": str, "smoke": bool,
      },
      "baseline": {                 # required (Held_Out_Set values)
        "full": float, "f1": float, "pq_iv": float, "pq_ex": float,
        "kept": int, "kept_wrong": int,
        "per_mouse_f1": {mouse: float},          # optional
        "reproduced": bool | None,  # None = tolerance not applicable (smoke)
        "expected": {"f1": 0.472, "full": 0.5186, "tol": 0.005},  # optional
      },
      "configs": [                  # every evaluated registration-only config
        {
          "name": str, "accepted": bool,
          "full": float, "f1": float, "pq_iv": float, "pq_ex": float,
          "per_mouse_f1": {mouse: float}, "kept": int, "kept_wrong": int,
          "csv": str | None,                     # written candidate CSV path
          "rejected_reason": str | None,         # e.g. "full <= baseline"
          "unlock_table": [{"sid": str, "baseline_pairs": int,
                            "candidate_pairs": int, "newly_unlocked": bool}],
        }, ...
      ],
      "selftrain": {                # optional; default status "disabled"
        "status": "disabled" | "no_candidate" | "accepted"
                  | "no_confident_regions" | <other, e.g. "skipped">,
        "full": float, "compared_to": float,     # measured vs comparison target
        "f1", "pq_iv", "pq_ex", "kept", "kept_wrong", "per_mouse_f1",
        "csv", "unlock_table": as for a config (used when "accepted"),
        "reason": str,                           # optional free text
        "no_confident_mice": [mouse, ...],       # 12.12
      },
      "diagnostics": {              # optional
        "pose_search": {"sigma": float, "n_with_gt": int,
                        "correct_in_candidates": int, "correct_first": int},   # 4.10
        "gpu_scan": {"n_with_gt": int, "correct_in_raw": int,
                     "correct_in_merged": int},                             # 5.10
        "joint": {"total": {"joint": int, "independent": int},
                  "per_mouse": {mouse: {"joint": int, "independent": int}}},  # 6.7
        "verifier": {"conservative_threshold": float | None,
                     "aggressive_threshold": float | None,
                     "conservative_unavailable": bool},                     # 7.8
        <other key>: any JSON value,
      },
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JSON_NAME = "hpc_unlock_report.json"
MD_NAME = "hpc_unlock_report.md"
BASELINE_CSV = "submission_v10_cpgate.csv"
FALLBACK_CSV = "submission_v7_grow15.csv"
PUBLIC_BAR = 0.48893
NO_CANDIDATE = "NO_CANDIDATE"
DISCLAIMER = ("Note: held-out gains are estimates; a public score above 0.65 "
              "is the target, not guaranteed.")
SCORE_KEYS = ("full", "f1", "pq_iv", "pq_ex")
SELFTRAIN_NAME = "selftrain"


# ---------------------------------------------------------------- helpers
def _num(x: Any) -> float | None:
    return None if x is None else float(x)


def _int(x: Any) -> int | None:
    return None if x is None else int(x)


def fmt(x: Any) -> str:
    """Score to 4 decimals (9.6); ``n/a`` when not measured."""
    return "n/a" if x is None else f"{float(x):.4f}"


def _metrics(d: dict) -> dict:
    out = {k: _num(d.get(k)) for k in SCORE_KEYS}
    out["kept"] = _int(d.get("kept"))
    out["kept_wrong"] = _int(d.get("kept_wrong"))
    out["per_mouse_f1"] = {str(m): _num(v)
                           for m, v in (d.get("per_mouse_f1") or {}).items()}
    return out


def _unlock(table: list | None) -> list[dict]:
    rows = []
    for r in table or []:
        b, c = int(r["baseline_pairs"]), int(r["candidate_pairs"])
        rows.append({"sid": str(r["sid"]), "baseline_pairs": b,
                     "candidate_pairs": c,
                     "newly_unlocked": bool(r.get("newly_unlocked",
                                                  b == 0 and c >= 1))})
    return rows


def rank_key(c: dict) -> tuple:
    """Higher full, then higher F1, then fewer kept wrong-Pose regions (11.2)."""
    return (-float(c["full"]), -float(c["f1"]), int(c.get("kept_wrong") or 0))


def _csv_name(c: dict) -> str:
    return Path(c["csv"]).name if c.get("csv") else c["name"]


# ---------------------------------------------------------------- build
def build(state: dict) -> dict:
    """Build the Run_Report dict from the run ``state`` (schema in module doc)."""
    run = dict(state.get("run") or {})
    smoke = bool(run.get("smoke", False))
    bstate = state["baseline"]
    baseline = {"name": f"Baseline ({BASELINE_CSV})", **_metrics(bstate)}

    # Baseline reproduction (9.3, 9.4)
    reproduced = bstate.get("reproduced")
    expected = dict(bstate.get("expected")
                    or {"f1": 0.472, "full": 0.5186, "tol": 0.005})
    if smoke or reproduced is None:
        repro_status = "n/a (smoke)"
    else:
        repro_status = "ok" if reproduced else "failed"
    repro = {"status": repro_status, "measured_full": baseline["full"],
             "measured_f1": baseline["f1"], "expected": expected}
    repro_failed = repro_status == "failed"

    # Evaluated configurations (9.6) and acceptance split (9.5)
    evaluated, candidates, rejected = [], [], []
    for cfg in state.get("configs") or []:
        entry = {"name": str(cfg["name"]), **_metrics(cfg),
                 "csv": cfg.get("csv"),
                 "accepted": bool(cfg.get("accepted")) and not repro_failed,
                 "unlock_table": _unlock(cfg.get("unlock_table"))}
        if entry["accepted"] and (entry["full"] is None or entry["f1"] is None):
            entry["accepted"] = False
            cfg = {**cfg, "rejected_reason": "missing held-out score"}
        evaluated.append(entry)
        if entry["accepted"]:
            candidates.append(entry)
        else:
            reason = cfg.get("rejected_reason")
            if repro_failed:
                reason = "Baseline reproduction failed; no CSV written"
            rejected.append({"name": entry["name"], "full": entry["full"],
                             "compared_to": baseline["full"],
                             "reason": reason or "full <= Baseline full"})

    # Self-training (12.2, 12.7, 12.11, 12.12)
    st = dict(state.get("selftrain") or {"status": "disabled"})
    status = str(st.get("status", "disabled"))
    no_conf = [str(m) for m in st.get("no_confident_mice") or []]
    selftrain = {"status": status, "full": _num(st.get("full")),
                 "compared_to": _num(st.get("compared_to")),
                 "csv": st.get("csv"), "no_confident_mice": no_conf}
    if status == "disabled":
        selftrain["summary"] = ("Self_Training_Stage disabled; "
                                "no self-trained CSV written.")
    elif status == "no_candidate":
        selftrain["summary"] = (f"{NO_CANDIDATE}: self-trained held-out full "
                                f"{fmt(selftrain['full'])} <= compared "
                                f"{fmt(selftrain['compared_to'])}.")
    elif status == "no_confident_regions":
        selftrain["summary"] = f"{NO_CANDIDATE}: no confident regions."
    elif status == "accepted" and not repro_failed:
        selftrain["summary"] = (f"accepted: self-trained held-out full "
                                f"{fmt(selftrain['full'])} > compared "
                                f"{fmt(selftrain['compared_to'])}.")
        entry = {"name": SELFTRAIN_NAME, **_metrics(st), "csv": st.get("csv"),
                 "accepted": True, "unlock_table": _unlock(st.get("unlock_table"))}
        if entry["full"] is not None and entry["f1"] is not None:
            candidates.append(entry)
    else:
        selftrain["summary"] = str(st.get("reason") or status)
    if st.get("reason") and status != "accepted":
        selftrain["reason"] = str(st["reason"])

    ranking = [{"rank": i + 1, **c}
               for i, c in enumerate(sorted(candidates, key=rank_key))]

    # Recommendations (11.5, 11.6)
    if ranking:
        nxt = ranking[1]["name"] if len(ranking) > 1 else None
        note = (f"Replace {BASELINE_CSV} with the next-ranked accepted "
                f"candidate{f' ({nxt})' if nxt else ''} only if you see a "
                f"candidate public Kaggle score above {PUBLIC_BAR}.")
        rec = {"status": "accepted",
               "finals": [_csv_name(ranking[0]), BASELINE_CSV], "note": note}
    else:
        rec = {"status": NO_CANDIDATE, "finals": [BASELINE_CSV, FALLBACK_CSV],
               "note": (f"{NO_CANDIDATE}: no configuration beat the Baseline "
                        f"held-out full {fmt(baseline['full'])}.")}

    return {
        "title": "HPC registration unlock: Run_Report",
        "run": run,
        "disclaimer": DISCLAIMER,
        "status": rec["status"],
        "baseline_reproduction": repro,
        "baseline": baseline,
        "ranking": ranking,
        "rejected": rejected,
        "evaluated": evaluated,
        "recommendation": rec,
        "selftrain": selftrain,
        "diagnostics": dict(state.get("diagnostics") or {}),
    }


# ---------------------------------------------------------------- markdown
def _score_row(name: str, m: dict, extra: list[str] = ()) -> str:
    kept = "n/a" if m.get("kept") is None else str(m["kept"])
    kw = "n/a" if m.get("kept_wrong") is None else str(m["kept_wrong"])
    cells = [*extra, name, fmt(m["full"]), fmt(m["f1"]), fmt(m["pq_iv"]),
             fmt(m["pq_ex"]), kept, kw]
    return "| " + " | ".join(cells) + " |"


def _per_mouse(m: dict) -> str:
    pm = m.get("per_mouse_f1") or {}
    return ", ".join(f"{k} {fmt(v)}" for k, v in pm.items()) or "n/a"


def _diag_lines(diag: dict) -> list[str]:
    out = []
    ps = diag.get("pose_search")
    if ps:
        n = ps.get("n_with_gt", 46)
        out.append(f"- Pose_Search (sigma {ps.get('sigma')} px): Correct_Pose "
                   f"among candidates {ps.get('correct_in_candidates')}/{n}, "
                   f"ranked first by Soft_Score {ps.get('correct_first')}/{n}")
    gs = diag.get("gpu_scan")
    if gs:
        n = gs.get("n_with_gt", 46)
        out.append(f"- GPU_Pose_Scan: Correct_Pose among raw scan candidates "
                   f"{gs.get('correct_in_raw')}/{n}, among merged and refined "
                   f"candidates {gs.get('correct_in_merged')}/{n}")
    jt = diag.get("joint")
    if jt:
        tot = jt.get("total") or {}
        out.append(f"- Joint_Registrar Correct_Pose selections: joint "
                   f"{tot.get('joint')}, independent {tot.get('independent')}")
        for m, v in (jt.get("per_mouse") or {}).items():
            out.append(f"  - {m}: joint {v.get('joint')}, "
                       f"independent {v.get('independent')}")
    vf = diag.get("verifier")
    if vf:
        if vf.get("conservative_unavailable"):
            out.append("- Pose_Verifier: conservative gate unavailable "
                       "(v10 gate only for the conservative configuration)")
        else:
            out.append(f"- Pose_Verifier: conservative threshold "
                       f"{vf.get('conservative_threshold')}")
        out.append(f"- Pose_Verifier: aggressive threshold "
                   f"{vf.get('aggressive_threshold')}")
    for k, v in diag.items():
        if k not in ("pose_search", "gpu_scan", "joint", "verifier"):
            out.append(f"- {k}: {json.dumps(v, sort_keys=True)}")
    return out or ["- none recorded"]


def render_md(report: dict) -> str:
    """Render the report dict as Markdown (same ranking and scores as the JSON)."""
    L: list[str] = [f"# {report['title']}", "", report["disclaimer"], ""]
    run = report.get("run") or {}
    if run:
        L += [f"Run: {', '.join(f'{k}={v}' for k, v in run.items())}", ""]

    rec = report["recommendation"]
    L += ["## Recommendation", ""]
    if rec["status"] == NO_CANDIDATE:
        L.append(f"**{NO_CANDIDATE}**")
        L.append("")
    L += [f"Finals: 1. `{rec['finals'][0]}`  2. `{rec['finals'][1]}`", "",
          rec["note"], ""]

    rp = report["baseline_reproduction"]
    L += ["## Baseline reproduction", ""]
    exp = rp["expected"]
    line = (f"Status: {rp['status']} (measured full {fmt(rp['measured_full'])},"
            f" F1 {fmt(rp['measured_f1'])}; expected full {exp.get('full')}, "
            f"F1 {exp.get('f1')}, tolerance {exp.get('tol')})")
    if rp["status"] == "failed":
        line += ". Baseline reproduction failure: no candidate CSV written."
    L += [line, ""]

    hdr = ["| config | full | F1 | PQ_iv | PQ_ex | kept | kept wrong |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    L += ["## Ranking (held-out)", "",
          "| rank " + hdr[0], "| --- " + hdr[1]]
    b = report["baseline"]
    L.append(_score_row(b["name"], b, ["-"]))
    for c in report["ranking"]:
        L.append(_score_row(c["name"], c, [str(c["rank"])]))
    L.append("")
    if report["ranking"]:
        L += ["CSV files:", ""]
        L += [f"- {c['rank']}. {c['name']}: `{c.get('csv')}`"
              for c in report["ranking"]]
        L.append("")

    if report["rejected"]:
        L += ["## Rejected configurations", "",
              "| config | full | compared to (Baseline full) | reason |",
              "| --- | --- | --- | --- |"]
        L += [f"| {r['name']} | {fmt(r['full'])} | {fmt(r['compared_to'])} "
              f"| {r['reason']} |" for r in report["rejected"]]
        L.append("")

    L += ["## All evaluated configurations", "", *hdr]
    L += [_score_row(e["name"], e) for e in report["evaluated"]]
    L += ["", "Per-mouse pair F1:", "", f"- Baseline: {_per_mouse(b)}"]
    L += [f"- {e['name']}: {_per_mouse(e)}" for e in report["evaluated"]]
    L.append("")

    st = report["selftrain"]
    L += ["## Self-training", "", f"Status: {st['status']}. {st['summary']}"]
    if st.get("reason"):
        L.append(f"Reason: {st['reason']}")
    if st["no_confident_mice"]:
        L.append("Mice with no confident regions (Baseline ex-vivo masks used): "
                 + ", ".join(st["no_confident_mice"]))
    L.append("")

    L += ["## Registration diagnostics", "", *_diag_lines(report["diagnostics"]), ""]

    for c in report["ranking"]:
        if not c["unlock_table"]:
            continue
        n_new = sum(r["newly_unlocked"] for r in c["unlock_table"])
        L += [f"## Unlock table: {c['name']} ({n_new} newly unlocked)", "",
              "| sample | Baseline pairs | candidate pairs | newly unlocked |",
              "| --- | --- | --- | --- |"]
        L += [f"| {r['sid']} | {r['baseline_pairs']} | {r['candidate_pairs']} "
              f"| {'yes' if r['newly_unlocked'] else 'no'} |"
              for r in c["unlock_table"]]
        L.append("")
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------- write
def write(report: dict, root: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``hpc_unlock_report.json`` and ``.md`` atomically into ``root``.

    ``root`` defaults to the Project_Folder root (``hpc_unlock.paths.ROOT``).
    Returns ``(json_path, md_path)``.
    """
    from hpc_unlock.checkpoint import _write_atomic

    if root is None:
        from hpc_unlock.paths import ROOT as root  # noqa: N811
    root = Path(root)
    jtext = json.dumps(report, indent=2) + "\n"
    mtext = render_md(report)
    jp, mp = root / JSON_NAME, root / MD_NAME
    _write_atomic(jp, lambda f: f.write(jtext.encode("utf-8")))
    _write_atomic(mp, lambda f: f.write(mtext.encode("utf-8")))
    return jp, mp
