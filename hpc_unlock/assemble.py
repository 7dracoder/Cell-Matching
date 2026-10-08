"""Assemble Stage (CPU, last): candidate CSVs, Format_Checker, unlock table, Run_Report.

Req 10.1-10.8, 9.4, 9.5, 11, 12.2, 12.8.

Registration-only candidates (10.1, 10.2, 10.8)
-----------------------------------------------
One CSV per *accepted* validate configuration, named
``submission_v13_<config>.csv`` with ``<config>`` = ``<selection>_<gate>``
(``indep_cons``, ``indep_aggr``, ``joint_cons``, ``joint_aggr``), so the file
name identifies both the selection and the gate (10.4). The Baseline rows
(``submission_v10_cpgate.csv``) are read with ``csv.DictReader``; for every row
the prep test record's ``iv_ids`` / ``ex_ids`` must equal the JSON key order of
the row's ``invivo_instances`` / ``exvivo_instances``. The config's
``test_pairs[sid]`` index pairs are mapped to those IDs and only ``match_pairs``
is replaced (``json.dumps``, the Baseline format). Every other column string is
copied unchanged and written with the Baseline's dialect (``csv`` excel dialect,
minimal quoting, CRLF), which round-trips the Baseline byte-for-byte. A region
without an entry in ``test_pairs`` gets ``[]``.

Self-trained candidate (12.8)
-----------------------------
``submission_v13_selftrain.csv``: ``invivo_instances`` copied from the Baseline,
``exvivo_instances = json.dumps(cellmatch.labels_to_rles(labels, prefix))`` with
the Baseline's ex-ID prefix (``EXP``), pairs on the new IDs. Interface of the
``selftrain_pairs`` Checkpoint read here (task 12.6 writes it)::

    {"accepted": bool,
     "status": str,                    # optional: "accepted" | "no_candidate"
                                       #   | "no_confident_regions" | other
     "full", "f1", "pq_iv", "pq_ex": float, "kept", "kept_wrong": int,
     "per_mouse_f1": {mouse: float},
     "compared_to": float,             # best accepted registration-only full, else Baseline
     "reason": str | None,
     "no_confident_mice": [mouse, ...],          # optional (12.12)
     "test_labels_side": {"__side_npz__": name, "sha256": hex},
                                       # npz in the run dir, key <sid> -> int32 label map
                                       # (grow15 already applied when cfg.st_grow15)
     "test_pairs": {sid: [[i, j], ...]}}
                                       # i: index into the record's iv_ids,
                                       # j: index into the new ex labels (label k -> j = k-1)

A test region absent from ``test_labels_side`` keeps its Baseline ex-vivo masks
and gets no pairs. The candidate is written only if ``accepted`` and
``full > compared_to`` (unrounded, 12.7).

Checks
------
* Each CSV is written temp-then-rename, re-read and compared with the Baseline
  (same sample IDs in the same order, identical non-``match_pairs`` strings;
  identical ``invivo_instances`` for the self-trained one), then checked with
  ``python validate_submission.py <csv>`` from the Project_Folder root (10.3).
  A failing CSV is deleted and recorded as rejected with the checker output
  (10.7).
* The SHA-256 of ``submission_v10_cpgate.csv`` and ``submission_v7_grow15.csv``
  must equal ``prep["baseline_sha256"]`` before any CSV is written and again at
  the end; a mismatch logs ``BASELINE_CSV_CHANGED``, deletes the CSVs written in
  this run and exits 1 without a done-marker (10.5).
* Unlock table per candidate and per test region: Baseline pair count,
  candidate pair count, ``newly_unlocked = baseline == 0 and candidate >= 1``
  (10.6).

Modes
-----
* Self-training disabled (``cfg.disable_selftrain``): only ``validate`` (plus the
  earlier Stages for diagnostics) is read; no ``selftrain_*`` Checkpoint is
  touched (12.2).
* ``--report-only`` (``ctx.options["report_only"]``, after
  ``BASELINE_REPRODUCTION_FAILED``): reads ``validate_failure.json`` from the run
  directory (validate has no done-marker then) and writes only the Run_Report,
  stating the reproduction failure and ``NO_CANDIDATE``; no CSV (9.4).
  ``stage.execute`` runs it without a Checkpoint or done-marker.
* Smoke (``ctx.smoke``): no ``submission_v13_*.csv``; the only CSV is
  ``smoke_candidate.csv`` in the smoke base directory (the run directory's
  parent, ``research/data/hpc_smoke/``), equal to the Baseline and checked by
  the Format_Checker. The Run_Report goes into the smoke run directory. Nothing
  is written into the Project_Folder root (13.5).

Checkpoint ``assemble.pkl``: ``{"candidates": [paths], "rejected": [{"name",
"csv", "reason"}], "report": report dict, "unlock": {name: table}}``.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hpc_unlock import checkpoint, paths, report

csv.field_size_limit(sys.maxsize)

BASELINE_CSV = report.BASELINE_CSV                 # submission_v10_cpgate.csv
GUARDED_CSVS = (report.BASELINE_CSV, report.FALLBACK_CSV)
FIELDS = ["sample_id", "invivo_instances", "exvivo_instances", "match_pairs"]
PREFIX = "submission_v13_"
SELFTRAIN_CSV = f"{PREFIX}selftrain.csv"
SMOKE_CSV = "smoke_candidate.csv"          # research/data/hpc_smoke/ (smoke only)
FAILURE_JSON = "validate_failure.json"
CHECKER = "validate_submission.py"
CHECK_TIMEOUT = 1800
OUTPUT_TAIL = 2000

Checker = Callable[[Path], tuple[bool, str]]


class CandidateError(RuntimeError):
    """A candidate cannot be written consistently (ID / pair / round-trip check)."""


class BaselineChanged(RuntimeError):
    """A guarded Baseline CSV differs from the hash recorded by ``prep``."""


# ---------------------------------------------------------------- names
def csv_name(config: str) -> str:
    """``submission_v13_<config>.csv``; ``<config>`` = ``<selection>_<gate>`` (10.4)."""
    if not config or any(c in config for c in "/\\ "):
        raise ValueError(f"invalid configuration name: {config!r}")
    return f"{PREFIX}{config}.csv"


# ---------------------------------------------------------------- Baseline I/O
def read_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """``(fieldnames, rows)`` of a submission CSV, every field as its raw string."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if fields != FIELDS:
        raise CandidateError(f"{path}: columns {fields} != {FIELDS}")
    return fields, rows


def render_rows(fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    """CSV bytes in the Baseline dialect (excel, minimal quoting, CRLF)."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(fields))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def write_csv_atomic(path: str | Path, fields: Sequence[str],
                     rows: Sequence[Mapping[str, str]]) -> str:
    """Temp-then-rename (``checkpoint._write_atomic``); returns the SHA-256."""
    data = render_rows(fields, rows)
    return checkpoint._write_atomic(Path(path), lambda f: f.write(data))


def guarded_sha256(root: str | Path) -> dict[str, str]:
    return {name: checkpoint.sha256_file(Path(root) / name) for name in GUARDED_CSVS}


def check_baseline(expected: Mapping[str, str] | None, root: str | Path) -> None:
    """Raise ``BaselineChanged`` unless every guarded CSV hash equals ``expected`` (10.5)."""
    if not expected:
        raise BaselineChanged("prep recorded no Baseline SHA-256 values")
    for name in GUARDED_CSVS:
        p = Path(root) / name
        if name not in expected:
            raise BaselineChanged(f"{name}: no SHA-256 recorded by prep")
        if not p.is_file():
            raise BaselineChanged(f"{name}: missing ({p})")
        got = checkpoint.sha256_file(p)
        if got != expected[name]:
            raise BaselineChanged(f"{name}: sha256 {got} != prep {expected[name]}")


def ex_prefix(rows: Sequence[Mapping[str, str]], default: str = "EXP") -> str:
    """Ex-vivo ID prefix of the Baseline (``EXP_000001`` -> ``EXP``)."""
    for row in rows:
        for k in json.loads(row["exvivo_instances"]):
            return str(k).rsplit("_", 1)[0] or default
    return default


def pair_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return {r["sample_id"]: len(json.loads(r["match_pairs"])) for r in rows}


# ---------------------------------------------------------------- pairs
def _check_one_to_one(sid: str, pairs: Sequence[Sequence[int]], n_iv: int, n_ex: int) -> None:
    ii = [int(p[0]) for p in pairs]
    jj = [int(p[1]) for p in pairs]
    if any(len(p) != 2 for p in pairs):
        raise CandidateError(f"{sid}: pairs must be [i, j]")
    if any(i < 0 or i >= n_iv for i in ii) or any(j < 0 or j >= n_ex for j in jj):
        raise CandidateError(f"{sid}: pair index out of range (n_iv={n_iv}, n_ex={n_ex})")
    if len(set(ii)) != len(ii) or len(set(jj)) != len(jj):
        raise CandidateError(f"{sid}: an instance appears in more than one pair")


def map_pairs(sid: str, pairs: Sequence[Sequence[int]], iv_ids: Sequence[str],
              ex_ids: Sequence[str]) -> list[list[str]]:
    """Index pairs -> ``[[iv_id, ex_id], ...]``, one-to-one and in range (10.8)."""
    pairs = [list(p) for p in pairs or []]
    _check_one_to_one(sid, pairs, len(iv_ids), len(ex_ids))
    return [[str(iv_ids[int(i)]), str(ex_ids[int(j)])] for i, j in pairs]


def _check_ids(row: Mapping[str, str], rec: Mapping) -> tuple[list[str], list[str]]:
    sid = row["sample_id"]
    iv = list(json.loads(row["invivo_instances"]))
    ex = list(json.loads(row["exvivo_instances"]))
    if rec.get("iv_ids") is None or rec.get("ex_ids") is None:
        raise CandidateError(f"{sid}: prep test record has no iv_ids / ex_ids")
    if iv != [str(x) for x in rec["iv_ids"]]:
        raise CandidateError(f"{sid}: Baseline in-vivo IDs differ from the prep record")
    if ex != [str(x) for x in rec["ex_ids"]]:
        raise CandidateError(f"{sid}: Baseline ex-vivo IDs differ from the prep record")
    return iv, ex


def _unknown_sids(test_pairs: Mapping, rows: Sequence[Mapping[str, str]]) -> list[str]:
    have = {r["sample_id"] for r in rows}
    return sorted(s for s, v in test_pairs.items() if v and s not in have)


# ---------------------------------------------------------------- builders
def registration_rows(base_rows: Sequence[Mapping[str, str]], records: Mapping[str, Mapping],
                      test_pairs: Mapping[str, Sequence]) -> list[dict[str, str]]:
    """Baseline rows with only ``match_pairs`` replaced (10.1, 10.8)."""
    bad = _unknown_sids(test_pairs, base_rows)
    if bad:
        raise CandidateError(f"test_pairs for regions not in the Baseline CSV: {bad[:5]}")
    out = []
    for row in base_rows:
        sid = row["sample_id"]
        pairs = test_pairs.get(sid) or []
        if sid in records:
            iv, ex = _check_ids(row, records[sid])
        elif pairs:
            raise CandidateError(f"{sid}: pairs given but no prep test record")
        else:
            iv = ex = []
        new = dict(row)
        new["match_pairs"] = json.dumps(map_pairs(sid, pairs, iv, ex))
        out.append(new)
    return out


def selftrain_rows(base_rows: Sequence[Mapping[str, str]], records: Mapping[str, Mapping],
                   labels: Mapping[str, Any], test_pairs: Mapping[str, Sequence],
                   prefix: str = "EXP") -> list[dict[str, str]]:
    """Baseline in-vivo, self-trained ex-vivo (``labels_to_rles``), pairs on new IDs (12.8)."""
    import numpy as np
    from cellmatch import labels_to_rles

    bad = _unknown_sids(test_pairs, base_rows)
    if bad:
        raise CandidateError(f"test_pairs for regions not in the Baseline CSV: {bad[:5]}")
    out = []
    for row in base_rows:
        sid = row["sample_id"]
        pairs = test_pairs.get(sid) or []
        new = dict(row)
        if sid not in labels:
            if pairs:
                raise CandidateError(f"{sid}: pairs given but no self-trained labels")
            new["match_pairs"] = "[]"
            out.append(new)
            continue
        if sid not in records:
            raise CandidateError(f"{sid}: self-trained labels but no prep test record")
        iv, _ = _check_ids(row, records[sid])
        lab = np.asarray(labels[sid])
        rles = labels_to_rles(lab, prefix)
        n_ex = int(lab.max()) if lab.size else 0
        ex_ids = [f"{prefix}_{k:06d}" for k in range(1, n_ex + 1)]
        pairs = [list(p) for p in pairs]
        _check_one_to_one(sid, pairs, len(iv), n_ex)
        mapped = [[iv[int(i)], ex_ids[int(j)]] for i, j in pairs]
        missing = [b for _, b in mapped if b not in rles]
        if missing:
            raise CandidateError(f"{sid}: pairs reference empty ex labels {missing[:3]}")
        new["exvivo_instances"] = json.dumps(rles)
        new["match_pairs"] = json.dumps(mapped)
        out.append(new)
    return out


# ---------------------------------------------------------------- verification
def verify_written(path: str | Path, base_rows: Sequence[Mapping[str, str]],
                   keep: Sequence[str]) -> list[dict[str, str]]:
    """Re-read ``path``; same sample IDs in order, ``keep`` columns identical (10.1, 10.2)."""
    _, rows = read_rows(path)
    if [r["sample_id"] for r in rows] != [r["sample_id"] for r in base_rows]:
        raise CandidateError(f"{path}: sample IDs differ from the Baseline")
    for got, ref in zip(rows, base_rows):
        for k in keep:
            if got[k] != ref[k]:
                raise CandidateError(f"{path}: {got['sample_id']} column {k} differs "
                                     f"from the Baseline")
    for got in rows:
        iv, ex = json.loads(got["invivo_instances"]), json.loads(got["exvivo_instances"])
        pairs = json.loads(got["match_pairs"])
        if (len({a for a, _ in pairs}) != len(pairs)
                or len({b for _, b in pairs}) != len(pairs)
                or any(a not in iv or b not in ex for a, b in pairs)):
            raise CandidateError(f"{path}: {got['sample_id']} pairs not one-to-one "
                                 f"on existing IDs")
    return rows


def format_check(path: str | Path, root: str | Path = paths.ROOT,
                 timeout: float = CHECK_TIMEOUT) -> tuple[bool, str]:
    """``python validate_submission.py <csv>`` from the Project_Folder root (10.3)."""
    try:
        proc = subprocess.run([sys.executable, CHECKER, str(Path(path).resolve())],
                              cwd=str(root), capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if len(out) > OUTPUT_TAIL:
        out = "..." + out[-OUTPUT_TAIL:]
    return proc.returncode == 0, f"rc={proc.returncode} {out}".strip()


def unlock_table(base_counts: Mapping[str, int], rows: Sequence[Mapping[str, str]]) -> list[dict]:
    """Per test region: Baseline / candidate pair counts and ``newly_unlocked`` (10.6)."""
    table = []
    for r in rows:
        sid = r["sample_id"]
        b, c = int(base_counts.get(sid, 0)), len(json.loads(r["match_pairs"]))
        table.append({"sid": sid, "baseline_pairs": b, "candidate_pairs": c,
                      "newly_unlocked": b == 0 and c >= 1})
    return table


def write_candidate(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]],
                    base_rows: Sequence[Mapping[str, str]], keep: Sequence[str],
                    checker: Checker) -> tuple[bool, str]:
    """Write, verify and Format_Check one CSV; delete it on any failure (10.3, 10.7)."""
    try:
        write_csv_atomic(path, fields, rows)
        verify_written(path, base_rows, keep)
    except CandidateError as e:
        _unlink(path)
        return False, f"round-trip check failed: {e}"
    ok, out = checker(path)
    if not ok:
        _unlink(path)
        return False, f"Format_Checker failed: {out}"
    return True, out


def _unlink(p: Path) -> None:
    try:
        Path(p).unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------- state pieces
_METRICS = ("full", "f1", "pq_iv", "pq_ex", "kept", "kept_wrong", "per_mouse_f1")


def baseline_state(base: Mapping) -> dict:
    out = {k: base.get(k) for k in _METRICS}
    out["reproduced"] = base.get("reproduced")
    if base.get("expected"):
        out["expected"] = dict(base["expected"])
    return out


def diagnostics(pose_ck: Mapping | None, joint_ck: Mapping | None,
                verifier_ck: Mapping | None, selection: str = "joint") -> dict:
    """Run_Report diagnostics (4.10, 5.10, 6.7, 7.8) from the earlier Checkpoints."""
    diag: dict = {}
    if pose_ck and pose_ck.get("diagnostics"):
        d = pose_ck["diagnostics"]
        diag["pose_search"] = {"sigma": d.get("sigma"), "n_with_gt": d.get("n_with_gt"),
                               "correct_in_candidates": d.get("correct_in_candidates"),
                               "correct_first": d.get("correct_first")}
        diag["gpu_scan"] = {"n_with_gt": d.get("n_with_gt"),
                            "correct_in_raw": d.get("correct_in_raw_gpu"),
                            "correct_in_merged": d.get("correct_in_candidates")}
    if joint_ck and joint_ck.get("diagnostics"):
        d = joint_ck["diagnostics"]
        diag["joint"] = {"total": dict(d.get("total") or {}),
                         "per_mouse": {str(m): dict(v)
                                       for m, v in (d.get("per_mouse") or {}).items()}}
    sel = ((verifier_ck or {}).get("selections") or {}).get(selection)
    if sel:
        cons, aggr = sel.get("conservative") or {}, sel.get("aggressive") or {}
        diag["verifier"] = {"conservative_threshold": cons.get("tau"),
                            "aggressive_threshold": aggr.get("tau"),
                            "conservative_unavailable": bool(cons.get("unavailable",
                                                                     cons.get("tau") is None))}
    return _plain(diag)


def _plain(o: Any) -> Any:
    """JSON-safe builtins (NumPy scalars -> float / int / bool)."""
    if isinstance(o, Mapping):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except (TypeError, ValueError):
            pass
    return o


# ---------------------------------------------------------------- main paths
def assemble(prep_ck: Mapping, validate_ck: Mapping, *, root: str | Path,
             baseline_root: str | Path = paths.ROOT, run_dir: str | Path = ".",
             selftrain: Mapping | None = None, selftrain_state: Mapping | None = None,
             diag: Mapping | None = None, run: Mapping | None = None,
             checker: Checker | None = None, write_report: bool = True,
             write_csvs: bool = True) -> dict:
    """Write candidate CSVs into ``root`` and the Run_Report; return the Checkpoint.

    ``baseline_root`` holds the guarded Baseline CSVs (read-only). ``selftrain``
    is the ``selftrain_pairs`` Checkpoint or ``None``; ``selftrain_state`` is the
    report state used when ``selftrain`` is ``None`` (disabled / not run).
    ``write_csvs=False`` (smoke) builds and checks every accepted candidate's
    rows and unlock table but writes no ``submission_v13_*.csv``.
    """
    root, baseline_root = Path(root), Path(baseline_root)
    checker = checker or (lambda p: format_check(p, paths.ROOT))
    expected_sha = prep_ck.get("baseline_sha256")
    check_baseline(expected_sha, baseline_root)                 # before any CSV

    fields, base_rows = read_rows(baseline_root / BASELINE_CSV)
    base_counts = pair_counts(base_rows)
    records = prep_ck.get("test") or {}
    base = validate_ck["baseline"]
    repro_failed = base.get("reproduced") is False
    written: list[Path] = []
    rejected: list[dict] = []
    unlock: dict[str, list] = {}
    keep_reg = [k for k in fields if k != "match_pairs"]

    configs = []
    for c in validate_ck.get("configs") or []:
        name = str(c["name"])
        entry = {"name": name, **{k: c.get(k) for k in _METRICS},
                 "accepted": bool(c.get("accepted")) and not repro_failed,
                 "csv": None, "rejected_reason": c.get("rejected_reason"),
                 "unlock_table": []}
        if entry["accepted"] and not write_csvs:
            try:
                rows = registration_rows(base_rows, records, c.get("test_pairs") or {})
                entry["unlock_table"] = unlock[name] = unlock_table(base_counts, rows)
                checkpoint.log_line("SMOKE_NO_CSV", f"{name}: accepted; rows built, "
                                                    f"no {csv_name(name)} written (smoke)")
            except CandidateError as e:
                entry["accepted"] = False
                entry["rejected_reason"] = f"rejected candidate: candidate build failed: {e}"
                rejected.append({"name": name, "csv": None, "reason": str(e)})
                checkpoint.log_line("CANDIDATE_REJECTED", f"{name}: {e}")
        elif entry["accepted"]:
            path = root / csv_name(name)
            try:
                rows = registration_rows(base_rows, records, c.get("test_pairs") or {})
                ok, out = write_candidate(path, fields, rows, base_rows, keep_reg, checker)
            except CandidateError as e:
                _unlink(path)
                ok, out = False, f"candidate build failed: {e}"
            if ok:
                written.append(path)
                entry["csv"] = path.name
                entry["unlock_table"] = unlock[name] = unlock_table(base_counts, rows)
                n_new = sum(r["newly_unlocked"] for r in unlock[name])
                checkpoint.log_line("CANDIDATE_WRITTEN",
                                    f"{name}: {path} newly_unlocked={n_new} {out}")
            else:
                entry["accepted"] = False
                entry["rejected_reason"] = f"rejected candidate: {out}"
                rejected.append({"name": name, "csv": path.name, "reason": out})
                checkpoint.log_line("CANDIDATE_REJECTED", f"{name}: {out}")
        else:
            checkpoint.log_line("NO_CANDIDATE", f"{name}: full={c.get('full')!r} "
                                                f"baseline={base.get('full')!r}")
        configs.append(entry)

    st_state = _selftrain(selftrain, selftrain_state, base_rows, records, fields,
                          base_counts, root, run_dir, checker, repro_failed,
                          written, rejected, unlock, write_csvs)

    try:
        check_baseline(expected_sha, baseline_root)             # at the end (10.5)
    except BaselineChanged:
        for p in written:
            _unlink(p)
        raise

    state = {"run": dict(run or {}), "baseline": baseline_state(base),
             "configs": configs, "selftrain": st_state, "diagnostics": dict(diag or {})}
    rep = report.build(_plain(state))
    if write_report:
        jp, mp_ = report.write(rep, root)
        checkpoint.log_line("REPORT_WRITTEN", f"{jp} {mp_} status={rep['status']}")
    return {"candidates": [str(p) for p in written], "rejected": rejected,
            "report": rep, "unlock": unlock}


def _selftrain(st: Mapping | None, st_state: Mapping | None, base_rows, records, fields,
               base_counts, root: Path, run_dir, checker, repro_failed: bool,
               written: list, rejected: list, unlock: dict,
               write_csvs: bool = True) -> dict:
    if st is None:
        return dict(st_state or {"status": "disabled"})
    state = {k: st.get(k) for k in _METRICS}
    state.update(compared_to=st.get("compared_to"), reason=st.get("reason"),
                 no_confident_mice=list(st.get("no_confident_mice") or []))
    full, target = st.get("full"), st.get("compared_to")
    accepted = (bool(st.get("accepted")) and full is not None and target is not None
                and float(full) > float(target) and not repro_failed)
    if not accepted:
        status = st.get("status")
        if status in (None, "accepted"):
            status = "no_candidate"
        state["status"] = str(status)
        checkpoint.log_line("NO_CANDIDATE", f"selftrain: full={full!r} compared_to={target!r}")
        return state
    if not write_csvs:
        state.update(status="smoke_no_csv",
                     reason=f"accepted, but no {SELFTRAIN_CSV} is written in smoke")
        checkpoint.log_line("SMOKE_NO_CSV", f"selftrain: {state['reason']}")
        return state
    path = root / SELFTRAIN_CSV
    try:
        labels = checkpoint.load_npz(st["test_labels_side"], run_dir)
        rows = selftrain_rows(base_rows, records, labels, st.get("test_pairs") or {},
                              ex_prefix(base_rows))
        ok, out = write_candidate(path, fields, rows, base_rows, ["invivo_instances"], checker)
    except (CandidateError, OSError, ValueError, KeyError) as e:
        _unlink(path)
        ok, out = False, f"candidate build failed: {type(e).__name__}: {e}"
    if not ok:
        rejected.append({"name": report.SELFTRAIN_NAME, "csv": path.name, "reason": out})
        checkpoint.log_line("CANDIDATE_REJECTED", f"selftrain: {out}")
        state.update(status="rejected", reason=f"rejected candidate: {out}")
        return state
    written.append(path)
    unlock[report.SELFTRAIN_NAME] = unlock_table(base_counts, rows)
    state.update(status="accepted", csv=path.name, unlock_table=unlock[report.SELFTRAIN_NAME])
    checkpoint.log_line("CANDIDATE_WRITTEN", f"selftrain: {path} {out}")
    return state


def smoke_rows(base_rows: Sequence[Mapping[str, str]],
               records: Mapping[str, Mapping]) -> list[dict[str, str]]:
    """Smoke candidate rows: the Baseline, with the prep test regions' pairs round-tripped.

    For every region with a prep test record the Baseline ``match_pairs`` are
    mapped to index pairs and back through ``map_pairs`` (exercising the ID
    check of a real candidate); every other row is copied. The result equals
    the Baseline (no new pairs in smoke).
    """
    out = []
    for row in base_rows:
        new = dict(row)
        sid = row["sample_id"]
        if sid in records:
            iv, ex = _check_ids(row, records[sid])
            iv_ix, ex_ix = {k: i for i, k in enumerate(iv)}, {k: j for j, k in enumerate(ex)}
            try:
                idx = [[iv_ix[a], ex_ix[b]] for a, b in json.loads(row["match_pairs"])]
            except KeyError as e:
                raise CandidateError(f"{sid}: Baseline pair references unknown ID {e}") from None
            new["match_pairs"] = json.dumps(map_pairs(sid, idx, iv, ex))
        out.append(new)
    return out


def write_smoke_candidate(path: str | Path, prep_ck: Mapping, *,
                          baseline_root: str | Path = paths.ROOT,
                          checker: Checker | None = None) -> str:
    """Write ``smoke_candidate.csv``, verify it equals the Baseline, Format_Check it.

    Returns the checker output; raises ``CandidateError`` (file deleted) on failure.
    """
    path, baseline_root = Path(path), Path(baseline_root)
    checker = checker or (lambda p: format_check(p, paths.ROOT))
    fields, base_rows = read_rows(baseline_root / BASELINE_CSV)
    rows = smoke_rows(base_rows, prep_ck.get("test") or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, out = write_candidate(path, fields, rows, base_rows, list(fields), checker)
    if not ok:
        raise CandidateError(f"smoke candidate {path}: {out}")
    return out


def report_only(run_dir: str | Path, *, root: str | Path, run: Mapping | None = None,
                diag: Mapping | None = None) -> dict:
    """Run_Report after ``BASELINE_REPRODUCTION_FAILED``: no CSV (9.4)."""
    p = Path(run_dir) / FAILURE_JSON
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found: --report-only needs the validate "
                                f"Stage's reproduction failure record")
    fail = json.loads(p.read_text())
    measured = dict(fail.get("measured") or {})
    baseline = {k: measured.get(k) for k in ("full", "f1", "pq_iv", "pq_ex",
                                             "kept", "kept_wrong")}
    baseline.update(reproduced=False,
                    expected=dict(fail.get("expected") or {"f1": 0.472, "full": 0.5186,
                                                            "tol": 0.005}))
    state = {"run": dict(run or {}), "baseline": baseline, "configs": [],
             "selftrain": {"status": "skipped",
                           "reason": "Baseline reproduction failed; no candidate CSV written"},
             "diagnostics": dict(diag or {})}
    rep = report.build(_plain(state))
    jp, mp_ = report.write(rep, root)
    checkpoint.log_line("REPORT_ONLY", f"{fail.get('status')} {jp} {mp_} "
                                       f"status={rep['status']}")
    return {"candidates": [], "rejected": [], "report": rep, "unlock": {}}


# ---------------------------------------------------------------- Stage
def _optional(ctx, name: str):
    from hpc_unlock.stage import DependencyError
    try:
        return ctx.load(name)
    except DependencyError as e:
        checkpoint.log_line("ASSEMBLE_OPTIONAL_MISSING", f"{name}: {e}")
        return None


def _selftrain_input(cfg, ctx) -> tuple[Mapping | None, dict | None]:
    """(selftrain_pairs Checkpoint, fallback state); never loads it when disabled (12.2)."""
    if getattr(cfg, "disable_selftrain", False):
        return None, {"status": "disabled"}
    from hpc_unlock.stage import DependencyError
    try:
        return ctx.load("selftrain_pairs"), None
    except DependencyError as e:
        return None, {"status": "skipped",
                      "reason": f"selftrain_pairs not finished: {e}"}


def compute(cfg, ctx) -> dict:
    run_dir = Path(ctx.run_dir)
    smoke = bool(getattr(ctx, "smoke", False))
    root = run_dir if smoke else paths.ROOT
    options = dict(getattr(ctx, "options", None) or {})
    run = {"fingerprint": cfg.fingerprint(), "smoke": smoke}
    diag = diagnostics(_optional(ctx, "pose_search"), _optional(ctx, "joint"),
                       _optional(ctx, "verifier"))
    if options.get("report_only"):
        return report_only(run_dir, root=root, run=run, diag=diag)

    prep_ck = ctx.load("prep")
    validate_ck = ctx.load("validate")
    st, st_state = _selftrain_input(cfg, ctx)
    try:
        out = assemble(prep_ck, validate_ck, root=root, baseline_root=paths.ROOT,
                       run_dir=run_dir, selftrain=st, selftrain_state=st_state,
                       diag=diag, run=run, write_csvs=not smoke)
        if smoke:
            sp = run_dir.parent / SMOKE_CSV
            res = write_smoke_candidate(sp, prep_ck, baseline_root=paths.ROOT)
            check_baseline(prep_ck.get("baseline_sha256"), paths.ROOT)
            checkpoint.log_line("SMOKE_CANDIDATE_WRITTEN", f"{sp} {res}")
            out["candidates"] = [str(sp)]
        return out
    except BaselineChanged as e:
        checkpoint.log_line("BASELINE_CSV_CHANGED", str(e))
        raise SystemExit(1) from None
