"""Required inputs, pinned versions and the env reuse rule (Req 1.2, 1.3, 1.6).

``required_inputs(cfg)`` lists every file the Job_Chain needs before any
computation Stage runs; ``check_inputs(cfg)`` returns every one of them that
is missing or unreadable (not only the first). ``reuse_env(installed)`` decides
whether an existing environment already has the six core pins.

The module top level imports only the standard library (numpy is imported
lazily for ``.npz`` checks), so ``python -m hpc_unlock.inputs pins`` also runs
in a fresh environment that has none of the pinned packages yet.

CLI::

    python -m hpc_unlock.inputs pins      # JSON {package: installed version or null}
    python -m hpc_unlock.inputs reuse     # prints true/false; exit 0 iff reuse_env
    python -m hpc_unlock.inputs check [--env-mode M] [--sif-name N] [--root R]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable, Mapping

from . import paths

PINS_FILE = paths.ROOT / "hpc" / "unlock" / "requirements-unlock.txt"

# An existing environment is reused only if all of these match exactly (Req 1.3).
CORE_PACKAGES = ("numpy", "scipy", "opencv-python-headless", "scikit-learn",
                 "tifffile", "pandas")

OVERLAY_NAME = "overlay-15GB-500K.ext3"
SIF_PLACEHOLDER = "*.sif"   # reported when no .sif is found in container/

# research/data caches. The first eight are named in the design; the research
# modules also load reg_hough.pkl (window_lab), reg_window_vote.pkl and
# vote_cands.pkl (margin_lab), lab.pkl + heldout_labels.npz (reg_lab) at import
# time, all covered here. reg_window_vote5.pkl is the training set of the v10
# pair classifier (test_apply.train_classifier, used by test_cp_apply).
RDATA_CACHES = ("lab.pkl", "vote_cands.pkl", "cp_pose_train.pkl",
                "reg_window_vote.pkl", "reg_hough.pkl", "heldout_labels.npz",
                "test_cp_base.npz", "submission.csv", "reg_window_vote5.pkl")
BASELINE_CSVS = ("submission_v10_cpgate.csv", "submission_v7_grow15.csv")
SPLIT_CSVS = (("training", "training/train_ground_truth.csv"),
              ("hidden_test", "sample_submission.csv"))
MODALITIES = ("invivo", "exvivo")


# --------------------------------------------------------------------- pins
def read_pins(path: str | Path | None = None) -> dict[str, str]:
    """``{package: version}`` for every ``name==version`` line, in file order."""
    p = PINS_FILE if path is None else Path(path)
    pins: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep or not name.strip() or not version.strip():
            raise ValueError(f"{p}: unpinned requirement line {line!r}")
        pins[name.strip()] = version.strip()
    missing = [n for n in CORE_PACKAGES if n not in pins]
    if missing:
        raise ValueError(f"{p}: core pin(s) missing: {', '.join(missing)}")
    return pins


def miniconda_installer(path: str | Path | None = None) -> str:
    """The installer file name from the ``# MINICONDA_INSTALLER=`` line."""
    p = PINS_FILE if path is None else Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        body = line.lstrip("#").strip()
        if body.startswith("MINICONDA_INSTALLER="):
            return body.split("=", 1)[1].strip()
    raise ValueError(f"{p}: no MINICONDA_INSTALLER line")


def installed_versions(names: Iterable[str] | None = None) -> dict[str, str | None]:
    """Installed version of each package via ``importlib.metadata`` (None if absent)."""
    from importlib import metadata
    out: dict[str, str | None] = {}
    for name in (read_pins() if names is None else names):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def reuse_env(installed: Mapping[str, str | None],
              pins: Mapping[str, str] | None = None) -> bool:
    """True iff every core pin is present in ``installed`` at exactly its pinned version."""
    pins = read_pins() if pins is None else pins
    return all(installed.get(name) == pins[name] for name in CORE_PACKAGES)


# ------------------------------------------------------------------- inputs
def _sample_ids(csv_path: Path) -> list[str]:
    """``sample_id`` column of a dataset CSV; [] if it cannot be read."""
    csv.field_size_limit(sys.maxsize)
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            return [row["sample_id"] for row in csv.DictReader(fh)
                    if row.get("sample_id")]
    except (OSError, csv.Error, KeyError, UnicodeDecodeError):
        return []


def _region_dirs(data: Path, split: str, csv_rel: str) -> list[Path]:
    """Region dirs on disk plus those listed in the split's CSV (if readable)."""
    base = data / split
    found = {d for d in base.glob("subject_*/region_*") if d.is_dir()}
    for sid in _sample_ids(data / csv_rel):
        subject, _, region = sid.partition("__")
        if subject and region:
            found.add(base / subject / region)
    return sorted(found)


def _sif_path(cfg, container: Path) -> Path:
    if getattr(cfg, "sif_name", None):
        return container / cfg.sif_name
    sif = paths.find_sif(container)
    return sif if sif is not None else container / SIF_PLACEHOLDER


def required_inputs(cfg=None, root: str | Path | None = None) -> list[Path]:
    """Every input file the Job_Chain needs, in a stable order without duplicates.

    ``root`` defaults to the Project_Folder (``paths.ROOT``); tests pass a
    temporary tree. The overlay and ``.sif`` are required only in
    ``singularity`` mode.
    """
    if cfg is None:
        from .config import UnlockConfig
        cfg = UnlockConfig()
    root = paths.ROOT if root is None else Path(root)
    data = root / "Project_2_Dataset"
    rdata = root / "research" / "data"
    container = root / "hpc" / "unlock" / "container"

    req: list[Path] = [data / rel for _, rel in SPLIT_CSVS]
    for split, rel in SPLIT_CSVS:
        for d in _region_dirs(data, split, rel):
            req.extend(d / f"{m}.tif" for m in MODALITIES)
    req.extend(rdata / name for name in RDATA_CACHES)
    req.extend(root / name for name in BASELINE_CSVS)
    if cfg.env_mode == "singularity":
        req.append(container / OVERLAY_NAME)
        req.append(_sif_path(cfg, container))
    return list(dict.fromkeys(req))


def _is_container_file(path: Path) -> bool:
    return path.suffix in (".ext3", ".sif")


def check_path(path: Path) -> str | None:
    """None if ``path`` is usable, else the reason it is missing or unreadable.

    Container files: existence and readability only. ``.pkl``: full unpickle.
    ``.npz``: lazy ``np.load`` plus the member list. Anything else: read 1 byte.
    """
    if not path.is_file():
        return "missing"
    try:
        with open(path, "rb") as fh:
            if _is_container_file(path):
                return None
            if path.suffix == ".pkl":
                pickle.load(fh)
            elif path.suffix == ".npz":
                import numpy as np
                with np.load(fh, allow_pickle=False) as z:
                    if not z.files:
                        return "unreadable: npz has no arrays"
            elif not fh.read(1):
                return "unreadable: empty file"
    except Exception as exc:  # any failure to open or load counts as unreadable
        return f"unreadable: {type(exc).__name__}: {exc}"
    return None


def input_failures(cfg=None, root: str | Path | None = None) -> list[tuple[Path, str]]:
    """``(path, reason)`` for every required input that is missing or unreadable."""
    out = []
    for p in required_inputs(cfg, root):
        reason = check_path(p)
        if reason is not None:
            out.append((p, reason))
    return out


def check_inputs(cfg=None, root: str | Path | None = None) -> list[Path]:
    """Every missing or unreadable required input path (Req 1.6)."""
    return [p for p, _ in input_failures(cfg, root)]


# ---------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpc_unlock.inputs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pins", help="print installed versions of every pin as JSON")
    sub.add_parser("reuse", help="exit 0 iff the core pins match (reuse_env)")
    chk = sub.add_parser("check", help="report every missing or unreadable input")
    chk.add_argument("--env-mode", default="singularity", choices=("singularity", "venv"))
    chk.add_argument("--sif-name", default=None)
    chk.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "pins":
        print(json.dumps(installed_versions(), sort_keys=False))
        return 0
    if args.cmd == "reuse":
        ok = reuse_env(installed_versions(CORE_PACKAGES))
        print("true" if ok else "false")
        return 0 if ok else 1
    from .config import UnlockConfig
    cfg = UnlockConfig(env_mode=args.env_mode, sif_name=args.sif_name)
    req = required_inputs(cfg, args.root)
    fails = input_failures(cfg, args.root)
    for p, reason in fails:
        print(f"INPUT_FAILED {p}: {reason}", file=sys.stderr)
    if fails:
        print(f"INPUT_CHECK_FAILED {len(fails)} of {len(req)} inputs", file=sys.stderr)
        return 1
    print(f"INPUT_CHECK_OK {len(req)} inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
