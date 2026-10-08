"""Atomic Checkpoints, done-markers and the Stage runner (Req 3.7-3.11).

Layout inside a run directory ``root`` (``research/data/hpc/<fingerprint>/``):

    <name>.pkl    Checkpoint, pickle protocol 5, written temp-then-rename
    <name>.done   done-marker JSON {"stage", "sha256", "fingerprint", "finished"},
                  written atomically and only after the Checkpoint

Large arrays go into ``.npz`` side files (same temp-then-rename rule). The
Checkpoint stores a plain-dict reference ``{"__side_npz__": <file name>,
"sha256": <hex>}`` per side file, so ``load`` fails when a side file is
missing or altered and ``run_stage`` then recomputes.

Imports only the standard library at module level (NumPy is imported lazily
by the ``.npz`` helpers), so a plain Python kernel can import it.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROTOCOL = 5
SIDE_KEY = "__side_npz__"
_CHUNK = 1 << 20


def log_line(tag: str, detail: str = "") -> str:
    """One log line in the fixed ``TAG detail`` format, printed and returned."""
    line = f"{tag} {detail}".rstrip()
    print(line, flush=True)
    return line


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{os.getpid()}")


def _fsync_dir(d: Path) -> None:
    """Persist a rename on POSIX file systems; ignored where unsupported."""
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path: Path, write: Callable[[Any], None]) -> str:
    """Write via ``write(fileobj)`` to ``path.tmp-<pid>``, fsync, ``os.replace``.

    Returns the SHA-256 of the final file. On any error the temp file is
    removed and ``path`` keeps its previous content (or stays absent).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        digest = sha256_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(path.parent)
    return digest


def save_atomic(path: str | Path, obj: Any) -> str:
    """Pickle ``obj`` (protocol 5) atomically to ``path``; return its SHA-256."""
    return _write_atomic(Path(path),
                         lambda f: pickle.dump(obj, f, protocol=PROTOCOL))


def _side_refs(obj: Any):
    """Yield every side-file reference dict nested in ``obj``."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            if SIDE_KEY in o and isinstance(o.get(SIDE_KEY), str):
                yield o
            else:
                stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)


def verify_side_files(obj: Any, base: str | Path) -> None:
    """Raise if a referenced side file is missing or its SHA-256 differs."""
    base = Path(base)
    for ref in _side_refs(obj):
        p = base / ref[SIDE_KEY]
        if not p.is_file():
            raise FileNotFoundError(f"side file missing: {p}")
        got = sha256_file(p)
        if got != ref.get("sha256"):
            raise ValueError(f"side file hash mismatch: {p} "
                             f"(expected {ref.get('sha256')}, got {got})")


def load(path: str | Path, verify_sides: bool = True) -> Any:
    """Unpickle ``path``. Raises on a missing or corrupt file.

    With ``verify_sides`` every side-file reference in the loaded object is
    checked (existence + SHA-256) relative to the Checkpoint's directory.
    """
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if verify_sides:
        verify_side_files(obj, path.parent)
    return obj


def save_npz_atomic(path: str | Path, **arrays: Any) -> dict:
    """Write arrays to an uncompressed ``.npz`` atomically.

    Returns the reference dict to store in the main Checkpoint:
    ``{"__side_npz__": <file name>, "sha256": <hex>}``. The name is relative
    to the Checkpoint directory, so the project folder can move.
    """
    import numpy as np

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"side file must end in .npz: {path}")
    digest = _write_atomic(path, lambda f: np.savez(f, **arrays))
    return {SIDE_KEY: path.name, "sha256": digest}


def load_npz(ref: dict, base: str | Path) -> dict:
    """Load a side file from its reference after checking its SHA-256."""
    import numpy as np

    verify_side_files(ref, base)
    with np.load(Path(base) / ref[SIDE_KEY], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def checkpoint_path(root: str | Path, name: str) -> Path:
    return Path(root) / f"{name}.pkl"


def marker_path(root: str | Path, name: str) -> Path:
    return Path(root) / f"{name}.done"


def write_marker(root: str | Path, name: str, sha256: str,
                 fingerprint: str | None = None) -> Path:
    """Atomically write ``<name>.done`` JSON."""
    rec = {"stage": name, "sha256": sha256, "fingerprint": fingerprint,
           "finished": datetime.now().isoformat(timespec="seconds")}
    data = json.dumps(rec, sort_keys=True).encode()
    p = marker_path(root, name)
    _write_atomic(p, lambda f: f.write(data))
    return p


def read_marker(root: str | Path, name: str) -> dict:
    """Parse ``<name>.done``; raises on missing or malformed JSON."""
    rec = json.loads(marker_path(root, name).read_text())
    if not isinstance(rec, dict) or not isinstance(rec.get("sha256"), str):
        raise ValueError("marker has no sha256")
    return rec


def _try_reuse(root: Path, name: str) -> tuple[bool, Any, str]:
    """(ok, obj, reason). ``ok`` only if marker hash matches and load succeeds."""
    ckpt = checkpoint_path(root, name)
    try:
        rec = read_marker(root, name)
    except Exception as e:  # noqa: BLE001 - any marker problem is an inconsistency
        return False, None, f"unreadable marker ({type(e).__name__}: {e})"
    if not ckpt.is_file():
        return False, None, f"checkpoint missing ({ckpt.name})"
    got = sha256_file(ckpt)
    if got != rec["sha256"]:
        return False, None, f"sha256 mismatch (marker {rec['sha256'][:12]}, file {got[:12]})"
    try:
        obj = load(ckpt)
    except Exception as e:  # noqa: BLE001
        return False, None, f"load failed ({type(e).__name__}: {e})"
    return True, obj, ""


def run_stage(name: str, compute: Callable[[], Any], root: str | Path,
              fingerprint: str | None = None) -> Any:
    """Reuse a finished Stage or compute it, save it and mark it done.

    1. Marker + Checkpoint present, hash matches and load succeeds:
       ``REUSE <name>`` and return the loaded object (Req 3.7).
    2. Marker present otherwise: ``INCONSISTENT <name>: <reason>``, delete the
       marker, then recompute (Req 3.8).
    3. ``compute()``, ``save_atomic`` the Checkpoint, then write the marker
       atomically (Req 3.9). Any exit before the marker write leaves no
       marker (Req 3.10).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    marker = marker_path(root, name)
    if marker.exists():
        ok, obj, reason = _try_reuse(root, name)
        if ok:
            log_line("REUSE", name)
            return obj
        log_line("INCONSISTENT", f"{name}: {reason}")
        marker.unlink()

    t0 = time.monotonic()
    log_line("COMPUTE", f"{name} fingerprint={fingerprint}")
    try:
        obj = compute()
    except BaseException as e:
        log_line("STAGE_FAILED", f"{name}: {type(e).__name__}: {e}")
        raise
    digest = save_atomic(checkpoint_path(root, name), obj)
    log_line("SAVED", f"{name} sha256={digest}")
    write_marker(root, name, digest, fingerprint)
    log_line("DONE", f"{name} elapsed={time.monotonic() - t0:.1f}s")
    return obj
