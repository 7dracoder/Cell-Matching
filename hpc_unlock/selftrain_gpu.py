"""Self-training GPU Stage: cpsam fine-tune, inference, flow decode grid (Req 12).

Inputs: only ``selftrain_prep.pkl`` and its side file ``selftrain_tiles.npz``
(12.10, 3.6). The job reads no TIFF and builds no label.

Jobs, in order ``[m1, m2, m3, test]`` (held-out mice sorted, then ``test``);
a job marked ``no_confident`` by ``selftrain_prep`` is skipped (12.11, 12.12):

1. ``model = pipeline.cellpose_model()``: a fresh model from the pretrained
   ``cpsam_v2`` weights for every job (12.4, 12.5, 12.9). The weights come from
   ``CELLPOSE_LOCAL_MODELS_PATH`` (set by ``run_in_env``).
2. ``cellpose.train.train_seg(model.net, train_data=<ex-vivo tiles>,
   train_labels=<ex-vivo pseudo-label tiles>, normalize=False, rescale=True,
   batch_size=cfg.st_batch, n_epochs=cfg.st_epochs,
   nimg_per_epoch=cfg.st_tiles_per_epoch, learning_rate=1e-5, weight_decay=0.1,
   min_train_masks=1, save_path=<run>/selftrain_models/<job>,
   model_name=f"exvivo_cpsam_selftrain_{job}")``. Only the job's
   ``tiles|<job>|image`` / ``tiles|<job>|label`` arrays are used: no in-vivo
   image and no ground-truth mask (12.4).
3. The saved model is reloaded and ``pipeline.cellpose_flows`` runs on every
   ex-vivo image of the job (all regions of mouse m, or every test region),
   then ``pipeline.cellpose_labels(flows, "exvivo", {"cellprob", "flow"})`` for
   each ``cfg.st_decode_grid`` setting (size gates 17-350 px).

Output per job, ``<run>/selftrain_flows/<job>.npz`` (``checkpoint.save_npz_atomic``)::

    "<sid>|dp"          (2, H, W) float16 flows
    "<sid>|cp"          (H, W) float16 cell probability
    "<sid>|n"           () int32 niter
    "<sid>|labels|<k>"  (H, W) uint16 label map for decode setting index k

Resume: a job whose npz exists and loads with every expected key is not
recomputed. A model whose training finished (``trained.json`` next to it) is
reused instead of retrained.

Checkpoint::

    {"jobs": {job: {"status": "done" | "skipped_no_confident" | "skipped_smoke",
                    "split", "regions", "flows_side" (ref | None), "decode_grid",
                    "train_seconds", "infer_seconds", "n_tiles", "reason",
                    "resumed", "model_reused", "model"}},
     "order": [...], "decode_grid": [...], "keys": {...}, "torch": {...},
     "params": {...}}

``flows_side`` is relative to the run directory, so
``checkpoint.load_npz(ref, run_dir)`` loads it and ``checkpoint.load`` verifies it.

Failures (12.14): any exception in a job (incl. CUDA OOM) logs
``SELFTRAIN_FAILED <job>: <Type>: <msg>`` and ``SystemExit(1)``, so no
done-marker is written. Outside smoke mode, no visible GPU logs
``NO_GPU_VISIBLE`` right after ``TORCH_INFO`` and exits 1 before anything is
loaded. Smoke mode fine-tunes nothing (13.3): every job is ``skipped_smoke``
and no flow file is written.

Imports: ``torch``, ``cellpose`` and ``pipeline`` only inside functions. This
module imports none of ``registration``, ``sklearn``, ``hpc_unlock.validate`` or
``hpc_unlock.assemble`` (3.5). ``pipeline`` itself loads the ``registration``
module when first used at run time; no registration function is called.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hpc_unlock import paths  # noqa: F401  (puts ROOT and research/ on sys.path)
from hpc_unlock import checkpoint
from hpc_unlock.checkpoint import log_line

NEEDS_GPU = True

PREP_STAGE = "selftrain_prep"
TEST_JOB = "test"
FLOWS_DIR = "selftrain_flows"
MODELS_DIR = "selftrain_models"
TRAINED_MARKER = "trained.json"
BASE_MODEL = "cpsam_v2"
MODALITY = "exvivo"
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
MIN_TRAIN_MASKS = 1
UINT16_MAX = np.iinfo(np.uint16).max
SMOKE_SKIP = "SKIPPED (no fine-tune in smoke)"
DEFAULT_KEYS = {"image": "image|<sid>", "tiles": "tiles|<job>|image",
                "labels": "tiles|<job>|label"}
OUT_KEYS = {"dp": "<sid>|dp", "cp": "<sid>|cp", "n": "<sid>|n",
            "labels": "<sid>|labels|<k>"}


# ----------------------------------------------------------------------------
# Seams (replaced in tests)
# ----------------------------------------------------------------------------

def _torch_info() -> dict:
    import torch

    avail = bool(torch.cuda.is_available())
    dev = "cpu"
    if avail:
        try:
            dev = torch.cuda.get_device_name(0)
        except Exception as e:  # noqa: BLE001 - only informational
            dev = f"unknown ({type(e).__name__})"
    return {"version": str(torch.__version__), "cuda": torch.version.cuda,
            "available": avail, "device": dev}


def _new_model(path: str | Path | None = None):
    """``pipeline.cellpose_model``: pretrained ``cpsam_v2`` (path None) or a saved fine-tune."""
    import pipeline

    return pipeline.cellpose_model(Path(path) if path else None)


def _train_seg() -> Callable[..., Any]:
    from cellpose import train

    return train.train_seg


def _flows(model, image: np.ndarray) -> tuple:
    import pipeline

    return pipeline.cellpose_flows(model, image)


def _decode(flows: tuple, cellprob: float, flow: float) -> np.ndarray:
    import pipeline

    return pipeline.cellpose_labels(flows, MODALITY, {"cellprob": float(cellprob),
                                                      "flow": float(flow)})


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

class LazySide:
    """A side ``.npz`` checked once by SHA-256, then read one key at a time."""

    def __init__(self, ref: Mapping, base: str | Path):
        checkpoint.verify_side_files(ref, base)
        self.path = Path(base) / ref[checkpoint.SIDE_KEY]
        self._z = np.load(self.path, allow_pickle=False)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._z[key]

    def __contains__(self, key: str) -> bool:
        return key in self._z.files

    def close(self) -> None:
        self._z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def decode_grid(cfg) -> list[tuple[float, float]]:
    """``cfg.st_decode_grid`` as ``[(cellprob, flow), ...]``."""
    grid = [(float(c), float(f)) for c, f in cfg.st_decode_grid]
    if not grid:
        raise ValueError("st_decode_grid is empty")
    return grid


def job_order(jobs: Mapping[str, Mapping]) -> list[str]:
    """Held-out mice (sorted), then ``test``."""
    held = sorted(j for j, v in jobs.items() if v.get("split") != "test")
    test = sorted(j for j, v in jobs.items() if v.get("split") == "test")
    return held + test


def expected_keys(regions: Sequence[str], n_decode: int) -> set[str]:
    keys = set()
    for sid in regions:
        keys |= {f"{sid}|dp", f"{sid}|cp", f"{sid}|n"}
        keys |= {f"{sid}|labels|{k}" for k in range(n_decode)}
    return keys


def flows_ref(run_dir: str | Path, job: str) -> dict:
    """Side reference for ``selftrain_flows/<job>.npz``, relative to the run directory."""
    p = Path(run_dir) / FLOWS_DIR / f"{job}.npz"
    return {checkpoint.SIDE_KEY: f"{FLOWS_DIR}/{p.name}", "sha256": checkpoint.sha256_file(p)}


def finished(path: str | Path, regions: Sequence[str], n_decode: int) -> bool:
    """True if ``path`` loads (every member readable) and has every expected key."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            if not expected_keys(regions, n_decode) <= set(z.files):
                return False
            for k in z.files:
                z[k]
    except Exception:  # noqa: BLE001 - unreadable file -> recompute
        return False
    return True


def model_paths(run_dir: str | Path, job: str) -> tuple[Path, Path, str]:
    """``(save_path, model file, model_name)``; train_seg writes ``save_path/models/name``."""
    name = f"exvivo_cpsam_selftrain_{job}"
    save = Path(run_dir) / MODELS_DIR / job
    return save, save / "models" / name, name


def _rel(p: str | Path, run_dir: str | Path) -> str:
    """``p`` relative to the run directory when inside it (the folder can move)."""
    p, root = Path(p).resolve(), Path(run_dir).resolve()
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)


def train_kwargs(cfg, save_path: Path, model_name: str) -> dict:
    return {"normalize": False, "rescale": True, "batch_size": int(cfg.st_batch),
            "n_epochs": int(cfg.st_epochs), "nimg_per_epoch": int(cfg.st_tiles_per_epoch),
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "min_train_masks": MIN_TRAIN_MASKS, "save_path": str(save_path),
            "model_name": model_name}


def to_uint16(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.size and (labels.min() < 0 or labels.max() > UINT16_MAX):
        raise ValueError(f"label IDs outside uint16 range: [{labels.min()}, {labels.max()}]")
    return labels.astype(np.uint16)


# ----------------------------------------------------------------------------
# One job
# ----------------------------------------------------------------------------

def train_job(job: str, tiles: np.ndarray, labels: np.ndarray, cfg, run_dir: Path) -> dict:
    """Fine-tune a fresh pretrained model on the job's ex-vivo tiles (or reuse a finished one)."""
    save, model_file, name = model_paths(run_dir, job)
    marker = save / TRAINED_MARKER
    if marker.is_file() and model_file.is_file():
        log_line("SELFTRAIN_MODEL_REUSE", f"{job}: {model_file}")
        return {"model": model_file, "seconds": 0.0, "reused": True}
    if tiles.shape != labels.shape or tiles.ndim != 3 or not len(tiles):
        raise ValueError(f"bad tiles {tiles.shape} / labels {labels.shape}")
    save.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    model = _new_model(None)  # fresh cpsam_v2 pretrained weights for every job
    kw = train_kwargs(cfg, save, name)
    log_line("SELFTRAIN_TRAIN", f"{job}: tiles={len(tiles)} epochs={kw['n_epochs']} "
                                f"per_epoch={kw['nimg_per_epoch']} batch={kw['batch_size']}")
    out = _train_seg()(model.net,
                       train_data=[np.ascontiguousarray(t, np.float32) for t in tiles],
                       train_labels=[np.ascontiguousarray(l, np.int32) for l in labels],
                       **kw)
    del model
    _free_gpu()
    saved = Path(out[0]) if isinstance(out, (tuple, list)) and out else model_file
    if not saved.is_file():
        raise FileNotFoundError(f"train_seg wrote no model at {saved}")
    secs = time.monotonic() - t0
    checkpoint._write_atomic(marker, lambda f: f.write(json.dumps(
        {"job": job, "model": saved.name, "seconds": secs}).encode()))
    log_line("SELFTRAIN_TRAINED", f"{job}: {saved} elapsed={secs:.1f}s")
    return {"model": saved, "seconds": secs, "reused": False}


def infer_job(job: str, model_file: Path, regions: Sequence[str], side: LazySide,
              image_key: str, grid: Sequence[tuple[float, float]]) -> tuple[dict, float]:
    """Flows + decoded labels for every region of the job."""
    t0 = time.monotonic()
    model = _new_model(model_file)
    arrays: dict[str, np.ndarray] = {}
    for sid in regions:
        t1 = time.monotonic()
        image = np.asarray(side[image_key.replace("<sid>", sid)], np.float32)
        dp, cp, n = _flows(model, image)
        dp = np.asarray(dp, np.float16)
        cp = np.asarray(cp, np.float16)
        arrays[f"{sid}|dp"] = dp
        arrays[f"{sid}|cp"] = cp
        arrays[f"{sid}|n"] = np.int32(n)
        counts = []
        for k, (c, f) in enumerate(grid):
            lab = to_uint16(_decode((dp, cp, int(n)), c, f))
            if lab.shape != image.shape:
                raise ValueError(f"{sid}: labels {lab.shape} vs image {image.shape}")
            arrays[f"{sid}|labels|{k}"] = lab
            counts.append(int(lab.max()) if lab.size else 0)
        log_line("SELFTRAIN_INFER", f"{job} sid={sid} shape={image.shape} niter={int(n)} "
                                    f"cells={counts} elapsed={time.monotonic() - t1:.1f}s")
    del model
    _free_gpu()
    return arrays, time.monotonic() - t0


def run_job(job: str, J: Mapping, side: LazySide, keys: Mapping[str, str], cfg,
            run_dir: Path, grid: Sequence[tuple[float, float]]) -> dict:
    regions = [str(s) for s in J["regions"]]
    base = {"split": J.get("split"), "regions": regions, "decode_grid": [list(g) for g in grid],
            "n_tiles": int(J.get("n_tiles", 0)), "reason": J.get("reason"),
            "flows_side": None, "train_seconds": None, "infer_seconds": None,
            "resumed": False, "model_reused": False, "model": None}
    if J.get("no_confident") or not int(J.get("n_tiles", 0)):
        reason = J.get("reason") or "no tiles"
        log_line("SELFTRAIN_SKIP", f"{job}: {reason}")
        return {**base, "status": "skipped_no_confident", "reason": reason}
    out = Path(run_dir) / FLOWS_DIR / f"{job}.npz"
    _, model_file, _ = model_paths(run_dir, job)
    if finished(out, regions, len(grid)):
        log_line("SELFTRAIN_RESUME", f"{job}: {out.name} complete, not recomputed")
        return {**base, "status": "done", "flows_side": flows_ref(run_dir, job),
                "resumed": True, "model": _rel(model_file, run_dir)}
    tiles = np.asarray(side[keys["tiles"].replace("<job>", job)], np.float32)
    labels = np.asarray(side[keys["labels"].replace("<job>", job)], np.int32)
    tr = train_job(job, tiles, labels, cfg, Path(run_dir))
    del tiles, labels
    arrays, infer_s = infer_job(job, tr["model"], regions, side, keys["image"], grid)
    ref = checkpoint.save_npz_atomic(out, **arrays)
    ref[checkpoint.SIDE_KEY] = f"{FLOWS_DIR}/{out.name}"
    del arrays
    gc.collect()
    log_line("SELFTRAIN_JOB_DONE", f"{job}: regions={len(regions)} decode={len(grid)} "
                                   f"train={tr['seconds']:.1f}s infer={infer_s:.1f}s")
    return {**base, "status": "done", "flows_side": ref, "train_seconds": float(tr["seconds"]),
            "infer_seconds": float(infer_s), "model_reused": bool(tr["reused"]),
            "model": _rel(tr["model"], run_dir)}


# ----------------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------------

def params(cfg, grid, smoke: bool) -> dict:
    return {"base_model": BASE_MODEL, "epochs": int(cfg.st_epochs),
            "tiles_per_epoch": int(cfg.st_tiles_per_epoch), "batch": int(cfg.st_batch),
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "min_train_masks": MIN_TRAIN_MASKS, "normalize": False, "rescale": True,
            "decode_grid": [list(g) for g in grid], "modality": MODALITY,
            "label_dtype": "uint16", "flow_dtype": "float16", "smoke": bool(smoke),
            "cellpose_models_path": os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")}


def compute(cfg, ctx) -> dict:
    smoke = bool(getattr(ctx, "smoke", False))
    info = _torch_info()
    log_line("TORCH_INFO", f"version={info['version']} cuda={info['cuda']} "
                           f"available={info['available']} device={info['device']}")
    if not smoke and not info["available"]:
        log_line("NO_GPU_VISIBLE", "selftrain_gpu: torch.cuda.is_available() is False; "
                                   "no fine-tune run")
        raise SystemExit(1)
    grid = decode_grid(cfg)
    prep = ctx.load(PREP_STAGE)
    jobs = prep["jobs"]
    order = job_order(jobs)
    keys = {**DEFAULT_KEYS, **(prep.get("keys") or {})}
    run_dir = Path(ctx.run_dir)
    result: dict[str, dict] = {}
    ck = {"jobs": result, "order": order, "decode_grid": [list(g) for g in grid],
          "keys": dict(OUT_KEYS), "torch": info, "params": params(cfg, grid, smoke)}

    if smoke:
        log_line("SELFTRAIN_GPU", f"{SMOKE_SKIP}: jobs={order}")
        for job in order:
            J = jobs[job]
            result[job] = {"status": "skipped_smoke", "split": J.get("split"),
                           "regions": [str(s) for s in J["regions"]], "flows_side": None,
                           "decode_grid": [list(g) for g in grid], "train_seconds": None,
                           "infer_seconds": None, "n_tiles": int(J.get("n_tiles", 0)),
                           "reason": SMOKE_SKIP, "resumed": False, "model_reused": False,
                           "model": None}
        return ck

    log_line("SELFTRAIN_GPU_START", f"jobs={order} decode={len(grid)} epochs={cfg.st_epochs} "
                                    f"per_epoch={cfg.st_tiles_per_epoch} batch={cfg.st_batch} "
                                    f"models_path={os.environ.get('CELLPOSE_LOCAL_MODELS_PATH')}")
    t0 = time.monotonic()
    with LazySide(prep["tiles_side"], run_dir) as side:
        for job in order:
            try:
                result[job] = run_job(job, jobs[job], side, keys, cfg, run_dir, grid)
            except Exception as e:  # noqa: BLE001 - incl. torch.cuda.OutOfMemoryError
                log_line("SELFTRAIN_FAILED", f"{job}: {type(e).__name__}: {e}")
                raise SystemExit(1)
    done = [j for j, v in result.items() if v["status"] == "done"]
    log_line("SELFTRAIN_GPU_DONE", f"done={done} "
                                   f"skipped={[j for j in order if j not in done]} "
                                   f"elapsed={time.monotonic() - t0:.1f}s")
    return ck
