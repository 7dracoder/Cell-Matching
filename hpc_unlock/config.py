"""Run configuration for the unlock job chain (Req 2.2, 2.3, 4.8, 5.11, 12.13).

``UnlockConfig`` holds every setting the notebook config cell exposes.
``validate()`` lists every invalid field, ``fingerprint()`` names the
Checkpoint directory ``research/data/hpc/<fingerprint>/`` and changes only
when a result-affecting field changes, and ``save()`` / ``load()`` round-trip
``run_config.json``.

Imports only the standard library, so a plain Python kernel can use it.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_CONFIG_NAME = "run_config.json"

GPU_PARTITIONS = ("g2-standard-12", "c12m85-a100-1")
A100_PARTITION = "c12m85-a100-1"
ENV_MODES = ("singularity", "venv")

SIGMA_RANGE = (1.5, 2.5)
K_RANGE = (1, 500)
RADIUS_RANGE = (1.0, 20.0)

GPU_STAGES = ("gpu_scan", "selftrain_gpu")

# Fields that change Stage results. Everything else (account, partitions,
# env mode, .sif name, resources, disable_selftrain) is excluded so that
# scheduling changes reuse existing Checkpoints.
RESULT_FIELDS = (
    "sigma",
    "scan_k",
    "scan_angle_step",
    "scan_scale_step",
    "scan_cell_px",
    "joint_lambdas",
    "joint_top_l",
    "match_radius",
    "seed_radius",
    "st_epochs",
    "st_tiles_per_epoch",
    "st_batch",
    "st_decode_grid",
    "st_grow15",
)
_FLOAT_FIELDS = ("sigma", "scan_angle_step", "scan_scale_step", "scan_cell_px",
                 "match_radius", "seed_radius")


def default_resources() -> dict[str, dict[str, Any]]:
    """Per-Stage SLURM requests (design Stage table).

    GPU Stages carry ``mem`` for the L4 partition and ``mem_a100`` for
    ``c12m85-a100-1``; see :meth:`UnlockConfig.stage_resources`.
    """
    return {
        "setup":           {"cpus": 4,  "mem": "16G", "time": "01:00:00"},
        "prep":            {"cpus": 32, "mem": "20G", "time": "01:00:00"},
        "gpu_scan":        {"cpus": 8,  "mem": "40G", "mem_a100": "64G", "time": "03:00:00"},
        "pose_search":     {"cpus": 32, "mem": "20G", "time": "03:00:00"},
        "joint":           {"cpus": 16, "mem": "16G", "time": "01:00:00"},
        "pairs":           {"cpus": 16, "mem": "16G", "time": "01:00:00"},
        "verifier":        {"cpus": 8,  "mem": "8G",  "time": "00:30:00"},
        "validate":        {"cpus": 16, "mem": "20G", "time": "01:00:00"},
        "selftrain_prep":  {"cpus": 32, "mem": "32G", "time": "01:00:00"},
        "selftrain_gpu":   {"cpus": 8,  "mem": "40G", "mem_a100": "64G", "time": "08:00:00"},
        "selftrain_pairs": {"cpus": 32, "mem": "32G", "time": "02:00:00"},
        "assemble":        {"cpus": 8,  "mem": "16G", "time": "00:30:00"},
    }


def _is_real(v: Any) -> bool:
    """A finite int or float (bool excluded)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _tupleize(v: Any) -> Any:
    """Lists (from JSON) back to tuples, recursively."""
    if isinstance(v, (list, tuple)):
        return tuple(_tupleize(x) for x in v)
    return v


@dataclass(frozen=True)
class UnlockConfig:
    # Cluster / scheduling (not result-affecting)
    account: str = "cs_gy_6923-2026fa"
    cpu_partition: str = "n2c48m24"
    gpu_partition: str = "g2-standard-12"        # or "c12m85-a100-1"
    env_mode: str = "singularity"                # or "venv"
    sif_name: str | None = None                  # None = paths.find_sif()
    disable_selftrain: bool = False
    # Soft_Score and GPU scan
    sigma: float = 2.5                           # [1.5, 2.5] px
    scan_k: int = 50                             # [1, 500]
    scan_angle_step: float = 0.5                 # degrees, <= 0.5
    scan_scale_step: float = 0.01                # <= 0.01
    scan_cell_px: float = 2.0                    # translation grid, <= 2 px
    # Joint selection
    joint_lambdas: tuple = (0.0, 2.0, 5.0, 10.0, 20.0)
    joint_top_l: int = 10
    # Self-training
    match_radius: float = 6.0                    # [1, 20] px
    seed_radius: float = 6.0                     # [1, 20] px
    st_epochs: int = 60
    st_tiles_per_epoch: int = 128
    st_batch: int = 8
    st_decode_grid: tuple = ((-0.5, 0.15), (-0.5, 0.4), (0.0, 0.15),
                             (0.0, 0.4), (0.5, 0.15), (0.5, 0.4))
    st_grow15: bool = True
    # Per-Stage cpus / mem / time; a dict, so excluded from hashing
    resources: dict = field(default_factory=default_resources, hash=False)

    # ------------------------------------------------------------------ checks
    def validate(self) -> list[str]:
        """Every error, one per invalid field, each naming the field and value."""
        errors: list[str] = []
        if self.gpu_partition not in GPU_PARTITIONS:
            errors.append(f"gpu_partition={self.gpu_partition!r} is invalid; "
                          f"expected one of {', '.join(GPU_PARTITIONS)}")
        if self.env_mode not in ENV_MODES:
            errors.append(f"env_mode={self.env_mode!r} is invalid; "
                          f"expected one of {', '.join(ENV_MODES)}")
        lo, hi = SIGMA_RANGE
        if not (_is_real(self.sigma) and lo <= self.sigma <= hi):
            errors.append(f"sigma={self.sigma!r} is invalid; "
                          f"expected a number in [{lo}, {hi}] px")
        lo, hi = K_RANGE
        if not (_is_int(self.scan_k) and lo <= self.scan_k <= hi):
            errors.append(f"scan_k={self.scan_k!r} is invalid; "
                          f"expected an integer in [{lo}, {hi}]")
        lo, hi = RADIUS_RANGE
        for name in ("match_radius", "seed_radius"):
            v = getattr(self, name)
            if not (_is_real(v) and lo <= v <= hi):
                errors.append(f"{name}={v!r} is invalid; "
                              f"expected a number in [{lo:g}, {hi:g}] px")
        return errors

    # ------------------------------------------------------------- identity
    def result_fields(self) -> dict[str, Any]:
        """Result-affecting fields in a canonical, JSON-ready form."""
        out: dict[str, Any] = {}
        for name in RESULT_FIELDS:
            v = getattr(self, name)
            if name in _FLOAT_FIELDS and _is_real(v):
                v = float(v)                     # 2 and 2.0 hash the same
            elif name == "joint_lambdas":
                v = [float(x) if _is_real(x) else x for x in v]
            elif name == "st_decode_grid":
                v = [[float(x) if _is_real(x) else x for x in pair] for pair in v]
            out[name] = v
        return out

    def fingerprint(self) -> str:
        """sha1 of the result-affecting fields, first 10 hex chars."""
        blob = json.dumps(self.result_fields(), sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]

    def run_dir(self, base: str | Path | None = None) -> Path:
        """Checkpoint directory ``<base>/<fingerprint>`` (base defaults to paths.HPC)."""
        if base is None:
            from . import paths
            base = paths.HPC
        return Path(base) / self.fingerprint()

    # ------------------------------------------------------------ resources
    def stage_resources(self, stage: str) -> dict[str, Any]:
        """``{"cpus", "mem", "time"}`` for ``stage``; GPU Stages use ``mem_a100`` on A100."""
        if stage not in self.resources:
            raise KeyError(f"no resources configured for stage {stage!r}")
        r = dict(self.resources[stage])
        a100_mem = r.pop("mem_a100", None)
        if a100_mem is not None and self.gpu_partition == A100_PARTITION:
            r["mem"] = a100_mem
        return r

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["resources"] = copy.deepcopy(self.resources)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnlockConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"unknown UnlockConfig field(s): {', '.join(unknown)}")
        kw = dict(d)
        for name in ("joint_lambdas", "st_decode_grid"):
            if name in kw:
                kw[name] = _tupleize(kw[name])
        return cls(**kw)

    def save(self, run_dir: str | Path | None = None) -> Path:
        """Write ``run_config.json`` atomically into ``run_dir`` (default ``self.run_dir()``)."""
        d = self.run_dir() if run_dir is None else Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / RUN_CONFIG_NAME
        payload = {"fingerprint": self.fingerprint(), "config": self.to_dict()}
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "UnlockConfig":
        """Read ``run_config.json`` (a file, or a run directory containing it).

        Raises ``ValueError`` if the stored fingerprint does not match the
        loaded fields.
        """
        p = Path(path)
        if p.is_dir():
            p = p / RUN_CONFIG_NAME
        with open(p, encoding="utf-8") as fh:
            payload = json.load(fh)
        cfg = cls.from_dict(payload["config"])
        stored = payload.get("fingerprint")
        if stored is not None and stored != cfg.fingerprint():
            raise ValueError(f"{p}: stored fingerprint {stored!r} does not match "
                             f"config fingerprint {cfg.fingerprint()!r}")
        return cfg
